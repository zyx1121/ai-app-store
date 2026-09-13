//! The model manager: one `llama-server` process per Model, shared by lease.
//!
//! The platform owns every inference parameter (decision D6). Apps declare a
//! model and a quant preference and get a URL back, nothing else.
//!
//! State lives in one process global [`Manager`]. Every mutation takes the
//! registry lock for as short as possible and never holds it across an await;
//! starts and stops are serialized by a separate async gate so two apps asking
//! for the same model at the same moment produce one process, not two.

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Child;

use rand::RngExt as _;

use crate::error::{Error, Result};
use crate::hardware::DeviceProfile;
use crate::models::{self, ModelRef};
use crate::paths;
use crate::runtime;

/// First port of the platform range.
pub const PORT_RANGE_START: u16 = 41000;
/// Last port of the platform range.
pub const PORT_RANGE_END: u16 = 41999;
/// Instances take the lower half, apps take the upper half.
pub const INSTANCE_PORT_END: u16 = 41499;

/// How long a zero lease instance stays warm, overridden by `AIAS_IDLE_SECS`.
pub const DEFAULT_IDLE_SECS: u64 = 600;

/// How often the idle reaper looks for an instance whose window has passed.
///
/// The reaper never sleeps longer than this and never longer than a quarter of
/// the window itself, so a short `AIAS_IDLE_SECS` is honoured by the same task
/// the default 600 s runs on.
const REAP_INTERVAL: Duration = Duration::from_secs(30);
/// Floor of the reaper's tick, so no window can make it spin.
const REAP_INTERVAL_MIN: Duration = Duration::from_millis(100);

/// A cold start loads the weights from disk, so the ceiling is generous.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(600);
/// Gap between two `/health` polls while an instance is loading.
const HEALTH_POLL: Duration = Duration::from_millis(500);
/// One `/health` poll gives up quickly, the loop is what waits.
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
/// Bytes of the log read back when a process dies before it is healthy.
const LOG_TAIL_BYTES: u64 = 16 * 1024;
/// Lines of that tail put in the error message.
const LOG_TAIL_LINES: usize = 25;

/// Bytes of entropy in the key that guards one instance, hex encoded to 64 chars.
const API_KEY_BYTES: usize = 32;

/// Substrings that mean `llama-server` ran out of memory while loading.
const OOM_MARKERS: &[&str] = &["out of memory", "cudaMalloc", "failed to allocate"];

/// One app holding one model open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lease {
    /// Manifest `name` of the app holding the lease.
    pub app: String,
    pub model: ModelRef,
    /// Unix seconds the lease was taken.
    pub acquired_at: u64,
}

/// One running `llama-server` process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Instance {
    pub model: ModelRef,
    pub port: u16,
    /// `None` while the process is starting or after it exited.
    pub pid: Option<u32>,
    /// Apps currently holding a lease. Empty means the instance is idle.
    pub leases: Vec<String>,
    pub params: Params,
    /// Unix seconds of the last acquire or release, the LRU key for eviction.
    #[serde(default)]
    pub last_used: u64,
    /// Memory this instance is planned to hold, weights plus KV cache.
    #[serde(default)]
    pub estimate_mb: u64,
    /// Unix seconds the last lease was dropped, `None` while one is held.
    ///
    /// This is what the idle reaper measures the warm window against, and what
    /// lets the Models page say how long an instance nobody is using has left.
    /// A new acquire clears it, so a window is only ever counted from the
    /// release that emptied the lease list.
    #[serde(default)]
    pub idle_since: Option<u64>,
    /// Bearer token `llama-server` was started with. Every request needs it.
    ///
    /// Never serialized. An `Instance` is what `instances_list` and
    /// `aias apps run` print, so a key in this struct was a key on a terminal
    /// and in a renderer that has no use for it. The token reaches the two
    /// callers that need it in Rust through [`Instance::api_key`], and reaches
    /// the frontend only through [`InstanceUrl`], which is handed out one
    /// acquire at a time.
    #[serde(default, skip_serializing)]
    api_key: String,
}

impl Instance {
    /// Bearer token of this instance, for the app environment and for the
    /// [`InstanceUrl`] an acquire hands back. Not part of the JSON.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// What an app is handed for one model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstanceUrl {
    /// OpenAI compatible base URL, ends in `/v1`.
    pub base_url: String,
    /// Value to put in the `model` field of the request body.
    pub model_id: String,
    pub port: u16,
    /// Bearer token for this instance, reached by an app as
    /// `AIAS_MODEL_<ALIAS>_KEY`.
    pub api_key: String,
}

/// The parameters the platform picks for an instance. Apps cannot set these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Params {
    /// Context size, `--ctx-size`.
    pub ctx: u32,
    /// Parallel slots, `--parallel`.
    pub n_parallel: u32,
    /// Layers offloaded to the GPU, `--n-gpu-layers`.
    pub ngl: u32,
}

/// The GGUF files of one installed model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    /// The weights.
    pub gguf: PathBuf,
    /// The vision projector, present for `kind: vlm` repos.
    pub mmproj: Option<PathBuf>,
    pub size_bytes: u64,
}

/// Everything the model manager owns on this machine.
#[derive(Default)]
struct Manager {
    /// Cached device profile, detection runs once per process.
    profile: Option<DeviceProfile>,
    running: Vec<Running>,
}

/// One entry of the registry: the snapshot plus what cannot be serialized.
struct Running {
    instance: Instance,
    child: Option<Child>,
    log_path: PathBuf,
}

static MANAGER: OnceLock<Mutex<Manager>> = OnceLock::new();

/// Serializes the start and stop side of the manager across tasks.
static GATE: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn manager() -> MutexGuard<'static, Manager> {
    MANAGER
        .get_or_init(|| Mutex::new(Manager::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn gate() -> &'static tokio::sync::Mutex<()> {
    GATE.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// What [`spawn_server`] is asked to start.
pub(crate) struct SpawnRequest<'a> {
    model_id: &'a str,
    files: &'a InstalledModel,
    port: u16,
    params: Params,
    api_key: &'a str,
    log_path: &'a Path,
    data_dir: &'a Path,
}

/// Swapped out in tests so the lease logic can be exercised without a GPU.
pub(crate) type SpawnHook = fn(&SpawnRequest<'_>) -> Result<Option<Child>>;
static SPAWN_HOOK: Mutex<Option<SpawnHook>> = Mutex::new(None);

/// Start the instance if needed and take a lease for `app`.
pub async fn acquire(app: &str, model: &ModelRef) -> Result<InstanceUrl> {
    let data_dir = paths::data_dir();
    acquire_in(app, model, &data_dir).await
}

/// [`acquire`] against an explicit data directory.
pub(crate) async fn acquire_in(
    app: &str,
    model: &ModelRef,
    data_dir: &Path,
) -> Result<InstanceUrl> {
    // 0. The reaper is spawned here and nowhere else: this is the first moment
    // there is anything to reap, and the first moment this code is on the tokio
    // runtime the desktop shell and `aias api serve` both own.
    start_reaper();

    // 1. The weights have to be on disk already. Downloading is the caller's job.
    let files = installed(model, data_dir)?;

    let _gate = gate().lock().await;

    // 2. A healthy instance is shared, that is the whole point of the lease.
    if let Some((port, api_key)) = registered_instance(model) {
        if health(port).await {
            add_lease(model, app);
            return Ok(url_for(model, port, &api_key));
        }
        // It died on us. Drop it and start a fresh one below.
        forget(model).await;
    }

    // 3. Plan the parameters and make the memory fit. The header is read here
    // and nowhere else: the context and the admission check have to be priced
    // against the same cache, or one of them is guessing.
    let profile = profile_cached()?;
    let layout = models::kv_layout(&files.gguf);
    let params = params_for(&profile, files.size_bytes, layout);
    let estimate_mb = models::estimate_mb(files.size_bytes, params.ctx, params.n_parallel, layout);
    make_room(model, estimate_mb, &profile).await?;

    // 4. Start the server and wait for it to answer /health.
    let port = pick_port(model)?;
    let model_id = model_id(model);
    // One key per instance, never reused across a restart: the port is
    // predictable by design, so the key is the only thing that keeps another
    // local process off the model.
    let api_key = new_api_key();
    // `runtime::spawn_server` writes here, one file per port.
    std::fs::create_dir_all(paths::logs_dir(data_dir))?;
    let log_path = paths::logs_dir(data_dir).join(format!("llama-{port}.log"));

    let mut child = spawn_server(&SpawnRequest {
        model_id: &model_id,
        files: &files,
        port,
        params,
        api_key: &api_key,
        log_path: &log_path,
        data_dir,
    })
    .await?;

    if let Err(err) = wait_healthy(&mut child, port, &model_id, estimate_mb, &log_path).await {
        if let Some(child) = child.as_mut() {
            let _ = child.start_kill();
        }
        return Err(err);
    }

    // 5. Record it and hand the app its URL.
    let now = now_secs();
    let instance = Instance {
        model: model.clone(),
        port,
        pid: child.as_ref().and_then(Child::id),
        leases: vec![app.to_string()],
        params,
        last_used: now,
        estimate_mb,
        idle_since: None,
        api_key: api_key.clone(),
    };
    manager().running.push(Running {
        instance,
        child,
        log_path,
    });

    Ok(url_for(model, port, &api_key))
}

/// Drop `app`'s lease. An instance at zero leases stays warm for the idle window.
///
/// The window is not slept on here. A release used to spawn a task that slept
/// the whole window and then looked, which meant the stop only happened if that
/// one task survived: a release from a CLI process that exits, or a second
/// release that moved `last_used`, left a zero lease `llama-server` holding its
/// memory for the rest of the session (issue #17). All this does now is stamp
/// `idle_since`; [`reap_idle`] is what stops it, and it runs for as long as the
/// process that owns the instance does.
pub async fn release(app: &str, model: &ModelRef) -> Result<()> {
    let now = now_secs();
    {
        let mut manager = manager();
        let Some(running) = manager
            .running
            .iter_mut()
            .find(|running| &running.instance.model == model)
        else {
            // Releasing what was never acquired is not an error, a crashed app
            // releases on process exit and may race with a stop.
            return Ok(());
        };
        if let Some(at) = running
            .instance
            .leases
            .iter()
            .position(|holder| holder == app)
        {
            running.instance.leases.remove(at);
        }
        running.instance.last_used = now;
        if running.instance.leases.is_empty() {
            // Only the release that empties the list starts the window. A
            // second release must not push the stop further out.
            running.instance.idle_since.get_or_insert(now);
        }
    }
    Ok(())
}

/// Snapshot of the instance registry, leases included.
pub fn list() -> Vec<Instance> {
    manager()
        .running
        .iter()
        .map(|running| running.instance.clone())
        .collect()
}

/// Every running app process is gone, so stop every instance. Used at shutdown.
pub async fn stop_all() -> Result<()> {
    let _gate = gate().lock().await;
    // Take the registry first, the lock must not be held across the wait.
    let running = {
        let mut manager = manager();
        std::mem::take(&mut manager.running)
    };
    for running in running {
        // Shutdown is only finished when every process is really gone.
        shutdown(running).await;
    }
    Ok(())
}

/// True when `llama-server` on this port reports itself healthy right now.
pub async fn health(port: u16) -> bool {
    runtime::wait_healthy(port, HEALTH_REQUEST_TIMEOUT)
        .await
        .is_ok()
}

/// The GGUF files of `model`, or [`Error::NotFound`] naming what to download.
///
/// The downloader is the authority: [`models::installed`] reads the sidecar
/// every finished download leaves behind, so a half written directory is not
/// mistaken for an installed model.
pub fn installed(model: &ModelRef, data_dir: &Path) -> Result<InstalledModel> {
    let found = models::installed(data_dir)
        .into_iter()
        .find(|(candidate, _, _)| candidate == model);

    let Some((_, gguf, size_bytes)) = found else {
        return Err(Error::NotFound(format!(
            "model {} is not downloaded, expected it in {}",
            model_id(model),
            models::model_dir(model, data_dir).display()
        )));
    };

    Ok(InstalledModel {
        gguf,
        mmproj: models::installed_mmproj(model, data_dir),
        size_bytes,
    })
}

/// Where a running instance writes its stdout and stderr.
pub fn log_path(model: &ModelRef) -> Option<PathBuf> {
    manager()
        .running
        .iter()
        .find(|running| &running.instance.model == model)
        .map(|running| running.log_path.clone())
}

/// How long a zero lease instance stays warm, in seconds.
///
/// `None` is `AIAS_IDLE_SECS=0`, which turns the reaper off: a zero lease
/// instance then stays warm until something else stops it, which is what a
/// developer watching one model wants and what PLAN.md section 5 documents.
/// A value that is not a number of seconds is a typo and not a policy, so it
/// says so once and the default window stands.
pub fn idle_secs() -> Option<u64> {
    let Ok(value) = std::env::var("AIAS_IDLE_SECS") else {
        return Some(DEFAULT_IDLE_SECS);
    };
    match value.trim().parse::<u64>() {
        Ok(0) => None,
        Ok(secs) => Some(secs),
        Err(_) => {
            warn_once_about(&value);
            Some(DEFAULT_IDLE_SECS)
        }
    }
}

/// Say once that `AIAS_IDLE_SECS` is not a number, whatever reads it.
///
/// The reaper reads the window on every tick, so without the latch this is a
/// line every 30 seconds for the life of the process.
fn warn_once_about(value: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if WARNED.swap(true, Ordering::SeqCst) {
        return;
    }
    eprintln!(
        "AIAS_IDLE_SECS={value} is not a number of seconds, keeping the default {DEFAULT_IDLE_SECS} s idle window (0 turns the idle reaper off)"
    );
}

/// Stable port for a model, so URLs in app logs stay meaningful across restarts.
pub fn port_for(model: &ModelRef) -> u16 {
    let span = u64::from(INSTANCE_PORT_END - PORT_RANGE_START + 1);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in model.slug().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    PORT_RANGE_START + u16::try_from(hash % span).unwrap_or(0)
}

/// Pick context size, slots and GPU layers for one model on one device.
///
/// The context used to come off a ladder of leftover megabytes that never
/// priced the cache it was asking for, so a model that fitted on its own was
/// started with a context several times its own size and the server died
/// loading. [`models::plan_context`] now walks the ladder and stops at the
/// first rung whose whole estimate, weights and cache together, fits 90% of the
/// budget. `layout` is the GGUF header when the file is on disk; without it the
/// cache is estimated from the file size alone.
pub fn params_for(
    profile: &DeviceProfile,
    model_size_bytes: u64,
    layout: Option<models::KvLayout>,
) -> Params {
    let (ctx, n_parallel) =
        models::plan_context(model_size_bytes, profile.effective_memory_mb(), layout);

    // llama.cpp clamps an over large layer count to the layers the model has.
    let ngl = if profile.has_gpu() { 999 } else { 0 };

    Params {
        ctx,
        n_parallel,
        ngl,
    }
}

/// `<repo>:<quant>`, the value an app puts in the `model` field.
pub fn model_id(model: &ModelRef) -> String {
    format!("{}:{}", model.repo, model.quant)
}

fn url_for(model: &ModelRef, port: u16, api_key: &str) -> InstanceUrl {
    InstanceUrl {
        base_url: format!("http://127.0.0.1:{port}/v1"),
        model_id: model_id(model),
        port,
        api_key: api_key.to_string(),
    }
}

/// A fresh bearer token for one instance: 32 random bytes, hex encoded.
pub fn new_api_key() -> String {
    let mut bytes = [0u8; API_KEY_BYTES];
    rand::rng().fill(&mut bytes);
    let key = hex::encode(bytes);
    remember_secret(&key);
    key
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

fn registered_instance(model: &ModelRef) -> Option<(u16, String)> {
    manager()
        .running
        .iter()
        .find(|running| &running.instance.model == model)
        .map(|running| {
            (
                running.instance.port,
                running.instance.api_key().to_string(),
            )
        })
}

fn add_lease(model: &ModelRef, app: &str) {
    let now = now_secs();
    let mut manager = manager();
    if let Some(running) = manager
        .running
        .iter_mut()
        .find(|running| &running.instance.model == model)
    {
        running.instance.leases.push(app.to_string());
        running.instance.last_used = now;
        // The window only ever runs while nobody holds a lease.
        running.instance.idle_since = None;
    }
}

/// Take an instance out of the registry without touching its process.
fn take(model: &ModelRef) -> Option<Running> {
    let mut manager = manager();
    let at = manager
        .running
        .iter()
        .position(|running| &running.instance.model == model)?;
    Some(manager.running.remove(at))
}

/// Remove an instance from the registry and wait for its process to be gone.
///
/// The wait is the point: the memory an evicted instance holds is only free
/// once the process exits, so the next `llama-server` must not be started
/// before this resolves.
async fn forget(model: &ModelRef) {
    if let Some(running) = take(model) {
        shutdown(running).await;
    }
}

/// Kill one instance and await its exit.
async fn shutdown(mut running: Running) {
    kill(&mut running);
    if let Some(child) = running.child.as_mut() {
        let _ = child.wait().await;
    }
}

/// Ask the process to die. The caller awaits the exit, this only signals it.
fn kill(running: &mut Running) {
    if let Some(child) = running.child.as_mut() {
        let _ = child.start_kill();
    }
    running.instance.pid = None;
}

/// True once the idle reaper is running on this process's runtime.
static REAPER: AtomicBool = AtomicBool::new(false);

/// Start the idle reaper, once per process.
///
/// PLAN.md section 2.4: an instance with zero leases is kept warm for an idle
/// window and then stopped. This is the task that does the stopping. It is
/// spawned from the first [`acquire_in`], so it lives on whichever tokio
/// runtime owns the manager, the desktop shell's or `aias api serve`'s, and it
/// outlives the request that started it. `AIAS_IDLE_SECS=0` leaves the task
/// running and stops it from ever stopping an instance, which is one branch
/// rather than a task that cannot be brought back.
fn start_reaper() {
    if REAPER.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async move {
        loop {
            // The window is read on every tick, so a process that starts with
            // the reaper off still answers a window set later.
            let window = idle_secs();
            tokio::time::sleep(window.map_or(REAP_INTERVAL, reap_interval_for)).await;
            if window.is_some() {
                reap_idle().await;
            }
        }
    });
}

/// Gap between two reaper passes, for the window `AIAS_IDLE_SECS` asks for.
fn reap_interval_for(window_secs: u64) -> Duration {
    REAP_INTERVAL
        .min(Duration::from_secs(window_secs) / 4)
        .max(REAP_INTERVAL_MIN)
}

/// Stop every instance whose idle window passed without a new lease.
///
/// The port is freed by the same call: [`forget`] takes the entry out of the
/// registry and awaits the child's exit, so the memory and the port are both
/// back before this returns.
async fn reap_idle() {
    // `AIAS_IDLE_SECS=0` turns the reaper off: nothing here stops an instance.
    let Some(window) = idle_secs() else {
        return;
    };
    let now = now_secs();
    let expired: Vec<ModelRef> = manager()
        .running
        .iter()
        .filter(|running| expired_at(&running.instance, now, window))
        .map(|running| running.instance.model.clone())
        .collect();
    if expired.is_empty() {
        return;
    }

    // The same gate every start and stop takes, so a reap cannot land in the
    // middle of an acquire that is still waiting for `/health` and has not
    // taken its lease yet.
    let _gate = gate().lock().await;
    for model in expired {
        // Read again under the lock: an acquire may have taken a lease while
        // this task was waiting for the gate.
        let now = now_secs();
        let still_idle = manager()
            .running
            .iter()
            .find(|running| running.instance.model == model)
            .is_some_and(|running| expired_at(&running.instance, now, window));
        if still_idle {
            forget(&model).await;
        }
    }
}

/// True when this instance has held no lease for the whole idle window.
fn expired_at(instance: &Instance, now: u64, window: u64) -> bool {
    instance.leases.is_empty()
        && instance
            .idle_since
            .is_some_and(|since| now.saturating_sub(since) >= window)
}

/// The device profile, detected once per process.
fn profile_cached() -> Result<DeviceProfile> {
    if let Some(profile) = manager().profile.clone() {
        return Ok(profile);
    }
    let profile = crate::hardware::detect()?;
    manager().profile = Some(profile.clone());
    Ok(profile)
}

/// Keep the sum of the estimates under the 90% ceiling, evicting LRU first.
async fn make_room(model: &ModelRef, estimate_mb: u64, profile: &DeviceProfile) -> Result<()> {
    let ceiling_mb = profile.usable_memory_mb();

    loop {
        let (held_mb, evictable) = {
            let manager = manager();
            let held: u64 = manager
                .running
                .iter()
                .map(|running| running.instance.estimate_mb)
                .sum();
            // Least recently used zero lease instance goes first.
            let evictable = manager
                .running
                .iter()
                .filter(|running| running.instance.leases.is_empty())
                .min_by_key(|running| running.instance.last_used)
                .map(|running| running.instance.model.clone());
            (held, evictable)
        };

        if held_mb + estimate_mb <= ceiling_mb {
            return Ok(());
        }

        match evictable {
            Some(idle) => forget(&idle).await,
            None => {
                let holders: Vec<String> = manager()
                    .running
                    .iter()
                    .map(|running| {
                        format!(
                            "{} ({} MB, held by {})",
                            model_id(&running.instance.model),
                            running.instance.estimate_mb,
                            running.instance.leases.join(", ")
                        )
                    })
                    .collect();
                return Err(Error::UnsupportedHardware(format!(
                    "{} needs about {estimate_mb} MB, only {} MB of the {ceiling_mb} MB budget is free; holding memory: {}",
                    model_id(model),
                    ceiling_mb.saturating_sub(held_mb),
                    if holders.is_empty() {
                        "nothing".to_string()
                    } else {
                        holders.join("; ")
                    }
                )));
            }
        }
    }
}

/// The model's stable port when it is free, otherwise the next free one.
fn pick_port(model: &ModelRef) -> Result<u16> {
    let preferred = port_for(model);
    if is_port_free(preferred) {
        return Ok(preferred);
    }
    for port in PORT_RANGE_START..=INSTANCE_PORT_END {
        if is_port_free(port) {
            return Ok(port);
        }
    }
    Err(Error::Process(format!(
        "no free port in {PORT_RANGE_START}-{INSTANCE_PORT_END} for {}",
        model_id(model)
    )))
}

fn is_port_free(port: u16) -> bool {
    let taken = manager()
        .running
        .iter()
        .any(|running| running.instance.port == port);
    if taken {
        return false;
    }
    TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).is_ok()
}

async fn spawn_server(request: &SpawnRequest<'_>) -> Result<Option<Child>> {
    let hook = *SPAWN_HOOK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(hook) = hook {
        return hook(request);
    }

    // The runtime module owns the executable, the argv and the log file. The
    // log is truncated first so a tail read after a crash is this run only.
    let install = runtime::installed(request.data_dir).ok_or_else(|| {
        Error::NotFound(format!(
            "no llama-server installed under {}, run `aias runtime install` first",
            paths::runtime_dir(request.data_dir).display()
        ))
    })?;
    let _ = std::fs::File::create(request.log_path);

    // A vision model is started with its projector, otherwise the same weights
    // would answer text only.
    let child = runtime::spawn_server(
        &install,
        &request.files.gguf,
        request.files.mmproj.as_deref(),
        request.port,
        &request.params,
        request.model_id,
        request.api_key,
    )
    .await?;
    Ok(Some(child))
}

/// Wait for the runtime health check, and fail early when the process dies.
///
/// `runtime::wait_healthy` owns the protocol; this races it against the child
/// so an out of memory exit is reported in seconds instead of ten minutes.
async fn wait_healthy(
    child: &mut Option<Child>,
    port: u16,
    model_id: &str,
    estimate_mb: u64,
    log_path: &Path,
) -> Result<()> {
    tokio::select! {
        health = runtime::wait_healthy(port, HEALTH_TIMEOUT) => health.map_err(|err| {
            Error::Process(format!(
                "{model_id}: {err}; log tail:\n{}",
                log_tail(log_path, LOG_TAIL_LINES)
            ))
        }),
        err = watch_for_exit(child, model_id, estimate_mb, log_path) => Err(err),
    }
}

/// Resolves only when the child exits, with the verdict read from its log.
async fn watch_for_exit(
    child: &mut Option<Child>,
    model_id: &str,
    estimate_mb: u64,
    log_path: &Path,
) -> Error {
    loop {
        if let Some(child) = child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let tail = log_tail(log_path, LOG_TAIL_LINES);
                    return Error::Process(if looks_like_oom(&tail) {
                        format!(
                            "{model_id} ran out of memory while loading, it needs about {estimate_mb} MB; log tail:\n{tail}"
                        )
                    } else {
                        format!(
                            "{model_id} exited with {status} before it was healthy; log tail:\n{tail}"
                        )
                    });
                }
                Ok(None) => {}
                Err(err) => return Error::Io(err),
            }
        }
        tokio::time::sleep(HEALTH_POLL).await;
    }
}

/// True when a log tail says the server died for lack of memory.
fn looks_like_oom(tail: &str) -> bool {
    let lowered = tail.to_lowercase();
    OOM_MARKERS
        .iter()
        .any(|marker| lowered.contains(&marker.to_lowercase()))
}

/// Status code of a loopback `GET`. Only the status line is read.
pub(crate) async fn http_status(port: u16, path: &str, timeout: Duration) -> Result<u16> {
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );

    let exchange = async move {
        let mut stream = TcpStream::connect(address).await?;
        stream.write_all(request.as_bytes()).await?;
        let mut head = vec![0u8; 256];
        let mut filled = 0;
        while filled < head.len() {
            let read = stream.read(&mut head[filled..]).await?;
            if read == 0 {
                break;
            }
            filled += read;
            if head[..filled].contains(&b'\n') {
                break;
            }
        }
        head.truncate(filled);
        Ok::<Vec<u8>, std::io::Error>(head)
    };

    let head = tokio::time::timeout(timeout, exchange)
        .await
        .map_err(|_| {
            Error::Process(format!(
                "127.0.0.1:{port}{path} did not answer within {} s",
                timeout.as_secs()
            ))
        })?
        .map_err(|err| Error::Process(format!("127.0.0.1:{port}{path}: {err}")))?;

    parse_status(&head).ok_or_else(|| {
        Error::Process(format!(
            "127.0.0.1:{port}{path} did not answer with an HTTP status line"
        ))
    })
}

fn parse_status(head: &[u8]) -> Option<u16> {
    let line = String::from_utf8_lossy(head);
    line.lines().next()?.split_whitespace().nth(1)?.parse().ok()
}

/// Last `lines` lines of a log file, with the platform's own secrets taken out.
///
/// This is the one door every log tail leaves the core through: the apps page,
/// the CLI and the tails embedded in build and start errors all come from here,
/// so it is where [`mask_secrets`] belongs. Masking in the frontend only would
/// leave the CLI printing the values and would put the rule in the layer that is
/// easiest to bypass.
pub(crate) fn log_tail(path: &Path, lines: usize) -> String {
    let Ok(mut file) = std::fs::File::open(path) else {
        return String::new();
    };
    let length = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let from = length.saturating_sub(LOG_TAIL_BYTES);
    if file.seek(SeekFrom::Start(from)).is_err() {
        return String::new();
    }
    let mut buffer = Vec::new();
    if file.read_to_end(&mut buffer).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&buffer);
    let tail: Vec<&str> = text.lines().rev().take(lines).collect();
    mask_secrets(&tail.into_iter().rev().collect::<Vec<&str>>().join("\n"))
}

/// What is written over a masked value.
const MASK: &str = "***";
/// The suffix of every variable this platform hands an app a secret in:
/// `AIAS_MODEL_<ALIAS>_KEY` and `LLAMA_API_KEY`.
const SECRET_SUFFIX: &str = "_KEY=";
/// The scheme of the one header these secrets are sent in.
const BEARER: &str = "bearer ";
/// A value shorter than this is not masked by value: a short string appears in
/// ordinary text and blanking it would damage the log for nothing.
const MIN_SECRET_LEN: usize = 12;

/// Every value this platform generated and would rather not read back.
///
/// Shapes catch an environment dump, which is how most of these reach a log,
/// but not an app that prints the header it sent or a URL it built. The values
/// are known here because this process made them, so they are blanked wherever
/// they appear. The set is in process and empty at start, the same life a
/// running Instance has (decision D10): a secret nobody generated this run is
/// a secret no log of this run can be printing.
static KNOWN_SECRETS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Add one value to what [`mask_secrets`] blanks wherever it appears.
///
/// Called where a secret is generated or read back: an Instance key, the local
/// API bearer token, a Postgres password.
pub(crate) fn remember_secret(value: &str) {
    let value = value.trim();
    if value.len() < MIN_SECRET_LEN {
        return;
    }
    KNOWN_SECRETS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(value.to_string());
}

/// The remembered values, longest first so a prefix never hides a longer one.
fn known_secrets() -> Vec<String> {
    let mut values: Vec<String> = KNOWN_SECRETS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .cloned()
        .collect();
    values.sort_by_key(|value| std::cmp::Reverse(value.len()));
    values
}

/// Blank the platform's secrets in a block of log text.
///
/// Two kinds, both of them values this platform generated and handed to a
/// process that prints its environment. The Postgres password inside a
/// `DATABASE_URL`, and the bearer token in `AIAS_MODEL_<ALIAS>_KEY`, which is
/// the whole of an Instance's access control: a screenshot of a log should not
/// carry either, and the user gains nothing from reading them.
///
/// The shape handled is `NAME=value` as an environment dump prints it, and any
/// `postgres://` or `postgresql://` URL. Token by token, so the spacing of the
/// log is left alone.
pub(crate) fn mask_secrets(text: &str) -> String {
    let mut masked: String = text
        .split_inclusive(char::is_whitespace)
        .map(mask_token)
        .collect();
    masked = mask_bearer(&masked);
    for secret in known_secrets() {
        if masked.contains(&secret) {
            masked = masked.replace(&secret, MASK);
        }
    }
    masked
}

/// Blank the token of every `Authorization: Bearer <token>` in a block of text.
///
/// An app that logs the request it made to its model prints the header, and the
/// token in it is the whole of that Instance's access control. The scheme name
/// is left readable, the token is not. Matching is case insensitive because
/// `bearer` is a scheme name and HTTP does not care how it is spelled.
fn mask_bearer(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(found) = lower[at..].find(BEARER) {
        let token_start = at + found + BEARER.len();
        out.push_str(&text[at..token_start]);
        let end = text[token_start..]
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .map_or(text.len(), |offset| token_start + offset);
        if end > token_start {
            out.push_str(MASK);
        }
        at = end;
    }
    out.push_str(&text[at..]);
    out
}

/// One whitespace delimited token of a log line, trailing whitespace included.
fn mask_token(token: &str) -> String {
    let (word, space) = match token.find(char::is_whitespace) {
        Some(at) => token.split_at(at),
        None => (token, ""),
    };

    if let Some(at) = word.find(SECRET_SUFFIX) {
        let keep = at + SECRET_SUFFIX.len();
        return format!("{}{MASK}{space}", &word[..keep]);
    }

    // `postgres://user:password@host`, the password being everything between
    // the colon that ends the user and the `@` that ends the userinfo.
    for scheme in ["postgres://", "postgresql://"] {
        let Some(at) = word.find(scheme) else {
            continue;
        };
        let userinfo = at + scheme.len();
        let Some(end) = word[userinfo..].find('@') else {
            continue;
        };
        let end = userinfo + end;
        let Some(colon) = word[userinfo..end].find(':') else {
            continue;
        };
        let colon = userinfo + colon;
        return format!("{}:{MASK}{}{space}", &word[..colon], &word[end..]);
    }

    token.to_string()
}

#[cfg(test)]
pub(crate) fn set_spawn_hook(hook: Option<SpawnHook>) {
    *SPAWN_HOOK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = hook;
}

/// Serializes every test that touches the process global registry, wherever it
/// lives: `apps` drives the same manager through leases.
#[cfg(test)]
pub(crate) static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A fake `llama-server`: answers `/health` with 200 and nothing else.
#[cfg(test)]
pub(crate) fn fake_spawn(request: &SpawnRequest<'_>) -> Result<Option<Child>> {
    let port = request.port;
    std::fs::write(request.log_path, "fake llama-server\n")?;
    tokio::spawn(async move {
        let Ok(listener) =
            tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).await
        else {
            return;
        };
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut scratch = [0u8; 512];
            let _ = stream.read(&mut scratch).await;
            let _ = stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}",
                )
                .await;
        }
    });
    Ok(None)
}

/// Put a sparse GGUF and the sidecar a finished download leaves where
/// [`installed`] looks for them.
#[cfg(test)]
pub(crate) fn install_fixture(
    data_dir: &Path,
    model: &ModelRef,
    size_mb: u64,
    mmproj: Option<&str>,
) -> PathBuf {
    let dir = models::model_dir(model, data_dir);
    std::fs::create_dir_all(&dir).unwrap();
    let filename = format!(
        "{}-{}.gguf",
        model.repo.rsplit('/').next().unwrap(),
        model.quant
    );
    let file = std::fs::File::create(dir.join(&filename)).unwrap();
    file.set_len(size_mb * 1024 * 1024).unwrap();
    if let Some(mmproj) = mmproj {
        std::fs::write(dir.join(mmproj), b"x").unwrap();
    }
    std::fs::write(
        dir.join("model.json"),
        serde_json::json!({
            "repo": model.repo,
            "quant": model.quant,
            "filename": filename,
            "mmproj": mmproj,
        })
        .to_string(),
    )
    .unwrap();
    dir.join(filename)
}

#[cfg(test)]
pub(crate) fn reset_for_test(profile: Option<DeviceProfile>) {
    // Each `#[tokio::test]` owns its runtime and drops it at the end, taking
    // the reaper spawned on it with it. Clearing the flag lets the next test
    // spawn one on its own runtime.
    REAPER.store(false, Ordering::SeqCst);
    let mut manager = manager();
    for mut running in std::mem::take(&mut manager.running) {
        kill(&mut running);
    }
    manager.profile = profile;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{MemoryModel, Vendor};

    const MB: u64 = 1024 * 1024;

    fn profile(vendor: Vendor, vram_mb: u64) -> DeviceProfile {
        DeviceProfile {
            gpu_vendor: vendor,
            gpu_name: "test".into(),
            vram_mb: Some(vram_mb),
            total_ram_mb: 65536,
            memory_model: MemoryModel::Dedicated,
            has_npu: false,
        }
    }

    #[test]
    fn the_log_tail_gives_up_neither_the_password_nor_the_instance_key() {
        let dir = std::env::temp_dir().join(format!("aias-mask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("app.log");
        std::fs::write(
            &path,
            "[start] DATABASE_URL=postgres://example-chat:Hn3kQ2vTpLsd8XwR@127.0.0.1:41999/example-chat\n\
             [start] AIAS_MODEL_CHAT_URL=http://127.0.0.1:41029/v1\n\
             [start] AIAS_MODEL_CHAT_KEY=4f2b9c8e1d6a3057\n\
             [start] LLAMA_API_KEY=4f2b9c8e1d6a3057\n\
             [start] listening on 127.0.0.1:41501\n",
        )
        .unwrap();

        let tail = log_tail(&path, 50);
        assert!(!tail.contains("Hn3kQ2vTpLsd8XwR"), "{tail}");
        assert!(!tail.contains("4f2b9c8e1d6a3057"), "{tail}");
        // Masking is not redaction: what the value was for is still readable.
        assert!(tail.contains("postgres://example-chat:***@127.0.0.1:41999/example-chat"));
        assert!(tail.contains("AIAS_MODEL_CHAT_KEY=***"));
        assert!(tail.contains("LLAMA_API_KEY=***"));
        assert!(tail.contains("AIAS_MODEL_CHAT_URL=http://127.0.0.1:41029/v1"));
        assert!(tail.contains("listening on 127.0.0.1:41501"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_authorization_header_gives_up_its_token_but_keeps_its_shape() {
        assert_eq!(
            mask_secrets("Authorization: Bearer sk-4f2b9c8e1d6a3057\n"),
            "Authorization: Bearer ***\n"
        );
        // HTTP does not care how the scheme is spelled, so neither does this.
        assert_eq!(
            mask_secrets("authorization: bearer 4f2b9c8e1d6a3057 sent"),
            "authorization: bearer *** sent"
        );
        // A JSON dump of the same header.
        assert_eq!(
            mask_secrets(r#"{"authorization":"Bearer 4f2b9c8e"}"#),
            r#"{"authorization":"Bearer ***"}"#
        );
        // The word on its own is not a token.
        assert_eq!(mask_secrets("bearer\n"), "bearer\n");
    }

    #[test]
    fn a_value_the_platform_generated_is_masked_wherever_it_appears() {
        let key = new_api_key();
        assert_eq!(key.len(), 64);
        let text = format!(
            "[app] GET http://127.0.0.1:41029/v1/models?key={key}\n[app] retrying with {key}\n"
        );
        let masked = mask_secrets(&text);
        assert!(!masked.contains(&key), "{masked}");
        assert_eq!(masked.matches(MASK).count(), 2, "{masked}");
        // The shape around it is untouched, so the log still reads.
        assert!(masked.contains("http://127.0.0.1:41029/v1/models?key=***"));

        // Something short is left alone: it is ordinary text somewhere.
        remember_secret("ok");
        assert_eq!(mask_secrets("ok then"), "ok then");
    }

    #[test]
    fn masking_leaves_everything_that_is_not_a_secret_alone() {
        assert_eq!(mask_secrets(""), "");
        assert_eq!(
            mask_secrets("nothing to hide here\n"),
            "nothing to hide here\n"
        );
        // A DSN with no password is left as it is rather than gaining one.
        assert_eq!(
            mask_secrets("postgres://app@127.0.0.1:41999/app"),
            "postgres://app@127.0.0.1:41999/app"
        );
        // Spacing survives, which matters for a log read as a block.
        assert_eq!(
            mask_secrets("  A_KEY=one\tB_KEY=two  \n"),
            "  A_KEY=***\tB_KEY=***  \n"
        );
    }

    #[test]
    fn a_roomy_device_gets_a_large_context_and_slots() {
        let p = params_for(&profile(Vendor::Amd, 98304), 9 * 1024 * MB, None);
        assert_eq!(p.ctx, 32768);
        assert_eq!(p.n_parallel, 4);
        assert_eq!(p.ngl, 999);
    }

    #[test]
    fn a_tight_device_gets_the_floor() {
        let p = params_for(&profile(Vendor::Intel, 8192), 7 * 1024 * MB, None);
        assert_eq!(p.ctx, 2048);
        assert_eq!(p.n_parallel, 1);
    }

    #[test]
    fn no_gpu_means_no_offload() {
        let p = params_for(&profile(Vendor::None, 0), 4 * 1024 * MB, None);
        assert_eq!(p.ngl, 0);
    }

    #[test]
    fn ports_are_stable_and_inside_the_instance_range() {
        let model = ModelRef::new("Qwen/Qwen3-14B-GGUF", "Q4_K_M");
        let port = port_for(&model);
        assert_eq!(port, port_for(&model));
        assert!((PORT_RANGE_START..=INSTANCE_PORT_END).contains(&port));
        const { assert!(INSTANCE_PORT_END < PORT_RANGE_END) };
    }

    #[test]
    fn the_model_id_is_repo_and_quant() {
        assert_eq!(
            model_id(&ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0")),
            "Qwen/Qwen3-0.6B-GGUF:Q8_0"
        );
    }

    #[test]
    fn an_allocation_failure_in_the_log_is_recognized() {
        assert!(looks_like_oom(
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory"
        ));
        assert!(looks_like_oom(
            "llama_model_load: failed to allocate buffer"
        ));
        assert!(!looks_like_oom(
            "main: server is listening on 127.0.0.1:41029"
        ));
    }

    #[test]
    fn a_status_line_is_parsed() {
        assert_eq!(parse_status(b"HTTP/1.1 200 OK\r\n"), Some(200));
        assert_eq!(
            parse_status(b"HTTP/1.1 503 Service Unavailable\r\n"),
            Some(503)
        );
        assert_eq!(parse_status(b"garbage"), None);
    }

    // Everything below shares the process global registry, so it runs under the
    // one lock in the parent module and resets the manager first.

    struct TempData(PathBuf);

    impl TempData {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "aias-instances-{}-{tag}-{}",
                std::process::id(),
                now_secs()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        /// Put a sparse file of `size_mb` where `installed` looks for it.
        ///
        /// The sidecar is what [`crate::models::installed`] reads, so a fixture
        /// writes the same pair a finished download leaves behind.
        fn install(&self, model: &ModelRef, size_mb: u64) -> PathBuf {
            self.install_with_mmproj(model, size_mb, None)
        }

        fn install_with_mmproj(
            &self,
            model: &ModelRef,
            size_mb: u64,
            mmproj: Option<&str>,
        ) -> PathBuf {
            install_fixture(&self.0, model, size_mb, mmproj)
        }
    }

    impl Drop for TempData {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_model_that_was_never_downloaded_is_reported_by_name() {
        let data = TempData::new("missing");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        let err = installed(&model, &data.0).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");
        assert!(
            err.to_string().contains("Qwen/Qwen3-0.6B-GGUF:Q8_0"),
            "{err}"
        );
    }

    #[test]
    fn the_weights_and_the_mmproj_are_told_apart() {
        let data = TempData::new("mmproj");
        let model = ModelRef::new("Qwen/Qwen2.5-VL-7B-Instruct-GGUF", "Q4_K_M");
        data.install_with_mmproj(&model, 4, Some("mmproj-F16.gguf"));

        let files = installed(&model, &data.0).unwrap();
        assert!(files.gguf.to_string_lossy().contains("Q4_K_M"));
        assert_eq!(
            files.mmproj,
            Some(models::model_dir(&model, &data.0).join("mmproj-F16.gguf"))
        );
        assert_eq!(files.size_bytes, 4 * MB);
    }

    #[tokio::test]
    async fn a_vision_model_is_started_with_its_projector() {
        let _guard = TEST_LOCK.lock().await;
        reset_for_test(Some(profile(Vendor::Nvidia, 16384)));
        // The hook stands in for the GPU, and records the argv the runtime
        // would have been handed.
        set_spawn_hook(Some(|request| {
            assert!(
                request.files.mmproj.is_some(),
                "the projector must reach spawn_server"
            );
            fake_spawn(request)
        }));

        let data = TempData::new("vlm");
        let model = ModelRef::new("ggml-org/gemma-3-4b-it-GGUF", "Q4_K_M");
        data.install_with_mmproj(&model, 8, Some("mmproj-gemma-3-4b-it-f16.gguf"));

        let url = acquire_in("vision-app", &model, &data.0).await.unwrap();
        assert_eq!(url.model_id, "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M");

        stop_all().await.unwrap();
        set_spawn_hook(None);
    }

    /// What the last [`fake_spawn`] was asked to start with, for assertions.
    static SEEN_KEY: Mutex<Option<String>> = Mutex::new(None);

    #[tokio::test]
    async fn every_instance_is_guarded_by_a_key_the_lease_holder_is_handed() {
        let _guard = TEST_LOCK.lock().await;
        reset_for_test(Some(profile(Vendor::Nvidia, 16384)));
        set_spawn_hook(Some(|request| {
            *SEEN_KEY.lock().unwrap() = Some(request.api_key.to_string());
            fake_spawn(request)
        }));

        let data = TempData::new("apikey");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        data.install(&model, 8);

        let first = acquire_in("app-one", &model, &data.0).await.unwrap();
        assert_eq!(first.api_key.len(), 64, "32 random bytes, hex encoded");
        assert!(first.api_key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            SEEN_KEY.lock().unwrap().as_deref(),
            Some(first.api_key.as_str()),
            "the key the app is handed is the key llama-server was started with"
        );
        assert_eq!(list()[0].api_key(), first.api_key);

        // ... and the snapshot the UI and the CLI print does not carry it.
        let json = serde_json::to_string(&list()[0]).unwrap();
        assert!(!json.contains("apiKey"), "{json}");
        assert!(!json.contains(first.api_key.as_str()), "{json}");

        // A second app on the same instance gets the same key, not a new one.
        let second = acquire_in("app-two", &model, &data.0).await.unwrap();
        assert_eq!(second.api_key, first.api_key);

        // A fresh instance gets a fresh key.
        let other = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q4_K_M");
        data.install(&other, 8);
        let third = acquire_in("app-three", &other, &data.0).await.unwrap();
        assert_ne!(third.api_key, first.api_key);

        stop_all().await.unwrap();
        set_spawn_hook(None);
    }

    #[tokio::test]
    async fn two_apps_share_one_instance_and_the_lease_survives_the_first_release() {
        let _guard = TEST_LOCK.lock().await;
        reset_for_test(Some(profile(Vendor::Nvidia, 16384)));
        set_spawn_hook(Some(fake_spawn));

        let data = TempData::new("share");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        data.install(&model, 8);

        let first = acquire_in("testapp", &model, &data.0).await.unwrap();
        let second = acquire_in("testapp2", &model, &data.0).await.unwrap();

        assert_eq!(first.port, second.port);
        assert_eq!(
            first.base_url,
            format!("http://127.0.0.1:{}/v1", first.port)
        );
        assert_eq!(first.model_id, "Qwen/Qwen3-0.6B-GGUF:Q8_0");
        assert!((PORT_RANGE_START..=INSTANCE_PORT_END).contains(&first.port));

        let running = list();
        assert_eq!(running.len(), 1, "{running:?}");
        assert_eq!(running[0].leases, vec!["testapp", "testapp2"]);

        release("testapp", &model).await.unwrap();
        assert_eq!(list()[0].leases, vec!["testapp2"]);

        release("testapp2", &model).await.unwrap();
        assert!(list()[0].leases.is_empty(), "stays warm at zero leases");
        assert!(
            list()[0].idle_since.is_some(),
            "the release that empties the list starts the window"
        );

        // The window has not passed, so a reaper pass leaves it alone.
        reap_idle().await;
        assert_eq!(list().len(), 1, "a warm instance is not stopped early");

        // A new lease takes it out of the window again.
        acquire_in("testapp", &model, &data.0).await.unwrap();
        assert_eq!(list()[0].idle_since, None);

        stop_all().await.unwrap();
        assert!(list().is_empty());
        set_spawn_hook(None);
    }

    #[tokio::test]
    async fn the_idle_reaper_stops_an_instance_nobody_leases_any_more() {
        let _guard = TEST_LOCK.lock().await;
        reset_for_test(Some(profile(Vendor::Nvidia, 16384)));
        set_spawn_hook(Some(fake_spawn));

        // A one second window, so the reaper's own tick is a quarter of a
        // second and the whole test is over in two.
        //
        // Safety: every test that reads `AIAS_IDLE_SECS` runs under the lock
        // held above, and the variable is removed before it is released.
        unsafe { std::env::set_var("AIAS_IDLE_SECS", "1") };

        let data = TempData::new("reaper");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        data.install(&model, 8);

        // The acquire is what spawns the reaper, on this test's runtime.
        acquire_in("reaped-app", &model, &data.0).await.unwrap();
        assert_eq!(list().len(), 1);
        release("reaped-app", &model).await.unwrap();
        assert!(list()[0].leases.is_empty());

        tokio::time::sleep(Duration::from_secs(2)).await;
        let running = list();
        unsafe { std::env::remove_var("AIAS_IDLE_SECS") };
        assert!(
            running.is_empty(),
            "a zero lease instance must be stopped once its window passes: {running:?}"
        );

        stop_all().await.unwrap();
        set_spawn_hook(None);
    }

    #[test]
    fn the_reaper_never_sleeps_past_the_window_it_is_measuring() {
        assert_eq!(reap_interval_for(600), REAP_INTERVAL);
        assert_eq!(reap_interval_for(1), Duration::from_millis(250));
        // The floor is what no window may go under. Zero never reaches it,
        // because zero turns the reaper off instead of making it spin.
        assert_eq!(reap_interval_for(0), REAP_INTERVAL_MIN);
    }

    #[tokio::test]
    async fn an_idle_window_that_is_not_a_number_keeps_the_default() {
        let _guard = TEST_LOCK.lock().await;
        assert_eq!(idle_secs(), Some(DEFAULT_IDLE_SECS), "unset is the default");

        // Safety: every test that reads `AIAS_IDLE_SECS` runs under the lock
        // held above, and the variable is removed before it is released.
        for value in ["", "soon", "600s", "-1", "1.5", "18446744073709551616"] {
            unsafe { std::env::set_var("AIAS_IDLE_SECS", value) };
            assert_eq!(
                idle_secs(),
                Some(DEFAULT_IDLE_SECS),
                "`{value}` is not a window, so the default stands"
            );
        }

        unsafe { std::env::set_var("AIAS_IDLE_SECS", " 42 ") };
        assert_eq!(idle_secs(), Some(42), "a number with spaces is a number");

        unsafe { std::env::remove_var("AIAS_IDLE_SECS") };
    }

    #[tokio::test]
    async fn a_zero_idle_window_turns_the_reaper_off() {
        let _guard = TEST_LOCK.lock().await;
        reset_for_test(Some(profile(Vendor::Nvidia, 16384)));
        set_spawn_hook(Some(fake_spawn));

        // Safety: as above, under the lock, removed before it is released.
        unsafe { std::env::set_var("AIAS_IDLE_SECS", "0") };
        assert_eq!(idle_secs(), None, "zero is off, not a zero second window");

        let data = TempData::new("no-reaper");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        data.install(&model, 8);

        acquire_in("kept-warm", &model, &data.0).await.unwrap();
        release("kept-warm", &model).await.unwrap();
        assert!(list()[0].leases.is_empty());

        // The stamp is an hour old, so any window would have expired, and the
        // reaper still leaves it alone.
        manager().running[0].instance.idle_since = Some(now_secs() - 3600);
        reap_idle().await;
        let running = list();
        unsafe { std::env::remove_var("AIAS_IDLE_SECS") };
        assert_eq!(
            running.len(),
            1,
            "the reaper is off, so nothing stops an idle instance: {running:?}"
        );

        stop_all().await.unwrap();
        set_spawn_hook(None);
    }

    #[tokio::test]
    async fn a_leased_instance_blocks_the_budget_and_an_idle_one_is_evicted() {
        let _guard = TEST_LOCK.lock().await;
        // 120 MB budget, 108 MB usable: one 60 MB model fits, two do not.
        reset_for_test(Some(profile(Vendor::Nvidia, 120)));
        set_spawn_hook(Some(fake_spawn));

        let data = TempData::new("budget");
        let first = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        let second = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q4_K_M");
        data.install(&first, 60);
        data.install(&second, 60);

        acquire_in("app-one", &first, &data.0).await.unwrap();

        // The first one is leased, so there is nothing to evict.
        let err = acquire_in("app-two", &second, &data.0).await.unwrap_err();
        assert!(matches!(err, Error::UnsupportedHardware(_)), "{err}");
        let message = err.to_string();
        assert!(message.contains("Qwen/Qwen3-0.6B-GGUF:Q8_0"), "{message}");
        assert!(message.contains("app-one"), "{message}");

        // Once it is idle it is the LRU victim and the second model starts.
        release("app-one", &first).await.unwrap();
        acquire_in("app-two", &second, &data.0).await.unwrap();

        let running = list();
        assert_eq!(running.len(), 1, "{running:?}");
        assert_eq!(running[0].model, second);

        stop_all().await.unwrap();
        set_spawn_hook(None);
    }
}
