//! Backend selection and `llama-server` installation.
//!
//! Selection is the fixed table in PLAN.md section 2.1, decision D8. It is never
//! a probe and never automatic after install.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::hardware::{DeviceProfile, Vendor};
use crate::instances::Params;
use crate::paths;
use crate::process::{LLAMA_ALLOWLIST, Quiet as _, Sealed as _};

/// Owner of the llama.cpp releases the platform installs from.
const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
/// Where a release asset is downloaded from, `<tag>/<asset>` appended.
const RELEASE_DOWNLOAD: &str = "https://github.com/ggml-org/llama.cpp/releases/download";
/// Pin the release tag instead of asking GitHub, for offline and for CI.
const TAG_ENV: &str = "AIAS_LLAMA_TAG";
/// Points at a `llama-server` binary that came from somewhere else.
const SERVER_ENV: &str = "AIAS_LLAMA_SERVER";
/// Tag of the install [`SERVER_ENV`] stands in for, so it is visible as one.
const EXTERNAL_TAG: &str = "external";
/// What llama.cpp reads `--api-key` from when the flag is absent.
const API_KEY_ENV: &str = "LLAMA_API_KEY";
/// GitHub rejects an API request without one.
const USER_AGENT: &str = concat!("aias/", env!("CARGO_PKG_VERSION"));
/// Written into the install directory, and the marker that makes it complete.
const INSTALL_JSON: &str = "install.json";

/// The `llama-server` build family installed on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Backend {
    Cuda,
    Vulkan,
    OpenVino,
    Cpu,
}

impl Backend {
    /// The ggml-org release artifact name for this backend on Windows x64.
    ///
    /// CUDA 13 is what recent releases ship and what the driver on every target
    /// machine supports, so the 12.4 artifact of the original table is not built
    /// any more and is not used.
    pub fn artifact(&self) -> &'static str {
        match self {
            Backend::Cuda => "win-cuda-13.3-x64",
            Backend::Vulkan => "win-vulkan-x64",
            Backend::OpenVino => "win-openvino-2026.3.1-x64",
            Backend::Cpu => "win-cpu-x64",
        }
    }

    /// Short name used in directory names and on the command line.
    pub fn slug(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::OpenVino => "openvino",
            Backend::Cpu => "cpu",
        }
    }

    /// The CUDA runtime archive that must sit next to the exe, if any.
    ///
    /// The CUDA build links against `cudart` and `cublas` dlls that the llama
    /// zip does not carry, so they come from a second release asset of the same
    /// CUDA version.
    fn cudart_artifact(&self) -> Option<String> {
        match self {
            // `win-cuda-13.3-x64` becomes `cudart-llama-bin-win-cuda-13.3-x64`.
            Backend::Cuda => Some(format!("cudart-llama-bin-{}", self.artifact())),
            _ => None,
        }
    }
}

/// An installed runtime on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInstall {
    pub backend: Backend,
    /// The llama.cpp release tag the artifact came from, for example `b1234`.
    pub tag: String,
    /// Directory the artifact was unpacked into.
    pub dir: PathBuf,
}

/// One downloaded zip and the hash of exactly what landed on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    /// File name of the release asset, including `.zip`.
    pub name: String,
    /// Lower case hex SHA-256 of the bytes that were downloaded.
    pub sha256: String,
}

/// `install.json`: what an install is made of, written once it is complete.
///
/// llama.cpp publishes no checksums with its releases, so the platform records
/// the SHA-256 of what it downloaded here and in a `.sha256` sidecar next to the
/// cached zip, which is what a later install compares against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallRecord {
    pub backend: Backend,
    pub tag: String,
    pub dir: PathBuf,
    pub artifacts: Vec<Artifact>,
}

impl InstallRecord {
    /// The install this record describes.
    pub fn install(&self) -> RuntimeInstall {
        RuntimeInstall {
            backend: self.backend,
            tag: self.tag.clone(),
            dir: self.dir.clone(),
        }
    }
}

/// Pick the backend for a device, with the NPU path left out.
pub fn select(profile: &DeviceProfile) -> Backend {
    select_with_npu(profile, false)
}

/// Pick the backend for a device.
///
/// `opt_in_npu` is the Settings toggle: OpenVINO is experimental and is only
/// offered when the user asked for it and the device has an NPU.
pub fn select_with_npu(profile: &DeviceProfile, opt_in_npu: bool) -> Backend {
    match profile.gpu_vendor {
        Vendor::Nvidia => Backend::Cuda,
        Vendor::Amd => Backend::Vulkan,
        Vendor::Intel => {
            if opt_in_npu && profile.has_npu {
                Backend::OpenVino
            } else {
                Backend::Vulkan
            }
        }
        // Apple and unknown GPUs have no Windows artifact in the table.
        Vendor::Apple | Vendor::Other | Vendor::None => Backend::Cpu,
    }
}

/// Download, verify and unpack the artifact for `backend`.
///
/// Idempotent: an install directory that already holds an `install.json` and the
/// executable is returned as it is, so this is safe to call on every start.
pub async fn install(backend: Backend, data_dir: &Path) -> Result<RuntimeInstall> {
    let tag = resolve_tag().await?;
    let dir = install_dir(data_dir, &tag, backend);

    if let Some(record) = read_record(&dir)
        && record.backend == backend
    {
        return Ok(record.install());
    }

    let cache = paths::runtime_dir(data_dir).join("cache");
    std::fs::create_dir_all(&cache)?;
    std::fs::create_dir_all(&dir)?;

    let mut artifacts = Vec::new();

    let llama_zip = format!("llama-{tag}-bin-{}.zip", backend.artifact());
    let path = fetch_asset(&tag, &llama_zip, &cache).await?;
    artifacts.push(Artifact {
        name: llama_zip,
        sha256: sha256_file(&path).await?,
    });
    unpack(&path, &dir).await?;

    // The cudart dlls unpack into the same directory, which is what puts them
    // next to `llama-server.exe`.
    if let Some(cudart) = backend.cudart_artifact() {
        let cudart_zip = format!("{cudart}.zip");
        let path = fetch_asset(&tag, &cudart_zip, &cache).await?;
        artifacts.push(Artifact {
            name: cudart_zip,
            sha256: sha256_file(&path).await?,
        });
        unpack(&path, &dir).await?;
    }

    let exe = dir.join(paths::exe_name("llama-server"));
    if !exe.is_file() {
        return Err(Error::NotFound(format!(
            "{} is missing from the unpacked artifact",
            exe.display()
        )));
    }

    let record = InstallRecord {
        backend,
        tag: tag.clone(),
        dir: dir.clone(),
        artifacts,
    };
    std::fs::write(dir.join(INSTALL_JSON), serde_json::to_vec_pretty(&record)?)?;

    Ok(record.install())
}

/// The install this machine already has, newest tag first, or `None`.
///
/// `AIAS_LLAMA_SERVER` overrides it with a binary the platform did not install.
/// There is no llama.cpp release artifact for macOS in the section 2.1 table,
/// so a developer on a Mac points this at the Homebrew `llama-server` and runs
/// the whole platform, models and apps included, against it. The install it
/// stands in for is reported with backend [`Backend::Cpu`] and the tag
/// `external`, because no artifact was downloaded and none was verified.
pub fn installed(data_dir: &Path) -> Option<RuntimeInstall> {
    external_install().or_else(|| downloaded(data_dir))
}

/// The newest install under `<data_dir>/runtime`, whatever the override says.
///
/// Kept apart from [`installed`] because `AIAS_LLAMA_SERVER` belongs to the
/// whole process: a test cannot unset it for itself, so the scan is what it
/// asserts on.
fn downloaded(data_dir: &Path) -> Option<RuntimeInstall> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(paths::runtime_dir(data_dir))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();

    dirs.iter()
        .rev()
        .filter_map(|dir| read_record(dir))
        .map(|record| record.install())
        .next()
}

/// Read the `install.json` of one install directory, if it is complete.
pub fn install_record(dir: &Path) -> Result<InstallRecord> {
    read_record(dir).ok_or_else(|| {
        Error::NotFound(format!(
            "no runtime install in {}, run `aias runtime install`",
            dir.display()
        ))
    })
}

/// The install `AIAS_LLAMA_SERVER` names, when it names a file.
///
/// An unset variable, or one pointing at something that is not a file, is not
/// an error here: the caller falls back to what is installed under the data
/// directory and fails with that message instead.
fn external_install() -> Option<RuntimeInstall> {
    external_at(Path::new(&std::env::var_os(SERVER_ENV)?))
}

/// The install one `llama-server` path stands for.
fn external_at(exe: &Path) -> Option<RuntimeInstall> {
    if !exe.is_file() {
        return None;
    }
    Some(RuntimeInstall {
        backend: Backend::Cpu,
        tag: EXTERNAL_TAG.to_string(),
        dir: exe.parent().unwrap_or(Path::new(".")).to_path_buf(),
    })
}

/// Path of the `llama-server` executable inside an install.
///
/// `AIAS_LLAMA_SERVER` wins, and it names the file rather than its directory,
/// so a binary that is not called `llama-server` is still found.
pub fn llama_server_path(install: &RuntimeInstall) -> PathBuf {
    if install.tag == EXTERNAL_TAG
        && let Some(exe) = std::env::var_os(SERVER_ENV).map(PathBuf::from)
        && exe.is_file()
    {
        return exe;
    }
    install.dir.join(paths::exe_name("llama-server"))
}

/// Start one `llama-server` on a loopback port with platform chosen parameters.
///
/// Every value goes in as an argv element (PLAN.md section 3 rule 6) and the
/// socket is bound to `127.0.0.1` so Windows never raises a firewall prompt.
/// stdout and stderr land in `logs/llama-<port>.log`.
///
/// `mmproj` is the vision projector of a VLM repo; without it llama.cpp serves
/// the same weights as a text only model. `--jinja` turns on the chat template
/// that ships inside the GGUF, which tool calling needs, and `--no-webui` keeps
/// the process an API: the only UI is this platform.
///
/// `api_key` is the bearer token the instance demands. Loopback is not a
/// permission boundary: every process on the machine can reach the port, so the
/// key is what keeps an Instance to the Apps that hold a lease on it. It goes in
/// through the environment rather than on argv: `--api-key` is declared with
/// `set_env("LLAMA_API_KEY")` in llama.cpp `common/arg.cpp`, and argv is world
/// readable on this machine through the process list, where a key that gates
/// every Instance has no business being.
///
/// The rest of the environment is [`crate::process::LLAMA_ALLOWLIST`] and
/// nothing else. A model server is the one process on this machine that holds
/// every Instance key, so it is also the one that must not be handed the
/// platform's own variables on top.
pub async fn spawn_server(
    install: &RuntimeInstall,
    model_path: &Path,
    mmproj: Option<&Path>,
    port: u16,
    params: &Params,
    alias: &str,
    api_key: &str,
) -> Result<tokio::process::Child> {
    let exe = llama_server_path(install);
    if !exe.is_file() {
        return Err(Error::NotFound(format!("{} is missing", exe.display())));
    }
    if !model_path.is_file() {
        return Err(Error::NotFound(format!(
            "{} is missing",
            model_path.display()
        )));
    }
    if let Some(mmproj) = mmproj
        && !mmproj.is_file()
    {
        return Err(Error::NotFound(format!("{} is missing", mmproj.display())));
    }

    let logs = logs_dir_of(install);
    std::fs::create_dir_all(&logs)?;
    let log = File::options()
        .create(true)
        .append(true)
        .open(logs.join(format!("llama-{port}.log")))?;

    let mut command = server_command(&exe, model_path, mmproj, port, params, alias, api_key);

    let child = command
        .current_dir(&install.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .quiet()
        .spawn()?;

    Ok(child)
}

/// The `llama-server` command line, with the key kept off it.
///
/// Split out of [`spawn_server`] so a test can read back what would be run: the
/// key is the one value that must never appear in argv.
fn server_command(
    exe: &Path,
    model_path: &Path,
    mmproj: Option<&Path>,
    port: u16,
    params: &Params,
    alias: &str,
    api_key: &str,
) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(exe);
    // Cleared before anything is set: the key below must survive it.
    command.sealed_env(LLAMA_ALLOWLIST);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--n-gpu-layers")
        .arg(params.ngl.to_string())
        .arg("--ctx-size")
        .arg(params.ctx.to_string())
        .arg("--parallel")
        .arg(params.n_parallel.to_string())
        .arg("--alias")
        .arg(alias)
        .arg("--jinja")
        .arg("--no-webui")
        .env(API_KEY_ENV, api_key);
    if let Some(mmproj) = mmproj {
        command.arg("--mmproj").arg(mmproj);
    }
    command
}

/// Poll `/health` until the server reports `ok` or `timeout` runs out.
///
/// llama.cpp answers 503 with `Loading model` while the weights are read, so a
/// failing request is a reason to wait, not a reason to stop.
pub async fn wait_healthy(port: u16, timeout: Duration) -> Result<()> {
    let client = http_client()?;
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last;

    loop {
        match client.get(&url).send().await {
            Ok(response) => {
                let status = response.status();
                match response.json::<serde_json::Value>().await {
                    Ok(body) => {
                        if body.get("status").and_then(serde_json::Value::as_str) == Some("ok") {
                            return Ok(());
                        }
                        last = format!("{status}: {body}");
                    }
                    Err(err) => last = format!("{status}: {err}"),
                }
            }
            Err(err) => last = err.to_string(),
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(Error::Process(format!(
                "llama-server on port {port} was not healthy within {}s, last answer {last}",
                timeout.as_secs()
            )));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// `<data_dir>/runtime/<tag>-<backend>`, one directory per build.
fn install_dir(data_dir: &Path, tag: &str, backend: Backend) -> PathBuf {
    paths::runtime_dir(data_dir).join(format!("{tag}-{}", backend.slug()))
}

/// The install directory sits at `<data_dir>/runtime/<tag>-<backend>`.
///
/// An external install is a binary somewhere else on the machine, and the
/// directory two levels above it belongs to whoever put it there, so its logs
/// go to the platform's own data directory instead.
fn logs_dir_of(install: &RuntimeInstall) -> PathBuf {
    if install.tag == EXTERNAL_TAG {
        return paths::logs_dir(&paths::data_dir());
    }
    match install.dir.parent().and_then(Path::parent) {
        Some(data_dir) => paths::logs_dir(data_dir),
        None => paths::logs_dir(&paths::data_dir()),
    }
}

/// A complete install directory, or `None` when it is absent or half written.
fn read_record(dir: &Path) -> Option<InstallRecord> {
    let json = std::fs::read_to_string(dir.join(INSTALL_JSON)).ok()?;
    let record: InstallRecord = serde_json::from_str(&json).ok()?;
    dir.join(paths::exe_name("llama-server"))
        .is_file()
        .then_some(record)
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(network)
}

/// Any network failure. `models.rs` carries the same helper.
fn network(err: impl std::fmt::Display) -> Error {
    Error::Http(err.to_string())
}

/// Minimal shape of a GitHub release.
#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
}

/// The release tag to install.
///
/// `AIAS_LLAMA_TAG` wins when it is set. Otherwise the newest release is asked
/// for first, because llama.cpp marks its rolling builds as prereleases and
/// `releases/latest` skips those; `releases/latest` is only the fallback.
async fn resolve_tag() -> Result<String> {
    if let Ok(tag) = std::env::var(TAG_ENV)
        && !tag.trim().is_empty()
    {
        return Ok(tag.trim().to_string());
    }

    let client = http_client()?;

    let newest = client
        .get(format!("{RELEASES_API}?per_page=1"))
        .send()
        .await
        .map_err(network)?;
    if newest.status().is_success()
        && let Ok(releases) = newest.json::<Vec<Release>>().await
        && let Some(release) = releases.into_iter().next()
    {
        return Ok(release.tag_name);
    }

    let latest = client
        .get(format!("{RELEASES_API}/latest"))
        .send()
        .await
        .map_err(network)?;
    let status = latest.status();
    if !status.is_success() {
        return Err(network(format!(
            "GitHub answered {status} for the latest llama.cpp release"
        )));
    }
    Ok(latest.json::<Release>().await.map_err(network)?.tag_name)
}

/// Download one release asset into `cache`, reusing a verified earlier copy.
///
/// The sidecar holds the SHA-256 of the cached zip; a cached file whose hash no
/// longer matches is downloaded again rather than trusted.
async fn fetch_asset(tag: &str, name: &str, cache: &Path) -> Result<PathBuf> {
    let path = cache.join(name);
    let sidecar = cache.join(format!("{name}.sha256"));

    if path.is_file()
        && let Ok(expected) = std::fs::read_to_string(&sidecar)
        && sha256_file(&path).await? == expected.trim()
    {
        return Ok(path);
    }

    let url = format!("{RELEASE_DOWNLOAD}/{tag}/{name}");
    let response = http_client()?.get(&url).send().await.map_err(network)?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(Error::NotFound(format!("{url} does not exist")));
    }
    if !status.is_success() {
        return Err(network(format!("GitHub answered {status} for {url}")));
    }

    // Write to a partial file first so an interrupted download is never cached.
    let partial = cache.join(format!("{name}.part"));
    let mut file = tokio::fs::File::create(&partial).await?;
    let mut response = response;
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    drop(file);
    tokio::fs::rename(&partial, &path).await?;

    let digest = sha256_file(&path).await?;
    std::fs::write(&sidecar, &digest)?;
    Ok(path)
}

/// SHA-256 of a file, hashed off the async runtime.
async fn sha256_file(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut file = File::open(&path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|err| Error::Process(err.to_string()))?
}

/// Unpack a release zip into `dir`. Release zips have no top level folder.
async fn unpack(zip_path: &Path, dir: &Path) -> Result<()> {
    let zip_path = zip_path.to_path_buf();
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let file = File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|err| Error::Process(format!("{}: {err}", zip_path.display())))?;
        archive
            .extract(&dir)
            .map_err(|err| Error::Process(format!("{}: {err}", zip_path.display())))
    })
    .await
    .map_err(|err| Error::Process(err.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MemoryModel;

    fn profile(vendor: Vendor, has_npu: bool) -> DeviceProfile {
        DeviceProfile {
            gpu_vendor: vendor,
            gpu_name: "test".into(),
            vram_mb: Some(16384),
            total_ram_mb: 32768,
            memory_model: MemoryModel::Dedicated,
            has_npu,
        }
    }

    #[test]
    fn an_external_binary_stands_in_for_an_install() {
        let dir = std::env::temp_dir().join(format!("aias-runtime-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join(paths::exe_name("llama-server"));
        std::fs::write(&exe, b"not a real binary").unwrap();

        let install = external_at(&exe).expect("a file is an install");
        assert_eq!(install.backend, Backend::Cpu);
        assert_eq!(install.tag, "external");
        assert_eq!(install.dir, dir);
        // Without the override set, the path is still the one inside the dir.
        if std::env::var_os(SERVER_ENV).is_none() {
            assert_eq!(llama_server_path(&install), exe);
        }

        assert!(
            external_at(&dir.join("missing")).is_none(),
            "a path that is not a file is not an install"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_instance_key_never_reaches_the_command_line() {
        let key = "a".repeat(64);
        let params = Params {
            ctx: 8192,
            n_parallel: 1,
            ngl: 999,
        };
        let command = server_command(
            Path::new("llama-server"),
            Path::new("model.gguf"),
            None,
            41029,
            &params,
            "Qwen/Qwen3-8B-GGUF:Q4_K_M",
            &key,
        );

        // argv is readable by every process on this machine.
        let argv: Vec<String> = command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(!argv.iter().any(|arg| arg == "--api-key"), "{argv:?}");
        assert!(!argv.iter().any(|arg| arg.contains(&key)), "{argv:?}");
        // The parameters the platform owns are still there.
        assert!(argv.iter().any(|arg| arg == "--ctx-size"), "{argv:?}");

        // llama.cpp reads the flag from this variable when the flag is absent,
        // `set_env("LLAMA_API_KEY")` in `common/arg.cpp`.
        let env: Vec<(String, Option<String>)> = command
            .as_std()
            .get_envs()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert!(
            env.contains(&("LLAMA_API_KEY".to_string(), Some(key))),
            "the key goes in through the environment: {env:?}"
        );
        // And it is the only thing there that this process did not allow: the
        // environment is cleared and refilled, so nothing of the platform's own
        // reaches a model server.
        for (name, _) in &env {
            assert!(
                name == "LLAMA_API_KEY" || LLAMA_ALLOWLIST.contains(&name.as_str()),
                "`{name}` is not on the allowlist: {env:?}"
            );
        }
    }

    #[test]
    fn table_from_plan_2_1() {
        assert_eq!(select(&profile(Vendor::Nvidia, false)), Backend::Cuda);
        assert_eq!(select(&profile(Vendor::Amd, false)), Backend::Vulkan);
        assert_eq!(select(&profile(Vendor::Intel, true)), Backend::Vulkan);
        assert_eq!(select(&profile(Vendor::None, false)), Backend::Cpu);
        assert_eq!(select(&profile(Vendor::Other, false)), Backend::Cpu);
    }

    #[test]
    fn openvino_needs_both_an_npu_and_the_opt_in() {
        assert_eq!(
            select_with_npu(&profile(Vendor::Intel, true), true),
            Backend::OpenVino
        );
        assert_eq!(
            select_with_npu(&profile(Vendor::Intel, false), true),
            Backend::Vulkan
        );
        assert_eq!(
            select_with_npu(&profile(Vendor::Nvidia, true), true),
            Backend::Cuda
        );
    }

    #[test]
    fn server_path_sits_in_the_install_dir() {
        let install = RuntimeInstall {
            backend: Backend::Vulkan,
            tag: "b1234".into(),
            dir: PathBuf::from("/data/aias/runtime/b1234"),
        };
        assert!(llama_server_path(&install).starts_with("/data/aias/runtime/b1234"));
    }

    #[test]
    fn only_cuda_needs_a_second_archive() {
        assert_eq!(
            Backend::Cuda.cudart_artifact().as_deref(),
            Some("cudart-llama-bin-win-cuda-13.3-x64")
        );
        assert_eq!(Backend::Vulkan.cudart_artifact(), None);
        assert_eq!(Backend::Cpu.cudart_artifact(), None);
    }

    #[test]
    fn install_dirs_carry_the_tag_and_the_backend() {
        let dir = install_dir(Path::new("/data/aias"), "b10905", Backend::Cuda);
        assert!(dir.ends_with("b10905-cuda"));
        assert_eq!(
            logs_dir_of(&RuntimeInstall {
                backend: Backend::Cuda,
                tag: "b10905".into(),
                dir,
            }),
            PathBuf::from("/data/aias/logs")
        );
    }

    #[test]
    fn an_empty_data_dir_has_no_install() {
        let dir = std::env::temp_dir().join(format!("aias-runtime-test-{}", std::process::id()));
        std::fs::create_dir_all(paths::runtime_dir(&dir)).unwrap();
        assert_eq!(downloaded(&dir), None);
        if std::env::var_os(SERVER_ENV).is_none() {
            assert_eq!(installed(&dir), None);
        }
        assert!(install_record(&paths::runtime_dir(&dir)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_install_record_round_trips() {
        let record = InstallRecord {
            backend: Backend::Cuda,
            tag: "b10905".into(),
            dir: PathBuf::from("/data/aias/runtime/b10905-cuda"),
            artifacts: vec![Artifact {
                name: "llama-b10905-bin-win-cuda-13.3-x64.zip".into(),
                sha256: "0".repeat(64),
            }],
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"backend\":\"cuda\""));
        assert_eq!(
            serde_json::from_str::<InstallRecord>(&json).unwrap(),
            record
        );
        assert_eq!(record.install().tag, "b10905");
    }

    #[tokio::test]
    async fn an_unreachable_server_is_never_healthy() {
        // Port 1 is outside the platform range and nothing listens on it.
        let err = wait_healthy(1, Duration::from_millis(10))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not healthy"));
    }
}
