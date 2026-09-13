//! The local API, `http://127.0.0.1:40999/v1`.
//!
//! PLAN.md section 6.1: the running app listens on loopback so the GUI, the
//! CLI and the agent see the same state. The routes mirror the Tauri commands
//! one to one, long operations answer with a job id (see [`crate::jobs`]), and
//! `aias mcp` is a stdio bridge in front of this, never a second owner of the
//! model manager (decision D11).
//!
//! Three rules hold this surface together:
//!
//! 1. **Loopback is not a permission.** Every process on the machine can reach
//!    this port, so every request carries the bearer token from
//!    `<data_dir>/api.token`, which is readable by this user only.
//! 2. **A client never names a path the platform then runs argv out of.** An
//!    app is named and resolved in Rust. The `dir` variants exist so an agent
//!    can work on a repo it just created, and a `dir` is canonicalized and has
//!    to sit under the user's home directory or under `apps_dir`, never inside
//!    the platform's own state.
//! 3. **One error shape.** `{ "error": { "code", "message" } }` with the status
//!    that [`Error::code`] maps to.
//! 4. **A change made here is announced.** The GUI and an agent drive the same
//!    state, so a page that only refetched on mount showed a stale card until
//!    the user navigated away and back (issue #16). Every handler that changes
//!    something calls an optional [`Emitter`] the host passes in; the desktop
//!    shell turns that into the [`CHANGED_EVENT`] Tauri event, and
//!    `aias api serve` has no window and passes `None`.

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path as UrlPath, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use rand::RngExt as _;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::jobs::{self, JobKind};
use crate::models::{self, ModelRef};
use crate::{apps, hardware, instances, paths, runtime};

/// The port PLAN.md section 5 reserves for this API.
pub const DEFAULT_PORT: u16 = 40999;
/// Overrides [`DEFAULT_PORT`], so two data directories can run side by side.
const PORT_ENV: &str = "AIAS_API_PORT";
/// The token file, next to the rest of the platform state.
const TOKEN_FILE: &str = "api.token";
/// Bytes of the token, hex encoded. The same width as an instance key.
const TOKEN_BYTES: usize = 32;
/// Tauri event the desktop shell announces a change on, mirrors
/// `CHANGED_EVENT` in `src/lib/api/events.ts`.
pub const CHANGED_EVENT: &str = "aias://changed";
/// Short git sha of this build, `dev` outside CI.
const GIT_SHA: &str = match option_env!("AIAS_GIT_SHA") {
    Some(sha) => sha,
    None => "dev",
};

/// What changed, the `kind` of a [`CHANGED_EVENT`] payload.
///
/// Coarse on purpose: a listener refetches the page it is on, so the kind is
/// there to let it skip a refetch it does not need, not to describe the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Changed {
    /// A subscribed app started, stopped or finished building.
    Apps,
    /// A model finished downloading.
    Models,
    /// An instance started or stopped, or its leases moved.
    Instances,
}

/// Payload of [`CHANGED_EVENT`], `{"kind": "apps"}`.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangedEvent {
    pub kind: Changed,
}

/// Told what changed, once per change this API made.
///
/// The host owns the delivery: the desktop shell emits a Tauri event from it
/// because the window is in the same process, and a headless host passes
/// `None` because there is nobody to tell. The callback is called from a
/// request handler and from a job, so it has to be cheap and must not block.
pub type Emitter = Arc<dyn Fn(Changed) + Send + Sync>;

/// Tell the host what changed, if there is a host that cares.
fn announce(emit: &Option<Emitter>, kind: Changed) {
    if let Some(emit) = emit {
        emit(kind);
    }
}

/// What every handler is given: where the state lives and what the token is.
#[derive(Clone)]
struct Api {
    data_dir: PathBuf,
    token: Arc<String>,
    /// `None` when the host has no window to tell, which is the CLI.
    emit: Option<Emitter>,
}

impl Api {
    /// Announce a change this API just made.
    fn changed(&self, kind: Changed) {
        announce(&self.emit, kind);
    }
}

/// The token is a bearer token and a `Debug` of the state used to print it.
impl std::fmt::Debug for Api {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Api")
            .field("data_dir", &self.data_dir)
            .field("announces", &self.emit.is_some())
            .finish_non_exhaustive()
    }
}

/// The port this API listens on, `AIAS_API_PORT` first.
pub fn port() -> u16 {
    std::env::var(PORT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// Base URL clients use, the one printed on the Setup page.
pub fn base_url() -> String {
    format!("http://127.0.0.1:{}/v1", port())
}

/// Path of the token file for a data directory.
pub fn token_path(data_dir: &Path) -> PathBuf {
    data_dir.join(TOKEN_FILE)
}

/// The bearer token, generated on first call.
///
/// 32 random bytes as hex, written with an owner only mode: `0600` on Unix, and
/// on Windows an ACL with inheritance broken and one grant to this user, which
/// is the same thing said in the only vocabulary that file system has. Every
/// value reaches `icacls` as an argv element.
///
/// The token is also remembered as a secret, so a log tail that quotes it,
/// from an app that was handed it or a tool that printed a request, comes back
/// masked.
pub fn token(data_dir: &Path) -> Result<String> {
    let path = token_path(data_dir);
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let existing = existing.trim().to_string();
        if !existing.is_empty() {
            instances::remember_secret(&existing);
            return Ok(existing);
        }
    }

    paths::ensure_data_dirs(data_dir)?;
    let mut bytes = [0u8; TOKEN_BYTES];
    rand::rng().fill(&mut bytes);
    let token = hex::encode(bytes);
    std::fs::write(&path, &token)?;
    paths::restrict(&path, 0o600)?;
    instances::remember_secret(&token);
    Ok(token)
}

/// Serve the API until `shutdown` resolves.
///
/// `emit` is how the host hears about a change made through this API. The
/// desktop shell passes one and forwards it to the window as
/// [`CHANGED_EVENT`]; `aias api serve` passes `None`.
///
/// Binds `127.0.0.1` only: PLAN.md section 9 says a listener on a routable
/// address raises the Windows Firewall prompt, and this one has no business
/// leaving the machine in the first place.
pub async fn serve(
    data_dir: &Path,
    shutdown: impl Future<Output = ()> + Send + 'static,
    emit: Option<Emitter>,
) -> Result<()> {
    let state = Api {
        data_dir: data_dir.to_path_buf(),
        token: Arc::new(token(data_dir)?),
        emit,
    };
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port()));
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|err| {
            Error::Process(format!("could not bind the local API on {address}: {err}"))
        })?;

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|err| Error::Process(format!("the local API stopped: {err}")))
}

/// Every route of the contract, behind the bearer check.
fn router(state: Api) -> Router {
    let v1 = Router::new()
        .route("/health", get(health))
        .route("/hardware", get(hardware_detect))
        .route("/runtime", get(runtime_installed))
        .route("/models/search", get(models_search))
        .route("/models/files", post(models_files))
        .route("/models/pull", post(models_pull))
        .route("/models/installed", get(models_installed))
        .route("/instances", get(instances_list))
        .route("/apps/index", get(apps_index))
        .route("/apps/installed", get(apps_installed))
        .route("/apps/clone", post(apps_clone))
        .route("/apps/init", post(apps_init))
        .route("/apps/validate", post(apps_validate))
        .route("/apps/build", post(apps_build))
        .route("/apps/start", post(apps_start))
        .route("/apps/stop", post(apps_stop))
        .route("/apps/missing", post(apps_missing))
        .route("/apps/fork", post(apps_fork))
        .route("/apps/{name}/logs", get(apps_logs))
        .route("/jobs/{id}", get(job))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize))
        .with_state(state);

    Router::new().nest("/v1", v1)
}

/// Reject anything without the bearer token, before any handler runs.
///
/// The comparison is over the whole token in constant time: a byte at a time
/// comparison on a loopback port is a real oracle, and the platform is the only
/// thing standing between another process on this machine and every model,
/// database and app it owns.
async fn authorize(State(state): State<Api>, request: Request, next: Next) -> Response {
    if presented(request.headers()).is_some_and(|token| same_token(&token, &state.token)) {
        return next.run(request).await;
    }
    ApiError(Error::Unauthorized(
        "a bearer token from <data_dir>/api.token is required".into(),
    ))
    .into_response()
}

/// The token of an `Authorization: Bearer <token>` header.
fn presented(headers: &HeaderMap) -> Option<String> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim().to_string())
}

/// Equality of two tokens that does not stop at the first byte that differs.
///
/// A length check first, because two tokens of different lengths are not equal
/// whatever else is true and the length of this one is not the secret.
fn same_token(presented: &str, expected: &str) -> bool {
    let (presented, expected) = (presented.as_bytes(), expected.as_bytes());
    if presented.len() != expected.len() {
        return false;
    }
    presented
        .iter()
        .zip(expected)
        .fold(0u8, |differs, (left, right)| differs | (left ^ right))
        == 0
}

/// What a failed request answers with.
struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(err: Error) -> Self {
        Self(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0.code() {
            "not_found" => StatusCode::NOT_FOUND,
            "invalid_manifest" => StatusCode::BAD_REQUEST,
            "unsupported_hardware" => StatusCode::CONFLICT,
            "unauthorized" => StatusCode::UNAUTHORIZED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = serde_json::json!({
            "error": { "code": self.0.code(), "message": self.0.to_string() }
        });
        (status, Json(body)).into_response()
    }
}

/// Result every handler returns.
type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

// ---------------------------------------------------------------- routes

/// `GET /v1/health`
async fn health() -> ApiResult<serde_json::Value> {
    Ok(Json(serde_json::json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "sha": GIT_SHA,
    })))
}

/// `GET /v1/hardware`
async fn hardware_detect() -> ApiResult<hardware::DeviceProfile> {
    Ok(Json(hardware::detect()?))
}

/// `GET /v1/runtime`, 404 when nothing is installed yet.
async fn runtime_installed(State(state): State<Api>) -> ApiResult<runtime::RuntimeInstall> {
    runtime::installed(&state.data_dir)
        .map(Json)
        .ok_or_else(|| {
            ApiError(Error::NotFound(
                "no llama-server is installed, run the Setup page or `aias runtime install`".into(),
            ))
        })
}

/// Query of `GET /v1/models/search`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchQuery {
    #[serde(default)]
    q: String,
    kind: Option<apps::ModelKind>,
    cursor: Option<String>,
}

/// `GET /v1/models/search?q=&kind=llm|vlm&cursor=`
async fn models_search(Query(query): Query<SearchQuery>) -> ApiResult<models::SearchPage> {
    Ok(Json(
        models::search_kind(&query.q, query.kind, query.cursor.as_deref()).await?,
    ))
}

/// Body of `POST /v1/models/files`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RepoBody {
    repo: String,
}

/// `POST /v1/models/files`
async fn models_files(Json(body): Json<RepoBody>) -> ApiResult<Vec<models::ModelFile>> {
    Ok(Json(models::files(&body.repo).await?))
}

/// `POST /v1/models/pull`, a job.
async fn models_pull(
    State(state): State<Api>,
    Json(model): Json<ModelRef>,
) -> ApiResult<JobStarted> {
    let data_dir = state.data_dir.clone();
    let emit = state.emit.clone();
    let id = jobs::spawn_job(JobKind::ModelsPull, move |id| async move {
        let report = move |progress: models::Progress| jobs::report(&id, progress);
        let path = models::download(&model, &data_dir, &report).await?;
        // On the way out of the job, so a page that refetches on this event
        // reads a model that is already on disk. A failed pull changed
        // nothing and says nothing.
        announce(&emit, Changed::Models);
        Ok(path)
    });
    Ok(Json(JobStarted { job_id: id }))
}

/// What a route that starts a job answers with.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobStarted {
    job_id: String,
}

/// One downloaded model, flat, as `GET /v1/models/installed` reports it.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InstalledModel {
    repo: String,
    quant: String,
    path: PathBuf,
    size_bytes: u64,
    /// The vision projector downloaded with the weights, for a `vlm` repo.
    mmproj: Option<PathBuf>,
}

/// `GET /v1/models/installed`
async fn models_installed(State(state): State<Api>) -> ApiResult<Vec<InstalledModel>> {
    let found = models::installed(&state.data_dir)
        .into_iter()
        .map(|(model, path, size_bytes)| InstalledModel {
            mmproj: models::installed_mmproj(&model, &state.data_dir),
            repo: model.repo,
            quant: model.quant,
            path,
            size_bytes,
        })
        .collect();
    Ok(Json(found))
}

/// `GET /v1/instances`. `Instance::api_key` is never serialized.
async fn instances_list() -> ApiResult<Vec<instances::Instance>> {
    Ok(Json(instances::list()))
}

/// `GET /v1/apps/index`
async fn apps_index() -> ApiResult<Vec<apps::IndexEntry>> {
    Ok(Json(apps::index(&apps::index_url()).await?))
}

/// `GET /v1/apps/installed`
async fn apps_installed(State(state): State<Api>) -> ApiResult<Vec<apps::InstalledApp>> {
    Ok(Json(apps::installed(&state.data_dir)))
}

/// Body of `POST /v1/apps/clone`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CloneBody {
    url: String,
    /// `ref` is a keyword in Rust and a branch or a tag everywhere else.
    #[serde(rename = "ref")]
    git_ref: Option<String>,
}

/// `POST /v1/apps/clone`
async fn apps_clone(
    State(state): State<Api>,
    Json(body): Json<CloneBody>,
) -> ApiResult<apps::InstalledApp> {
    Ok(Json(
        apps::subscribe(&body.url, body.git_ref.as_deref(), &state.data_dir).await?,
    ))
}

/// Body of `POST /v1/apps/init`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitBody {
    name: String,
    template: apps::Template,
    dir: PathBuf,
}

/// `POST /v1/apps/init`
async fn apps_init(
    State(state): State<Api>,
    Json(body): Json<InitBody>,
) -> ApiResult<apps::ScaffoldedApp> {
    let dir = paths::resolve_dir(&body.dir, &state.data_dir)?;
    Ok(Json(apps::init(&body.name, body.template, &dir).await?))
}

/// Body of `POST /v1/apps/fork`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ForkBody {
    /// An installed app name or an https git URL.
    source: String,
    name: String,
    dir: PathBuf,
}

/// `POST /v1/apps/fork`
async fn apps_fork(
    State(state): State<Api>,
    Json(body): Json<ForkBody>,
) -> ApiResult<apps::ScaffoldedApp> {
    let dir = paths::resolve_dir(&body.dir, &state.data_dir)?;
    Ok(Json(
        apps::fork_in(&body.source, &body.name, &dir, &state.data_dir).await?,
    ))
}

/// `{name}` or `{dir}`: the two ways to point at an app.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppTarget {
    name: Option<String>,
    dir: Option<PathBuf>,
}

impl AppTarget {
    /// The directory this target names, checked.
    ///
    /// A name is resolved in Rust against `<data_dir>/apps` and can reach
    /// nothing else. A directory is canonicalized and has to pass
    /// [`paths::resolve_dir`].
    fn dir(&self, data_dir: &Path) -> Result<PathBuf> {
        match (&self.name, &self.dir) {
            (Some(name), None) => apps::app_dir(data_dir, name),
            (None, Some(dir)) => paths::resolve_dir(dir, data_dir),
            _ => Err(Error::InvalidManifest(
                "give exactly one of `name` and `dir`".into(),
            )),
        }
    }
}

/// `POST /v1/apps/validate`
async fn apps_validate(
    State(state): State<Api>,
    Json(body): Json<AppTarget>,
) -> ApiResult<apps::Manifest> {
    let dir = body.dir(&state.data_dir)?;
    Ok(Json(apps::validate_dir(&dir)?))
}

/// `POST /v1/apps/build`, a job.
async fn apps_build(
    State(state): State<Api>,
    Json(body): Json<AppTarget>,
) -> ApiResult<JobStarted> {
    let dir = body.dir(&state.data_dir)?;
    // Validation happens before the job starts, so a manifest that breaks a
    // rule is a 400 on this request and not a job that fails a second later.
    let manifest = apps::validate_dir(&dir)?;
    let data_dir = state.data_dir.clone();
    let emit = state.emit.clone();
    let id = jobs::spawn_job(JobKind::AppsBuild, move |_id| async move {
        apps::build_in(&dir, &manifest, &data_dir).await?;
        announce(&emit, Changed::Apps);
        Ok(serde_json::json!({ "built": manifest.name, "dir": dir }))
    });
    Ok(Json(JobStarted { job_id: id }))
}

/// `POST /v1/apps/start`
async fn apps_start(
    State(state): State<Api>,
    Json(body): Json<AppTarget>,
) -> ApiResult<apps::AppProcess> {
    let dir = body.dir(&state.data_dir)?;
    let manifest = apps::validate_dir(&dir)?;
    let started = apps::start_in(&dir, &manifest, &state.data_dir).await?;
    // A start takes a lease on every model the manifest declares, so the
    // Models page is as stale as the Apps page after this.
    state.changed(Changed::Apps);
    state.changed(Changed::Instances);
    Ok(Json(started))
}

/// Body of `POST /v1/apps/stop`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NameBody {
    name: String,
}

/// `POST /v1/apps/stop`
async fn apps_stop(
    State(state): State<Api>,
    Json(body): Json<NameBody>,
) -> ApiResult<serde_json::Value> {
    apps::stop(&body.name).await?;
    // The stop gave back every lease the app held, which is the case issue #16
    // was opened for: `app_stop` through `aias mcp` left a Running card up.
    state.changed(Changed::Apps);
    state.changed(Changed::Instances);
    Ok(Json(serde_json::json!({})))
}

/// `POST /v1/apps/missing`
async fn apps_missing(
    State(state): State<Api>,
    Json(body): Json<AppTarget>,
) -> ApiResult<Vec<ModelRef>> {
    let dir = body.dir(&state.data_dir)?;
    let manifest = apps::validate_dir(&dir)?;
    Ok(Json(apps::missing_models(&manifest, &state.data_dir)))
}

/// Query of `GET /v1/apps/{name}/logs`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LogsQuery {
    /// `build` or `run`, `run` by default.
    kind: Option<String>,
    /// Lines of the tail.
    tail: Option<usize>,
}

/// `GET /v1/apps/{name}/logs?kind=build|run&tail=200`
async fn apps_logs(
    State(state): State<Api>,
    UrlPath(name): UrlPath<String>,
    Query(query): Query<LogsQuery>,
) -> ApiResult<serde_json::Value> {
    let kind = match query.kind.as_deref() {
        None | Some("run") => apps::LogKind::Run,
        Some("build") => apps::LogKind::Build,
        Some(other) => {
            return Err(ApiError(Error::InvalidManifest(format!(
                "log kind `{other}` is not `build` or `run`"
            ))));
        }
    };
    let text = apps::logs_in(&name, kind, query.tail.unwrap_or(200), &state.data_dir)?;
    Ok(Json(serde_json::json!({ "text": text })))
}

/// `GET /v1/jobs/{id}`
async fn job(UrlPath(id): UrlPath<String>) -> ApiResult<jobs::Job> {
    jobs::get(&id)
        .map(Json)
        .ok_or_else(|| ApiError(Error::NotFound(format!("job `{id}`"))))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// A state with an emitter that records what it was told, and the record.
    fn watched() -> (Api, Arc<Mutex<Vec<Changed>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let state = Api {
            data_dir: PathBuf::from("/nowhere"),
            token: Arc::new("t".repeat(64)),
            emit: Some(Arc::new(move |kind| {
                sink.lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(kind);
            })),
        };
        (state, seen)
    }

    #[test]
    fn a_host_with_a_window_is_told_what_changed() {
        let (state, seen) = watched();
        state.changed(Changed::Apps);
        state.changed(Changed::Instances);
        announce(&state.emit, Changed::Models);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Changed::Apps, Changed::Instances, Changed::Models]
        );
    }

    #[test]
    fn a_host_without_a_window_is_not_told_anything() {
        // `aias api serve` passes no emitter, and a change is then a no-op
        // rather than a panic or a channel nobody reads.
        let state = Api {
            data_dir: PathBuf::from("/nowhere"),
            token: Arc::new("t".repeat(64)),
            emit: None,
        };
        state.changed(Changed::Apps);
        announce(&None, Changed::Models);
    }

    #[tokio::test]
    async fn a_change_that_did_not_happen_is_not_announced() {
        let (state, seen) = watched();
        // Nothing is running, so the stop fails and the window must not be
        // told to refetch a state that did not move.
        let err = apps_stop(
            State(state),
            Json(NameBody {
                name: "never-started".into(),
            }),
        )
        .await
        .expect_err("stopping an app that is not running fails");
        assert_eq!(err.0.code(), "not_found");
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn the_event_payload_is_the_kind_the_frontend_switches_on() {
        let payload = |kind| serde_json::to_string(&ChangedEvent { kind }).unwrap();
        assert_eq!(payload(Changed::Apps), r#"{"kind":"apps"}"#);
        assert_eq!(payload(Changed::Models), r#"{"kind":"models"}"#);
        assert_eq!(payload(Changed::Instances), r#"{"kind":"instances"}"#);
        assert_eq!(CHANGED_EVENT, "aias://changed");
    }

    #[test]
    fn the_debug_of_the_state_does_not_print_the_bearer_token() {
        let (state, _) = watched();
        let printed = format!("{state:?}");
        assert!(!printed.contains(&"t".repeat(64)), "{printed}");
        assert!(printed.contains("announces: true"), "{printed}");
    }

    #[test]
    fn the_error_codes_map_onto_the_statuses_of_the_contract() {
        let status = |err: Error| ApiError(err).into_response().status();
        assert_eq!(status(Error::NotFound("x".into())), StatusCode::NOT_FOUND);
        assert_eq!(
            status(Error::InvalidManifest("x".into())),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            status(Error::UnsupportedHardware("x".into())),
            StatusCode::CONFLICT
        );
        assert_eq!(
            status(Error::Unauthorized("x".into())),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(Error::Process("x".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            status(Error::Http("x".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn only_a_bearer_scheme_presents_a_token() {
        let header = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(
                axum::http::header::AUTHORIZATION,
                value.parse().expect("a header value"),
            );
            presented(&headers)
        };
        assert_eq!(header("Bearer abc"), Some("abc".to_string()));
        assert_eq!(header("bearer abc"), Some("abc".to_string()));
        assert_eq!(header("Basic abc"), None);
        assert_eq!(header("abc"), None);
        assert_eq!(presented(&HeaderMap::new()), None);
    }

    #[test]
    fn a_token_is_compared_whole() {
        let token = "a".repeat(64);
        assert!(same_token(&token, &token));
        assert!(!same_token(&token, &"a".repeat(63)));
        assert!(!same_token(&"a".repeat(63), &token));
        assert!(!same_token("", &token));
        assert!(!same_token(&format!("{token}b"), &token));
    }

    #[test]
    fn a_token_file_is_created_once_and_kept() {
        let dir = std::env::temp_dir().join(format!("aias-api-token-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = token(&dir).expect("a token is generated");
        assert_eq!(first.len(), TOKEN_BYTES * 2);
        assert_eq!(token(&dir).unwrap(), first, "the token is stable");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(token_path(&dir))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "the token is readable by this user only"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
