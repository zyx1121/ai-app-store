//! The stdio MCP bridge an agent talks to (PLAN.md decision D11).
//!
//! `aias mcp` holds no state. Every tool is a thin call to the local API of the
//! running app, so the desktop shell stays the single owner of instances, apps
//! and the memory budget (decision D10). The bridge only shapes the request,
//! names the result and turns a failure into something an agent can act on.
//!
//! The contract is `http://127.0.0.1:40999/v1` with a bearer token from
//! `<data_dir>/api.token`. `AIAS_API_URL` and `AIAS_API_TOKEN` override both,
//! which is what lets a test point the bridge at a mock server.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo};
use rmcp::{ErrorData, ServerHandler, ServiceExt as _, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{Error, Result as CoreResult};

/// Port the running app serves the local API on, PLAN.md section 5.
pub const DEFAULT_API_PORT: u16 = 40999;
/// Overrides the base URL, for a test or a non default port.
pub const API_URL_ENV: &str = "AIAS_API_URL";
/// Overrides the token file, for a test or a machine with several data dirs.
pub const API_TOKEN_ENV: &str = "AIAS_API_TOKEN";
/// Bearer token of the local API, written by the app on first launch.
pub const TOKEN_FILE: &str = "api.token";

/// The one sentence every tool answers with when the app is not listening.
///
/// It is the same text for every tool on purpose: an agent that sees it knows
/// the platform is down rather than that this particular call is unsupported.
pub const NOT_RUNNING: &str =
    "AI App Store is not running. Start the app (or run: aias api serve) and try again.";

/// One HTTP call gives up here. A build or a pull is a job, never a long request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// `app_run` waits at most this long for the build job it started.
const BUILD_TIMEOUT: Duration = Duration::from_secs(600);
/// Gap between two `jobs/<id>` polls while `app_run` waits.
const JOB_POLL: Duration = Duration::from_secs(2);
/// Default tail length of `app_logs`, the same default as the API route.
const DEFAULT_LOG_TAIL: u32 = 200;

/// Why a call to the local API did not produce a result.
#[derive(Debug)]
enum ApiFailure {
    /// Nothing is listening on the API port, or it stopped mid call.
    Unreachable,
    /// The API answered with a 4xx or a 5xx and its `{code, message}` body.
    Api { code: String, message: String },
    /// The bridge itself could not carry out the call.
    Local(String),
}

impl ApiFailure {
    /// What the agent reads. Every failure is a tool error, never a protocol
    /// error: the agent has to see the text to know what to do next.
    fn text(&self) -> String {
        match self {
            ApiFailure::Unreachable => NOT_RUNNING.to_string(),
            ApiFailure::Api { code, message } => format!("{code}: {message}"),
            ApiFailure::Local(message) => message.clone(),
        }
    }
}

type ApiResult<T> = std::result::Result<T, ApiFailure>;

/// A client of the running app's local API.
///
/// Cloning is cheap: `reqwest::Client` is an Arc over one connection pool.
#[derive(Debug, Clone)]
pub struct ApiClient {
    base_url: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl ApiClient {
    /// Read the base URL and the token out of the environment and the data dir.
    ///
    /// A missing token file is not an error here. The call then goes out
    /// without a bearer, the API answers 401, and the agent is told that
    /// rather than being told the bridge is misconfigured.
    pub fn from_env(data_dir: &Path) -> CoreResult<Self> {
        let base_url = std::env::var(API_URL_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("http://127.0.0.1:{DEFAULT_API_PORT}/v1"));
        let token = match std::env::var(API_TOKEN_ENV) {
            Ok(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
            _ => read_token(data_dir),
        };
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|err| Error::Http(format!("could not build the API client: {err}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
            http,
        })
    }

    /// `GET <base>/<path>` with the query appended.
    async fn get(&self, path: &str, query: &[(&str, String)]) -> ApiResult<Value> {
        let request = self
            .http
            .get(format!(
                "{}/{}",
                self.base_url,
                path.trim_start_matches('/')
            ))
            .query(query);
        self.send(request).await
    }

    /// `POST <base>/<path>` with a JSON body.
    async fn post(&self, path: &str, body: Value) -> ApiResult<Value> {
        let request = self
            .http
            .post(format!(
                "{}/{}",
                self.base_url,
                path.trim_start_matches('/')
            ))
            .json(&body);
        self.send(request).await
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> ApiResult<Value> {
        let request = match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        };
        // Every transport level failure is the same answer: the port is not
        // answering, so the app is not running.
        let response = request.send().await.map_err(|_| ApiFailure::Unreachable)?;
        let status = response.status();
        let body = response.text().await.map_err(|_| ApiFailure::Unreachable)?;
        if status.is_success() {
            if body.trim().is_empty() {
                return Ok(Value::Null);
            }
            return serde_json::from_str(&body).map_err(|err| {
                ApiFailure::Local(format!(
                    "the API answered with something that is not JSON: {err}"
                ))
            });
        }
        Err(api_error(status, &body))
    }
}

/// Turn an error response into `{code, message}`, whatever shape it arrived in.
fn api_error(status: reqwest::StatusCode, body: &str) -> ApiFailure {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().and_then(|value| value.get("error"));
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| status.as_str())
        .to_string();
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            let body = body.trim();
            if body.is_empty() {
                format!("the API answered {status}")
            } else {
                body.to_string()
            }
        });
    ApiFailure::Api { code, message }
}

/// The bearer token written by the app at `<data_dir>/api.token`.
fn read_token(data_dir: &Path) -> Option<String> {
    let token = std::fs::read_to_string(data_dir.join(TOKEN_FILE)).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// `{name}` or `{dir}`, the two ways the API addresses one app.
///
/// Exactly one of them is required. A name is resolved by the server under its
/// apps directory; a dir is for a repo the agent just created somewhere else.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppTarget {
    /// Manifest name of an installed app, `^[a-z][a-z0-9-]{0,50}$`.
    pub name: Option<String>,
    /// Absolute path of a checked out app repo. Use this for a repo you just
    /// created outside the platform's apps directory.
    pub dir: Option<String>,
}

impl AppTarget {
    fn body(&self) -> ApiResult<Value> {
        match (self.name.as_deref(), self.dir.as_deref()) {
            (Some(name), None) => Ok(json!({ "name": name })),
            (None, Some(dir)) => Ok(json!({ "dir": dir })),
            (Some(_), Some(_)) => Err(ApiFailure::Local(
                "pass either `name` or `dir`, not both: a name is resolved under the platform's apps directory, a dir is a repo somewhere else".into(),
            )),
            (None, None) => Err(ApiFailure::Local(
                "pass `name` for an installed app or `dir` for a checked out repo".into(),
            )),
        }
    }
}

/// Input of `store_search_models`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchModelsRequest {
    /// Free text matched against Hugging Face repo names, for example `qwen3`.
    pub query: String,
    /// Narrow the result to one kind: `llm` or `vlm`.
    pub kind: Option<String>,
}

/// Input of `store_search_apps`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchAppsRequest {
    /// Free text matched against the name, description and tags of the index
    /// entries. Omit it to list every published app.
    pub query: Option<String>,
}

/// Input of `model_pull`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ModelPullRequest {
    /// Hugging Face repo holding the GGUF files, for example `Qwen/Qwen3-8B-GGUF`.
    pub repo: String,
    /// Quant file name to download, for example `Q4_K_M`.
    pub quant: String,
}

/// Input of `job_status`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JobStatusRequest {
    /// The `jobId` a pull or a build returned.
    pub job_id: String,
}

/// Input of `app_init`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppInitRequest {
    /// Name of the app, `^[a-z][a-z0-9-]{0,50}$`. It becomes the manifest name.
    pub name: String,
    /// `next` for the Next.js template, `fastapi` for the Python one.
    pub template: String,
    /// Absolute path of the directory to scaffold into.
    pub dir: String,
}

/// Input of `app_run`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppRunRequest {
    /// Manifest name of an installed app.
    pub name: Option<String>,
    /// Absolute path of a checked out app repo.
    pub dir: Option<String>,
    /// HTTPS git URL to clone first. The clone is what is then built and run,
    /// so `name` and `dir` are not needed with it.
    pub url: Option<String>,
    /// Branch or tag to check out with `url`, `main` when omitted.
    #[serde(rename = "ref")]
    pub git_ref: Option<String>,
}

/// Input of `app_logs`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppLogsRequest {
    /// Manifest name of the app.
    pub name: String,
    /// `build` for the output of the manifest build commands, `run` for the
    /// app's own stdout and stderr. Defaults to `run`.
    pub kind: Option<String>,
    /// How many lines from the end, 200 when omitted.
    pub tail: Option<u32>,
}

/// Input of `app_stop`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppStopRequest {
    /// Manifest name of the running app.
    pub name: String,
}

/// Input of `app_fork`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppForkRequest {
    /// An installed app name or an HTTPS git URL to copy from.
    pub source: String,
    /// Name of the new app, `^[a-z][a-z0-9-]{0,50}$`.
    pub name: String,
    /// Absolute path of the directory the fork is written to.
    pub dir: String,
}

/// Input of `app_publish`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AppPublishRequest {
    /// Absolute path of the app repo to publish. It must be committed and
    /// pushed to an HTTPS GitHub remote.
    pub dir: String,
}

/// The stdio MCP server. One process per agent session, no state of its own.
#[derive(Debug, Clone)]
pub struct AiasServer {
    api: ApiClient,
    data_dir: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl AiasServer {
    /// Build a server that talks to the API described by the environment.
    pub fn new(data_dir: PathBuf) -> CoreResult<Self> {
        Ok(Self {
            api: ApiClient::from_env(&data_dir)?,
            data_dir,
            tool_router: Self::tool_router(),
        })
    }

    /// Search Hugging Face for GGUF model repos this machine can actually run,
    /// each marked ready, maybe or incompatible against its memory budget. Use
    /// it before writing a manifest, to pick the repo and the quant an app will
    /// declare, and to check that a model the user asked for exists as GGUF.
    /// Returns a page of repos with their files, sizes and fit badge.
    #[tool]
    async fn store_search_models(
        &self,
        Parameters(request): Parameters<SearchModelsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut query = vec![("q", request.query)];
        if let Some(kind) = request.kind {
            query.push(("kind", kind));
        }
        reply(self.api.get("models/search", &query).await)
    }

    /// Search the store index, the published apps a user can install. Use it to
    /// find an app to fork, to check whether a name is already taken before
    /// publishing, and to show the user what exists. Returns the index entries
    /// with their name, repo URL, ref, description and tags.
    #[tool]
    async fn store_search_apps(
        &self,
        Parameters(request): Parameters<SearchAppsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let entries = match self.api.get("apps/index", &[]).await {
            Ok(entries) => entries,
            Err(failure) => return reply(Err(failure)),
        };
        reply(Ok(filter_index(entries, request.query.as_deref())))
    }

    /// Download one quant of one Hugging Face GGUF repo onto this machine. Use
    /// it when an app declares a model that is not downloaded yet, which is
    /// what `app_run` reports before it refuses to start. The download is long,
    /// so this returns a `jobId` immediately: poll it with `job_status`.
    #[tool]
    async fn model_pull(
        &self,
        Parameters(request): Parameters<ModelPullRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(
            self.api
                .post(
                    "models/pull",
                    json!({ "repo": request.repo, "quant": request.quant }),
                )
                .await,
        )
    }

    /// Report the progress and the result of a job started by `model_pull` or
    /// by a build. Use it in a loop, a few seconds apart, until `state` is
    /// `done` or `failed`. Returns the state, the downloaded and total bytes
    /// while it runs, and either the result or the error at the end.
    #[tool]
    async fn job_status(
        &self,
        Parameters(request): Parameters<JobStatusRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(self.api.get(&format!("jobs/{}", request.job_id), &[]).await)
    }

    /// Scaffold a new app repo from one of the two templates, with a valid
    /// `aias.yaml`, a Dockerfile and the model wiring already in place. Use it
    /// as the first step of building an app, before editing any code. Returns
    /// the directory it wrote and the manifest it generated.
    #[tool]
    async fn app_init(
        &self,
        Parameters(request): Parameters<AppInitRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(
            self.api
                .post(
                    "apps/init",
                    json!({
                        "name": request.name,
                        "template": request.template,
                        "dir": request.dir,
                    }),
                )
                .await,
        )
    }

    /// Check an app repo against the manifest rules: required fields, unique
    /// model aliases, no inference parameters, a Dockerfile at the root. Use it
    /// after every edit of `aias.yaml` and before `app_run` or `app_publish`,
    /// because both refuse an invalid manifest. Returns the parsed manifest, or
    /// an error naming the rule that was broken.
    #[tool]
    async fn app_validate(
        &self,
        Parameters(request): Parameters<AppTarget>,
    ) -> Result<CallToolResult, ErrorData> {
        let body = match request.body() {
            Ok(body) => body,
            Err(failure) => return reply(Err(failure)),
        };
        reply(self.api.post("apps/validate", body).await)
    }

    /// Build and start an app, then hand back the URL to open. Use it to try an
    /// app after editing it. It clones first when given a `url`, checks the
    /// models it declares, builds it, waits for the build, acquires the model
    /// leases, applies migrations and starts the process. When a declared model
    /// is not downloaded it returns that list instead of starting: call
    /// `model_pull` for each one, wait with `job_status`, then call this again.
    #[tool]
    async fn app_run(
        &self,
        Parameters(request): Parameters<AppRunRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(self.run_app(request).await)
    }

    /// Read the tail of one of an app's two logs: `build` for the output of the
    /// manifest build commands, `run` for the app's own stdout and stderr. Use
    /// it whenever `app_run` fails or the app behaves oddly, since this is the
    /// only place the failure is written down. Returns the last lines as text.
    /// Secrets the platform generated are masked: model keys, the password in
    /// a `DATABASE_URL` and the token of an `Authorization: Bearer` header all
    /// come back as `***`.
    #[tool]
    async fn app_logs(
        &self,
        Parameters(request): Parameters<AppLogsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let query = [
            ("kind", request.kind.unwrap_or_else(|| "run".into())),
            ("tail", request.tail.unwrap_or(DEFAULT_LOG_TAIL).to_string()),
        ];
        let path = format!("apps/{}/logs", request.name);
        match self.api.get(&path, &query).await {
            Ok(value) => {
                let text = value
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string());
                Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
            }
            Err(failure) => reply(Err(failure)),
        }
    }

    /// Stop a running app and release the leases it holds on its models. Use it
    /// when the user is done with an app, and before rebuilding one that is
    /// already running. Returns an empty object once the process is gone.
    #[tool]
    async fn app_stop(
        &self,
        Parameters(request): Parameters<AppStopRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(
            self.api
                .post("apps/stop", json!({ "name": request.name }))
                .await,
        )
    }

    /// Copy a published or installed app into a new repo under a new name, with
    /// a fresh git history. Use it when the user wants their own version of an
    /// app: fork first, then edit the prompt, the schema or the UI, then
    /// `app_run` and `app_publish`. Returns the new directory and its manifest.
    #[tool]
    async fn app_fork(
        &self,
        Parameters(request): Parameters<AppForkRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        reply(
            self.api
                .post(
                    "apps/fork",
                    json!({
                        "source": request.source,
                        "name": request.name,
                        "dir": request.dir,
                    }),
                )
                .await,
        )
    }

    /// Publish an app to the store index. It validates the manifest, checks the
    /// repo is committed and pushed to an HTTPS GitHub remote, builds the index
    /// entry, and opens a pull request against the index repo through `gh`. Use
    /// it as the last step, after the app runs. Without `gh` it returns the
    /// entry and the URL to paste it into by hand. This runs locally and works
    /// whether or not the platform is running.
    #[tool]
    async fn app_publish(
        &self,
        Parameters(request): Parameters<AppPublishRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let outcome = crate::publish::publish(Path::new(&request.dir)).await;
        match outcome {
            Ok(result) => reply(Ok(to_value(&result))),
            Err(err) => reply(Err(ApiFailure::Api {
                code: err.code().to_string(),
                message: err.to_string(),
            })),
        }
    }

    /// Clone if asked, refuse early when a model is missing, build, start.
    ///
    /// The polling loop lives here and not in the agent because a build is the
    /// one step with no useful intermediate state: an agent that polls it by
    /// hand only burns turns. A pull is the opposite, which is why `model_pull`
    /// returns a job id and this one does not.
    async fn run_app(&self, request: AppRunRequest) -> ApiResult<Value> {
        let target = match request.url.as_deref() {
            Some(url) => {
                let mut body = json!({ "url": url });
                if let Some(git_ref) = request.git_ref.as_deref() {
                    body["ref"] = json!(git_ref);
                }
                let app = self.api.post("apps/clone", body).await?;
                let name = app
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ApiFailure::Local(format!("the clone of {url} has no name")))?;
                json!({ "name": name })
            }
            None => AppTarget {
                name: request.name,
                dir: request.dir,
            }
            .body()?,
        };

        let missing = self.api.post("apps/missing", target.clone()).await?;
        if missing.as_array().is_some_and(|list| !list.is_empty()) {
            return Ok(json!({
                "started": false,
                "missingModels": missing,
                "instruction": "These models are not downloaded. Call model_pull for each one, poll job_status until every job is done, then call app_run again.",
            }));
        }

        let build = self.api.post("apps/build", target.clone()).await?;
        if let Some(job_id) = build.get("jobId").and_then(Value::as_str) {
            self.wait_for_job(job_id).await?;
        }

        let process = self.api.post("apps/start", target).await?;
        let port = process.get("port").cloned().unwrap_or(Value::Null);
        let url = process
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_default();
        Ok(json!({
            "started": true,
            "name": process.get("name").cloned().unwrap_or(Value::Null),
            "url": url,
            "port": port,
            "instruction": "Open the URL in a browser to test the app. Read app_logs when it misbehaves.",
        }))
    }

    /// Poll one job until it finishes, giving up after [`BUILD_TIMEOUT`].
    async fn wait_for_job(&self, job_id: &str) -> ApiResult<Value> {
        let deadline = Instant::now() + BUILD_TIMEOUT;
        loop {
            let job = self.api.get(&format!("jobs/{job_id}"), &[]).await?;
            match job.get("state").and_then(Value::as_str) {
                Some("done") => return Ok(job),
                Some("failed") => return Err(job_failure(job_id, &job)),
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(ApiFailure::Api {
                    code: "build_timeout".into(),
                    message: format!(
                        "the build did not finish within {} seconds. It may still be running: poll job_status with `{job_id}`.",
                        BUILD_TIMEOUT.as_secs()
                    ),
                });
            }
            tokio::time::sleep(JOB_POLL).await;
        }
    }

    /// Where this bridge looks for `api.token`.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for AiasServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }
}

/// What the agent reads once, before any tool is called.
const INSTRUCTIONS: &str = "\
AI App Store builds and runs local AI apps on this machine. An app is a git repo with an \
aias.yaml manifest that declares which models it needs; the platform downloads the models, \
starts one llama-server per model and hands the app a loopback port and an OpenAI compatible \
URL in AIAS_MODEL_<ALIAS>_URL. The usual path is app_init, edit the code, app_run, test in a \
browser, app_publish. Every tool here forwards to the running app; when it is not running they \
all answer with the same sentence saying to start it.";

/// What a `failed` job says, as a failure this bridge can report.
///
/// `error` is the `{code, message}` pair of the API contract, never a string.
/// Reading it as one turned every failed build into the same bare sentence and
/// left the agent with nothing to act on, which is the whole value of the tool.
fn job_failure(job_id: &str, job: &Value) -> ApiFailure {
    let error = job.get("error");
    let field = |name: &str| {
        error
            .and_then(|error| error.get(name))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    ApiFailure::Api {
        code: field("code").unwrap_or_else(|| "build_failed".into()),
        message: format!(
            "{}. Read the build log with app_logs.",
            field("message").unwrap_or_else(|| format!("job {job_id} failed"))
        ),
    }
}

/// Serve the MCP protocol on stdin and stdout until the client disconnects.
pub async fn serve_stdio(data_dir: PathBuf) -> CoreResult<()> {
    let server = AiasServer::new(data_dir)?;
    let running = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|err| Error::Process(format!("could not start the MCP server: {err}")))?;
    running
        .waiting()
        .await
        .map_err(|err| Error::Process(format!("the MCP server stopped: {err}")))?;
    Ok(())
}

/// One result shape for every tool: JSON on success, text on failure.
fn reply(result: ApiResult<Value>) -> Result<CallToolResult, ErrorData> {
    match result {
        Ok(value) => {
            let text = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
            Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
        }
        // A failed call is a tool error and not a protocol error: a protocol
        // error is rendered opaquely by the client, and the whole point of the
        // `not running` sentence is that the agent reads it.
        Err(failure) => Ok(CallToolResult::error(vec![ContentBlock::text(
            failure.text(),
        )])),
    }
}

/// Serialize a core value, falling back to its debug text.
fn to_value<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Keep the index entries that match a free text query.
///
/// The API serves the whole index, which is a few kilobytes, so the filter is
/// here rather than in a route: the agent asked a question, not for a page.
fn filter_index(entries: Value, query: Option<&str>) -> Value {
    let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
        return entries;
    };
    let Value::Array(entries) = entries else {
        return entries;
    };
    let needle = query.to_lowercase();
    let kept: Vec<Value> = entries
        .into_iter()
        .filter(|entry| entry_matches(entry, &needle))
        .collect();
    Value::Array(kept)
}

fn entry_matches(entry: &Value, needle: &str) -> bool {
    let field = |key: &str| {
        entry
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_lowercase()
    };
    if field("name").contains(needle) || field("description").contains(needle) {
        return true;
    }
    entry
        .get("tags")
        .and_then(Value::as_array)
        .is_some_and(|tags| {
            tags.iter()
                .filter_map(Value::as_str)
                .any(|tag| tag.to_lowercase().contains(needle))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Value {
        json!([
            { "name": "contract-review", "description": "Flag deviations", "tags": ["legal"] },
            { "name": "note-taker", "description": "Summarize meetings", "tags": ["office"] },
        ])
    }

    #[test]
    fn an_absent_query_keeps_every_entry() {
        assert_eq!(filter_index(entries(), None), entries());
        assert_eq!(filter_index(entries(), Some("  ")), entries());
    }

    #[test]
    fn a_query_matches_name_description_and_tags() {
        let by_name = filter_index(entries(), Some("Contract"));
        assert_eq!(by_name.as_array().unwrap().len(), 1);
        let by_description = filter_index(entries(), Some("summarize"));
        assert_eq!(by_description.as_array().unwrap().len(), 1);
        let by_tag = filter_index(entries(), Some("legal"));
        assert_eq!(by_tag.as_array().unwrap().len(), 1);
        assert!(
            filter_index(entries(), Some("nothing"))
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_target_is_a_name_or_a_dir_but_not_both() {
        let by_name = AppTarget {
            name: Some("chat".into()),
            dir: None,
        };
        assert_eq!(by_name.body().unwrap(), json!({ "name": "chat" }));
        let by_dir = AppTarget {
            name: None,
            dir: Some("/tmp/chat".into()),
        };
        assert_eq!(by_dir.body().unwrap(), json!({ "dir": "/tmp/chat" }));
        let both = AppTarget {
            name: Some("chat".into()),
            dir: Some("/tmp/chat".into()),
        };
        assert!(both.body().is_err());
        let neither = AppTarget {
            name: None,
            dir: None,
        };
        assert!(neither.body().is_err());
    }

    #[test]
    fn an_error_body_becomes_its_code_and_message() {
        let failure = api_error(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error":{"code":"not_found","message":"app `chat`"}}"#,
        );
        assert_eq!(failure.text(), "not_found: app `chat`");
    }

    #[test]
    fn an_error_without_a_body_still_names_the_status() {
        let failure = api_error(reqwest::StatusCode::UNAUTHORIZED, "");
        assert_eq!(failure.text(), "401: the API answered 401 Unauthorized");
    }

    #[test]
    fn a_failed_job_is_reported_with_the_code_and_the_message_of_its_error() {
        let job = json!({
            "id": "abc",
            "kind": "appsBuild",
            "state": "failed",
            "error": {
                "code": "process_failed",
                "message": "could not run `bun`: No such file or directory (os error 2)",
            },
        });
        let failure = job_failure("abc", &job);
        assert_eq!(
            failure.text(),
            "process_failed: could not run `bun`: No such file or directory (os error 2). \
             Read the build log with app_logs."
        );

        // A job without an error is still worth a sentence, not a panic.
        let bare = json!({ "id": "abc", "state": "failed" });
        assert_eq!(
            job_failure("abc", &bare).text(),
            "build_failed: job abc failed. Read the build log with app_logs."
        );
    }

    #[test]
    fn every_unreachable_api_answers_with_one_sentence() {
        assert_eq!(ApiFailure::Unreachable.text(), NOT_RUNNING);
    }

    #[test]
    fn the_base_url_comes_from_the_environment_or_the_default() {
        // The env var is process wide, so this test owns it and puts it back.
        let previous = std::env::var(API_URL_ENV).ok();
        unsafe {
            std::env::remove_var(API_URL_ENV);
        }
        let client = ApiClient::from_env(Path::new("/nowhere")).unwrap();
        assert_eq!(
            client.base_url,
            format!("http://127.0.0.1:{DEFAULT_API_PORT}/v1")
        );
        unsafe {
            std::env::set_var(API_URL_ENV, "http://127.0.0.1:41234/v1/");
        }
        let client = ApiClient::from_env(Path::new("/nowhere")).unwrap();
        assert_eq!(client.base_url, "http://127.0.0.1:41234/v1");
        unsafe {
            match previous {
                Some(value) => std::env::set_var(API_URL_ENV, value),
                None => std::env::remove_var(API_URL_ENV),
            }
        }
    }

    #[test]
    fn the_token_falls_back_to_the_data_dir_file() {
        let dir = std::env::temp_dir().join(format!("aias-mcp-token-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(TOKEN_FILE), "  secret\n").unwrap();
        assert_eq!(read_token(&dir), Some("secret".to_string()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
