//! The manifest and the app lifecycle.
//!
//! `aias.yaml` is the whole contract between an app and the platform. This
//! module parses and validates it, subscribes to a repo (clone, validate,
//! build), starts and stops the process, and reads the store index.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use include_dir::{Dir, DirEntry, include_dir};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::instances::{self, InstanceUrl};
use crate::models::ModelRef;
use crate::paths;
use crate::process::{APP_ALLOWLIST, Quiet as _, Sealed as _};
use crate::services;

/// File name of the manifest at the repo root.
pub const MANIFEST_FILE: &str = "aias.yaml";

/// Keys that are rejected at validate time, rule 4 of PLAN.md section 3.
///
/// Sampling belongs in the request body, everything else belongs to the platform.
pub const REJECTED_KEYS: &[&str] = &[
    "ctx",
    "ctx_size",
    "n_parallel",
    "parallel",
    "gpu_layers",
    "n_gpu_layers",
    "ngl",
    "threads",
    "batch_size",
    "kv_cache_type",
    "flash_attn",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "port",
    "host",
];

/// Separator element allowed inside `build`, splitting it into commands.
const COMMAND_SEPARATOR: &str = "&&";

/// The one Postgres cluster owns the first port of the app half of the range.
pub const APP_PORT_START: u16 = 41500;
/// Last port of the platform range, see [`crate::instances::PORT_RANGE_END`].
pub const APP_PORT_END: u16 = 41999;

/// An app has this long to answer its health endpoint, PLAN.md section 3.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(60);
/// Gap between two health polls while the app boots.
const HEALTH_POLL: Duration = Duration::from_millis(500);
/// One health poll gives up quickly, the loop is what waits.
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
/// Lines of the log put in an error message.
const LOG_TAIL_LINES: usize = 25;
/// The only address an app is allowed to listen on, PLAN.md section 3.
const LOOPBACK: &str = "127.0.0.1";

/// The store index, PLAN.md decision D5: a git repo of manifests, not images.
const DEFAULT_INDEX_URL: &str =
    "https://raw.githubusercontent.com/zyx1121/aias-index/main/apps.yaml";
/// Overrides [`DEFAULT_INDEX_URL`], for a fork or a local index.
const INDEX_URL_ENV: &str = "AIAS_INDEX_URL";
/// The index is small, so one short timeout is enough.
const INDEX_TIMEOUT: Duration = Duration::from_secs(30);
/// Written next to a clone so [`installed`] can name the repo it came from.
const ORIGIN_FILE: &str = ".aias-origin";

/// Longest app name, first character included: `^[a-z][a-z0-9-]{0,50}$`.
const MAX_APP_NAME_LEN: usize = 51;

/// Which interpreter the platform starts the app under (decision D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppRuntime {
    Node,
    Python,
}

/// What a declared model is used for. Version 2 adds `stt`, `tts` and `image`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Llm,
    Vlm,
}

/// One model an app declares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDecl {
    /// Becomes the `AIAS_MODEL_<ALIAS>_URL` suffix in upper case.
    pub alias: String,
    pub kind: ModelKind,
    /// Hugging Face repo holding the GGUF files.
    pub repo: String,
    /// Ordered preference, the platform picks the first that fits.
    pub quant: Vec<String>,
    /// Tried with the same quant list when nothing above fits.
    pub fallback: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_yaml_ng::Value>,
}

impl ModelDecl {
    /// The env var suffix this alias produces.
    pub fn env_suffix(&self) -> String {
        self.alias.to_uppercase()
    }
}

/// The Postgres dependency an app can declare.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostgresService {
    /// Folder of `*.sql`, applied in filename order on every start.
    pub migrations: PathBuf,
}

/// Platform provided dependencies.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Services {
    pub postgres: Option<PostgresService>,
}

/// One secret the user is asked for once at install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Secret {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// A parsed `aias.yaml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub description: String,
    pub version: String,
    pub license: Option<String>,
    pub homepage: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub runtime: AppRuntime,
    #[serde(default)]
    pub build: Vec<String>,
    pub start: Vec<String>,
    /// Path of the health endpoint, for example `/api/health`.
    pub health: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelDecl>,
    #[serde(default)]
    pub services: Services,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub secrets: Vec<Secret>,
    /// Keys the schema does not know. Validation rejects every one of them.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_yaml_ng::Value>,
}

/// One entry of the store index, `apps.yaml` in the index repo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexEntry {
    pub name: String,
    /// HTTPS git URL the machine clones and builds locally (decision D5).
    pub repo: String,
    /// Branch or tag to clone, `main` when the entry omits it.
    ///
    /// Checked while the index is parsed, so no [`IndexEntry`] can carry a ref
    /// git would read as an option. See [`check_git_ref`].
    #[serde(
        rename = "ref",
        default = "default_ref",
        deserialize_with = "de_git_ref"
    )]
    pub git_ref: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub homepage: Option<String>,
}

fn default_ref() -> String {
    "main".to_string()
}

/// Reject a bad `ref` where the index is read, not where git is run.
fn de_git_ref<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    check_git_ref(&value).map_err(serde::de::Error::custom)?;
    Ok(value)
}

/// `apps.yaml` as it is published: a bare list, or a list under `apps:`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IndexFile {
    Wrapped { apps: Vec<IndexEntry> },
    Bare(Vec<IndexEntry>),
}

/// One subscribed app: the clone on disk and what its manifest declares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledApp {
    /// Manifest `name`, not the directory name.
    pub name: String,
    pub dir: PathBuf,
    pub manifest: Manifest,
    /// Where the clone came from, `None` for a directory put there by hand.
    pub repo_url: Option<String>,
    /// Branch or tag the index asked for, `None` when it asked for nothing.
    pub git_ref: Option<String>,
    /// Commit the working tree is at. This is what actually ran.
    pub sha: Option<String>,
}

/// `.aias-origin`: where a clone came from and what is checked out.
///
/// A name is a moving target and a `ref` is a moving target too; the SHA is the
/// only thing that identifies the code that was built and started, so it is
/// recorded next to them and shown to the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    pub url: String,
    #[serde(rename = "ref", default, skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
}

/// Which of an app's two logs to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogKind {
    /// Output of the manifest `build` commands.
    Build,
    /// stdout and stderr of the running app.
    Run,
}

/// A running app process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppProcess {
    /// Manifest `name`.
    pub name: String,
    /// Loopback port the app was told to listen on.
    pub port: u16,
    /// `http://127.0.0.1:<port>`.
    pub url: String,
    pub pid: Option<u32>,
}

/// How often a waiter asks whether the app it watches is still alive.
const EXIT_POLL: Duration = Duration::from_millis(500);
/// How often that same waiter re-reads what the app is listening on.
///
/// Binding loopback at boot and a routable address a minute later is one line
/// of code, so the check at start time is a start and not the whole answer.
const LOOPBACK_POLL: Duration = Duration::from_secs(30);

/// One app this process started: the snapshot plus what cannot be serialized.
struct RunningApp {
    /// Distinguishes two runs of the same app, so a waiter left over from the
    /// first one cannot reap the second.
    id: u64,
    process: AppProcess,
    child: Option<Child>,
    /// Models this app holds a lease on, released when it stops.
    models: Vec<ModelRef>,
    dir: PathBuf,
    /// Where this run writes its log, so the waiter can say why it stopped it.
    log_path: PathBuf,
}

/// Apps started by this process. The Tauri app is the single owner on a real
/// machine; a CLI invocation only sees what it started itself.
static APPS: OnceLock<Mutex<Vec<RunningApp>>> = OnceLock::new();

fn apps() -> MutexGuard<'static, Vec<RunningApp>> {
    APPS.get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Hands out [`RunningApp::id`].
static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

/// Swapped out in tests so an app can be started without a Postgres cluster.
type DatabaseHook = fn(&str) -> Result<String>;
static DATABASE_HOOK: Mutex<Option<DatabaseHook>> = Mutex::new(None);

impl Manifest {
    /// Parse YAML. Does not validate.
    pub fn parse(yaml: &str) -> Result<Self> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }

    /// Read and parse `<dir>/aias.yaml`. Does not validate.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        if !path.is_file() {
            return Err(Error::NotFound(path.display().to_string()));
        }
        Self::parse(&std::fs::read_to_string(&path)?)
    }

    /// Enforce the rules of PLAN.md section 3 that do not need the repo or the network.
    ///
    /// Rule 1 required keys, rule 2 alias shape and uniqueness, rule 3 a non
    /// empty quant preference (file existence is checked by the store at publish
    /// time), rule 4 rejected keys. Rule 5 needs the repo, see [`validate_dir`].
    pub fn validate(&self) -> Result<()> {
        let mut issues: Vec<String> = Vec::new();

        // Rule 1. Required keys are non empty.
        if !is_app_name(&self.name) {
            issues.push(format!(
                "name `{}` must match ^[a-z][a-z0-9-]*$, it becomes a directory and a database name",
                self.name
            ));
        }
        if self.description.trim().is_empty() {
            issues.push("description must not be empty".into());
        }
        if self.version.trim().is_empty() {
            issues.push("version must not be empty".into());
        }
        if self.start.is_empty() {
            issues.push("start must have at least one argv element".into());
        }
        if let Some(health) = &self.health
            && !is_health_path(health)
        {
            issues.push(format!(
                "health `{health}` must match ^/[A-Za-z0-9._~/-]*$, it is put in a request line"
            ));
        }

        // Rule 2. Aliases are unique and produce a usable env var suffix.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for model in &self.models {
            if !is_alias(&model.alias) {
                issues.push(format!(
                    "models[].alias `{}` must match ^[a-z][a-z0-9_]*$",
                    model.alias
                ));
            }
            if !seen.insert(model.alias.as_str()) {
                issues.push(format!(
                    "models[].alias `{}` is declared twice",
                    model.alias
                ));
            }

            // Rule 3. A quant preference is required.
            if model.quant.is_empty() {
                issues.push(format!(
                    "models[{}].quant must list at least one quant",
                    model.alias
                ));
            }
            if model.repo.trim().is_empty() {
                issues.push(format!("models[{}].repo must not be empty", model.alias));
            }

            // Rule 4, inside a model entry.
            issues.extend(rejected_key_issues(
                &model.extra,
                &format!("models[{}]", model.alias),
            ));
        }

        if let Some(postgres) = &self.services.postgres {
            issues.extend(migrations_issues(&postgres.migrations));
        }

        // Rule 4, at the top level.
        issues.extend(rejected_key_issues(&self.extra, "manifest"));

        if issues.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidManifest(issues.join("; ")))
        }
    }

    /// `build` split into argv vectors on the literal `&&` element.
    ///
    /// Rule 6: every element is passed to the process as an argv element. The
    /// platform never builds a shell string, so a chained build is several
    /// commands run in order, not one string handed to `cmd.exe`.
    pub fn build_commands(&self) -> Vec<Vec<String>> {
        self.build
            .split(|arg| arg == COMMAND_SEPARATOR)
            .filter(|argv| !argv.is_empty())
            .map(<[String]>::to_vec)
            .collect()
    }
}

/// Load, validate and check the repo level rules against a checked out app.
///
/// Adds rule 5 of PLAN.md section 3: a Dockerfile is required in the repo.
pub fn validate_dir(dir: &Path) -> Result<Manifest> {
    let manifest = Manifest::load(dir)?;
    manifest.validate()?;
    if !dir.join("Dockerfile").is_file() {
        return Err(Error::InvalidManifest(
            "Dockerfile is required at the repo root, it is the portability contract".into(),
        ));
    }
    Ok(manifest)
}

/// Clone an app repo into `<data_dir>/apps/<name>`, or fast forward the clone.
///
/// The directory name is the tail of the URL. Every value reaches git as an
/// argv element (PLAN.md section 3 rule 6), the URL and the ref are checked
/// before git sees them, and both are passed after `--` so git reads them as
/// operands rather than as options. The manifest is validated after the
/// checkout, so a repo that is not an app fails here and not at start time.
pub async fn clone_app(repo_url: &str, git_ref: Option<&str>, data_dir: &Path) -> Result<PathBuf> {
    check_clone_url(repo_url)?;
    if let Some(git_ref) = git_ref {
        check_git_ref(git_ref)?;
    }
    let name = repo_dir_name(repo_url)?;
    let apps = paths::apps_dir(data_dir);
    std::fs::create_dir_all(&apps)?;
    let dir = apps.join(name);

    match (dir.exists(), git_ref) {
        // The index asked for a branch or a tag, so that is what is checked out,
        // on a fresh clone and on an existing one alike. `git clone --branch`
        // takes a branch or a tag, not a raw commit.
        (false, Some(git_ref)) => {
            git(&[
                "clone".as_ref(),
                "--branch".as_ref(),
                git_ref.as_ref(),
                "--depth".as_ref(),
                "1".as_ref(),
                "--".as_ref(),
                repo_url.as_ref(),
                dir.as_os_str(),
            ])
            .await?;
        }
        (false, None) => {
            git(&[
                "clone".as_ref(),
                "--depth".as_ref(),
                "1".as_ref(),
                "--".as_ref(),
                repo_url.as_ref(),
                dir.as_os_str(),
            ])
            .await?;
        }
        (true, Some(git_ref)) => {
            git(&[
                "-C".as_ref(),
                dir.as_os_str(),
                "fetch".as_ref(),
                "--depth".as_ref(),
                "1".as_ref(),
                "origin".as_ref(),
                "--".as_ref(),
                git_ref.as_ref(),
            ])
            .await?;
            git(&[
                "-C".as_ref(),
                dir.as_os_str(),
                "checkout".as_ref(),
                "--force".as_ref(),
                "FETCH_HEAD".as_ref(),
            ])
            .await?;
        }
        (true, None) => {
            git(&[
                "-C".as_ref(),
                dir.as_os_str(),
                "pull".as_ref(),
                "--ff-only".as_ref(),
            ])
            .await?;
        }
    }

    validate_dir(&dir)?;
    // The origin is not in the manifest, and `installed` needs it to offer the
    // repository link and a fast forward. `git config` would need the process,
    // one small file is cheaper and survives a copied directory.
    let origin = Origin {
        url: repo_url.to_string(),
        git_ref: git_ref.map(str::to_string),
        sha: head_sha(&dir).await,
    };
    std::fs::write(dir.join(ORIGIN_FILE), serde_json::to_vec_pretty(&origin)?)?;
    Ok(dir)
}

/// The commit a clone is checked out at, `None` when git cannot say.
async fn head_sha(dir: &Path) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args([
            "-C".as_ref(),
            dir.as_os_str(),
            "rev-parse".as_ref(),
            "HEAD".as_ref(),
        ])
        .quiet()
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

/// The published apps, read from `apps.yaml` in the store index repo.
///
/// `AIAS_INDEX_URL` overrides the default, which is the raw file in
/// `zyx1121/aias-index`. Nothing is cached: the file is a few kilobytes and a
/// stale store index is worse than one extra request.
pub async fn index(url: &str) -> Result<Vec<IndexEntry>> {
    let response = index_client()
        .get(url)
        .timeout(INDEX_TIMEOUT)
        .send()
        .await
        .map_err(|err| Error::Http(format!("{url}: {err}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(Error::Http(format!("http {status} for {url}")));
    }
    let body = response
        .text()
        .await
        .map_err(|err| Error::Http(format!("{url}: {err}")))?;

    match serde_yaml_ng::from_str::<IndexFile>(&body)? {
        IndexFile::Wrapped { apps } => Ok(apps),
        IndexFile::Bare(apps) => Ok(apps),
    }
}

/// The store index URL, `AIAS_INDEX_URL` first.
pub fn index_url() -> String {
    std::env::var(INDEX_URL_ENV)
        .ok()
        .filter(|url| !url.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_INDEX_URL.to_string())
}

fn index_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("aias/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("http client builds")
    })
}

/// Every app checked out under `<data_dir>/apps`, in manifest name order.
///
/// A directory whose manifest does not parse is skipped rather than failing the
/// whole list: one broken clone must not hide the others.
pub fn installed(data_dir: &Path) -> Vec<InstalledApp> {
    let root = paths::apps_dir(data_dir);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut found: Vec<InstalledApp> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Ok(manifest) = Manifest::load(&dir) else {
            continue;
        };
        found.push(installed_app(dir, manifest));
    }
    found.sort_by(|left, right| left.name.cmp(&right.name));
    found
}

/// The directory of one subscribed app, resolved from its name.
///
/// The desktop shell used to take the directory itself from the renderer, which
/// made every command that touches an app a file system primitive: a web
/// context could ask the platform to validate, build or start argv out of any
/// directory on the machine. The renderer now names an app and this resolves
/// it, so the only reachable directories are the ones under `<data_dir>/apps`.
///
/// The name is matched against the manifest first, because a clone is named
/// after the tail of its repo URL and an app is named by its manifest, and the
/// two differ whenever a repo is called `aias-<name>`. The plain join is the
/// fallback, so a clone whose manifest stopped parsing can still be validated
/// and told what is wrong with it.
pub fn app_dir(data_dir: &Path, app_name: &str) -> Result<PathBuf> {
    if !is_app_name(app_name) {
        return Err(Error::InvalidManifest(format!(
            "app name `{app_name}` must match ^[a-z][a-z0-9-]{{0,50}}$"
        )));
    }
    if let Some(app) = installed(data_dir)
        .into_iter()
        .find(|app| app.name == app_name)
    {
        return Ok(app.dir);
    }
    let dir = paths::apps_dir(data_dir).join(app_name);
    if dir.is_dir() {
        return Ok(dir);
    }
    Err(Error::NotFound(format!("app {app_name} is not subscribed")))
}

/// Clone the repo and validate the manifest. Subscribing stops there.
///
/// The build is deliberately not part of this. `build` and `start` are argv the
/// repo author wrote and the platform runs on the user's machine, so the user
/// has to read them first: the caller shows the manifest this returns and asks,
/// then calls [`build`]. The CLI does the same, behind `--build`.
pub async fn subscribe(
    repo_url: &str,
    git_ref: Option<&str>,
    data_dir: &Path,
) -> Result<InstalledApp> {
    let dir = clone_app(repo_url, git_ref, data_dir).await?;
    let manifest = validate_dir(&dir)?;
    Ok(installed_app(dir, manifest))
}

/// One clone plus what `.aias-origin` records about where it came from.
fn installed_app(dir: PathBuf, manifest: Manifest) -> InstalledApp {
    let origin = origin_of(&dir);
    InstalledApp {
        name: manifest.name.clone(),
        repo_url: origin.as_ref().map(|origin| origin.url.clone()),
        git_ref: origin.as_ref().and_then(|origin| origin.git_ref.clone()),
        sha: origin.as_ref().and_then(|origin| origin.sha.clone()),
        dir,
        manifest,
    }
}

/// The two templates `init` writes, PLAN.md section 6.2.
///
/// Both are minimal and both are already wired to `AIAS_MODEL_CHAT_*` and to
/// `DATABASE_URL` with a migrations folder. A template is the default an agent
/// reaches for, not a requirement: any repo that satisfies section 3 is an app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Template {
    /// Next.js standalone, `runtime: node`.
    Next,
    /// FastAPI plus a static frontend, `runtime: python`.
    Fastapi,
}

/// `templates/next` as it is in this repository, compiled into the binary.
static NEXT_TEMPLATE: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../templates/next");
/// `templates/fastapi` as it is in this repository, compiled into the binary.
static FASTAPI_TEMPLATE: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../templates/fastapi");

/// A template file called this is written as `.gitignore`.
///
/// The name is held back one character because the template lives inside this
/// repository: a real `.gitignore` under `templates/` would be read by git as a
/// rule for the platform's own tree, not as a file to hand to an app.
const GITIGNORE_SOURCE: &str = "gitignore";

/// What a scaffolded file may be handed, `{{name}}` and `{{description}}`.
const NAME_PLACEHOLDER: &str = "{{name}}";
const DESCRIPTION_PLACEHOLDER: &str = "{{description}}";

/// Build output and installed dependencies, never copied by a fork.
///
/// A fork is source: the new repo builds on the machine it lands on, and
/// copying a `node_modules` or a `.venv` from the source would carry another
/// machine's binaries into it.
const NOT_SOURCE: &[&str] = &[".git", "node_modules", ".venv", ".next", "__pycache__"];

impl Template {
    /// Name of the template as the API and the CLI spell it.
    pub fn slug(&self) -> &'static str {
        match self {
            Template::Next => "next",
            Template::Fastapi => "fastapi",
        }
    }

    /// The embedded directory this template is written from.
    fn files(&self) -> &'static Dir<'static> {
        match self {
            Template::Next => &NEXT_TEMPLATE,
            Template::Fastapi => &FASTAPI_TEMPLATE,
        }
    }
}

/// A directory `init` or `fork` wrote, and the manifest that is now in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScaffoldedApp {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// Write one of the two templates into `dir` as a git repository.
///
/// The files are compiled into the binary, so this works offline and on a
/// machine that has never seen this repository. `{{name}}` and
/// `{{description}}` are the only placeholders, `git init` and one commit are
/// what make the result a repo an agent can edit and publish, and the manifest
/// is validated afterwards: `init` never hands back a directory that would fail
/// `apps::validate_dir`.
pub async fn init(name: &str, template: Template, dir: &Path) -> Result<ScaffoldedApp> {
    if !is_app_name(name) {
        return Err(Error::InvalidManifest(format!(
            "app name `{name}` must match ^[a-z][a-z0-9-]{{0,50}}$"
        )));
    }
    ensure_writable_target(dir)?;

    let description = format!(
        "{name}, a local AI app scaffolded from the aias {} template.",
        template.slug()
    );
    write_template(template, dir, name, &description)?;

    git_init(
        dir,
        &format!("Scaffold {name} from the {} template", template.slug()),
    )
    .await?;

    let manifest = validate_dir(dir)?;
    Ok(ScaffoldedApp {
        dir: dir.to_path_buf(),
        manifest,
    })
}

/// Copy an installed app or a git repository into `dir` under a new name.
///
/// `source` is the manifest name of a subscribed app, an https git URL, or a
/// local directory the user is allowed to work in, which is what
/// [`check_fork_source`] decides. Fork is a product requirement and fork needs
/// source (decision D5): the copy keeps the code and drops the history, so the
/// new owner starts from one commit of their own.
pub async fn fork(source: &str, name: &str, dir: &Path) -> Result<ScaffoldedApp> {
    fork_in(source, name, dir, &paths::data_dir()).await
}

/// [`fork`] against an explicit data directory.
pub(crate) async fn fork_in(
    source: &str,
    name: &str,
    dir: &Path,
    data_dir: &Path,
) -> Result<ScaffoldedApp> {
    if !is_app_name(name) {
        return Err(Error::InvalidManifest(format!(
            "app name `{name}` must match ^[a-z][a-z0-9-]{{0,50}}$"
        )));
    }
    ensure_writable_target(dir)?;

    // An installed app is named, never given as a path: the name is resolved in
    // Rust against `<data_dir>/apps`, which is the only place it can be.
    if is_app_name(source) {
        let from = app_dir(data_dir, source)?;
        copy_source(&from, dir)?;
    } else {
        let from = check_fork_source(source, data_dir)?;
        git(&[
            "clone".as_ref(),
            "--depth".as_ref(),
            "1".as_ref(),
            "--".as_ref(),
            from.as_os_str(),
            dir.as_os_str(),
        ])
        .await?;
    }

    // The history belongs to the app that was forked, not to this one.
    let git_dir = dir.join(".git");
    if git_dir.exists() {
        std::fs::remove_dir_all(&git_dir)?;
    }
    // `.aias-origin` says where a clone came from, and this is not that clone.
    let _ = std::fs::remove_file(dir.join(ORIGIN_FILE));

    rename_in_manifest(dir, name)?;
    git_init(dir, &format!("Fork {source} as {name}")).await?;

    let manifest = validate_dir(dir)?;
    Ok(ScaffoldedApp {
        dir: dir.to_path_buf(),
        manifest,
    })
}

/// Rewrite the top level `name:` of `aias.yaml`, comments and layout kept.
///
/// A manifest is read by people as much as by this platform, so the edit is the
/// one line that has to change. Re-serializing the parsed document would work
/// too and would throw away every comment in the file, which is most of what
/// the templates put there.
fn rename_in_manifest(dir: &Path, name: &str) -> Result<()> {
    let path = dir.join(MANIFEST_FILE);
    if !path.is_file() {
        return Err(Error::NotFound(format!(
            "{} has no {MANIFEST_FILE}, it is not an app",
            dir.display()
        )));
    }
    let text = std::fs::read_to_string(&path)?;

    let mut edited = String::with_capacity(text.len());
    let mut renamed = false;
    for line in text.lines() {
        // Top level only: an indented `name:` belongs to a model alias or to a
        // secret, and neither is the name of the app.
        if !renamed && line.starts_with("name:") {
            edited.push_str(&format!("name: {name}"));
            renamed = true;
        } else {
            edited.push_str(line);
        }
        edited.push('\n');
    }

    if !renamed {
        // A manifest that writes its name in a flow mapping or over several
        // lines is not the shape the templates produce, so the document is
        // rebuilt from what it parses to rather than patched blindly.
        let mut manifest = Manifest::load(dir)?;
        manifest.name = name.to_string();
        edited = serde_yaml_ng::to_string(&manifest)?;
    }

    std::fs::write(&path, edited)?;
    Ok(())
}

/// Copy an app directory, leaving build output and history behind.
fn copy_source(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let name = entry.file_name();
        if NOT_SOURCE.iter().any(|skip| *skip == name) {
            continue;
        }
        let target = to.join(&name);
        if entry.file_type()?.is_dir() {
            copy_source(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Write every file of a template into `dir`, placeholders filled in.
fn write_template(template: Template, dir: &Path, name: &str, description: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    write_entries(template.files().entries(), dir, name, description)
}

/// One level of the embedded tree, then the level below it.
fn write_entries(
    entries: &[DirEntry<'_>],
    dir: &Path,
    name: &str,
    description: &str,
) -> Result<()> {
    for entry in entries {
        match entry {
            DirEntry::Dir(sub) => {
                let target = dir.join(file_name_of(sub.path())?);
                std::fs::create_dir_all(&target)?;
                write_entries(sub.entries(), &target, name, description)?;
            }
            DirEntry::File(file) => {
                let source = file_name_of(file.path())?;
                let target = dir.join(if source == GITIGNORE_SOURCE {
                    ".gitignore"
                } else {
                    source
                });
                let text = file.contents_utf8().ok_or_else(|| {
                    Error::InvalidManifest(format!(
                        "template file {} is not text",
                        file.path().display()
                    ))
                })?;
                std::fs::write(&target, render(text, source, name, description))?;
            }
        }
    }
    Ok(())
}

/// The last component of a path inside a template.
fn file_name_of(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            Error::InvalidManifest(format!("template entry {} has no name", path.display()))
        })
}

/// Fill in the two placeholders.
///
/// Every placeholder outside a Markdown file sits inside a double quoted string
/// in YAML, JSON, TOML, TypeScript or Python, and all five escape a quote and a
/// backslash the same way, so one rule covers them. Markdown is prose and is
/// handed the text as it is. The name is `^[a-z][a-z0-9-]{0,50}$` and needs
/// neither.
fn render(text: &str, file_name: &str, name: &str, description: &str) -> String {
    let quoted = !file_name.ends_with(".md");
    let description = if quoted {
        description.replace('\\', "\\\\").replace('"', "\\\"")
    } else {
        description.to_string()
    };
    text.replace(NAME_PLACEHOLDER, name)
        .replace(DESCRIPTION_PLACEHOLDER, &description)
}

/// Refuse a destination that already holds something.
///
/// Scaffolding into a directory with files in it would mix two apps together,
/// and the failure would only show up at validate time with a manifest nobody
/// wrote.
fn ensure_writable_target(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    if !dir.is_dir() {
        return Err(Error::InvalidManifest(format!(
            "{} is a file, an app needs a directory",
            dir.display()
        )));
    }
    if std::fs::read_dir(dir)?.next().is_some() {
        return Err(Error::InvalidManifest(format!(
            "{} is not empty, an app is scaffolded into a new directory",
            dir.display()
        )));
    }
    Ok(())
}

/// `git init` plus the first commit, every value an argv element.
///
/// The identity is set on the commit rather than in the repository, because a
/// machine with no `user.email` configured would otherwise fail here, and the
/// author of a scaffold is the platform.
async fn git_init(dir: &Path, message: &str) -> Result<()> {
    git(&[
        "init".as_ref(),
        "--initial-branch=main".as_ref(),
        dir.as_os_str(),
    ])
    .await?;
    git(&[
        "-C".as_ref(),
        dir.as_os_str(),
        "add".as_ref(),
        "--all".as_ref(),
    ])
    .await?;
    git(&[
        "-C".as_ref(),
        dir.as_os_str(),
        "-c".as_ref(),
        "user.name=aias".as_ref(),
        "-c".as_ref(),
        "user.email=aias@localhost".as_ref(),
        "commit".as_ref(),
        "--message".as_ref(),
        message.as_ref(),
    ])
    .await?;
    Ok(())
}

/// Delete one subscribed app's clone. The app must not be running.
///
/// Downloaded models and the app database are left alone: a model is shared and
/// a database is the user's data, both are removed from their own page.
pub fn remove(name: &str, data_dir: &Path) -> Result<()> {
    if running().iter().any(|app| app.name == name) {
        return Err(Error::Process(format!(
            "app {name} is running, stop it before removing it"
        )));
    }
    let app = installed(data_dir)
        .into_iter()
        .find(|app| app.name == name)
        .ok_or_else(|| Error::NotFound(format!("app `{name}` is not subscribed")))?;
    std::fs::remove_dir_all(&app.dir)?;
    Ok(())
}

/// Tail of one app log. An absent log reads as empty, not as an error.
///
/// The name is validated the same way [`app_dir`] validates it, and for the
/// same reason: it is interpolated into a file name under `<data_dir>/logs`,
/// so a caller passing `../x` would otherwise read a file of its choosing.
///
/// Every reader goes through here, the API route, the `app_logs` tool and the
/// desktop shell, so [`instances::log_tail`] is the one place the masking is
/// applied and the one place it can be got wrong.
pub fn logs(name: &str, kind: LogKind, tail_lines: usize) -> Result<String> {
    logs_in(name, kind, tail_lines, &paths::data_dir())
}

/// [`logs`] against an explicit data directory.
pub(crate) fn logs_in(
    name: &str,
    kind: LogKind,
    tail_lines: usize,
    data_dir: &Path,
) -> Result<String> {
    Ok(instances::log_tail(
        &log_path(name, kind, data_dir)?,
        tail_lines,
    ))
}

/// Where one app writes the log of `kind`, for a name that is an app name.
fn log_path(name: &str, kind: LogKind, data_dir: &Path) -> Result<PathBuf> {
    if !is_app_name(name) {
        return Err(Error::InvalidManifest(format!(
            "app name `{name}` must match ^[a-z][a-z0-9-]{{0,50}}$"
        )));
    }
    let file = match kind {
        LogKind::Build => format!("app-{name}-build.log"),
        LogKind::Run => format!("app-{name}.log"),
    };
    Ok(paths::logs_dir(data_dir).join(file))
}

/// The models an app declares that nothing on disk satisfies.
///
/// One entry per unsatisfied alias, and it is the first choice of that alias:
/// the preferred quant of the preferred repo, which is what the store should
/// download. An alias with any candidate installed produces nothing.
pub fn missing_models(manifest: &Manifest, data_dir: &Path) -> Vec<ModelRef> {
    let installed = crate::models::installed(data_dir);
    let mut missing = Vec::new();

    for declared in &manifest.models {
        let candidates = candidates_of(declared);
        let satisfied = candidates
            .iter()
            .any(|model| installed.iter().any(|(found, _, _)| found == model));
        if let (false, Some(first)) = (satisfied, candidates.first()) {
            missing.push(first.clone());
        }
    }
    missing
}

/// What `.aias-origin` says about a clone, when the sidecar is there.
///
/// Clones made before the sidecar became JSON hold one bare URL line, so that
/// shape is still read: an old clone keeps its repository link and simply has
/// no ref and no SHA until it is subscribed again.
fn origin_of(dir: &Path) -> Option<Origin> {
    let text = std::fs::read_to_string(dir.join(ORIGIN_FILE)).ok()?;
    parse_origin(&text)
}

fn parse_origin(text: &str) -> Option<Origin> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Ok(origin) = serde_json::from_str::<Origin>(text) {
        return (!origin.url.is_empty()).then_some(origin);
    }
    Some(Origin {
        url: text.to_string(),
        git_ref: None,
        sha: None,
    })
}

/// Every model reference one declared alias accepts, in preference order.
fn candidates_of(declared: &ModelDecl) -> Vec<ModelRef> {
    let mut candidates = Vec::new();
    for repo in std::iter::once(&declared.repo).chain(declared.fallback.iter()) {
        for quant in &declared.quant {
            candidates.push(ModelRef::new(repo.clone(), quant.clone()));
        }
    }
    candidates
}

/// Run [`Manifest::build_commands`] in the app directory, in order.
///
/// Output of every command lands in `<data_dir>/logs/app-<name>-build.log`. A
/// non zero exit stops the build and the error carries the tail of that log.
///
/// The environment is built, not inherited: [`crate::process::APP_ALLOWLIST`]
/// plus the manifest `env`, so a build command sees neither the local API
/// bearer token nor the user's Hugging Face token. Model URLs and stored
/// secrets are not there either; they belong to a running app, and no lease is
/// held while a build runs.
pub async fn build(dir: &Path, manifest: &Manifest) -> Result<()> {
    build_in(dir, manifest, &paths::data_dir()).await
}

/// [`build`] against an explicit data directory.
pub(crate) async fn build_in(dir: &Path, manifest: &Manifest, data_dir: &Path) -> Result<()> {
    let logs = paths::logs_dir(data_dir);
    std::fs::create_dir_all(&logs)?;
    let log_path = logs.join(format!("app-{}-build.log", manifest.name));
    std::fs::write(
        &log_path,
        format!("build {} in {}\n", manifest.name, dir.display()),
    )?;

    for argv in manifest.build_commands() {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| Error::InvalidManifest("build has an empty command".into()))?;

        let mut log = std::fs::OpenOptions::new().append(true).open(&log_path)?;
        writeln!(log, "\n> {}", argv.join(" "))?;
        let errors = log.try_clone()?;

        let mut command = tokio::process::Command::new(program);
        command
            .args(args)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(errors))
            .quiet()
            .sealed_env(APP_ALLOWLIST);
        with_tools_path(command.as_std_mut());
        for (key, value) in &manifest.env {
            command.env(key, value);
        }

        let status = match command.status().await {
            Ok(status) => status,
            Err(err) => {
                // A program that is not installed never opened the log, so the
                // only record of the failure is the one written here. Without
                // it `app_logs` shows the header and nothing else, and a
                // missing `bun` is indistinguishable from a build that hung.
                append_to_log(&log_path, &format!("spawn failed: {program}: {err}\n"));
                return Err(Error::Process(format!("could not run `{program}`: {err}")));
            }
        };
        if !status.success() {
            return Err(Error::Process(format!(
                "build step `{}` of {} failed with {status}; log tail:\n{}",
                argv.join(" "),
                manifest.name,
                instances::log_tail(&log_path, LOG_TAIL_LINES)
            )));
        }
    }
    Ok(())
}

/// Append one line to a log, best effort.
///
/// Used where the failure being reported is already the interesting one: a log
/// that cannot be written is not worth replacing it with.
fn append_to_log(log_path: &Path, line: &str) {
    if let Ok(mut log) = std::fs::OpenOptions::new().append(true).open(log_path) {
        let _ = log.write_all(line.as_bytes());
    }
}

/// Start the app with its injected environment and wait for `health`.
///
/// One lease per declared model, the database when it is declared, then the
/// process with `PORT`, `AIAS_MODEL_*`, `DATABASE_URL`, the manifest `env` and
/// the stored secrets. Anything that fails releases every lease taken here.
///
/// That list is the whole environment: it is cleared first and refilled from
/// [`crate::process::APP_ALLOWLIST`], so an app reads no platform variable
/// except the `AIAS_MODEL_*` ones meant for it.
pub async fn start(dir: &Path, manifest: &Manifest) -> Result<AppProcess> {
    start_in(dir, manifest, &paths::data_dir()).await
}

/// [`start`] against an explicit data directory.
pub(crate) async fn start_in(
    dir: &Path,
    manifest: &Manifest,
    data_dir: &Path,
) -> Result<AppProcess> {
    let mut acquired: Vec<ModelRef> = Vec::new();
    match start_inner(dir, manifest, data_dir, &mut acquired).await {
        Ok(process) => Ok(process),
        Err(err) => {
            for model in &acquired {
                let _ = instances::release(&manifest.name, model).await;
            }
            Err(err)
        }
    }
}

async fn start_inner(
    dir: &Path,
    manifest: &Manifest,
    data_dir: &Path,
    acquired: &mut Vec<ModelRef>,
) -> Result<AppProcess> {
    paths::ensure_data_dirs(data_dir)?;
    if running().iter().any(|app| app.name == manifest.name) {
        return Err(Error::Process(format!(
            "app {} is already running in this process",
            manifest.name
        )));
    }

    // Models first: a missing one is the cheapest failure.
    let mut env: BTreeMap<String, String> = BTreeMap::new();
    for declared in &manifest.models {
        let (model, url) = acquire_declared(&manifest.name, declared, data_dir).await?;
        acquired.push(model);
        for (key, value) in model_env(declared, url) {
            env.insert(key, value);
        }
    }

    if let Some(postgres) = &manifest.services.postgres {
        let migrations = resolve_migrations(dir, &postgres.migrations)?;
        let database_url = provision_database(&manifest.name, data_dir, &migrations).await?;
        env.insert("DATABASE_URL".into(), database_url);
    }

    for (key, value) in &manifest.env {
        env.insert(key.clone(), value.clone());
    }
    for (key, value) in secrets(&manifest.name, data_dir)? {
        env.insert(key, value);
    }

    let port = pick_app_port()?;
    for (key, value) in platform_env(port) {
        env.insert(key, value);
    }

    let log_path = paths::logs_dir(data_dir).join(format!("app-{}.log", manifest.name));
    let log = std::fs::File::create(&log_path)?;
    let errors = log.try_clone()?;

    let (program, args) = manifest
        .start
        .split_first()
        .ok_or_else(|| Error::InvalidManifest("start has an empty command".into()))?;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(errors))
        .quiet()
        .sealed_env(APP_ALLOWLIST);
    with_tools_path(&mut command);
    for (key, value) in &env {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .map_err(|err| Error::Process(format!("could not start `{program}`: {err}")))?;

    if let Err(err) = wait_ready(&mut child, port, manifest, &log_path).await {
        kill_tree(&mut child).await;
        return Err(err);
    }

    // `HOST` is a request; this is the check. An app that ignored it and bound
    // a routable interface is stopped here rather than left on the network.
    if let Err(err) = verify_loopback(Some(child.id()), port).await {
        kill_tree(&mut child).await;
        return Err(err);
    }

    let process = AppProcess {
        name: manifest.name.clone(),
        port,
        url: format!("http://127.0.0.1:{port}"),
        pid: Some(child.id()),
    };
    let id = NEXT_RUN_ID.fetch_add(1, Ordering::SeqCst);
    apps().push(RunningApp {
        id,
        process: process.clone(),
        child: Some(child),
        models: acquired.clone(),
        dir: dir.to_path_buf(),
        log_path: log_path.clone(),
    });
    // An app that dies on its own has to give back what it holds, the same way
    // a stopped one does.
    spawn_exit_waiter(id);
    Ok(process)
}

/// Stop the app and release every lease it holds.
pub async fn stop(app_name: &str) -> Result<()> {
    let id = apps()
        .iter()
        .find(|app| app.process.name == app_name)
        .map(|app| app.id)
        .ok_or_else(|| Error::NotFound(format!("app {app_name} is not running")))?;
    stop_run(id).await
}

/// [`stop`] for one specific run, so a waiter cannot stop a later namesake.
///
/// Killing a process that already exited is a no-op, which is what makes this
/// the same path for a stop and for a crash. The log is left where it is: it is
/// the only thing that says why the app died.
async fn stop_run(id: u64) -> Result<()> {
    let removed = {
        let mut apps = apps();
        apps.iter()
            .position(|app| app.id == id)
            .map(|at| apps.remove(at))
    };
    let Some(mut app) = removed else {
        return Err(Error::NotFound(format!("run {id} is not running")));
    };

    if let Some(child) = app.child.as_mut() {
        kill_tree(child).await;
    }
    for model in &app.models {
        instances::release(&app.process.name, model).await?;
    }
    Ok(())
}

/// Release what an app holds the moment it exits, not when someone notices, and
/// keep asking what it is listening on for as long as it runs.
///
/// A crashed app used to stay in the registry with its leases held, which kept
/// a `llama-server` alive forever and made its model look busy to the next app
/// that wanted it. The waiter polls instead of owning the child, because
/// [`stop`] has to keep the right to kill it; one `try_wait` under the registry
/// lock costs nothing and the lock is never held across an await.
///
/// Every [`LOOPBACK_POLL`] it also re-runs [`verify_loopback`]. An app that
/// binds `0.0.0.0` a minute after it answered health is on the network just as
/// much as one that did it at boot, and the only difference is that nobody was
/// looking. A violation stops the run and the reason is appended to the app log,
/// which is what the Apps page shows.
fn spawn_exit_waiter(id: u64) {
    tokio::spawn(async move {
        let mut since_check = Duration::ZERO;
        loop {
            tokio::time::sleep(EXIT_POLL).await;
            match run_has_exited(id) {
                // Gone from the registry: a stop got there first.
                None => return,
                Some(false) => {}
                Some(true) => {
                    let _ = stop_run(id).await;
                    return;
                }
            }

            since_check += EXIT_POLL;
            if since_check < LOOPBACK_POLL {
                continue;
            }
            since_check = Duration::ZERO;
            let Some((pid, port, log_path)) = run_endpoint(id) else {
                return;
            };
            if let Err(err) = verify_loopback(pid, port).await {
                note_in_log(&log_path, &err.to_string());
                let _ = stop_run(id).await;
                return;
            }
        }
    });
}

/// What the waiter needs to re-check one run, or `None` when the run is gone.
fn run_endpoint(id: u64) -> Option<(Option<u32>, u16, PathBuf)> {
    apps()
        .iter()
        .find(|app| app.id == id)
        .map(|app| (app.process.pid, app.process.port, app.log_path.clone()))
}

/// Append one platform line to an app log. Best effort: a log that cannot be
/// written is not a reason to leave the app running.
fn note_in_log(log_path: &Path, message: &str) {
    if let Ok(mut log) = std::fs::OpenOptions::new().append(true).open(log_path) {
        let _ = writeln!(log, "\n[aias] stopping the app: {message}");
    }
}

/// `Some(true)` when that run's process is gone, `None` when the run is.
fn run_has_exited(id: u64) -> Option<bool> {
    let mut apps = apps();
    let app = apps.iter_mut().find(|app| app.id == id)?;
    match app.child.as_mut() {
        // An error from `try_wait` means the child cannot be asked about any
        // more, which is as good as gone.
        Some(child) => Some(!matches!(child.try_wait(), Ok(None))),
        None => Some(true),
    }
}

/// Every app this process started and has not stopped.
pub fn running() -> Vec<AppProcess> {
    apps()
        .iter()
        .map(|app| app.process.clone())
        .collect::<Vec<AppProcess>>()
}

/// The directory an app was started from, for the CLI and the store page.
pub fn running_dir(app_name: &str) -> Option<PathBuf> {
    apps()
        .iter()
        .find(|app| app.process.name == app_name)
        .map(|app| app.dir.clone())
}

/// Take a lease on the first declared candidate whose files are on disk.
///
/// Order is the quant list on the declared repo, then the same list on
/// `fallback`. Nothing is downloaded here: a store install pulls the weights,
/// starting an app only uses what is already there.
async fn acquire_declared(
    app: &str,
    declared: &ModelDecl,
    data_dir: &Path,
) -> Result<(ModelRef, InstanceUrl)> {
    let candidates = candidates_of(declared);

    // The downloader is the authority on what is on disk.
    let downloaded = crate::models::installed(data_dir);
    let installed: Vec<ModelRef> = candidates
        .iter()
        .filter(|model| downloaded.iter().any(|(found, _, _)| &found == model))
        .cloned()
        .collect();

    let Some(model) = installed.first() else {
        return Err(Error::NotFound(format!(
            "app model `{}` has nothing downloaded, pull one of: {}",
            declared.alias,
            candidates
                .iter()
                .map(instances::model_id)
                .collect::<Vec<String>>()
                .join(", ")
        )));
    };

    let url = instances::acquire_in(app, model, data_dir).await?;
    Ok((model.clone(), url))
}

/// Start the cluster if needed, create this app's database, apply its migrations.
///
/// This is the only place apps.rs touches Postgres. The hook exists because the
/// real path downloads and initializes a cluster, which `services` covers in its
/// own tests; here it stands in for one so the lifecycle can be exercised.
async fn provision_database(app: &str, data_dir: &Path, migrations: &Path) -> Result<String> {
    let hook = *DATABASE_HOOK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(hook) = hook {
        return hook(app);
    }
    let postgres = services::ensure_postgres(data_dir).await?;
    let database_url = services::provision_app_db(&postgres, app).await?;
    if migrations.is_dir() {
        services::apply_migrations(&database_url, migrations).await?;
    }
    Ok(database_url)
}

/// Secrets the user stored for this app, `<data_dir>/secrets/<app>.json`.
///
/// PLAN.md section 10 puts these in the Windows Credential Manager in v1.
/// Until then the file is the only copy, so it is taken away from every other
/// user before it is read: a file written by hand, or by an older build that
/// did not set a mode, is repaired here rather than left readable.
fn secrets(app: &str, data_dir: &Path) -> Result<BTreeMap<String, String>> {
    let path = paths::secrets_dir(data_dir).join(format!("{app}.json"));
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    paths::restrict(&path, paths::FILE_MODE)?;
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let object = value.as_object().ok_or_else(|| {
        Error::InvalidManifest(format!(
            "{} must be a JSON object of name to value",
            path.display()
        ))
    })?;

    let mut secrets = BTreeMap::new();
    for (key, value) in object {
        let text = value.as_str().ok_or_else(|| {
            Error::InvalidManifest(format!("{}: `{key}` must be a string", path.display()))
        })?;
        secrets.insert(key.clone(), text.to_string());
    }
    Ok(secrets)
}

/// Poll the health endpoint until the app answers 2xx, dies, or the window ends.
async fn wait_ready(
    child: &mut Child,
    port: u16,
    manifest: &Manifest,
    log_path: &Path,
) -> Result<()> {
    let Some(health) = manifest.health.as_deref() else {
        // No endpoint declared, so the only contract left is `it did not exit`.
        tokio::time::sleep(HEALTH_POLL).await;
        return match child.try_wait()? {
            Some(status) => Err(Error::Process(format!(
                "{} exited with {status} right after start; log tail:\n{}",
                manifest.name,
                instances::log_tail(log_path, LOG_TAIL_LINES)
            ))),
            None => Ok(()),
        };
    };

    let deadline = Instant::now() + HEALTH_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            return Err(Error::Process(format!(
                "{} exited with {status} before {health} answered; log tail:\n{}",
                manifest.name,
                instances::log_tail(log_path, LOG_TAIL_LINES)
            )));
        }
        if let Ok(status) = instances::http_status(port, health, HEALTH_REQUEST_TIMEOUT).await
            && (200..300).contains(&status)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Error::Process(format!(
                "{} did not answer {health} on port {port} with 2xx within {} s; log tail:\n{}",
                manifest.name,
                HEALTH_TIMEOUT.as_secs(),
                instances::log_tail(log_path, LOG_TAIL_LINES)
            )));
        }
        tokio::time::sleep(HEALTH_POLL).await;
    }
}

/// Everything wrong with a declared `migrations` path, as validation messages.
///
/// The path is joined onto the clone and every `*.sql` under it is run against
/// the app's database, so it has to name a place inside the repo. A rooted path
/// or a `..` would read files the app was never given, and neither is checked
/// by `Path` alone: on a developer machine `\` is an ordinary character and
/// `C:\Users\me` looks like one relative component, so both shapes of
/// separator are inspected by hand.
fn migrations_issues(declared: &Path) -> Vec<String> {
    let text = declared.to_string_lossy();
    let mut issues = Vec::new();

    if text.trim().is_empty() {
        issues.push("services.postgres.migrations must not be empty".into());
        return issues;
    }
    let rooted = declared.is_absolute()
        || text.starts_with('/')
        || text.starts_with('\\')
        || text.chars().nth(1) == Some(':');
    if rooted {
        issues.push(format!(
            "services.postgres.migrations `{text}` must be relative to the repo root"
        ));
    }
    if text.split(['/', '\\']).any(|part| part == "..") {
        issues.push(format!(
            "services.postgres.migrations `{text}` must not leave the app directory with `..`"
        ));
    }
    issues
}

/// The migrations directory of one app, proven to sit inside its clone.
///
/// Public because the desktop shell derives it from the app name rather than
/// taking a path from the frontend.
///
/// Validation rejects the shapes that are wrong on their face; this is the
/// check that survives a symlink, which no amount of reading the string can
/// catch. A path that does not exist is handed back untouched: an app may
/// declare a folder it has not written yet, and applying nothing is fine.
pub fn resolve_migrations(dir: &Path, declared: &Path) -> Result<PathBuf> {
    let issues = migrations_issues(declared);
    if !issues.is_empty() {
        return Err(Error::InvalidManifest(issues.join("; ")));
    }

    let joined = dir.join(declared);
    if !joined.exists() {
        return Ok(joined);
    }
    let root = dir.canonicalize()?;
    let real = joined.canonicalize()?;
    if !real.starts_with(&root) {
        return Err(Error::InvalidManifest(format!(
            "services.postgres.migrations `{}` resolves to {}, which is outside the app directory {}",
            declared.display(),
            real.display(),
            root.display()
        )));
    }
    Ok(real)
}

/// The three variables one declared alias produces.
///
/// The key is not optional: `llama-server` is started with `--api-key`, so an
/// app that only reads the URL and the id gets a 401.
fn model_env(declared: &ModelDecl, url: InstanceUrl) -> [(String, String); 3] {
    let suffix = declared.env_suffix();
    [
        (format!("AIAS_MODEL_{suffix}_URL"), url.base_url),
        (format!("AIAS_MODEL_{suffix}_ID"), url.model_id),
        (format!("AIAS_MODEL_{suffix}_KEY"), url.api_key),
    ]
}

/// The variables the platform owns, whatever the manifest says.
///
/// Applied after the manifest `env` and the stored secrets, so neither can move
/// the app off loopback. `HOST` and `HOSTNAME` cover both conventions: Node and
/// Next.js read `HOSTNAME`, most other servers read `HOST`.
fn platform_env(port: u16) -> [(String, String); 3] {
    [
        ("PORT".to_string(), port.to_string()),
        ("HOST".to_string(), LOOPBACK.to_string()),
        ("HOSTNAME".to_string(), LOOPBACK.to_string()),
    ]
}

/// Fail unless every TCP socket the app's process tree listens on is bound to
/// loopback.
///
/// Exactly what is checked, and nothing wider: the **TCP listening sockets** of
/// every process in the app's tree, plus every TCP listening socket on the port
/// the platform assigned whoever owns it. **UDP is not checked**, nor are
/// outbound connections, nor sockets opened after this returns.
///
/// The health check only proves the app answers on `127.0.0.1`; a server bound
/// to `0.0.0.0` answers there too and is reachable from the whole network. The
/// operating system is the only honest witness, so two tables are read back as
/// argv, never through a shell: the process table (`ps -eo pid,ppid`, or
/// `Get-CimInstance Win32_Process` on Windows) and the listener table
/// (`netstat -ano -p tcp` on Windows, `lsof` everywhere else).
///
/// The process tree is what makes the pid question honest. A launcher such as
/// `bun run start` is the only child this platform ever sees, and the server
/// that binds a socket is its child or grandchild; matching the direct pid
/// alone let that grandchild open a second routable port unnoticed. The port
/// question stays as a second net, for the case where the tree is read a moment
/// before a process appears in it.
///
/// A tool that cannot be run is a failure, not a pass: the whole point of this
/// check is that an app which ignored `HOST` gets stopped, and it used to be
/// waved through whenever `lsof` was missing.
async fn verify_loopback(pid: Option<u32>, port: u16) -> Result<()> {
    let tree = match pid {
        Some(pid) => process_tree(pid).await?,
        None => Vec::new(),
    };
    let listeners = listening_addresses(&tree, port).await?;
    check_listeners(&listeners, pid, port)
}

/// Kill an app and everything it started, the launcher included.
///
/// The pid this platform holds is the one it spawned, and for `uv run python
/// main.py` or `bun run start` that is a launcher: killing it leaves the server
/// that owns the port running with nobody to stop it. So the tree is read
/// first, while the parent links still exist, then the root is killed so it
/// starts nothing more, then what it left behind.
///
/// Every failure here is ignored on purpose: a process that already exited is
/// the normal case, and this is also the crash path.
async fn kill_tree(child: &mut Child) {
    // A child that is already over has no tree left to read: its own children,
    // if it had any, were reparented the moment it exited. This is the crash
    // path as much as the stop path, and reading the process table there would
    // cost a `ps` on Unix and a CIM query on Windows to learn nothing.
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }

    let root = child.id();
    let tree = process_tree(root).await.unwrap_or_default();

    let _ = child.kill();
    let _ = child.wait();

    for pid in tree.into_iter().filter(|pid| *pid != root) {
        kill_pid(pid).await;
    }
}

/// Kill one pid by number, with the tool the platform ships with.
///
/// The pid comes from the process table of the app's own tree and reaches the
/// command as an argv element, never as part of a string a shell reads.
async fn kill_pid(pid: u32) {
    let (program, args) = if cfg!(windows) {
        (
            "taskkill",
            vec!["/F".to_string(), "/PID".into(), pid.to_string()],
        )
    } else {
        ("kill", vec!["-KILL".to_string(), pid.to_string()])
    };
    let _ = tokio::process::Command::new(program)
        .args(args)
        .quiet()
        .output()
        .await;
}

/// Every pid of the app's process tree: `root` plus all of its descendants.
///
/// Read from the OS process table, because a launcher such as `bun run start`
/// or `uv run` is the pid this platform holds while the socket belongs to a
/// process one or two levels below it.
async fn process_tree(root: u32) -> Result<Vec<u32>> {
    let table = if cfg!(windows) {
        // One argv element, a constant with nothing interpolated, printing the
        // same two columns `ps` does.
        tool_output(
            "powershell.exe",
            &[
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-ExecutionPolicy".into(),
                "Bypass".into(),
                "-Command".into(),
                WINDOWS_PROCESS_TABLE.into(),
            ],
        )
        .await?
    } else {
        tool_output("ps", &["-eo".into(), "pid,ppid".into()]).await?
    };
    Ok(descendants(&parse_process_table(&table), root))
}

/// Prints `<pid> <ppid>` per process, the shape [`parse_process_table`] reads.
const WINDOWS_PROCESS_TABLE: &str = "Get-CimInstance -ClassName Win32_Process | \
     ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }";

/// `(pid, ppid)` of every row of a process table, headers and noise skipped.
///
/// Both shapes are two leading integer columns: `ps -eo pid,ppid` and the
/// PowerShell line above. Anything whose first two fields are not numbers, the
/// `PID PPID` header included, is not a process.
fn parse_process_table(output: &str) -> Vec<(u32, u32)> {
    let mut rows = Vec::new();
    for line in output.lines() {
        let mut fields = line.split_whitespace();
        let (Some(pid), Some(ppid)) = (fields.next(), fields.next()) else {
            continue;
        };
        if let (Ok(pid), Ok(ppid)) = (pid.parse::<u32>(), ppid.parse::<u32>()) {
            rows.push((pid, ppid));
        }
    }
    rows
}

/// `root` and every process under it, breadth first.
///
/// Each pid is taken once, so a table that reports a cycle, or a pid that is
/// its own parent, terminates instead of spinning.
fn descendants(table: &[(u32, u32)], root: u32) -> Vec<u32> {
    let mut tree = vec![root];
    let mut at = 0;
    while at < tree.len() {
        let parent = tree[at];
        at += 1;
        for (pid, ppid) in table {
            if *ppid == parent && !tree.contains(pid) {
                tree.push(*pid);
            }
        }
    }
    tree
}

/// The message [`verify_loopback`] fails with, split out so a fake table can
/// drive it in a test.
fn check_listeners(listeners: &[String], pid: Option<u32>, port: u16) -> Result<()> {
    let routable: Vec<&str> = listeners
        .iter()
        .filter(|address| !is_loopback_address(address))
        .map(String::as_str)
        .collect();
    if routable.is_empty() {
        return Ok(());
    }
    Err(Error::Process(format!(
        "the app listens on {} instead of {LOOPBACK}:{port}; \
         it must bind the loopback address the platform injects as HOST and PORT{}",
        routable.join(", "),
        match pid {
            Some(pid) => format!(" (pid {pid})"),
            None => String::new(),
        }
    )))
}

/// Local addresses of every TCP socket the app is listening on, as the OS
/// reports them: everything owned by a pid in `tree`, plus everything on
/// `port`.
async fn listening_addresses(tree: &[u32], port: u16) -> Result<Vec<String>> {
    if cfg!(windows) {
        let table = tool_output("netstat", &["-ano".into(), "-p".into(), "tcp".into()]).await?;
        Ok(parse_netstat_listeners(&table, tree, port))
    } else {
        let table = tool_output(
            "lsof",
            &["-nP".into(), "-iTCP".into(), "-sTCP:LISTEN".into()],
        )
        .await?;
        Ok(parse_lsof_listeners(&table, tree, port))
    }
}

/// Run one of the inspection tools and hand back its stdout, or fail naming it.
///
/// `lsof` exits non zero when it matched nothing, which is why the status alone
/// is not the test: only a non zero exit with no output at all is treated as
/// the tool being unable to answer.
async fn tool_output(program: &str, args: &[String]) -> Result<String> {
    let output = tokio::process::Command::new(program)
        .args(args)
        .quiet()
        .output()
        .await
        .map_err(|err| {
            Error::Process(format!(
                "could not run `{program}` to check what the app is listening on: {err}"
            ))
        })?;
    if !output.status.success() && output.stdout.is_empty() {
        return Err(Error::Process(format!(
            "`{program}` could not report what the app is doing: {} {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Local addresses of the `LISTENING` TCP rows of `netstat -ano -p tcp` that
/// belong to the process tree or to `port`.
fn parse_netstat_listeners(output: &str, tree: &[u32], port: u16) -> Vec<String> {
    let mut found = Vec::new();
    for line in output.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        // Proto, Local Address, Foreign Address, State, PID.
        if fields.len() < 5 || !fields[0].eq_ignore_ascii_case("tcp") {
            continue;
        }
        if !fields[3].eq_ignore_ascii_case("LISTENING") {
            continue;
        }
        if is_ours(fields[4], fields[1], tree, port) {
            found.push(fields[1].to_string());
        }
    }
    found
}

/// Local addresses of the `(LISTEN)` rows of `lsof -nP -iTCP -sTCP:LISTEN`
/// that belong to the process tree or to `port`.
fn parse_lsof_listeners(output: &str, tree: &[u32], port: u16) -> Vec<String> {
    let mut found = Vec::new();
    for line in output.lines() {
        if !line.contains("(LISTEN)") {
            continue;
        }
        // COMMAND, PID, ..., and NAME is the field before `(LISTEN)`, for
        // example `127.0.0.1:41501`.
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(at) = fields.iter().position(|field| *field == "(LISTEN)") else {
            continue;
        };
        let Some(name) = at.checked_sub(1).and_then(|at| fields.get(at)) else {
            continue;
        };
        let Some(owner) = fields.get(1) else {
            continue;
        };
        if is_ours(owner, name, tree, port) {
            found.push((*name).to_string());
        }
    }
    found
}

/// True when a row of the listener table is the app's: a pid anywhere in its
/// process tree, or the port the platform assigned it.
fn is_ours(owner: &str, address: &str, tree: &[u32], port: u16) -> bool {
    let ours = matches!(owner.parse::<u32>(), Ok(owner) if tree.contains(&owner));
    ours || port_of(address) == Some(port)
}

/// The port of a `host:port` pair, IPv6 brackets and `*` included.
fn port_of(address: &str) -> Option<u16> {
    address.rsplit(':').next()?.parse().ok()
}

/// True when a listening address only accepts connections from this machine.
///
/// `0.0.0.0`, `[::]` and `*` are the wildcards that put an app on the network,
/// and anything else is a real interface, so only the two loopback addresses
/// pass.
fn is_loopback_address(address: &str) -> bool {
    let host = match address.rsplit_once(':') {
        Some((host, _)) => host,
        None => address,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host == LOOPBACK || host == "::1" || host.starts_with("127.")
}

/// A free loopback port for an app, above the instance range.
fn pick_app_port() -> Result<u16> {
    let taken: Vec<u16> = apps().iter().map(|app| app.process.port).collect();
    for port in (APP_PORT_START + 1)..=APP_PORT_END {
        if taken.contains(&port) {
            continue;
        }
        if TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).is_ok() {
            return Ok(port);
        }
    }
    Err(Error::Process(format!(
        "no free port in {}-{APP_PORT_END} for an app",
        APP_PORT_START + 1
    )))
}

/// Prepend the platform tool directories to `PATH` when they are bundled.
///
/// On a machine where `bun`, `uv` and `node` are already on `PATH`, for example
/// a developer box, `AIAS_TOOLS_DIR` is unset and the inherited `PATH` is used.
fn with_tools_path(command: &mut Command) {
    let Some(root) = std::env::var_os("AIAS_TOOLS_DIR") else {
        return;
    };
    let root = PathBuf::from(root);
    let mut dirs: Vec<PathBuf> = vec![root.clone()];
    for tool in ["bun", "uv", "node"] {
        let dir = root.join(tool);
        if dir.is_dir() {
            dirs.push(dir);
        }
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    if let Ok(path) = std::env::join_paths(dirs.into_iter().chain(std::env::split_paths(&current)))
    {
        command.env("PATH", path);
    }
}

/// Refuse anything git would treat as a transport other than HTTPS.
///
/// `git clone` accepts `ext::`, `file://` and an SSH shorthand, and `ext::`
/// runs a command the URL names. The store index is a file this platform
/// fetches over the network, so an entry in it must not be able to choose how
/// git connects. A local directory stays allowed: it is how an app is developed
/// and tested, and it names a place the user already has.
fn check_clone_url(repo_url: &str) -> Result<()> {
    if repo_url.starts_with("https://") || Path::new(repo_url).is_dir() {
        return Ok(());
    }
    Err(Error::InvalidManifest(format!(
        "`{repo_url}` is not an app source: it must be an https:// git URL or a local directory"
    )))
}

/// The three shapes `fork` accepts a source in, resolved to what git is given.
///
/// An installed app name is handled by the caller, which resolves it against
/// `<data_dir>/apps`. What is left is an `https://` URL, which git clones as
/// itself, and a local directory, which has to pass [`paths::resolve_dir`]
/// like every other directory a client names: under the user's home or under
/// `apps_dir`, and never inside the platform's own state.
///
/// Without that check `source` was the one path a client could point anywhere
/// on the disk. `check_clone_url` accepts any directory, and git clones a
/// directory happily, so `/etc` or another user's repository could be forked
/// into an app and read back through the manifest and the build log.
fn check_fork_source(source: &str, data_dir: &Path) -> Result<PathBuf> {
    if source.starts_with("https://") {
        return Ok(PathBuf::from(source));
    }
    let path = Path::new(source);
    if path.is_dir() {
        return paths::resolve_dir(path, data_dir);
    }
    Err(Error::InvalidManifest(format!(
        "`{source}` is not an app source: it must be an installed app name, an https:// git URL, or a local directory"
    )))
}

/// Refuse a `ref` git would read as anything but a branch or a tag.
///
/// `git fetch origin <ref>` takes the ref as a positional, and git parses
/// options before positionals: a ref of `--upload-pack=<command>` makes git run
/// that command, and a local directory origin is enough to reach it. The `--`
/// separator in [`clone_app`] already stops that, and this is the second lock:
/// `^[A-Za-z0-9][A-Za-z0-9._/-]*$` cannot start with `-`, so nothing here can
/// be read as an option even if a future call site forgets the separator.
///
/// The shape is a subset of what `git check-ref-format` allows. A ref outside
/// it is not a ref this platform needs to clone.
fn check_git_ref(git_ref: &str) -> Result<()> {
    let mut chars = git_ref.chars();
    let head_ok = matches!(chars.next(), Some(first) if first.is_ascii_alphanumeric());
    let tail_ok = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'));
    if head_ok && tail_ok {
        return Ok(());
    }
    Err(Error::InvalidManifest(format!(
        "`{git_ref}` is not a usable git ref: it must match ^[A-Za-z0-9][A-Za-z0-9._/-]*$"
    )))
}

/// The directory name a repo URL checks out into.
fn repo_dir_name(repo_url: &str) -> Result<String> {
    let trimmed = repo_url.trim_end_matches(['/', '\\']);
    let tail = trimmed
        .rsplit(['/', '\\', ':'])
        .next()
        .unwrap_or_default()
        .trim_end_matches(".git");
    let safe = !tail.is_empty()
        && tail != "."
        && tail != ".."
        && tail
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !safe {
        return Err(Error::NotFound(format!(
            "no usable app directory name in `{repo_url}`"
        )));
    }
    Ok(tail.to_string())
}

/// Run git with argv elements only and fail with what it printed.
async fn git(args: &[&std::ffi::OsStr]) -> Result<()> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .quiet()
        .output()
        .await
        .map_err(|err| Error::Process(format!("could not run git: {err}")))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(Error::Process(format!(
        "git {} failed with {}: {}",
        args.iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<String>>()
            .join(" "),
        output.status,
        stderr.trim()
    )))
}

/// `^/[A-Za-z0-9._~/-]*$`.
///
/// The value is written straight into `GET <path> HTTP/1.1`, so a space or a
/// carriage return in it would let a manifest write its own request line. Only
/// the unreserved path characters are allowed; a query string is not one of
/// them, and a health endpoint does not need one.
fn is_health_path(value: &str) -> bool {
    value.starts_with('/')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '~' | '/' | '-'))
}

/// `^[a-z][a-z0-9_]*$`, written out to keep the crate free of a regex dependency.
fn is_alias(value: &str) -> bool {
    matches_shape(value, |c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'
    })
}

/// `^[a-z][a-z0-9-]{0,50}$`.
///
/// The cap is not decoration: the name becomes a directory under
/// `<data_dir>/apps` and a Postgres identifier, and `services::db_name` already
/// refuses anything longer once it has added its `app_` prefix.
pub fn is_app_name(value: &str) -> bool {
    value.len() <= MAX_APP_NAME_LEN
        && matches_shape(value, |c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'
        })
}

fn matches_shape(value: &str, tail: impl Fn(char) -> bool) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => chars.all(tail),
        _ => false,
    }
}

fn rejected_key_issues(extra: &BTreeMap<String, serde_yaml_ng::Value>, scope: &str) -> Vec<String> {
    extra
        .keys()
        .map(|key| {
            if REJECTED_KEYS.contains(&key.as_str()) {
                format!(
                    "{scope}: `{key}` is an inference parameter, the platform owns it (PLAN section 3 rule 4)"
                )
            } else {
                format!("{scope}: `{key}` is not a manifest key")
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn set_database_hook(hook: Option<DatabaseHook>) {
    *DATABASE_HOOK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = hook;
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = include_str!("../../../spec/aias.example.yaml");

    #[test]
    fn the_spec_example_parses_and_validates() {
        let manifest = Manifest::parse(EXAMPLE).expect("example parses");
        manifest.validate().expect("example validates");

        assert_eq!(manifest.name, "contract-review");
        assert_eq!(manifest.runtime, AppRuntime::Node);
        assert_eq!(manifest.models.len(), 2);
        assert_eq!(manifest.models[0].env_suffix(), "REVIEWER");
        assert_eq!(manifest.models[1].kind, ModelKind::Vlm);
        assert_eq!(
            manifest
                .services
                .postgres
                .as_ref()
                .map(|p| p.migrations.clone()),
            Some(PathBuf::from("db/migrations"))
        );
        assert_eq!(
            manifest
                .env
                .get("NEXT_TELEMETRY_DISABLED")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(manifest.secrets.len(), 1);
        assert!(!manifest.secrets[0].required);
    }

    #[test]
    fn build_is_split_into_argv_vectors() {
        let manifest = Manifest::parse(EXAMPLE).unwrap();
        assert_eq!(
            manifest.build_commands(),
            vec![
                vec![
                    "bun".to_string(),
                    "install".into(),
                    "--frozen-lockfile".into()
                ],
                vec!["bun".to_string(), "run".into(), "build".into()],
            ]
        );
        assert_eq!(manifest.start, vec!["node", ".next/standalone/server.js"]);
    }

    fn minimal(extra: &str) -> String {
        format!(
            "name: demo\ndescription: A demo app.\nversion: 0.1.0\nruntime: node\nstart: [\"node\", \"server.js\"]\n{extra}"
        )
    }

    #[test]
    fn a_migrations_path_that_leaves_the_app_is_rejected() {
        let declared =
            |value: &str| format!("services:\n  postgres:\n    migrations: \"{value}\"\n");

        // What a manifest is allowed to say.
        Manifest::parse(&minimal(&declared("db/migrations")))
            .unwrap()
            .validate()
            .expect("a relative folder inside the repo is the whole point");

        for bad in [
            "C:\\\\Users\\\\me",
            "../x",
            "..\\\\x",
            "/etc",
            "\\\\\\\\server\\\\share",
        ] {
            let result = Manifest::parse(&minimal(&declared(bad)))
                .unwrap_or_else(|err| panic!("`{bad}` must parse: {err}"))
                .validate();
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains("migrations"),
                "`{bad}` must be refused by name: {err}"
            );
        }
    }

    #[test]
    fn a_migrations_symlink_out_of_the_app_is_refused_at_start() {
        let app = TempDir::new("migrations-app");
        let outside = TempDir::new("migrations-outside");
        std::fs::create_dir_all(outside.0.join("sql")).unwrap();

        // A folder inside the clone resolves to itself.
        let inside = app.0.join("db").join("migrations");
        std::fs::create_dir_all(&inside).unwrap();
        assert_eq!(
            resolve_migrations(&app.0, Path::new("db/migrations")).unwrap(),
            inside.canonicalize().unwrap()
        );

        // A folder the app has not written yet is not an error, there is
        // simply nothing to apply.
        assert_eq!(
            resolve_migrations(&app.0, Path::new("db/later")).unwrap(),
            app.0.join("db").join("later")
        );

        // The string is innocent and the destination is not, which is what the
        // shape checks cannot see.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.0.join("sql"), app.0.join("escape")).unwrap();
            let err = resolve_migrations(&app.0, Path::new("escape"))
                .expect_err("a symlink out of the app must be refused");
            assert!(
                err.to_string().contains("outside the app directory"),
                "{err}"
            );
        }
    }

    #[test]
    fn an_inference_parameter_is_rejected() {
        let manifest = Manifest::parse(&minimal("ctx: 4096\n")).expect("parses");
        let err = manifest.validate().expect_err("ctx must fail");
        let message = err.to_string();
        assert!(message.contains("ctx"), "{message}");
        assert!(message.contains("rule 4"), "{message}");
    }

    #[test]
    fn an_inference_parameter_inside_a_model_is_rejected() {
        let yaml = minimal(
            "models:\n  - alias: main\n    kind: llm\n    repo: Qwen/Qwen3-8B-GGUF\n    quant: [Q4_K_M]\n    n_parallel: 4\n",
        );
        let err = Manifest::parse(&yaml)
            .unwrap()
            .validate()
            .expect_err("n_parallel must fail");
        assert!(err.to_string().contains("n_parallel"), "{err}");
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let err = Manifest::parse(&minimal("gpu: true\n"))
            .unwrap()
            .validate()
            .expect_err("unknown key must fail");
        assert!(err.to_string().contains("not a manifest key"), "{err}");
    }

    #[test]
    fn aliases_must_be_unique_and_lowercase() {
        let yaml = minimal(
            "models:\n  - alias: Main\n    kind: llm\n    repo: a/b\n    quant: [Q4_K_M]\n  - alias: Main\n    kind: llm\n    repo: a/b\n    quant: [Q4_K_M]\n",
        );
        let err = Manifest::parse(&yaml)
            .unwrap()
            .validate()
            .expect_err("alias must fail");
        let message = err.to_string();
        assert!(message.contains("^[a-z][a-z0-9_]*$"), "{message}");
        assert!(message.contains("declared twice"), "{message}");
    }

    #[test]
    fn a_quant_preference_is_required() {
        let yaml =
            minimal("models:\n  - alias: main\n    kind: llm\n    repo: a/b\n    quant: []\n");
        let err = Manifest::parse(&yaml)
            .unwrap()
            .validate()
            .expect_err("quant must fail");
        assert!(err.to_string().contains("at least one quant"), "{err}");
    }

    #[test]
    fn a_missing_required_key_fails_to_parse() {
        let yaml = "name: demo\ndescription: A demo app.\nversion: 0.1.0\nruntime: node\n";
        assert!(Manifest::parse(yaml).is_err());
    }

    #[test]
    fn load_reports_a_missing_manifest() {
        let err = Manifest::load(Path::new("/nonexistent-app-dir")).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");
    }

    /// Both templates have to satisfy the same rules an app cloned from the
    /// store does, rule 5 and its Dockerfile included, or `init` hands an agent
    /// a repo that fails the moment it is validated.
    #[tokio::test]
    async fn both_templates_scaffold_a_valid_app() {
        for template in [Template::Next, Template::Fastapi] {
            let temp = TempDir::new(&format!("init-{}", template.slug()));
            let dir = temp.0.join("demo-app");

            let app = init("demo-app", template, &dir)
                .await
                .unwrap_or_else(|err| panic!("{} scaffolds: {err}", template.slug()));

            assert_eq!(app.dir, dir);
            assert_eq!(app.manifest.name, "demo-app");
            assert!(app.manifest.description.contains("demo-app"));
            assert_eq!(app.manifest.health.as_deref(), Some("/api/health"));
            assert_eq!(app.manifest.models.len(), 1);
            assert_eq!(app.manifest.models[0].alias, "chat");
            assert!(app.manifest.services.postgres.is_some());

            // The rules of PLAN section 3, Dockerfile included.
            validate_dir(&dir).expect("the scaffold validates");
            assert!(dir.join("Dockerfile").is_file());
            assert!(dir.join("db/migrations/0001_init.sql").is_file());
            // The template ships it one character short so git does not read it
            // as a rule for this repository.
            assert!(dir.join(".gitignore").is_file());
            assert!(!dir.join("gitignore").exists());

            // Nothing is left unrendered anywhere in the tree.
            for file in files_under(&dir) {
                if let Ok(text) = std::fs::read_to_string(&file) {
                    assert!(
                        !text.contains("{{name}}") && !text.contains("{{description}}"),
                        "{} still holds a placeholder",
                        file.display()
                    );
                }
            }

            assert_eq!(commit_count(&dir), 1, "{} has one commit", template.slug());
        }
    }

    #[tokio::test]
    async fn the_start_command_of_the_python_template_expands_no_variable() {
        let temp = TempDir::new("init-python-argv");
        let dir = temp.0.join("demo-app");
        let app = init("demo-app", Template::Fastapi, &dir).await.unwrap();

        assert_eq!(app.manifest.runtime, AppRuntime::Python);
        assert_eq!(app.manifest.build, vec!["uv", "sync", "--frozen"]);
        // `$PORT` in argv is four literal characters, so the entry point reads
        // the variable itself.
        assert_eq!(app.manifest.start, vec!["uv", "run", "python", "main.py"]);
        assert!(!app.manifest.start.iter().any(|arg| arg.contains('$')));
        let main = std::fs::read_to_string(dir.join("main.py")).unwrap();
        assert!(main.contains("PORT"), "{main}");
        assert!(main.contains("127.0.0.1"), "{main}");
    }

    #[tokio::test]
    async fn a_directory_with_files_in_it_is_not_scaffolded_over() {
        let temp = TempDir::new("init-occupied");
        let dir = temp.0.join("taken");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();

        let err = init("demo-app", Template::Next, &dir).await.unwrap_err();
        assert!(matches!(err, Error::InvalidManifest(_)), "{err}");
        assert!(dir.join("notes.txt").is_file(), "the files are still there");
    }

    /// The pid the platform holds is a launcher, and the launcher is not the
    /// server: `uv run python main.py` leaves a python process holding the port
    /// when only the parent is killed, which is how an app used to survive its
    /// own stop. `sh` stands in for `uv` here, and the grandchild for the app.
    #[cfg(unix)]
    #[tokio::test]
    async fn stopping_an_app_kills_what_its_launcher_started() {
        let mut launcher = Command::new("sh")
            .args(["-c", "sleep 60 & wait"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("a launcher starts");

        let mut grandchild = None;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let tree = process_tree(launcher.id()).await.expect("a process table");
            grandchild = tree.into_iter().find(|pid| *pid != launcher.id());
            if grandchild.is_some() {
                break;
            }
        }
        let grandchild = grandchild.expect("the launcher started a process of its own");

        kill_tree(&mut launcher).await;

        let mut alive = true;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            alive = std::process::Command::new("kill")
                .args(["-0", &grandchild.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);
            if !alive {
                break;
            }
        }
        assert!(!alive, "pid {grandchild} outlived the app it belonged to");
    }

    #[tokio::test]
    async fn a_fork_renames_the_manifest_and_starts_a_new_history() {
        let temp = TempDir::new("fork");
        let data_dir = temp.0.join("data");
        let source = paths::apps_dir(&data_dir).join("demo-app");
        init("demo-app", Template::Next, &source).await.unwrap();
        // Build output of the source must not travel with the source.
        std::fs::create_dir_all(source.join("node_modules/left-pad")).unwrap();
        std::fs::write(source.join("node_modules/left-pad/index.js"), "x").unwrap();

        let dir = temp.0.join("forked");
        let app = fork_in("demo-app", "second-firm", &dir, &data_dir)
            .await
            .expect("an installed app forks by name");

        assert_eq!(app.manifest.name, "second-firm");
        assert_eq!(app.dir, dir);
        // The rename is a line edit, so the comments of the template survive.
        let manifest = std::fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap();
        assert!(manifest.contains("name: second-firm"), "{manifest}");
        assert!(manifest.contains("# AI App Store manifest"), "{manifest}");
        assert!(!manifest.contains("name: demo-app"), "{manifest}");

        assert!(
            !dir.join("node_modules").exists(),
            "build output is not source"
        );
        assert!(
            !dir.join(ORIGIN_FILE).exists(),
            "the origin is the source app's"
        );
        assert_eq!(commit_count(&dir), 1, "the history starts here");
    }

    /// Number of commits on the current branch, `0` when there is no repository.
    fn commit_count(dir: &Path) -> usize {
        let output = std::process::Command::new("git")
            .args(["-C", &dir.to_string_lossy(), "rev-list", "--count", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0)
    }

    /// Every file under a directory, `.git` left out.
    fn files_under(dir: &Path) -> Vec<PathBuf> {
        let mut found = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return found;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name().is_some_and(|name| name == ".git") {
                continue;
            }
            if path.is_dir() {
                found.extend(files_under(&path));
            } else {
                found.push(path);
            }
        }
        found
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "aias-apps-{}-{tag}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const TINY_MANIFEST: &str = "name: demo\ndescription: A demo app.\nversion: 0.1.0\nruntime: node\nstart: [\"node\", \"server.js\"]\n";

    fn git_repo(dir: &Path) {
        std::fs::write(dir.join(MANIFEST_FILE), TINY_MANIFEST).unwrap();
        std::fs::write(dir.join("Dockerfile"), "FROM scratch\nEXPOSE $PORT\n").unwrap();
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .expect("git runs");
            assert!(status.success(), "git {args:?} failed");
        };
        git(&["init"]);
        git(&["add", "."]);
        git(&[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-m",
            "initial",
        ]);
    }

    #[test]
    fn only_https_and_a_local_directory_are_cloned_from() {
        assert!(check_clone_url("https://github.com/zyx1121/aias-example-chat").is_ok());

        // `ext::` hands git a command to run, which is the one that matters.
        for bad in [
            "ext::sh -c whoami",
            "file:///etc",
            "git@github.com:zyx1121/aias-example-chat.git",
            "http://example.test/repo.git",
            "ssh://example.test/repo.git",
        ] {
            let err = check_clone_url(bad).unwrap_err();
            assert!(
                err.to_string().contains("not an app source"),
                "{bad}: {err}"
            );
        }

        // A directory on this machine is how an app is developed and tested.
        let dir = TempDir::new("clone-url");
        assert!(check_clone_url(&dir.0.display().to_string()).is_ok());
    }

    #[test]
    fn a_ref_cannot_be_read_as_a_git_option() {
        for good in [
            "main",
            "v1.2.3",
            "release/2026-01",
            "feature_x",
            "0",
            "a-b.c/d",
        ] {
            check_git_ref(good).unwrap_or_else(|err| panic!("{good} is a ref: {err}"));
        }

        // `--upload-pack` is the one that matters: git runs what it names.
        for bad in [
            "--upload-pack=touch /tmp/pwned",
            "-x",
            "--exec=whoami",
            "",
            "-",
            "main;whoami",
            "main ref",
            "main\n--upload-pack=x",
        ] {
            let err = check_git_ref(bad).unwrap_err();
            assert!(
                err.to_string().contains("not a usable git ref"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn an_index_entry_cannot_publish_a_flag_shaped_ref() {
        let entry = "- name: example-chat\n  repo: https://github.com/zyx1121/aias-example-chat\n  ref: \"--upload-pack=touch /tmp/pwned\"\n";
        serde_yaml_ng::from_str::<IndexFile>(entry)
            .expect_err("an index entry with a flag shaped ref must not parse");
    }

    #[tokio::test]
    async fn a_ref_from_the_index_never_reaches_git_as_an_option() {
        // The reviewer's exploit: a local path origin lets `--upload-pack=<cmd>`
        // run `<cmd>`, and the ref used to reach `git fetch` as a positional.
        let source = TempDir::new("upload-pack-source");
        let source_repo = source.0.join("demo-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);

        let data = TempDir::new("upload-pack");
        let url = source_repo.display().to_string();
        clone_app(&url, None, &data.0).await.expect("first clone");

        let marker = data.0.join("pwned");
        for bad in [
            format!("--upload-pack=touch {}", marker.display()),
            "-x".to_string(),
        ] {
            let err = clone_app(&url, Some(&bad), &data.0).await.unwrap_err();
            assert!(
                err.to_string().contains("not a usable git ref"),
                "{bad}: {err}"
            );
            assert!(!marker.exists(), "`{bad}` ran a command through git");
        }
    }

    #[test]
    fn a_health_path_cannot_write_its_own_request_line() {
        let with = |health: &str| {
            Manifest::parse(&minimal(&format!("health: \"{health}\"\n")))
                .unwrap()
                .validate()
        };

        with("/api/health").expect("an ordinary path is the point");
        with("/").expect("the root is a path");

        for bad in [
            "api/health",
            "/health HTTP/1.1\\r\\nX-Evil: 1",
            "/health?deep=1",
            "/health with a space",
        ] {
            let err = with(bad).unwrap_err();
            assert!(err.to_string().contains("health"), "{bad}: {err}");
        }
    }

    #[test]
    fn the_directory_name_comes_from_the_url_tail() {
        assert_eq!(
            repo_dir_name("https://github.com/example/contract-review.git").unwrap(),
            "contract-review"
        );
        assert_eq!(
            repo_dir_name("https://github.com/example/contract-review/").unwrap(),
            "contract-review"
        );
        assert_eq!(
            repo_dir_name("git@github.com:example/contract-review.git").unwrap(),
            "contract-review"
        );
        assert!(repo_dir_name("https://").is_err());
        assert!(repo_dir_name("https://github.com/example/..").is_err());
    }

    /// Run git in `dir` and give back its trimmed stdout.
    fn git_out(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git runs");
        assert!(output.status.success(), "git {args:?} failed");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    #[tokio::test]
    async fn a_ref_decides_what_is_checked_out_and_the_sha_is_recorded() {
        let source = TempDir::new("ref-source");
        let source_repo = source.0.join("demo-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);
        let first = git_out(&source_repo, &["rev-parse", "HEAD"]);
        git_out(&source_repo, &["tag", "v1"]);

        // A second commit on the default branch, so `v1` and the tip differ.
        std::fs::write(
            source_repo.join("Dockerfile"),
            "FROM scratch\nEXPOSE $PORT\n# second\n",
        )
        .unwrap();
        git_out(
            &source_repo,
            &[
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=test",
                "commit",
                "-am",
                "second",
            ],
        );
        let tip = git_out(&source_repo, &["rev-parse", "HEAD"]);
        git_out(&source_repo, &["tag", "v2"]);
        assert_ne!(first, tip);

        let data = TempDir::new("ref-clone");
        let url = source_repo.display().to_string();
        let dir = clone_app(&url, Some("v1"), &data.0).await.unwrap();
        assert_eq!(
            git_out(&dir, &["rev-parse", "HEAD"]),
            first,
            "`ref` must decide the checkout, not the default branch"
        );

        let origin: Origin =
            serde_json::from_str(&std::fs::read_to_string(dir.join(ORIGIN_FILE)).unwrap()).unwrap();
        assert_eq!(origin.url, url);
        assert_eq!(origin.git_ref.as_deref(), Some("v1"));
        assert_eq!(origin.sha.as_deref(), Some(first.as_str()));

        // `installed` carries the ref and the SHA to the CLI and the Apps page.
        let listed = installed(&data.0);
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].git_ref.as_deref(), Some("v1"));
        assert_eq!(listed[0].sha.as_deref(), Some(first.as_str()));

        // Re-subscribing at a new ref moves the existing clone and re-records it.
        clone_app(&url, Some("v2"), &data.0).await.unwrap();
        assert_eq!(git_out(&dir, &["rev-parse", "HEAD"]), tip);
        let moved = installed(&data.0);
        assert_eq!(moved[0].git_ref.as_deref(), Some("v2"));
        assert_eq!(moved[0].sha.as_deref(), Some(tip.as_str()));
    }

    #[test]
    fn an_origin_file_written_before_the_sidecar_was_json_still_reads() {
        let legacy = parse_origin("https://github.com/zyx1121/aias-example-chat\n").unwrap();
        assert_eq!(legacy.url, "https://github.com/zyx1121/aias-example-chat");
        assert_eq!(legacy.git_ref, None);
        assert_eq!(legacy.sha, None);

        let current = parse_origin(r#"{"url":"https://example.test/a","ref":"v2","sha":"abc"}"#)
            .expect("json parses");
        assert_eq!(current.git_ref.as_deref(), Some("v2"));
        assert_eq!(current.sha.as_deref(), Some("abc"));

        assert_eq!(parse_origin("   "), None);
    }

    #[tokio::test]
    async fn a_local_repo_clones_and_is_validated() {
        let source = TempDir::new("source-repo");
        let source_repo = source.0.join("demo-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);

        let data = TempDir::new("clone");
        let dir = clone_app(&source_repo.display().to_string(), None, &data.0)
            .await
            .unwrap();
        assert_eq!(dir, paths::apps_dir(&data.0).join("demo-app"));
        assert!(dir.join(MANIFEST_FILE).is_file());

        // Second call fast forwards the existing clone instead of failing.
        let again = clone_app(&source_repo.display().to_string(), None, &data.0)
            .await
            .unwrap();
        assert_eq!(again, dir);
    }

    #[tokio::test]
    async fn an_app_is_named_and_never_pointed_at() {
        let source = TempDir::new("app-dir-source");
        // The clone is named after the repo, the app after its manifest: the
        // two differ, which is why the lookup reads the manifest.
        let source_repo = source.0.join("aias-demo");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);

        let data = TempDir::new("app-dir");
        let dir = clone_app(&source_repo.display().to_string(), None, &data.0)
            .await
            .expect("clone");
        assert_eq!(dir, paths::apps_dir(&data.0).join("aias-demo"));
        assert_eq!(app_dir(&data.0, "demo").unwrap(), dir);

        // Nothing the renderer can send reaches a directory of its choosing.
        for bad in [
            "../../../etc",
            "..",
            "/etc",
            "C:\\Windows",
            "demo/../../..",
            "Demo",
            "-demo",
            "",
            &"d".repeat(52),
        ] {
            let err = app_dir(&data.0, bad).unwrap_err();
            assert!(
                err.to_string().contains("must match"),
                "`{bad}` was not refused on its shape: {err}"
            );
        }

        // A well shaped name that is not subscribed is a plain not found.
        let err = app_dir(&data.0, "not-subscribed").unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");

        // A clone whose manifest stopped parsing still resolves, so the user is
        // told what is wrong with it rather than that it does not exist.
        let broken = paths::apps_dir(&data.0).join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join(MANIFEST_FILE), "name: [not a manifest\n").unwrap();
        assert_eq!(app_dir(&data.0, "broken").unwrap(), broken);
        assert!(validate_dir(&broken).is_err());
    }

    #[tokio::test]
    async fn a_repo_without_a_manifest_is_refused() {
        let source = TempDir::new("bare-repo");
        let source_repo = source.0.join("not-an-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);
        std::fs::remove_file(source_repo.join(MANIFEST_FILE)).unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=test",
                "commit",
                "-am",
                "drop the manifest",
            ])
            .current_dir(&source_repo)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();

        let data = TempDir::new("clone-bad");
        let err = clone_app(&source_repo.display().to_string(), None, &data.0)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");
    }

    #[tokio::test]
    async fn a_build_runs_every_command_and_reports_the_failing_one() {
        let data = TempDir::new("build");
        let app = TempDir::new("build-app");

        let ok: Manifest = Manifest::parse(&format!(
            "{TINY_MANIFEST}build: [\"git\", \"--version\", \"&&\", \"git\", \"--version\"]\n"
        ))
        .unwrap();
        build_in(&app.0, &ok, &data.0).await.unwrap();
        let log =
            std::fs::read_to_string(paths::logs_dir(&data.0).join("app-demo-build.log")).unwrap();
        assert_eq!(log.matches("git version").count(), 2, "{log}");

        let bad: Manifest = Manifest::parse(&format!(
            "{TINY_MANIFEST}build: [\"git\", \"--not-a-flag\"]\n"
        ))
        .unwrap();
        let err = build_in(&app.0, &bad, &data.0).await.unwrap_err();
        assert!(matches!(err, Error::Process(_)), "{err}");
        assert!(err.to_string().contains("--not-a-flag"), "{err}");
    }

    #[tokio::test]
    async fn a_fork_source_outside_the_users_own_files_is_refused() {
        let data = TempDir::new("fork-source");
        let target = TempDir::new("fork-target");

        // A directory that exists and is not the user's to fork. On Windows
        // this is a relative path that exists nowhere, which is refused by the
        // same branch for the same reason.
        let err = fork_in("/etc", "demo", &target.0.join("demo"), &data.0)
            .await
            .expect_err("/etc is not an app source");
        assert_eq!(err.code(), "invalid_manifest", "{err}");
        assert!(err.to_string().contains("/etc"), "{err}");
        assert!(
            !target.0.join("demo").exists(),
            "nothing was cloned into the target"
        );
    }

    #[test]
    fn a_fork_source_is_a_url_or_a_directory_the_user_may_work_in() {
        let data = TempDir::new("fork-check");

        // The two shapes that are always allowed.
        assert_eq!(
            check_fork_source("https://github.com/zyx1121/aias-example-chat", &data.0).unwrap(),
            PathBuf::from("https://github.com/zyx1121/aias-example-chat")
        );
        let installed = paths::apps_dir(&data.0).join("example-chat");
        std::fs::create_dir_all(&installed).unwrap();
        check_fork_source(&installed.display().to_string(), &data.0)
            .expect("a clone under apps_dir is the user's own");

        // Everything git would otherwise accept.
        for bad in [
            "ext::sh -c whoami",
            "file:///etc",
            "git@github.com:zyx1121/aias-index.git",
            "",
        ] {
            let err =
                check_fork_source(bad, &data.0).expect_err(&format!("`{bad}` must be refused"));
            assert_eq!(err.code(), "invalid_manifest", "{err}");
        }

        // A real directory outside the user's files, which is what the plain
        // `is_dir` check used to wave through.
        #[cfg(unix)]
        {
            let outside = PathBuf::from("/tmp").join(format!("aias-fork-{}", std::process::id()));
            std::fs::create_dir_all(&outside).unwrap();
            let err = check_fork_source(&outside.display().to_string(), &data.0)
                .expect_err("a directory outside the home is not a fork source");
            assert_eq!(err.code(), "invalid_manifest", "{err}");
            std::fs::remove_dir_all(&outside).unwrap();
        }
    }

    #[tokio::test]
    async fn a_build_command_that_does_not_exist_says_so_in_the_build_log() {
        let data = TempDir::new("build-spawn");
        let app = TempDir::new("build-spawn-app");

        let manifest = Manifest::parse(&format!(
            "{TINY_MANIFEST}build: [\"aias-no-such-program\", \"install\"]\n"
        ))
        .unwrap();
        let err = build_in(&app.0, &manifest, &data.0).await.unwrap_err();
        assert!(err.to_string().contains("aias-no-such-program"), "{err}");

        // The log is the only place an agent can read the failure from, so the
        // failure has to be in it.
        let log = logs_in("demo", LogKind::Build, 50, &data.0).unwrap();
        assert!(
            log.contains("spawn failed: aias-no-such-program"),
            "the build log says nothing about the missing program: {log}"
        );
    }

    #[test]
    fn secrets_are_read_when_the_file_is_there() {
        let data = TempDir::new("secrets");
        assert!(secrets("demo", &data.0).unwrap().is_empty());

        std::fs::create_dir_all(data.0.join("secrets")).unwrap();
        std::fs::write(
            data.0.join("secrets").join("demo.json"),
            r#"{"DOCUSIGN_API_KEY": "abc"}"#,
        )
        .unwrap();
        assert_eq!(
            secrets("demo", &data.0).unwrap().get("DOCUSIGN_API_KEY"),
            Some(&"abc".to_string())
        );

        std::fs::write(data.0.join("secrets").join("demo.json"), r#"{"K": 1}"#).unwrap();
        assert!(secrets("demo", &data.0).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_secrets_file_is_read_by_this_user_and_nobody_else() {
        use std::os::unix::fs::PermissionsExt as _;

        let data = TempDir::new("secrets-mode");
        let path = paths::secrets_dir(&data.0).join("demo.json");
        std::fs::create_dir_all(paths::secrets_dir(&data.0)).unwrap();
        std::fs::write(&path, r#"{"DOCUSIGN_API_KEY": "abc"}"#).unwrap();
        // What an editor or an older build leaves behind.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        secrets("demo", &data.0).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "the secrets file is world readable");
    }

    #[test]
    fn an_app_port_is_above_the_instance_range_and_never_the_postgres_one() {
        let port = pick_app_port().unwrap();
        assert!(port > APP_PORT_START, "{port} must leave 41500 to Postgres");
        assert!(port <= APP_PORT_END);
        assert!(port > crate::instances::INSTANCE_PORT_END);
    }

    #[tokio::test]
    async fn the_database_url_reaches_the_app_environment() {
        let data = TempDir::new("database");
        // `services` covers the real cluster in its own tests; standing in for
        // one here keeps this about who hands the URL to the app.
        set_database_hook(Some(|app| {
            Ok(format!("postgres://{app}@127.0.0.1:41500/{app}"))
        }));
        let url = provision_database("demo", &data.0, &data.0).await.unwrap();
        assert_eq!(url, "postgres://demo@127.0.0.1:41500/demo");
        set_database_hook(None);
    }

    #[test]
    fn the_index_parses_both_published_shapes() {
        let bare = "- name: example-chat\n  repo: https://github.com/zyx1121/aias-example-chat\n  description: Chat with one local model\n  tags: [chat]\n";
        let entries: Vec<IndexEntry> = match serde_yaml_ng::from_str(bare).unwrap() {
            IndexFile::Wrapped { apps } => apps,
            IndexFile::Bare(apps) => apps,
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "example-chat");
        assert_eq!(entries[0].git_ref, "main", "ref defaults to main");
        assert_eq!(entries[0].homepage, None);

        let wrapped = "apps:\n  - name: example-chat\n    repo: https://github.com/zyx1121/aias-example-chat\n    ref: v1\n    description: Chat with one local model\n";
        let entries: Vec<IndexEntry> = match serde_yaml_ng::from_str(wrapped).unwrap() {
            IndexFile::Wrapped { apps } => apps,
            IndexFile::Bare(apps) => apps,
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].repo,
            "https://github.com/zyx1121/aias-example-chat"
        );
        assert_eq!(entries[0].git_ref, "v1");
        assert!(entries[0].tags.is_empty());
    }

    #[test]
    fn the_index_url_is_overridable() {
        assert!(index_url().starts_with("https://"));
        assert_eq!(index_url(), DEFAULT_INDEX_URL);
    }

    #[tokio::test]
    async fn a_subscribed_app_is_listed_with_its_origin_and_can_be_removed() {
        let source = TempDir::new("origin-source");
        let source_repo = source.0.join("demo-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        git_repo(&source_repo);

        let data = TempDir::new("origin");
        let url = source_repo.display().to_string();
        let app = subscribe(&url, None, &data.0).await.unwrap();
        assert_eq!(app.name, "demo");
        assert_eq!(app.repo_url.as_deref(), Some(url.as_str()));

        let listed = installed(&data.0);
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].name, "demo");
        assert_eq!(listed[0].dir, paths::apps_dir(&data.0).join("demo-app"));
        assert_eq!(listed[0].repo_url.as_deref(), Some(url.as_str()));

        remove("demo", &data.0).unwrap();
        assert!(installed(&data.0).is_empty());
        let err = remove("demo", &data.0).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");
    }

    #[tokio::test]
    async fn subscribing_clones_but_runs_nothing_from_the_repo() {
        let source = TempDir::new("consent-source");
        let source_repo = source.0.join("demo-app");
        std::fs::create_dir_all(&source_repo).unwrap();
        // A build step with a visible side effect: if it ever runs, the log says so.
        std::fs::write(
            source_repo.join(MANIFEST_FILE),
            format!("{TINY_MANIFEST}build: [\"git\", \"--version\"]\n"),
        )
        .unwrap();
        std::fs::write(
            source_repo.join("Dockerfile"),
            "FROM scratch\nEXPOSE $PORT\n",
        )
        .unwrap();
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(&source_repo)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .expect("git runs");
            assert!(status.success(), "git {args:?} failed");
        };
        git(&["init"]);
        git(&["add", "."]);
        git(&[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-m",
            "initial",
        ]);

        let data = TempDir::new("consent");
        let app = subscribe(&source_repo.display().to_string(), None, &data.0)
            .await
            .unwrap();
        assert_eq!(app.name, "demo");
        assert!(
            app.dir.join(MANIFEST_FILE).is_file(),
            "the clone is on disk"
        );
        // The build log is the proof: subscribing must not have run the argv,
        // the caller has to show `build` and `start` and ask first.
        assert_eq!(
            logs_in("demo", LogKind::Build, 50, &data.0).unwrap(),
            "",
            "subscribe must not run the manifest build"
        );
        assert_eq!(app.manifest.build, vec!["git", "--version"]);

        // Consent given, the build runs and leaves its log.
        build_in(&app.dir, &app.manifest, &data.0).await.unwrap();
        assert!(
            logs_in("demo", LogKind::Build, 50, &data.0)
                .unwrap()
                .contains("git version"),
            "the build runs once it is asked for"
        );
    }

    #[test]
    fn a_missing_model_is_the_first_choice_of_the_alias() {
        let data = TempDir::new("missing-models");
        let yaml = minimal(
            "models:\n  - alias: main\n    kind: llm\n    repo: Qwen/Qwen3-8B-GGUF\n    quant: [Q4_K_M, Q8_0]\n    fallback: Qwen/Qwen3-4B-GGUF\n",
        );
        let manifest = Manifest::parse(&yaml).unwrap();
        assert_eq!(
            missing_models(&manifest, &data.0),
            vec![ModelRef::new("Qwen/Qwen3-8B-GGUF", "Q4_K_M")]
        );

        // The fallback counts as satisfying the alias, so nothing is missing.
        let dir = crate::models::model_dir(&ModelRef::new("Qwen/Qwen3-4B-GGUF", "Q8_0"), &data.0);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Qwen3-4B-Q8_0.gguf"), b"x").unwrap();
        std::fs::write(
            dir.join("model.json"),
            r#"{"repo":"Qwen/Qwen3-4B-GGUF","quant":"Q8_0","filename":"Qwen3-4B-Q8_0.gguf","mmproj":null}"#,
        )
        .unwrap();
        assert!(missing_models(&manifest, &data.0).is_empty());
    }

    #[tokio::test]
    async fn the_build_log_is_readable_by_name_and_kind() {
        let data = TempDir::new("logs");
        let app = TempDir::new("logs-app");
        let manifest =
            Manifest::parse(&format!("{TINY_MANIFEST}build: [\"git\", \"--version\"]\n")).unwrap();
        build_in(&app.0, &manifest, &data.0).await.unwrap();

        let build_log = logs_in("demo", LogKind::Build, 50, &data.0).unwrap();
        assert!(build_log.contains("git version"), "{build_log}");
        // Nothing ran, so the run log is empty rather than an error.
        assert_eq!(logs_in("demo", LogKind::Run, 50, &data.0).unwrap(), "");
    }

    #[test]
    fn the_log_an_agent_reads_is_the_masked_one() {
        let data = TempDir::new("log-mask");
        // What the API route and the `app_logs` tool both call is `logs_in`,
        // so this is the one place the promise in that tool's description is
        // kept or broken.
        let key = instances::new_api_key();
        let token = crate::api::token(&data.0).unwrap();
        std::fs::create_dir_all(paths::logs_dir(&data.0)).unwrap();
        std::fs::write(
            paths::logs_dir(&data.0).join("app-demo.log"),
            format!(
                "[app] AIAS_MODEL_CHAT_KEY={key}\n\
                 [app] DATABASE_URL=postgres://demo:Hn3kQ2vTpLsd8XwR@127.0.0.1:41500/demo\n\
                 [app] fetch /v1/chat/completions Authorization: Bearer {key}\n\
                 [app] calling the platform with {token}\n\
                 [app] listening on 127.0.0.1:41501\n"
            ),
        )
        .unwrap();

        let tail = logs_in("demo", LogKind::Run, 50, &data.0).unwrap();
        assert!(!tail.contains(&key), "the instance key is readable: {tail}");
        assert!(!tail.contains(&token), "the API token is readable: {tail}");
        assert!(!tail.contains("Hn3kQ2vTpLsd8XwR"), "{tail}");
        // Masking is not redaction: the log still says what happened.
        assert!(tail.contains("Authorization: Bearer ***"), "{tail}");
        assert!(tail.contains("listening on 127.0.0.1:41501"), "{tail}");
    }

    #[test]
    fn a_log_is_only_read_for_something_that_could_be_an_app_name() {
        let data = TempDir::new("log-escape");
        // The name lands in a file name under `<data_dir>/logs`, so the same
        // shape `app_dir` demands is demanded here. It used to be interpolated
        // raw, which made `../` a way out of the log directory.
        let escape = std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("aias-log-escape-{}", std::process::id()));
        std::fs::write(&escape, "not a log of yours\n").unwrap();

        for name in ["../x", "..", "/etc/passwd", "demo/../../x", "Demo", ""] {
            let err =
                logs_in(name, LogKind::Run, 50, &data.0).expect_err("{name} is not an app name");
            assert!(err.to_string().contains("^[a-z][a-z0-9-]"), "{err}");
        }

        // The traversal that used to work, spelled against a real file.
        let up = format!("../../{}", escape.file_name().unwrap().to_string_lossy());
        assert!(logs_in(&up, LogKind::Run, 50, &data.0).is_err());
        // A real name still reads, and an absent log is still empty.
        assert_eq!(logs_in("demo", LogKind::Run, 50, &data.0).unwrap(), "");

        std::fs::remove_file(&escape).ok();
    }

    const NETSTAT: &str = "\r
Active Connections\r
\r
  Proto  Local Address          Foreign Address        State           PID\r
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1044\r
  TCP    127.0.0.1:41501        0.0.0.0:0              LISTENING       9100\r
  TCP    0.0.0.0:41502          0.0.0.0:0              LISTENING       9200\r
  TCP    [::]:41503             [::]:0                 LISTENING       9300\r
  TCP    127.0.0.1:41501        127.0.0.1:52110        ESTABLISHED     9100\r
";

    const LSOF: &str = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
bun     91001 user   20u  IPv4 0xabc      0t0  TCP 127.0.0.1:41501 (LISTEN)\n";

    const LSOF_WILDCARD: &str = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
bun     91002 user   20u  IPv4 0xdef      0t0  TCP *:41502 (LISTEN)\n";

    /// One app, two sockets: the port the platform assigned, on loopback, and a
    /// second one the app opened on the network that nobody asked about.
    const LSOF_SECOND_SOCKET: &str = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
node    91003 user   20u  IPv4 0xabc      0t0  TCP 127.0.0.1:41501 (LISTEN)\n\
node    91003 user   23u  IPv4 0xdef      0t0  TCP 0.0.0.0:8080 (LISTEN)\n\
other   91004 user   11u  IPv4 0x123      0t0  TCP 0.0.0.0:3000 (LISTEN)\n";

    /// The same shape on Windows: PID is the last field of a netstat row.
    const NETSTAT_SECOND_SOCKET: &str = "\r
  Proto  Local Address          Foreign Address        State           PID\r
  TCP    127.0.0.1:41501        0.0.0.0:0              LISTENING       9100\r
  TCP    0.0.0.0:8080           0.0.0.0:0              LISTENING       9100\r
  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       9999\r
";

    /// `ps -eo pid,ppid` on Unix: the app is 91003, it launched 91010, and that
    /// one launched 91020. 91004 belongs to somebody else entirely.
    const PS_TABLE: &str = "  PID  PPID\n\
 91003     1\n\
 91010 91003\n\
 91020 91010\n\
 91004     1\n";

    /// The same tree as the PowerShell one liner prints it on Windows.
    const WIN_PROCESS_TABLE: &str = "9100 1\r\n9110 9100\r\n9120 9110\r\n9999 1\r\n";

    /// The launcher case: the pid this platform holds listens on nothing, the
    /// grandchild answers the assigned port on loopback and also opened 8080
    /// on the wildcard. Only walking the tree finds that second socket.
    const LSOF_GRANDCHILD: &str = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
node    91020 user   20u  IPv4 0xabc      0t0  TCP 127.0.0.1:41501 (LISTEN)\n\
node    91020 user   23u  IPv4 0xdef      0t0  TCP 0.0.0.0:8080 (LISTEN)\n\
other   91004 user   11u  IPv4 0x123      0t0  TCP 0.0.0.0:3000 (LISTEN)\n";

    #[test]
    fn a_process_table_reads_the_same_on_both_platforms() {
        assert_eq!(
            parse_process_table(PS_TABLE),
            [(91003, 1), (91010, 91003), (91020, 91010), (91004, 1)],
            "the `PID PPID` header is not a process"
        );
        assert_eq!(
            parse_process_table(WIN_PROCESS_TABLE),
            [(9100, 1), (9110, 9100), (9120, 9110), (9999, 1)]
        );
        assert!(parse_process_table("").is_empty());
        assert!(parse_process_table("no columns here\n").is_empty());
    }

    #[test]
    fn the_tree_of_a_pid_is_itself_and_everything_under_it() {
        let table = parse_process_table(PS_TABLE);
        assert_eq!(descendants(&table, 91003), [91003, 91010, 91020]);
        // A leaf is its own tree, and somebody else's process is not in it.
        assert_eq!(descendants(&table, 91020), [91020]);
        assert!(!descendants(&table, 91003).contains(&91004));

        // A table that reports a cycle terminates instead of spinning.
        assert_eq!(descendants(&[(1, 2), (2, 1)], 1), [1, 2]);
        assert_eq!(descendants(&[(7, 7)], 7), [7]);
    }

    #[tokio::test]
    async fn the_process_table_of_this_machine_parses() {
        // The fake tables above prove the logic; this proves the argv, on both
        // platforms, because the Windows one liner is only ever run by CI.
        let tree = process_tree(std::process::id())
            .await
            .expect("the process table must be readable");
        assert_eq!(
            tree.first(),
            Some(&std::process::id()),
            "a tree starts at its root"
        );
    }

    #[test]
    fn a_grandchild_on_a_second_routable_port_is_caught() {
        let tree = descendants(&parse_process_table(PS_TABLE), 91003);

        // What the old check asked: the direct child pid, or the assigned
        // port. Both miss 8080, because the pid is not 91003 and the port is
        // not 41501.
        let missed = parse_lsof_listeners(LSOF_GRANDCHILD, &[91003], 41501);
        assert_eq!(missed, ["127.0.0.1:41501"]);
        check_listeners(&missed, Some(91003), 41501).expect("this is what used to pass");

        // What the tree asks: every pid under the launcher, 8080 included.
        let found = parse_lsof_listeners(LSOF_GRANDCHILD, &tree, 41501);
        assert_eq!(found, ["127.0.0.1:41501", "0.0.0.0:8080"]);
        let err = check_listeners(&found, Some(91003), 41501)
            .expect_err("a grandchild on 0.0.0.0 is on the network");
        assert!(err.to_string().contains("0.0.0.0:8080"), "{err}");

        // Another user's process on the wildcard is still not this app's.
        assert!(!found.iter().any(|address| address.contains(":3000")));
    }

    #[test]
    fn a_declared_alias_produces_a_url_an_id_and_a_key() {
        let manifest = Manifest::parse(&minimal(
            "models:\n  - alias: chat\n    kind: llm\n    repo: Qwen/Qwen3-8B-GGUF\n    quant: [Q4_K_M]\n",
        ))
        .unwrap();
        let url = InstanceUrl {
            base_url: "http://127.0.0.1:41029/v1".into(),
            model_id: "Qwen/Qwen3-8B-GGUF:Q4_K_M".into(),
            port: 41029,
            api_key: "a".repeat(64),
        };

        let env: BTreeMap<String, String> =
            model_env(&manifest.models[0], url).into_iter().collect();
        assert_eq!(
            env.get("AIAS_MODEL_CHAT_URL").map(String::as_str),
            Some("http://127.0.0.1:41029/v1")
        );
        assert_eq!(
            env.get("AIAS_MODEL_CHAT_ID").map(String::as_str),
            Some("Qwen/Qwen3-8B-GGUF:Q4_K_M")
        );
        assert_eq!(
            env.get("AIAS_MODEL_CHAT_KEY").map(String::as_str),
            Some("a".repeat(64).as_str())
        );
    }

    /// The argv that prints the environment it was started with.
    #[cfg(windows)]
    const PRINT_ENV: &str = "start: [\"cmd\", \"/c\", \"set\"]\n";
    #[cfg(not(windows))]
    const PRINT_ENV: &str = "start: [\"env\"]\n";

    #[tokio::test]
    async fn an_app_is_started_with_a_built_environment_and_not_an_inherited_one() {
        let _guard = instances::TEST_LOCK.lock().await;
        let data = TempDir::new("app-env");
        let app = TempDir::new("app-env-app");

        // Safety: no other test in this crate reads `AIAS_API_TOKEN`, and the
        // registry lock above keeps the two tests that spawn processes apart.
        unsafe { std::env::set_var("AIAS_API_TOKEN", "canary-must-not-be-inherited") };

        let manifest = Manifest::parse(&format!(
            "name: demo\ndescription: A demo app.\nversion: 0.1.0\nruntime: node\n{PRINT_ENV}env:\n  AIAS_MODEL_CHAT_URL: http://127.0.0.1:41029/v1\n"
        ))
        .unwrap();

        // The command prints and exits, so starting it always fails; the log is
        // what this test is after.
        start_in(&app.0, &manifest, &data.0)
            .await
            .expect_err("`env` exits at once, so it never becomes a running app");

        unsafe { std::env::remove_var("AIAS_API_TOKEN") };

        let dump = std::fs::read_to_string(paths::logs_dir(&data.0).join("app-demo.log")).unwrap();
        assert!(
            !dump.contains("canary-must-not-be-inherited"),
            "the local API bearer token reached the app: {dump}"
        );
        assert!(!dump.contains("AIAS_API_TOKEN"), "{dump}");

        let has = |name: &str, value: &str| {
            dump.lines()
                .any(|line| line.eq_ignore_ascii_case(&format!("{name}={value}")))
        };
        assert!(
            dump.lines()
                .any(|line| line.to_ascii_uppercase().starts_with("PORT=4")),
            "the assigned port is missing: {dump}"
        );
        assert!(has("HOST", LOOPBACK), "{dump}");
        assert!(
            has("AIAS_MODEL_CHAT_URL", "http://127.0.0.1:41029/v1"),
            "{dump}"
        );
    }

    #[test]
    fn the_platform_variables_win_over_a_manifest_that_declares_them() {
        let mut env: BTreeMap<String, String> = BTreeMap::new();
        // What a hostile or careless manifest would put in `env`.
        env.insert("HOST".into(), "0.0.0.0".into());
        env.insert("HOSTNAME".into(), "0.0.0.0".into());
        env.insert("PORT".into(), "8080".into());

        for (key, value) in platform_env(41501) {
            env.insert(key, value);
        }

        assert_eq!(env.get("HOST").map(String::as_str), Some("127.0.0.1"));
        assert_eq!(env.get("HOSTNAME").map(String::as_str), Some("127.0.0.1"));
        assert_eq!(env.get("PORT").map(String::as_str), Some("41501"));
    }

    #[test]
    fn a_listening_socket_is_read_back_from_the_operating_system() {
        let any = |port| parse_netstat_listeners(NETSTAT, &[], port);
        assert_eq!(any(41501), ["127.0.0.1:41501"]);
        assert_eq!(any(41502), ["0.0.0.0:41502"]);
        assert_eq!(any(41503), ["[::]:41503"]);
        // Nothing listens there, and an established connection is not a listener.
        assert!(any(41504).is_empty());

        assert_eq!(parse_lsof_listeners(LSOF, &[], 41501), ["127.0.0.1:41501"]);
        assert_eq!(parse_lsof_listeners(LSOF_WILDCARD, &[], 41502), ["*:41502"]);
        assert!(parse_lsof_listeners(LSOF, &[], 41502).is_empty());
    }

    #[test]
    fn every_socket_the_app_owns_is_checked_not_only_the_assigned_one() {
        // The app answered health on the port it was given and also opened
        // 8080 on the wildcard. Asking about 41501 alone calls that clean.
        let only_the_port = parse_lsof_listeners(LSOF_SECOND_SOCKET, &[], 41501);
        assert_eq!(only_the_port, ["127.0.0.1:41501"]);
        check_listeners(&only_the_port, Some(91003), 41501).expect("the old question passes");

        // Asking about the pid finds both, and the routable one is the answer.
        let owned = parse_lsof_listeners(LSOF_SECOND_SOCKET, &[91003], 41501);
        assert_eq!(owned, ["127.0.0.1:41501", "0.0.0.0:8080"]);
        let err = check_listeners(&owned, Some(91003), 41501).expect_err("8080 is on the network");
        assert!(err.to_string().contains("0.0.0.0:8080"), "{err}");

        // Another process listening on the wildcard is not this app's problem.
        assert!(!owned.iter().any(|address| address.contains(":3000")));

        // Windows reports the same thing in its own shape.
        let owned = parse_netstat_listeners(NETSTAT_SECOND_SOCKET, &[9100], 41501);
        assert_eq!(owned, ["127.0.0.1:41501", "0.0.0.0:8080"]);
        assert!(check_listeners(&owned, Some(9100), 41501).is_err());

        // A grandchild listening on the assigned port is still found by port,
        // which is how a `bun run start` launcher is covered.
        let launched = parse_lsof_listeners(LSOF_WILDCARD, &[91003], 41502);
        assert_eq!(launched, ["*:41502"]);
        assert!(check_listeners(&launched, Some(91003), 41502).is_err());
    }

    #[tokio::test]
    async fn a_listener_tool_that_cannot_run_is_a_failure_not_a_pass() {
        // The check used to swallow this and return an empty list, which read
        // as "nothing routable" and let the app through.
        let err = tool_output("aias-no-such-listener-tool", &[])
            .await
            .expect_err("a missing tool must not pass the app");
        assert!(
            err.to_string().contains("aias-no-such-listener-tool"),
            "the error must name the tool: {err}"
        );
    }

    #[test]
    fn only_the_loopback_addresses_pass_the_bind_check() {
        assert!(is_loopback_address("127.0.0.1:41501"));
        assert!(is_loopback_address("[::1]:41501"));
        assert!(is_loopback_address("127.0.0.1"));
        assert!(!is_loopback_address("0.0.0.0:41501"));
        assert!(!is_loopback_address("[::]:41501"));
        assert!(!is_loopback_address("*:41501"));
        assert!(!is_loopback_address("192.168.1.20:41501"));
    }

    #[tokio::test]
    async fn an_app_bound_to_a_routable_address_is_refused_by_name() {
        // A real listener, so the check runs against the operating system and
        // not against a fixture: binding the wildcard is what an app must not do.
        let listener =
            TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).expect("wildcard bind");
        let port = listener.local_addr().unwrap().port();

        let err = verify_loopback(Some(std::process::id()), port)
            .await
            .expect_err("a wildcard bind must be refused");
        let message = err.to_string();
        assert!(message.contains(&port.to_string()), "{message}");
        assert!(
            message.contains("0.0.0.0") || message.contains("*:"),
            "the error must name the address it found: {message}"
        );

        drop(listener);
        // A loopback listener is what the platform asks for, so it passes.
        let listener =
            TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("loopback bind");
        let port = listener.local_addr().unwrap().port();
        verify_loopback(Some(std::process::id()), port)
            .await
            .expect("a loopback bind must pass");
    }

    #[tokio::test]
    async fn an_app_that_dies_gives_back_its_leases() {
        let _guard = instances::TEST_LOCK.lock().await;
        instances::reset_for_test(Some(crate::hardware::DeviceProfile {
            gpu_vendor: crate::hardware::Vendor::Nvidia,
            gpu_name: "test".into(),
            vram_mb: Some(16384),
            total_ram_mb: 65536,
            memory_model: crate::hardware::MemoryModel::Dedicated,
            has_npu: false,
        }));
        instances::set_spawn_hook(Some(instances::fake_spawn));

        let data = TempDir::new("crash");
        let model = ModelRef::new("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        instances::install_fixture(&data.0, &model, 8, None);

        // The lease an app takes at start, taken here directly so the test is
        // about what happens when the process dies and nothing else.
        instances::acquire_in("crasher", &model, &data.0)
            .await
            .unwrap();
        assert_eq!(instances::list()[0].leases, vec!["crasher"]);

        // A child that is already over by the time anyone looks.
        let child = Command::new("git")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("git runs");
        let id = NEXT_RUN_ID.fetch_add(1, Ordering::SeqCst);
        let process = AppProcess {
            name: "crasher".into(),
            port: 41999,
            url: "http://127.0.0.1:41999".into(),
            pid: Some(child.id()),
        };
        apps().push(RunningApp {
            id,
            process,
            child: Some(child),
            models: vec![model.clone()],
            dir: data.0.clone(),
            log_path: data.0.join("crasher.log"),
        });
        spawn_exit_waiter(id);

        // The waiter polls, so give it a few windows to notice. The app leaves
        // the registry before its leases are given back, so both are waited on.
        for _ in 0..20 {
            let gone = running().iter().all(|app| app.name != "crasher");
            if gone && instances::list()[0].leases.is_empty() {
                break;
            }
            tokio::time::sleep(EXIT_POLL).await;
        }

        assert!(
            running().iter().all(|app| app.name != "crasher"),
            "a dead app must leave the registry"
        );
        assert!(
            instances::list()[0].leases.is_empty(),
            "a dead app must not keep an instance leased: {:?}",
            instances::list()
        );

        instances::stop_all().await.unwrap();
        instances::set_spawn_hook(None);
    }

    #[tokio::test]
    async fn stopping_an_app_that_is_not_running_says_so() {
        let err = stop("never-started").await.unwrap_err();
        assert!(matches!(err, Error::NotFound(_)), "{err}");
        assert!(running().iter().all(|app| app.name != "never-started"));
    }
}
