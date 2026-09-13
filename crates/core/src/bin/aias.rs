//! Headless shell over `aias-core`.
//!
//! Every module is reachable without the GUI, so a machine reached over SSH can
//! verify the same code the desktop app runs. Output is JSON on stdout; errors
//! are JSON on stderr. Exit code 2 means the command exists but is not
//! implemented yet, 1 means it failed.

use std::path::PathBuf;
use std::time::Duration;

use aias_core::error::{Error, Result};
use aias_core::{api, apps, hardware, instances, models, paths, runtime, services};
use clap::{Parser, Subcommand};

/// Exit code for a scaffolded command with no implementation behind it.
const EXIT_NOT_IMPLEMENTED: i32 = 2;
/// Exit code for a real failure.
const EXIT_FAILED: i32 = 1;
/// How long a model may take to load before `runtime serve` gives up.
const HEALTH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

#[derive(Parser)]
#[command(name = "aias", version, about = "AI App Store, headless")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Device detection
    Hardware {
        #[command(subcommand)]
        command: HardwareCommand,
    },
    /// Backend selection and the llama-server install
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
    /// The GGUF catalogue
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Platform provided dependencies
    Services {
        #[command(subcommand)]
        command: ServicesCommand,
    },
    /// Manifests and the app lifecycle
    Apps {
        #[command(subcommand)]
        command: AppsCommand,
    },
    /// Running llama-server processes
    Instances {
        #[command(subcommand)]
        command: InstancesCommand,
    },
    /// The loopback API agents and the desktop shell share
    Api {
        #[command(subcommand)]
        command: ApiCommand,
    },
    /// The stdio MCP server the agent plugin starts, PLAN.md decision D11
    Mcp,
    /// Open the store index pull request for an app repo
    Publish {
        /// Directory of the app repo, committed and pushed
        dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum HardwareCommand {
    /// Print the device profile of this machine
    Detect,
}

#[derive(Subcommand)]
enum RuntimeCommand {
    /// Print the backend the section 2.1 table picks for this machine
    Select {
        /// Offer OpenVINO when the device has an NPU
        #[arg(long)]
        npu: bool,
    },
    /// Download and unpack the llama-server artifact
    Install {
        /// Override the selected backend
        #[arg(long, value_parser = ["cuda", "vulkan", "openvino", "cpu"])]
        backend: Option<String>,
    },
    /// Serve one GGUF from the installed runtime until Ctrl-C, for development
    Serve {
        /// Path of the GGUF file to load
        model: PathBuf,
        /// Path of the mmproj file, for a vision model
        #[arg(long)]
        mmproj: Option<PathBuf>,
        /// Loopback port to bind, inside the platform range
        #[arg(long, default_value_t = 41900)]
        port: u16,
        /// Value of `--alias`, the model name apps put in the request body
        #[arg(long)]
        alias: Option<String>,
    },
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// Search Hugging Face for GGUF repos
    Search {
        query: String,
        /// Narrow to one model kind
        #[arg(long, value_parser = ["llm", "vlm"])]
        kind: Option<String>,
        #[arg(long)]
        cursor: Option<String>,
    },
    /// List the GGUF files of one repo
    Files { repo: String },
    /// Download one quant of one repo
    Pull { repo: String, quant: String },
    /// List the models already downloaded
    List,
    /// Delete one downloaded model
    Rm { repo: String, quant: String },
}

#[derive(Subcommand)]
enum ServicesCommand {
    /// The one Postgres cluster
    Postgres {
        #[command(subcommand)]
        command: PostgresCommand,
    },
    /// The database and role of one app
    Db {
        #[command(subcommand)]
        command: DbCommand,
    },
}

#[derive(Subcommand)]
enum PostgresCommand {
    /// Initialize and start the cluster
    Ensure,
    /// Stop the cluster
    Stop,
}

#[derive(Subcommand)]
enum DbCommand {
    /// Create the database and role for an app and print its DATABASE_URL
    Provision { app: String },
    /// Apply *.sql from a folder in filename order
    Migrate { url: String, dir: PathBuf },
}

#[derive(Subcommand)]
enum AppsCommand {
    /// Check a checked out app against the PLAN section 3 rules
    Validate { dir: PathBuf },
    /// Clone or fast forward an app repo under the data directory
    Clone {
        url: String,
        /// Branch or tag to check out, the index `ref`
        #[arg(long = "ref")]
        git_ref: Option<String>,
    },
    /// List the apps published to the store index
    Index {
        /// Override AIAS_INDEX_URL and the default index
        #[arg(long)]
        url: Option<String>,
    },
    /// List the apps subscribed on this machine
    Installed,
    /// Clone and validate one app repo, printing what it would run
    Subscribe {
        url: String,
        /// Branch or tag to check out, the index `ref`
        #[arg(long = "ref")]
        git_ref: Option<String>,
        /// Also run the manifest `build` commands, after reading them yourself
        #[arg(long)]
        build: bool,
    },
    /// Delete one subscribed app's clone
    Rm { name: String },
    /// Print the tail of an app log
    Logs {
        name: String,
        /// Read the build log instead of the run log
        #[arg(long)]
        build: bool,
        /// Lines of the tail
        #[arg(long, default_value_t = 200)]
        lines: usize,
    },
    /// List the models an app declares that are not downloaded
    Missing { dir: PathBuf },
    /// Run the manifest build commands in an app directory
    Build { dir: PathBuf },
    /// Start an app and hold it until Ctrl-C
    Run {
        dir: PathBuf,
        /// Stop after this many seconds instead of waiting for Ctrl-C
        #[arg(long)]
        exit_after: Option<u64>,
    },
    /// Start several apps in one process, so they share one instance per model
    RunMany {
        #[arg(required = true, num_args = 1..)]
        dirs: Vec<PathBuf>,
        /// Stop after this many seconds instead of waiting for Ctrl-C
        #[arg(long)]
        exit_after: Option<u64>,
    },
}

#[derive(Subcommand)]
enum InstancesCommand {
    /// List the instances this process owns
    List,
    /// Take a lease on a model and hold it until Ctrl-C
    Acquire {
        repo: String,
        quant: String,
        /// Name of the app the lease is taken for
        #[arg(long)]
        app: String,
        /// Release and stop after this many seconds instead of waiting for Ctrl-C
        #[arg(long)]
        exit_after: Option<u64>,
    },
    /// Drop a lease this process holds
    Release {
        repo: String,
        quant: String,
        #[arg(long)]
        app: String,
    },
}

#[derive(Subcommand)]
enum ApiCommand {
    /// Serve the local API in the foreground until Ctrl-C
    Serve,
    /// Print the bearer token, creating it if this is the first call
    Token,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match dispatch(cli).await {
        // `aias mcp` owns stdout for the JSON-RPC transport, so it answers with
        // an empty string and nothing at all is written there.
        Ok(json) if json.is_empty() => {}
        Ok(json) => println!("{json}"),
        Err(err) => {
            let code = if err.is_not_implemented() {
                EXIT_NOT_IMPLEMENTED
            } else {
                EXIT_FAILED
            };
            eprintln!(
                "{}",
                serde_json::json!({ "error": { "code": err.code(), "message": err.to_string() } })
            );
            std::process::exit(code);
        }
    }
}

async fn dispatch(cli: Cli) -> Result<String> {
    let data_dir = paths::data_dir();

    match cli.command {
        Command::Hardware { command } => match command {
            HardwareCommand::Detect => print(&hardware::detect()?),
        },
        Command::Runtime { command } => match command {
            RuntimeCommand::Select { npu } => {
                let profile = hardware::detect()?;
                print(&runtime::select_with_npu(&profile, npu))
            }
            RuntimeCommand::Install { backend } => {
                let backend = match backend.as_deref() {
                    Some(name) => parse_backend(name)?,
                    None => runtime::select(&hardware::detect()?),
                };
                let install = runtime::install(backend, &data_dir).await?;
                print(&runtime::install_record(&install.dir)?)
            }
            RuntimeCommand::Serve {
                model,
                mmproj,
                port,
                alias,
            } => serve(&data_dir, &model, mmproj.as_deref(), port, alias.as_deref()).await,
        },
        Command::Models { command } => match command {
            ModelsCommand::Search {
                query,
                kind,
                cursor,
            } => {
                let kind = kind.as_deref().map(parse_kind).transpose()?;
                print(&models::search_kind(&query, kind, cursor.as_deref()).await?)
            }
            ModelsCommand::Files { repo } => print(&models::files(&repo).await?),
            ModelsCommand::Pull { repo, quant } => {
                let model = models::ModelRef::new(repo, quant);
                let path = models::download(&model, &data_dir, &report_progress).await?;
                print(&path)
            }
            ModelsCommand::List => {
                let installed: Vec<_> = models::installed(&data_dir)
                    .into_iter()
                    .map(|(model, path, size_bytes)| {
                        serde_json::json!({
                            "repo": model.repo,
                            "quant": model.quant,
                            "path": path,
                            "sizeBytes": size_bytes,
                        })
                    })
                    .collect();
                print(&installed)
            }
            ModelsCommand::Rm { repo, quant } => {
                let model = models::ModelRef::new(repo, quant);
                models::remove(&model, &data_dir)?;
                print(&serde_json::json!({ "removed": model.slug() }))
            }
        },
        Command::Services { command } => match command {
            ServicesCommand::Postgres { command } => match command {
                PostgresCommand::Ensure => print(&services::ensure_postgres(&data_dir).await?),
                PostgresCommand::Stop => {
                    let pg = services::postgres_at(&data_dir);
                    services::stop_postgres(&pg).await?;
                    print(&pg)
                }
            },
            ServicesCommand::Db { command } => match command {
                DbCommand::Provision { app } => {
                    let pg = services::postgres_at(&data_dir);
                    print(&services::provision_app_db(&pg, &app).await?)
                }
                DbCommand::Migrate { url, dir } => {
                    print(&services::apply_migrations(&url, &dir).await?)
                }
            },
        },
        Command::Apps { command } => match command {
            AppsCommand::Validate { dir } => print(&apps::validate_dir(&dir)?),
            AppsCommand::Clone { url, git_ref } => {
                print(&apps::clone_app(&url, git_ref.as_deref(), &data_dir).await?)
            }
            AppsCommand::Build { dir } => {
                let manifest = apps::validate_dir(&dir)?;
                apps::build(&dir, &manifest).await?;
                print(&serde_json::json!({ "built": manifest.name }))
            }
            AppsCommand::Index { url } => {
                let url = url.unwrap_or_else(apps::index_url);
                print(&apps::index(&url).await?)
            }
            AppsCommand::Installed => print(&apps::installed(&data_dir)),
            AppsCommand::Subscribe {
                url,
                git_ref,
                build,
            } => {
                // Cloning is safe, building is not: `build` runs argv from the
                // repo. So the manifest is printed and the build only happens
                // when the caller asked for it after reading that.
                let app = apps::subscribe(&url, git_ref.as_deref(), &data_dir).await?;
                if build {
                    apps::build(&app.dir, &app.manifest).await?;
                } else {
                    eprintln!(
                        "cloned but not built: read `build` and `start` above, then rerun with --build"
                    );
                }
                print(&app)
            }
            AppsCommand::Rm { name } => {
                apps::remove(&name, &data_dir)?;
                print(&serde_json::json!({ "removed": name }))
            }
            AppsCommand::Logs { name, build, lines } => {
                let kind = if build {
                    apps::LogKind::Build
                } else {
                    apps::LogKind::Run
                };
                // The tail is text, so it goes out as text and not as a JSON
                // string with every newline escaped.
                Ok(apps::logs(&name, kind, lines)?)
            }
            AppsCommand::Missing { dir } => {
                let manifest = apps::validate_dir(&dir)?;
                print(&apps::missing_models(&manifest, &data_dir))
            }
            AppsCommand::Run { dir, exit_after } => run_apps(&[dir], exit_after).await,
            AppsCommand::RunMany { dirs, exit_after } => run_apps(&dirs, exit_after).await,
        },
        Command::Instances { command } => match command {
            InstancesCommand::List => print(&instances::list()),
            InstancesCommand::Acquire {
                repo,
                quant,
                app,
                exit_after,
            } => {
                let model = models::ModelRef::new(repo, quant);
                let url = instances::acquire(&app, &model).await?;
                emit(&url)?;
                emit(&instances::list())?;
                // The registry is process local, so holding the lease here is
                // what keeps the instance alive and reachable.
                wait_for_exit(exit_after).await;
                instances::release(&app, &model).await?;
                instances::stop_all().await?;
                print(&serde_json::json!({ "released": app, "instances": instances::list() }))
            }
            InstancesCommand::Release { repo, quant, app } => {
                let model = models::ModelRef::new(repo, quant);
                instances::release(&app, &model).await?;
                print(&instances::list())
            }
        },
        Command::Api { command } => match command {
            ApiCommand::Serve => serve_api(&data_dir).await,
            ApiCommand::Token => Ok(api::token(&data_dir)?),
        },
        // The only command that prints nothing: stdout is the MCP transport,
        // so anything written there that is not a JSON-RPC frame corrupts the
        // session for the client.
        Command::Mcp => {
            aias_core::mcp::serve_stdio(data_dir).await?;
            Ok(String::new())
        }
        Command::Publish { dir } => print(&aias_core::publish::publish(&dir).await?),
    }
}

/// Start every app in one process, hold them, then stop and release.
///
/// One process is one registry: apps started together share an instance per
/// model. Two `aias apps run` processes each have their own registry, which is
/// why the desktop app is the single owner on a real machine.
async fn run_apps(dirs: &[PathBuf], exit_after: Option<u64>) -> Result<String> {
    let mut started: Vec<apps::AppProcess> = Vec::new();

    for dir in dirs {
        let manifest = apps::validate_dir(dir)?;
        match apps::start(dir, &manifest).await {
            Ok(process) => {
                emit(&process)?;
                started.push(process);
            }
            Err(err) => {
                shutdown(&started).await;
                return Err(err);
            }
        }
    }
    emit(&instances::list())?;

    wait_for_exit(exit_after).await;
    shutdown(&started).await;
    print(&serde_json::json!({
        "stopped": started.iter().map(|app| app.name.clone()).collect::<Vec<String>>(),
        "instances": instances::list(),
    }))
}

async fn shutdown(started: &[apps::AppProcess]) {
    for app in started {
        if let Err(err) = apps::stop(&app.name).await {
            eprintln!("stopping {}: {err}", app.name);
        }
    }
    if let Err(err) = instances::stop_all().await {
        eprintln!("stopping instances: {err}");
    }
}

/// Block until Ctrl-C, or until the deadline when one was given.
async fn wait_for_exit(exit_after: Option<u64>) {
    match exit_after {
        Some(seconds) => {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(seconds)) => {}
                _ = stop_signal() => {}
            }
        }
        None => stop_signal().await,
    }
}

/// Block until the user or the system asks this process to stop.
///
/// Ctrl-C alone is not enough. A terminal sends SIGINT to the whole process
/// group, but `kill` and every service manager send SIGTERM to this process
/// only, and the default action kills it before the shutdown below runs: the
/// `llama-server` and the Postgres cluster it started would be left listening
/// with no owner. Windows has no SIGTERM, so there Ctrl-C is the whole story.
async fn stop_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            // A handler this process is not allowed to install is not a reason
            // to exit: wait for Ctrl-C the way the rest of the CLI does.
            Err(err) => {
                eprintln!("could not listen for SIGTERM, waiting for Ctrl-C: {err}");
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Print one JSON value while a long running command is still going.
fn emit<T: serde::Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// Development aid: start one llama-server by hand and hold it in the
/// foreground, so a machine reached over SSH can answer a real completion.
async fn serve(
    data_dir: &std::path::Path,
    model: &std::path::Path,
    mmproj: Option<&std::path::Path>,
    port: u16,
    alias: Option<&str>,
) -> Result<String> {
    let install = runtime::installed(data_dir).ok_or_else(|| {
        Error::NotFound(format!(
            "no runtime installed under {}, run `aias runtime install`",
            data_dir.display()
        ))
    })?;

    let profile = hardware::detect()?;
    let size_bytes = std::fs::metadata(model)?.len();
    // The file is right there, so the context is planned against its real cache.
    let params = instances::params_for(&profile, size_bytes, models::kv_layout(model));
    let alias = alias
        .map(str::to_string)
        .or_else(|| {
            model
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "model".into());

    // The same rule as a real Instance: the port is reachable by every process
    // on the machine, so the server asks for a bearer token.
    let api_key = instances::new_api_key();
    let mut child =
        runtime::spawn_server(&install, model, mmproj, port, &params, &alias, &api_key).await?;
    let health = runtime::wait_healthy(port, HEALTH_TIMEOUT).await;
    if let Err(err) = health {
        child.kill().await.ok();
        return Err(err);
    }

    let base_url = format!("http://127.0.0.1:{port}/v1");
    println!(
        "{}",
        serde_json::json!({
            "baseUrl": base_url,
            "modelId": alias,
            "port": port,
            "apiKey": api_key,
            "pid": child.id(),
            "params": params,
        })
    );
    eprintln!("serving until Ctrl-C");

    stop_signal().await;
    child.kill().await.ok();
    print(&serde_json::json!({ "stopped": true, "port": port }))
}

/// One line per callback, on stderr, so stdout stays parseable JSON.
fn report_progress(progress: models::Progress) {
    let done = progress.downloaded_bytes;
    match progress.total_bytes {
        Some(total) if total > 0 => {
            eprintln!("{done} / {total} bytes ({}%)", done * 100 / total)
        }
        _ => eprintln!("{done} bytes"),
    }
}

fn parse_kind(name: &str) -> Result<apps::ModelKind> {
    match name {
        "llm" => Ok(apps::ModelKind::Llm),
        "vlm" => Ok(apps::ModelKind::Vlm),
        other => Err(Error::NotFound(format!("model kind `{other}`"))),
    }
}

fn parse_backend(name: &str) -> Result<runtime::Backend> {
    match name {
        "cuda" => Ok(runtime::Backend::Cuda),
        "vulkan" => Ok(runtime::Backend::Vulkan),
        "openvino" => Ok(runtime::Backend::OpenVino),
        "cpu" => Ok(runtime::Backend::Cpu),
        other => Err(Error::UnsupportedHardware(format!(
            "unknown backend `{other}`"
        ))),
    }
}

fn print<T: serde::Serialize>(value: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(value)?)
}

/// Host the local API without the GUI, which is how a headless machine and CI
/// run it (PLAN.md section 6.1).
///
/// This process is the owner of everything it starts while it serves, so Ctrl-C
/// stops the apps, then the instances their leases kept alive, then the one
/// Postgres cluster. Leaving any of them behind would leave a model server with
/// no owner on a machine whose only owner just exited.
async fn serve_api(data_dir: &std::path::Path) -> Result<String> {
    // Generated on the first serve, so the URL printed here is usable at once.
    api::token(data_dir)?;
    eprintln!(
        "serving {} until Ctrl-C, token in {}",
        api::base_url(),
        api::token_path(data_dir).display()
    );

    // No window in this host, so no emitter: the `aias://changed` event of
    // issue #16 is the desktop shell's business.
    api::serve(data_dir, stop_signal(), None).await?;

    let started: Vec<apps::AppProcess> = apps::running();
    shutdown(&started).await;
    let postgres = services::postgres_at(data_dir);
    if let Err(err) = services::stop_postgres(&postgres).await {
        eprintln!("stopping postgres: {err}");
    }
    print(&serde_json::json!({
        "stopped": started.iter().map(|app| app.name.clone()).collect::<Vec<String>>(),
        "url": api::base_url(),
    }))
}
