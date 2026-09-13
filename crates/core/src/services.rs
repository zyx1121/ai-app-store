//! Platform provided dependencies. Version 1 has exactly one: Postgres.
//!
//! One native cluster for the whole machine, one database and one role per app
//! (decision D4). The app only ever sees its own `DATABASE_URL`.

use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use rand::RngExt as _;
use rand::distr::Alphanumeric;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use tokio::process::Command;
use tokio_postgres::{Client, NoTls};

use crate::error::{Error, Result};
use crate::paths;
use crate::process::{POSTGRES_ALLOWLIST, Quiet as _, Sealed as _};

/// Loopback port of the one cluster. Fixed, so a `DATABASE_URL` written into an
/// app log stays meaningful across restarts.
pub const POSTGRES_PORT: u16 = 41500;

/// Superuser role `initdb` creates. The platform holds it, apps never see it.
pub const SUPERUSER: &str = "aias";

/// Table the applied migrations are recorded in, inside the app's own database.
pub const MIGRATIONS_TABLE: &str = "_aias_migrations";

/// Windows x64 binaries. There is no upstream SHA-256 next to this file, so the
/// hash of what was downloaded is recorded in a sidecar instead.
const BINARIES_URL: &str =
    "https://get.enterprisedb.com/postgresql/postgresql-17.6-1-windows-x64-binaries.zip";

/// Points at an existing Postgres 17 `bin` directory and skips the download.
/// Used on developer machines where the binaries come from a package manager.
const BIN_DIR_ENV: &str = "AIAS_PG_BIN";

/// Length of every password the platform generates. Alphanumeric only, so it
/// needs no escaping inside a DSN.
const PASSWORD_LEN: usize = 32;

/// Longest role and database name the platform accepts, `app_` prefix included.
const MAX_IDENT_LEN: usize = 51;

/// The running Postgres cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Postgres {
    /// Cluster directory, binaries and data.
    pub dir: PathBuf,
    /// Loopback port the cluster listens on.
    pub port: u16,
}

/// Where the cluster lives under a data directory, running or not.
///
/// Handed to [`stop_postgres`] and [`provision_app_db`] without going through
/// [`ensure_postgres`] first.
pub fn postgres_at(data_dir: &Path) -> Postgres {
    Postgres {
        dir: paths::postgres_dir(data_dir),
        port: POSTGRES_PORT,
    }
}

/// Start the cluster, initializing it on first call.
///
/// Installs the binaries when they are missing, runs `initdb` with a generated
/// superuser password, starts the server bound to `127.0.0.1` and takes
/// `CONNECT` on the maintenance databases away from `PUBLIC`.
pub async fn ensure_postgres(data_dir: &Path) -> Result<Postgres> {
    paths::ensure_data_dirs(data_dir)?;
    let pg = postgres_at(data_dir);

    ensure_binaries(&pg.dir).await?;
    ensure_cluster(&pg.dir).await?;
    if !is_ready(&pg.dir).await {
        start_cluster(&pg.dir, &paths::logs_dir(data_dir)).await?;
    }
    harden_cluster(&pg).await?;

    Ok(pg)
}

/// Stop the cluster with `pg_ctl stop -m fast`. A cluster already down is fine.
pub async fn stop_postgres(pg: &Postgres) -> Result<()> {
    if !is_ready(&pg.dir).await {
        return Ok(());
    }
    let data = cluster_dir(&pg.dir);
    run(
        &tool(&pg.dir, "pg_ctl"),
        &[
            OsStr::new("-D"),
            data.as_os_str(),
            OsStr::new("stop"),
            OsStr::new("-m"),
            OsStr::new("fast"),
        ],
        "pg_ctl stop",
    )
    .await?;
    Ok(())
}

/// DSN of the superuser against the maintenance database.
///
/// Never leaves the platform. An unreadable password file yields a DSN with an
/// empty password, which fails at connect time rather than silently succeeding.
pub fn superuser_url(pg: &Postgres) -> String {
    let password = read_secret(&superuser_pw_path(&pg.dir)).unwrap_or_default();
    format!(
        "postgres://{SUPERUSER}:{password}@127.0.0.1:{}/postgres",
        pg.port
    )
}

/// The role and database name for an app, `app_` plus the manifest name.
///
/// Hyphens become underscores, and the result must match
/// `^[a-z][a-z0-9_]{0,50}$`. Everything that reaches SQL as an identifier comes
/// from here.
pub fn db_name(app_name: &str) -> Result<String> {
    let name = format!("app_{}", app_name.replace('-', "_"));
    // An empty app name would leave the bare prefix, which passes the shape.
    if app_name.is_empty() || !is_identifier(&name) {
        return Err(Error::InvalidManifest(format!(
            "app name `{app_name}` is not usable as a Postgres identifier"
        )));
    }
    Ok(name)
}

/// Create the database and the owning role for one app, and return its DSN.
///
/// The role owns its database and nothing else, and `PUBLIC` keeps no privilege
/// on it. The generated password is kept on disk and reused, so the DSN an app
/// is started with is stable across calls.
pub async fn provision_app_db(pg: &Postgres, app_name: &str) -> Result<String> {
    let name = db_name(app_name)?;
    let (password, generated) = load_or_create_secret(&app_pw_path(&pg.dir, app_name))?;
    let client = connect(&superuser_url(pg), "provision").await?;

    let role_exists = client
        .query_opt("SELECT 1 FROM pg_roles WHERE rolname = $1", &[&name])
        .await
        .map_err(|err| db_error("look up the role", err))?
        .is_some();
    let ident = quote_ident(&name);
    if !role_exists {
        client
            .batch_execute(&format!(
                "CREATE ROLE {ident} LOGIN PASSWORD {}",
                quote_literal(&password)
            ))
            .await
            .map_err(|err| db_error("create the role", err))?;
    } else if generated {
        // The password file was gone, so the stored one is now the only truth.
        client
            .batch_execute(&format!(
                "ALTER ROLE {ident} WITH LOGIN PASSWORD {}",
                quote_literal(&password)
            ))
            .await
            .map_err(|err| db_error("reset the role password", err))?;
    }

    let db_exists = client
        .query_opt("SELECT 1 FROM pg_database WHERE datname = $1", &[&name])
        .await
        .map_err(|err| db_error("look up the database", err))?
        .is_some();
    if !db_exists {
        client
            .batch_execute(&format!("CREATE DATABASE {ident} OWNER {ident}"))
            .await
            .map_err(|err| db_error("create the database", err))?;
    }
    client
        .batch_execute(&format!("REVOKE ALL ON DATABASE {ident} FROM PUBLIC"))
        .await
        .map_err(|err| db_error("revoke PUBLIC on the database", err))?;

    Ok(app_url(pg, &name, &password))
}

/// Drop one app's database and role, and forget its password.
pub async fn drop_app_db(pg: &Postgres, app_name: &str) -> Result<()> {
    let name = db_name(app_name)?;
    let ident = quote_ident(&name);
    let client = connect(&superuser_url(pg), "drop").await?;

    client
        .batch_execute(&format!("DROP DATABASE IF EXISTS {ident} WITH (FORCE)"))
        .await
        .map_err(|err| db_error("drop the database", err))?;
    client
        .batch_execute(&format!("DROP ROLE IF EXISTS {ident}"))
        .await
        .map_err(|err| db_error("drop the role", err))?;

    let pw_path = app_pw_path(&pg.dir, app_name);
    if pw_path.exists() {
        std::fs::remove_file(&pw_path)?;
    }
    Ok(())
}

/// Apply `*.sql` from `dir` in filename order as the app's own role.
///
/// Each file runs in one transaction and is recorded, so a second call applies
/// nothing. Returns the number of files applied.
pub async fn apply_migrations(database_url: &str, dir: &Path) -> Result<usize> {
    let files = sql_files(dir)?;
    let mut client = connect(database_url, "migrations").await?;

    client
        .batch_execute(&format!(
            "CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                 filename text PRIMARY KEY,
                 applied_at timestamptz NOT NULL DEFAULT now()
             )"
        ))
        .await
        .map_err(|err| db_error("create the migrations table", err))?;

    let applied: BTreeSet<String> = client
        .query(&format!("SELECT filename FROM {MIGRATIONS_TABLE}"), &[])
        .await
        .map_err(|err| db_error("read the migrations table", err))?
        .iter()
        .map(|row| row.get(0))
        .collect();

    let mut count = 0;
    for (filename, path) in files {
        if applied.contains(&filename) {
            continue;
        }
        let sql = std::fs::read_to_string(&path)
            .map_err(|err| Error::Process(format!("migration {filename}: {err}")))?;

        let tx = client
            .transaction()
            .await
            .map_err(|err| migration_error(&filename, err))?;
        tx.batch_execute(&sql)
            .await
            .map_err(|err| migration_error(&filename, err))?;
        tx.execute(
            &format!("INSERT INTO {MIGRATIONS_TABLE} (filename) VALUES ($1)"),
            &[&filename],
        )
        .await
        .map_err(|err| migration_error(&filename, err))?;
        tx.commit()
            .await
            .map_err(|err| migration_error(&filename, err))?;

        count += 1;
    }
    Ok(count)
}

// -- binaries ---------------------------------------------------------------

/// Directory holding `initdb`, `pg_ctl` and `pg_isready`.
fn bin_dir(dir: &Path) -> PathBuf {
    match std::env::var_os(BIN_DIR_ENV) {
        Some(over) => PathBuf::from(over),
        None => dir.join("pgsql").join("bin"),
    }
}

/// One Postgres executable, named for this platform.
fn tool(dir: &Path, stem: &str) -> PathBuf {
    bin_dir(dir).join(paths::exe_name(stem))
}

/// The cluster's data directory, `initdb -D`.
fn cluster_dir(dir: &Path) -> PathBuf {
    dir.join("data")
}

async fn ensure_binaries(dir: &Path) -> Result<()> {
    if tool(dir, "initdb").is_file() {
        return Ok(());
    }
    if std::env::var_os(BIN_DIR_ENV).is_some() {
        return Err(Error::NotFound(format!(
            "{BIN_DIR_ENV} holds no `initdb`: {}",
            bin_dir(dir).display()
        )));
    }
    // Windows is the only target with a published archive, so anywhere else the
    // binaries have to be pointed at rather than downloaded.
    if std::env::consts::OS != "windows" {
        return Err(Error::NotFound(format!(
            "no Postgres binaries: set {BIN_DIR_ENV} to a Postgres 17 `bin` directory, \
             the platform only downloads the Windows build"
        )));
    }

    let archive = dir.join("pg.zip");
    if !archive.is_file() {
        download(BINARIES_URL, &archive).await?;
    }
    // No upstream checksum is published next to the file, so record what landed.
    record_sha256(&archive).await?;
    // The archive carries a single `pgsql/` folder, which lands next to it.
    unpack(&archive, dir).await?;

    if !tool(dir, "initdb").is_file() {
        return Err(Error::NotFound(format!(
            "no `initdb` after unpacking {}",
            archive.display()
        )));
    }
    Ok(())
}

async fn download(url: &str, path: &Path) -> Result<()> {
    use tokio::io::AsyncWriteExt as _;

    let mut response = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|err| Error::Process(format!("download {url}: {err}")))?;

    let partial = path.with_extension("part");
    let mut file = tokio::fs::File::create(&partial).await?;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| Error::Process(format!("download {url}: {err}")))?
    {
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&partial, path).await?;
    Ok(())
}

/// Write `<file>.sha256` next to a download, or check it when it already exists.
async fn record_sha256(path: &Path) -> Result<()> {
    let path = path.to_path_buf();
    let sidecar = sidecar_path(&path);
    tokio::task::spawn_blocking(move || {
        let digest = sha256_of(&path)?;

        match std::fs::read_to_string(&sidecar) {
            Ok(recorded) if recorded.trim() != digest => Err(Error::Process(format!(
                "{} does not match its recorded SHA-256",
                path.display()
            ))),
            Ok(_) => Ok(()),
            Err(_) => {
                std::fs::write(&sidecar, format!("{digest}\n"))?;
                Ok(())
            }
        }
    })
    .await
    .map_err(|err| Error::Process(format!("hashing failed: {err}")))?
}

/// SHA-256 of a file, streamed so a 330 MB archive never lands in memory.
fn sha256_of(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".sha256");
    PathBuf::from(name)
}

async fn unpack(archive: &Path, into: &Path) -> Result<()> {
    let archive = archive.to_path_buf();
    let into = into.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(&archive)?;
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|err| Error::Process(format!("open {}: {err}", archive.display())))?;
        zip.extract(&into)
            .map_err(|err| Error::Process(format!("unpack {}: {err}", archive.display())))
    })
    .await
    .map_err(|err| Error::Process(format!("unpacking failed: {err}")))?
}

// -- cluster ----------------------------------------------------------------

async fn ensure_cluster(dir: &Path) -> Result<()> {
    let data = cluster_dir(dir);
    if data.join("PG_VERSION").is_file() {
        return Ok(());
    }

    let (password, _) = load_or_create_secret(&superuser_pw_path(dir))?;
    // initdb reads the password from a file so it never reaches an argv element
    // or the process list.
    let pwfile = dir.join("initdb.pw");
    write_secret(&pwfile, &password)?;

    let result = run(
        &tool(dir, "initdb"),
        &[
            OsStr::new("-D"),
            data.as_os_str(),
            OsStr::new("-U"),
            OsStr::new(SUPERUSER),
            OsStr::new("--pwfile"),
            pwfile.as_os_str(),
            OsStr::new("-A"),
            OsStr::new("scram-sha-256"),
            OsStr::new("-E"),
            OsStr::new("UTF8"),
            OsStr::new("--locale=C"),
        ],
        "initdb",
    )
    .await;
    if pwfile.exists() {
        std::fs::remove_file(&pwfile)?;
    }
    result.map(|_| ())
}

/// The postmaster options `pg_ctl -o` is given.
///
/// A packaged postgres puts its unix socket in a directory the package owns,
/// `/var/run/postgresql` on Debian, which this user cannot write and which has
/// nothing to do with a cluster the platform started for itself. So the socket
/// goes next to the cluster's own data. Windows has no unix socket, and the
/// option is a Unix one, so the line it is left out of is the shipping one.
fn server_options(data: &Path) -> String {
    let common = format!("-p {POSTGRES_PORT} -h 127.0.0.1");
    if cfg!(unix) {
        // pg_ctl hands this string to a shell on Unix, so the path is quoted.
        format!("{common} -k '{}'", data.display())
    } else {
        common
    }
}

async fn start_cluster(dir: &Path, logs_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(logs_dir)?;
    let log = logs_dir.join("postgres.log");
    // pg_ctl opens the server log itself and will not share it, so its own
    // chatter goes to a second file.
    let start_log = logs_dir.join("postgres-start.log");
    let data = cluster_dir(dir);
    // pg_ctl takes the server options as one argument and splits them itself.
    let options: OsString = server_options(&data).into();

    run_into_log(
        &tool(dir, "pg_ctl"),
        &[
            OsStr::new("-D"),
            data.as_os_str(),
            OsStr::new("-o"),
            &options,
            OsStr::new("-l"),
            log.as_os_str(),
            OsStr::new("-w"),
            OsStr::new("start"),
        ],
        &start_log,
        &format!("pg_ctl start (server log {})", log.display()),
    )
    .await?;

    for _ in 0..30 {
        if is_ready(dir).await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    Err(Error::Process(format!(
        "postgres did not answer on 127.0.0.1:{POSTGRES_PORT}, see {}",
        log.display()
    )))
}

async fn is_ready(dir: &Path) -> bool {
    let port = POSTGRES_PORT.to_string();
    Command::new(tool(dir, "pg_isready"))
        .quiet()
        .sealed_env(POSTGRES_ALLOWLIST)
        .arg("-h")
        .arg("127.0.0.1")
        .arg("-p")
        .arg(&port)
        .output()
        .await
        .is_ok_and(|output| output.status.success())
}

/// Take `CONNECT` on the maintenance databases away from `PUBLIC`.
///
/// Without this an app role could open `postgres` or `template1` and read the
/// catalogue of every other app. Database level isolation is the whole of
/// decision D4, so it is applied on every start.
async fn harden_cluster(pg: &Postgres) -> Result<()> {
    let client = connect(&superuser_url(pg), "harden").await?;
    client
        .batch_execute(
            "REVOKE ALL ON DATABASE postgres FROM PUBLIC;
             REVOKE ALL ON DATABASE template1 FROM PUBLIC;",
        )
        .await
        .map_err(|err| db_error("revoke PUBLIC on the maintenance databases", err))
}

/// Run one executable that leaves a server behind, sending its output to `log`.
///
/// The started server inherits the handles, so a pipe here would only reach end
/// of file when the server itself exits. On Windows that means `pg_ctl start`
/// never returns. A file handle has no such wait.
///
/// Windows hands the server every inheritable handle the caller holds, this
/// crate cannot narrow that from `std`, so a caller that pipes its own stdout
/// somewhere and waits for end of file waits for the server to stop.
async fn run_into_log(program: &Path, args: &[&OsStr], log: &Path, what: &str) -> Result<()> {
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    let err = out.try_clone()?;

    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .quiet()
        .sealed_env(POSTGRES_ALLOWLIST)
        .status()
        .await
        .map_err(|err| {
            Error::Process(format!(
                "{what}: {} did not start: {err}",
                program.display()
            ))
        })?;

    if !status.success() {
        return Err(Error::Process(format!(
            "{what}: {status}, see {}",
            log.display()
        )));
    }
    Ok(())
}

/// Run one platform owned executable, every value its own argv element.
///
/// The environment is [`crate::process::POSTGRES_ALLOWLIST`], which holds no
/// `PG*` variable: every setting these tools take is passed on argv here, and
/// an exported `PGDATA` or `PGPORT` would silently override it.
async fn run(program: &Path, args: &[&OsStr], what: &str) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .quiet()
        .sealed_env(POSTGRES_ALLOWLIST)
        .output()
        .await
        .map_err(|err| {
            Error::Process(format!(
                "{what}: {} did not start: {err}",
                program.display()
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = if stderr.trim().is_empty() {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        } else {
            stderr.trim().to_string()
        };
        return Err(Error::Process(format!(
            "{what}: {}: {detail}",
            output.status
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// -- sql --------------------------------------------------------------------

async fn connect(url: &str, what: &str) -> Result<Client> {
    let (client, connection) = tokio_postgres::connect(url, NoTls)
        .await
        .map_err(|err| Error::Process(format!("{what}: connect failed: {err}")))?;
    // The connection future drives the socket and ends when the client drops.
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

fn db_error(what: &str, err: tokio_postgres::Error) -> Error {
    Error::Process(format!("could not {what}: {}", describe(err)))
}

fn migration_error(filename: &str, err: tokio_postgres::Error) -> Error {
    Error::Process(format!("migration {filename}: {}", describe(err)))
}

/// `tokio_postgres::Error` prints as `db error`; the server message the caller
/// needs is one level down in the source chain.
fn describe(err: tokio_postgres::Error) -> String {
    use std::error::Error as _;

    match err.source() {
        Some(source) => format!("{err}: {source}"),
        None => err.to_string(),
    }
}

/// `*.sql` in one folder, sorted by filename, which is the apply order.
fn sql_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    if !dir.is_dir() {
        return Err(Error::NotFound(dir.display().to_string()));
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let is_sql = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("sql"));
        if !path.is_file() || !is_sql {
            continue;
        }
        let filename = match path.file_name() {
            Some(name) => name.to_string_lossy().into_owned(),
            None => continue,
        };
        files.push((filename, path));
    }
    files.sort();
    Ok(files)
}

/// Quote an SQL identifier. Doubling `"` is what makes it safe to interpolate.
fn quote_ident(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Quote an SQL string literal. Doubling `'` is what makes it safe.
fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// `^[a-z][a-z0-9_]{0,50}$`, written out the same way `apps::is_alias` is.
fn is_identifier(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_IDENT_LEN {
        return false;
    }
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {
            chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        }
        _ => false,
    }
}

/// The `DATABASE_URL` an app is started with. The password is alphanumeric, so
/// it needs no percent encoding.
fn app_url(pg: &Postgres, name: &str, password: &str) -> String {
    format!("postgres://{name}:{password}@127.0.0.1:{}/{name}", pg.port)
}

// -- secrets ----------------------------------------------------------------

fn superuser_pw_path(dir: &Path) -> PathBuf {
    dir.join("superuser.pw")
}

/// One file per app. `app_name` already passed [`db_name`], so it cannot escape
/// this folder.
fn app_pw_path(dir: &Path, app_name: &str) -> PathBuf {
    dir.join("apps").join(format!("{app_name}.pw"))
}

/// Read the password at `path`, generating and storing one when it is missing.
///
/// The second field is true when the password was just generated, which is the
/// only case where an existing role has to be updated to match it.
fn load_or_create_secret(path: &Path) -> Result<(String, bool)> {
    if let Some(secret) = read_secret(path) {
        crate::instances::remember_secret(&secret);
        return Ok((secret, false));
    }
    let secret: String = rand::rng()
        .sample_iter(Alphanumeric)
        .take(PASSWORD_LEN)
        .map(char::from)
        .collect();
    write_secret(path, &secret)?;
    crate::instances::remember_secret(&secret);
    Ok((secret, true))
}

fn read_secret(path: &Path) -> Option<String> {
    let secret = std::fs::read_to_string(path).ok()?;
    let secret = secret.trim().to_string();
    (!secret.is_empty()).then_some(secret)
}

fn write_secret(path: &Path, secret: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    let mut file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?
    };

    // On Windows the file inherits the user profile ACL, which keeps it out of
    // reach of other users. PLAN section 7 moves this password into the
    // Credential Manager; until that lands the file is the only copy.
    #[cfg(not(unix))]
    let mut file = std::fs::File::create(path)?;

    file.write_all(secret.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_app_name_becomes_a_prefixed_identifier() {
        assert_eq!(db_name("example-chat").unwrap(), "app_example_chat");
        assert_eq!(db_name("chat").unwrap(), "app_chat");
        assert_eq!(db_name("a1-b2-c3").unwrap(), "app_a1_b2_c3");
    }

    #[test]
    fn anything_that_is_not_a_plain_name_is_rejected() {
        for name in [
            "",
            "Example",
            "app; DROP DATABASE postgres",
            "we\"ird",
            "../escape",
            "app'name",
            "ünicode",
            &"x".repeat(MAX_IDENT_LEN),
        ] {
            assert!(db_name(name).is_err(), "accepted `{name}`");
        }
    }

    #[test]
    fn the_identifier_shape_is_the_documented_one() {
        assert!(is_identifier("a"));
        assert!(is_identifier("app_example_chat"));
        assert!(!is_identifier("1app"));
        assert!(!is_identifier("app-example"));
        assert!(!is_identifier("APP"));
        assert!(is_identifier(&"a".repeat(MAX_IDENT_LEN)));
        assert!(!is_identifier(&"a".repeat(MAX_IDENT_LEN + 1)));
    }

    #[test]
    fn identifiers_are_quoted_and_embedded_quotes_doubled() {
        assert_eq!(quote_ident("app_chat"), "\"app_chat\"");
        assert_eq!(quote_ident("we\"ird"), "\"we\"\"ird\"");
        assert_eq!(quote_ident("a\"; DROP"), "\"a\"\"; DROP\"");
    }

    #[test]
    fn literals_are_quoted_and_embedded_quotes_doubled() {
        assert_eq!(quote_literal("secret"), "'secret'");
        assert_eq!(quote_literal("it's"), "'it''s'");
    }

    #[test]
    fn the_server_listens_on_loopback_and_keeps_its_socket_to_itself() {
        let options = server_options(Path::new("/data/aias/postgres/data"));
        assert!(options.contains("-p 41500"), "{options}");
        assert!(options.contains("-h 127.0.0.1"), "{options}");
        #[cfg(unix)]
        assert!(
            options.contains("-k '/data/aias/postgres/data'"),
            "the socket belongs to the cluster, not to a packaged directory: {options}"
        );
        #[cfg(not(unix))]
        assert!(
            !options.contains("-k"),
            "windows has no unix socket: {options}"
        );
    }

    #[test]
    fn urls_point_at_loopback_on_the_fixed_port() {
        let pg = Postgres {
            dir: PathBuf::from("/data/aias/postgres"),
            port: POSTGRES_PORT,
        };
        assert_eq!(
            app_url(&pg, "app_chat", "pw"),
            "postgres://app_chat:pw@127.0.0.1:41500/app_chat"
        );
        // No password file under that directory, so only the shape is asserted.
        let url = superuser_url(&pg);
        assert!(url.starts_with("postgres://aias:"), "{url}");
        assert!(url.ends_with("@127.0.0.1:41500/postgres"), "{url}");
    }

    #[test]
    fn the_cluster_hangs_off_the_data_dir() {
        let pg = postgres_at(Path::new("/data/aias"));
        assert_eq!(pg.dir, PathBuf::from("/data/aias/postgres"));
        assert_eq!(pg.port, POSTGRES_PORT);
        assert_eq!(
            cluster_dir(&pg.dir),
            PathBuf::from("/data/aias/postgres/data")
        );
    }

    #[test]
    fn a_password_file_is_generated_once_and_then_reused() {
        let dir = std::env::temp_dir().join(format!("aias-services-{}", std::process::id()));
        let path = dir.join("apps").join("example-chat.pw");
        let _ = std::fs::remove_file(&path);

        let (first, generated) = load_or_create_secret(&path).unwrap();
        assert!(generated);
        assert_eq!(first.len(), PASSWORD_LEN);
        assert!(first.chars().all(|c| c.is_ascii_alphanumeric()), "{first}");

        let (second, generated) = load_or_create_secret(&path).unwrap();
        assert!(!generated);
        assert_eq!(first, second);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sql_files_come_back_in_filename_order() {
        let dir = std::env::temp_dir().join(format!("aias-migrations-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["0002_b.sql", "0001_a.sql", "notes.md", "0010_c.SQL"] {
            std::fs::write(dir.join(name), "SELECT 1;").unwrap();
        }

        let names: Vec<String> = sql_files(&dir)
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["0001_a.sql", "0002_b.sql", "0010_c.SQL"]);

        std::fs::remove_dir_all(&dir).unwrap();
        assert!(sql_files(&dir).is_err());
    }

    #[test]
    fn the_sidecar_sits_next_to_the_download() {
        assert_eq!(
            sidecar_path(Path::new("/data/aias/postgres/pg.zip")),
            PathBuf::from("/data/aias/postgres/pg.zip.sha256")
        );
    }

    /// Needs a cluster, so CI skips it. Run it with the binaries in place:
    /// `AIAS_PG_BIN=... AIAS_DATA_DIR=... cargo test -p aias-core -- --ignored`.
    #[tokio::test]
    #[ignore = "needs a Postgres cluster"]
    async fn an_app_database_is_provisioned_migrated_and_dropped() {
        let data_dir = paths::data_dir();
        let pg = ensure_postgres(&data_dir).await.unwrap();
        let app = "drop-probe";

        let url = provision_app_db(&pg, app).await.unwrap();
        assert!(url.ends_with("/app_drop_probe"), "{url}");
        assert_eq!(provision_app_db(&pg, app).await.unwrap(), url);

        let dir = std::env::temp_dir().join(format!("aias-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("0001_probe.sql"), "CREATE TABLE probe (id int);").unwrap();
        assert_eq!(apply_migrations(&url, &dir).await.unwrap(), 1);
        assert_eq!(apply_migrations(&url, &dir).await.unwrap(), 0);
        std::fs::remove_dir_all(&dir).unwrap();

        drop_app_db(&pg, app).await.unwrap();
        assert!(!app_pw_path(&pg.dir, app).exists());
        // Dropping twice is how an unsubscribe retry behaves.
        drop_app_db(&pg, app).await.unwrap();
    }
}
