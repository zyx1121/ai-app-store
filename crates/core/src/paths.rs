//! Where the platform keeps its state on disk.
//!
//! This is the only module allowed to carry `#[cfg(windows)]`. Everything else
//! asks for a path and stays portable.

use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// Root of every file the platform owns.
///
/// Windows: `%LOCALAPPDATA%\aias`. Everywhere else: `$XDG_DATA_HOME/aias`,
/// falling back to `~/.local/share/aias`.
pub fn data_dir() -> PathBuf {
    // Explicit override first, so tests and parallel developers can isolate state.
    if let Some(dir) = std::env::var_os("AIAS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local).join("aias");
        }
        PathBuf::from(r"C:\aias")
    }

    #[cfg(not(windows))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(xdg).join("aias");
        }
        let home = std::env::var_os("HOME").map(PathBuf::from);
        match home {
            Some(home) => home.join(".local").join("share").join("aias"),
            None => PathBuf::from("/tmp").join("aias"),
        }
    }
}

/// The installed `llama-server` build lives here, one directory per release tag.
pub fn runtime_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("runtime")
}

/// Downloaded GGUF files and their mmproj siblings.
pub fn models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

/// Local clones of subscribed apps, one directory per app name.
pub fn apps_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("apps")
}

/// The one Postgres cluster, binaries and data directory.
pub fn postgres_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("postgres")
}

/// stdout and stderr of every process the platform starts.
pub fn logs_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("logs")
}

/// Stored secrets, one JSON file per app.
pub fn secrets_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("secrets")
}

/// Mode of a directory only this user may enter.
pub(crate) const DIR_MODE: u32 = 0o700;
/// Mode of a file only this user may read.
pub(crate) const FILE_MODE: u32 = 0o600;

/// Create the root and every subdirectory, owner only. Safe to call repeatedly.
///
/// `create_dir_all` applies the ambient umask, which on a shared machine is
/// often `022` and leaves every other user able to list the models, read the
/// app clones and open `secrets/`. So each directory is restricted after it is
/// made, and the mode is put back on every call rather than only at creation:
/// that is one `chmod` syscall per directory and it repairs a data directory
/// made before this existed. Windows pays for a process per directory, so
/// there the ACL is only written where the directory is new.
pub fn ensure_data_dirs(data_dir: &Path) -> Result<()> {
    for dir in [
        data_dir.to_path_buf(),
        runtime_dir(data_dir),
        models_dir(data_dir),
        apps_dir(data_dir),
        postgres_dir(data_dir),
        logs_dir(data_dir),
        secrets_dir(data_dir),
    ] {
        let fresh = !dir.exists();
        std::fs::create_dir_all(&dir)?;
        if cfg!(unix) || fresh {
            restrict(&dir, DIR_MODE)?;
        }
    }
    Ok(())
}

/// Take a file or a directory away from everyone but this user.
///
/// `mode` is the Unix answer, `0o600` for a file and `0o700` for a directory.
/// Windows has no mode bits and gets an ACL instead, which says the same thing
/// either way, so the argument is only read on Unix.
#[cfg(unix)]
pub(crate) fn restrict(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

/// Windows has no mode bits, so the ACL is rewritten instead: `/inheritance:r`
/// drops what the parent folder handed down, `/grant:r <user>:F` replaces any
/// existing entry for this user with full control, and nobody else is named.
/// Every value reaches `icacls` as an argv element.
#[cfg(windows)]
pub(crate) fn restrict(path: &Path, _mode: u32) -> Result<()> {
    use crate::process::Quiet as _;

    let user = std::env::var("USERNAME")
        .map_err(|_| Error::Process("USERNAME is not set, cannot restrict a path".into()))?;
    let output = std::process::Command::new("icacls")
        .arg(path)
        .arg("/inheritance:r")
        .arg("/grant:r")
        .arg(format!("{user}:F"))
        .quiet()
        .output()
        .map_err(|err| Error::Process(format!("could not run icacls: {err}")))?;
    if !output.status.success() {
        return Err(Error::Process(format!(
            "icacls could not restrict {}: {} {}",
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn restrict(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

/// Turn a `dir` from a client into a directory it is allowed to work in.
///
/// An agent scaffolds a repo where it is already working, so `dir` cannot be
/// limited to `apps_dir`. What it is limited to: somewhere under the user's
/// home directory, or under `apps_dir` wherever that is; never inside the
/// platform's own state, because the runtime, the models, the Postgres cluster
/// and the logs are not places an app may be written into or started from.
///
/// Symlinks are resolved before the check, so a link inside the home directory
/// that points at `runtime/` is refused like the path it really is.
pub(crate) fn resolve_dir(raw: &Path, data_dir: &Path) -> Result<PathBuf> {
    let dir = absolute(raw)?;
    let home = home_dir()?;
    let apps = absolute(&apps_dir(data_dir))?;
    let data = absolute(data_dir)?;

    if !dir.starts_with(&home) && !dir.starts_with(&apps) {
        return Err(Error::InvalidManifest(format!(
            "{} is outside {} and {}; an app is worked on in the user's own files",
            dir.display(),
            home.display(),
            apps.display()
        )));
    }
    if dir.starts_with(&data) && !dir.starts_with(&apps) {
        return Err(Error::InvalidManifest(format!(
            "{} is inside the platform's own state; only {} is an app directory",
            dir.display(),
            apps.display()
        )));
    }
    Ok(dir)
}

/// Home directory of this user, the one boundary `resolve_dir` is built on.
pub(crate) fn home_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| {
            Error::NotFound("neither HOME nor USERPROFILE names a home directory".into())
        })?;
    absolute(Path::new(&home))
}

/// An absolute path with `.` and `..` gone and every existing part resolved.
///
/// A directory that does not exist yet is the normal case here, `init` and
/// `fork` both write one, so the longest existing ancestor is canonicalized
/// and the rest is appended to it. That is what makes the check in
/// [`resolve_dir`] survive a symlink without demanding the target exist.
fn absolute(raw: &Path) -> Result<PathBuf> {
    let mut path = if raw.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    for component in raw.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                path.pop();
            }
            other => path.push(other.as_os_str()),
        }
    }

    // Canonicalize what exists, keep what does not.
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut head = path.clone();
    loop {
        if let Ok(real) = head.canonicalize() {
            let mut resolved = real;
            for part in tail.iter().rev() {
                resolved.push(part);
            }
            return Ok(resolved);
        }
        let Some(name) = head.file_name().map(std::ffi::OsString::from) else {
            return Ok(path);
        };
        tail.push(name);
        head.pop();
    }
}

/// Platform specific executable file name for a bare stem such as `llama-server`.
pub fn exe_name(stem: &str) -> String {
    #[cfg(windows)]
    {
        format!("{stem}.exe")
    }

    #[cfg(not(windows))]
    {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdirs_hang_off_the_root() {
        let root = PathBuf::from("/data/aias");
        assert_eq!(runtime_dir(&root), PathBuf::from("/data/aias/runtime"));
        assert_eq!(models_dir(&root), PathBuf::from("/data/aias/models"));
        assert_eq!(apps_dir(&root), PathBuf::from("/data/aias/apps"));
        assert_eq!(postgres_dir(&root), PathBuf::from("/data/aias/postgres"));
        assert_eq!(logs_dir(&root), PathBuf::from("/data/aias/logs"));
    }

    #[test]
    fn data_dir_ends_in_aias() {
        if std::env::var_os("AIAS_DATA_DIR").is_some() {
            return;
        }
        assert!(data_dir().ends_with("aias"));
    }

    #[cfg(unix)]
    #[test]
    fn the_data_directory_and_its_subdirectories_are_owner_only() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = std::env::temp_dir().join(format!("aias-paths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        // A directory made before this existed, with the ambient umask, is
        // repaired rather than left as it is.
        std::fs::create_dir_all(models_dir(&root)).unwrap();
        std::fs::set_permissions(models_dir(&root), std::fs::Permissions::from_mode(0o755))
            .unwrap();

        ensure_data_dirs(&root).unwrap();
        for dir in [
            root.clone(),
            runtime_dir(&root),
            models_dir(&root),
            apps_dir(&root),
            postgres_dir(&root),
            logs_dir(&root),
            secrets_dir(&root),
        ] {
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
            assert_eq!(
                mode & 0o777,
                0o700,
                "{} is readable by somebody else",
                dir.display()
            );
        }

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_directory_outside_the_home_is_refused() {
        let data_dir = home().join(".local/share/aias-test");
        // Somewhere under the home directory is what an agent works in.
        let ok = resolve_dir(&home().join("projects/demo-app"), &data_dir).expect("under home");
        assert!(ok.starts_with(home()));

        // Outside the home directory, and inside the platform's own state.
        for bad in [
            PathBuf::from("/etc/aias-demo"),
            home().join("projects/../../etc/aias-demo"),
            data_dir.join("runtime/demo-app"),
            data_dir.join("models/demo-app"),
        ] {
            let err = resolve_dir(&bad, &data_dir)
                .expect_err(&format!("{} must be refused", bad.display()));
            assert_eq!(err.code(), "invalid_manifest", "{err}");
        }

        // The apps directory is an app directory, wherever it sits.
        resolve_dir(&data_dir.join("apps/demo-app"), &data_dir).expect("apps_dir is allowed");
    }

    fn home() -> PathBuf {
        home_dir().expect("a home directory")
    }
}
