//! What every platform owned child process gets, and what it does not.
//!
//! The desktop shell is the owner of `llama-server`, of every app process and
//! of the Postgres command line tools. Two things are true of all of them:
//! they must not open a console window on Windows, and they must not inherit
//! this process's environment. [`Quiet`] is the first, [`Sealed`] is the
//! second.

/// `CREATE_NO_WINDOW` from the Windows process creation flags.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Start a child without a console window. A no op off Windows.
pub(crate) trait Quiet {
    /// Set the flag and return the command, so it chains with the builder.
    fn quiet(&mut self) -> &mut Self;
}

impl Quiet for std::process::Command {
    fn quiet(&mut self) -> &mut Self {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            self.creation_flags(CREATE_NO_WINDOW);
        }
        self
    }
}

impl Quiet for tokio::process::Command {
    fn quiet(&mut self) -> &mut Self {
        #[cfg(windows)]
        {
            self.creation_flags(CREATE_NO_WINDOW);
        }
        self
    }
}

/// Variables a platform owned child keeps from this process's environment.
///
/// Everything else is dropped, because this process holds secrets an app has
/// no business reading: `AIAS_API_TOKEN` is the bearer of the local API and
/// `AIAS_HF_TOKEN` is the user's Hugging Face credential, and both would
/// otherwise be inherited by every app and every build command. So the
/// environment of a child is built rather than passed on: cleared first, then
/// filled from a list, then given what the platform means it to have.
///
/// What is on the list is what a program needs to run at all: where to find
/// executables and libraries (`PATH`, `PATHEXT`, `ComSpec`), where the user's
/// files are (`HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `ProgramData`),
/// where scratch space is (`TEMP`, `TMP`), how to format text and time (`LANG`,
/// `LC_ALL`, `TZ`), and the Windows facts every process reads (`SystemRoot`,
/// `SystemDrive`, `NUMBER_OF_PROCESSORS`). Windows environment names are case
/// insensitive, so the Windows entries are spelled as Windows spells them.
pub(crate) const APP_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USERPROFILE",
    "SystemRoot",
    "SystemDrive",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "TZ",
    "APPDATA",
    "LOCALAPPDATA",
    "ProgramData",
    "ComSpec",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
];

/// What `llama-server` is started with, [`APP_ALLOWLIST`] minus what a model
/// server has no use for, plus the one GPU selector.
///
/// `CUDA_VISIBLE_DEVICES` is how a user with two cards says which one the
/// platform may use, and it is read by the CUDA runtime inside the process, so
/// it is the one variable that has to survive. `LLAMA_API_KEY` is not on the
/// list: it is set explicitly on the command, because it is a value the
/// platform generates rather than one it passes on.
pub(crate) const LLAMA_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USERPROFILE",
    "SystemRoot",
    "SystemDrive",
    "TEMP",
    "TMP",
    "CUDA_VISIBLE_DEVICES",
];

/// What the Postgres command line tools are started with.
///
/// Deliberately without any `PG*` variable: `PGDATA`, `PGPORT`, `PGHOST`,
/// `PGUSER` and `PGPASSWORD` all override an argument this crate passes on
/// argv, so a user who runs another Postgres would otherwise redirect the
/// platform's own cluster by exporting one of them. `LANG` stays because
/// `initdb` reads it when no locale is given, and this crate gives one.
pub(crate) const POSTGRES_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USERPROFILE",
    "SystemRoot",
    "SystemDrive",
    "TEMP",
    "TMP",
    "LANG",
];

/// The allowlisted variables this process actually has, in list order.
fn inherited<'a>(allow: &'a [&'a str]) -> Vec<(&'a str, std::ffi::OsString)> {
    allow
        .iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (*name, value)))
        .collect()
}

/// Replace the inherited environment with an allowlist of it.
///
/// Call it before any `env` of your own: it clears everything, so a value set
/// earlier on the same command would be dropped.
pub(crate) trait Sealed {
    /// Clear the environment and put back only `allow`.
    fn sealed_env(&mut self, allow: &[&str]) -> &mut Self;
}

impl Sealed for std::process::Command {
    fn sealed_env(&mut self, allow: &[&str]) -> &mut Self {
        self.env_clear();
        for (name, value) in inherited(allow) {
            self.env(name, value);
        }
        self
    }
}

impl Sealed for tokio::process::Command {
    fn sealed_env(&mut self, allow: &[&str]) -> &mut Self {
        self.env_clear();
        for (name, value) in inherited(allow) {
            self.env(name, value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_platform_secret_is_on_any_allowlist() {
        for list in [APP_ALLOWLIST, LLAMA_ALLOWLIST, POSTGRES_ALLOWLIST] {
            for name in list {
                assert!(
                    !name.starts_with("AIAS_"),
                    "`{name}` would hand a platform variable to a child"
                );
            }
        }
        // The Postgres tools take every setting on argv, so no `PG*` override
        // may reach them.
        for name in POSTGRES_ALLOWLIST {
            assert!(!name.starts_with("PG"), "`{name}` overrides an argument");
        }
    }
}
