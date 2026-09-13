//! Error shape crossing the `invoke` boundary.
//!
//! `aias_core::Error` is not `Serialize`, and the frontend only needs a code and
//! a message, so every command returns this.

use serde::Serialize;

/// What the frontend receives when a command fails.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CmdError {
    /// Stable tag, for example `not_implemented`.
    pub code: String,
    pub message: String,
}

impl From<aias_core::Error> for CmdError {
    fn from(err: aias_core::Error) -> Self {
        Self {
            code: err.code().to_string(),
            message: err.to_string(),
        }
    }
}

/// Result every `#[tauri::command]` returns.
pub type CmdResult<T> = std::result::Result<T, CmdError>;
