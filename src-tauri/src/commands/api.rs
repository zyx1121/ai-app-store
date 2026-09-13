//! Commands over `aias_core::api`.

use aias_core::{api, paths};
use serde::Serialize;

use crate::error::CmdResult;

/// How an agent reaches this running app.
///
/// The token is a secret and the renderer is the one place it is handed to, on
/// a page whose only use for it is a copy button: the developer pastes it into
/// the MCP client's configuration, which is what `aias mcp` reads.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiToken {
    /// Base URL of the local API, ends in `/v1`.
    pub url: String,
    pub token: String,
}

#[tauri::command]
pub fn api_token() -> CmdResult<ApiToken> {
    Ok(ApiToken {
        url: api::base_url(),
        token: api::token(&paths::data_dir())?,
    })
}
