//! Commands over `aias_core::apps`.
//!
//! No command here takes a path. The renderer names an app and
//! [`aias_core::apps::app_dir`] resolves it under `<data_dir>/apps`, so a web
//! context cannot point `validate`, `build` or `start` at a directory of its
//! choosing. The CLI still takes a directory: it is the developer's own shell.

use aias_core::apps::{self, AppProcess, Manifest};
use aias_core::models::ModelRef;
use aias_core::paths;
use serde::Serialize;

use crate::error::CmdResult;

#[tauri::command]
pub fn apps_validate(app_name: String) -> CmdResult<Manifest> {
    let dir = apps::app_dir(&paths::data_dir(), &app_name)?;
    Ok(apps::validate_dir(&dir)?)
}

#[tauri::command]
pub async fn apps_clone(repo_url: String, git_ref: Option<String>) -> CmdResult<String> {
    let dir = apps::clone_app(&repo_url, git_ref.as_deref(), &paths::data_dir()).await?;
    Ok(dir.display().to_string())
}

#[tauri::command]
pub async fn apps_build(app_name: String) -> CmdResult<()> {
    let dir = apps::app_dir(&paths::data_dir(), &app_name)?;
    let manifest = apps::validate_dir(&dir)?;
    Ok(apps::build(&dir, &manifest).await?)
}

#[tauri::command]
pub async fn apps_start(app_name: String) -> CmdResult<AppProcess> {
    let dir = apps::app_dir(&paths::data_dir(), &app_name)?;
    let manifest = apps::validate_dir(&dir)?;
    Ok(apps::start(&dir, &manifest).await?)
}

#[tauri::command]
pub async fn apps_stop(app_name: String) -> CmdResult<()> {
    Ok(apps::stop(&app_name).await?)
}

/// One subscribed app as the Apps page needs it: the clone plus its process.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledAppView {
    #[serde(flatten)]
    pub app: apps::InstalledApp,
    /// Present while the app runs in this process.
    pub process: Option<AppProcess>,
}

#[tauri::command]
pub async fn apps_index() -> CmdResult<Vec<apps::IndexEntry>> {
    Ok(apps::index(&apps::index_url()).await?)
}

#[tauri::command]
pub fn apps_installed() -> CmdResult<Vec<InstalledAppView>> {
    let running = apps::running();
    Ok(apps::installed(&paths::data_dir())
        .into_iter()
        .map(|app| InstalledAppView {
            process: running
                .iter()
                .find(|process| process.name == app.name)
                .cloned(),
            app,
        })
        .collect())
}

/// Clone and validate. The build is a separate command on purpose: the store
/// shows the manifest this returns and asks before running anything from it.
#[tauri::command]
pub async fn apps_subscribe(
    repo_url: String,
    git_ref: Option<String>,
) -> CmdResult<apps::InstalledApp> {
    Ok(apps::subscribe(&repo_url, git_ref.as_deref(), &paths::data_dir()).await?)
}

#[tauri::command]
pub fn apps_remove(app_name: String) -> CmdResult<()> {
    Ok(apps::remove(&app_name, &paths::data_dir())?)
}

/// The name is validated inside [`apps::logs`], the same shape `app_dir`
/// demands, so a renderer cannot tail a file outside `<data_dir>/logs`.
#[tauri::command]
pub fn apps_logs(app_name: String, kind: apps::LogKind, tail_lines: usize) -> CmdResult<String> {
    Ok(apps::logs(&app_name, kind, tail_lines)?)
}

#[tauri::command]
pub fn apps_missing_models(app_name: String) -> CmdResult<Vec<ModelRef>> {
    let data_dir = paths::data_dir();
    let manifest = apps::validate_dir(&apps::app_dir(&data_dir, &app_name)?)?;
    Ok(apps::missing_models(&manifest, &data_dir))
}
