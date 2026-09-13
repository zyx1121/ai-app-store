//! Commands over `aias_core::models`.

use std::path::PathBuf;

use aias_core::apps::ModelKind;
use aias_core::models::{self, Fit, ModelFile, ModelRef, SearchPage};
use aias_core::paths;
use serde::Serialize;
use tauri::Emitter;

use crate::error::CmdResult;

/// Tauri event carrying the progress of one download.
///
/// The frontend listens on it while `models_download` is in flight; one event
/// per callback of `models::download`, which is floored at 200 ms.
pub const DOWNLOAD_PROGRESS_EVENT: &str = "models://progress";

/// Payload of [`DOWNLOAD_PROGRESS_EVENT`].
///
/// The model is named in the payload because several downloads can be running
/// at once and the page has to tell their bars apart.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub repo: String,
    pub quant: String,
    pub downloaded_bytes: u64,
    /// `None` when the server sends no content length.
    pub total_bytes: Option<u64>,
}

/// One downloaded model, flattened for the frontend.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub model: ModelRef,
    /// Absolute path of the weights file.
    pub path: PathBuf,
    pub size_bytes: u64,
    /// The mmproj downloaded with the weights, for vision models.
    pub mmproj: Option<PathBuf>,
}

#[tauri::command]
pub async fn models_search(
    query: String,
    kind: Option<ModelKind>,
    cursor: Option<String>,
) -> CmdResult<SearchPage> {
    Ok(models::search_kind(&query, kind, cursor.as_deref()).await?)
}

#[tauri::command]
pub async fn models_files(repo: String) -> CmdResult<Vec<ModelFile>> {
    Ok(models::files(&repo).await?)
}

#[tauri::command]
pub async fn models_download(app: tauri::AppHandle, model: ModelRef) -> CmdResult<String> {
    let repo = model.repo.clone();
    let quant = model.quant.clone();
    let progress = move |progress: models::Progress| {
        // A failed emit means the window is gone, which the download survives.
        let _ = app.emit(
            DOWNLOAD_PROGRESS_EVENT,
            DownloadProgress {
                repo: repo.clone(),
                quant: quant.clone(),
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes,
            },
        );
    };
    let path = models::download(&model, &paths::data_dir(), &progress).await?;
    Ok(path.display().to_string())
}

#[tauri::command]
pub fn models_installed() -> CmdResult<Vec<InstalledModel>> {
    let data_dir = paths::data_dir();
    Ok(models::installed(&data_dir)
        .into_iter()
        .map(|(model, path, size_bytes)| InstalledModel {
            mmproj: models::installed_mmproj(&model, &data_dir),
            model,
            path,
            size_bytes,
        })
        .collect())
}

#[tauri::command]
pub fn models_remove(model: ModelRef) -> CmdResult<()> {
    Ok(models::remove(&model, &paths::data_dir())?)
}

#[tauri::command]
pub fn models_fit(size_bytes: u64, budget_mb: u64) -> CmdResult<Fit> {
    Ok(models::fit(size_bytes, budget_mb))
}
