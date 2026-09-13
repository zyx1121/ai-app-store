//! Commands over `aias_core::services`.
//!
//! The frontend names an app and nothing else. It cannot hand in a DSN, a
//! cluster or a directory: those are derived here from `paths::data_dir` and
//! from the app's own manifest, so a compromised web context cannot point the
//! platform at another database or another folder of SQL.

use aias_core::services::{self, Postgres};
use aias_core::{apps, paths};

use crate::error::{CmdError, CmdResult};

#[tauri::command]
pub async fn services_postgres_ensure() -> CmdResult<Postgres> {
    Ok(services::ensure_postgres(&paths::data_dir()).await?)
}

/// Create the database and role for one subscribed app, and give back its DSN.
#[tauri::command]
pub async fn services_provision_app_db(app_name: String) -> CmdResult<String> {
    let data_dir = paths::data_dir();
    let pg = services::ensure_postgres(&data_dir).await?;
    Ok(services::provision_app_db(&pg, &app_name).await?)
}

/// Apply one subscribed app's declared migrations to its own database.
///
/// The folder comes from the manifest of the clone on disk and is resolved
/// against the app directory, so it cannot be pointed anywhere else.
#[tauri::command]
pub async fn services_apply_migrations(app_name: String) -> CmdResult<usize> {
    let data_dir = paths::data_dir();
    let app = apps::installed(&data_dir)
        .into_iter()
        .find(|app| app.name == app_name)
        .ok_or_else(|| CmdError {
            code: "not_found".into(),
            message: format!("app `{app_name}` is not subscribed"),
        })?;
    let Some(postgres) = app.manifest.services.postgres else {
        return Ok(0);
    };

    let dir = apps::resolve_migrations(&app.dir, &postgres.migrations)?;
    let pg = services::ensure_postgres(&data_dir).await?;
    let database_url = services::provision_app_db(&pg, &app_name).await?;
    Ok(services::apply_migrations(&database_url, &dir).await?)
}

#[tauri::command]
pub async fn services_postgres_stop() -> CmdResult<Postgres> {
    let pg = services::postgres_at(&paths::data_dir());
    services::stop_postgres(&pg).await?;
    Ok(pg)
}
