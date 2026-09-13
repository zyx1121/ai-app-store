//! Desktop shell. Every command is a thin wrapper over `aias-core`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use aias_core::{api, apps, instances, paths, services};
use tauri::{Emitter, RunEvent};
use tokio::sync::oneshot;

mod commands;
mod error;

pub use error::{CmdError, CmdResult};

/// Shutdown runs once. `ExitRequested` is followed by `Exit`, and stopping the
/// same processes twice would only slow the window down.
static STOPPED: AtomicBool = AtomicBool::new(false);

/// Tells the local API to stop serving. Sent once, on the way out.
///
/// The API has to end with this process: it is the surface every agent reaches
/// the model manager through, and decision D10 has exactly one owner of that.
static API_SHUTDOWN: Mutex<Option<oneshot::Sender<()>>> = Mutex::new(None);

/// Start the desktop application.
pub fn run() {
    let app = tauri::Builder::default()
        // Apps open in the user's browser, not inside this window.
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Every module assumes its directory exists. Postgres is not started
            // here: `apps::start` calls `services::ensure_postgres` on the first
            // app that declares it, so a machine that runs no such app never
            // pays for a cluster.
            let data_dir = paths::data_dir();
            paths::ensure_data_dirs(&data_dir)?;

            // The loopback API of PLAN section 6.1. It is what `aias mcp`
            // forwards to, so it starts with the window and stops with it.
            let (stop, stopped) = oneshot::channel();
            *API_SHUTDOWN
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(stop);

            // An agent drives the same platform this window shows, so a change
            // it makes through the API is pushed to the pages (issue #16). A
            // failed emit means the window is gone, which the API survives.
            let handle = app.handle().clone();
            let announce: api::Emitter = std::sync::Arc::new(move |kind| {
                let _ = handle.emit(api::CHANGED_EVENT, api::ChangedEvent { kind });
            });

            tauri::async_runtime::spawn(async move {
                let shutdown = async {
                    let _ = stopped.await;
                };
                if let Err(err) = api::serve(&data_dir, shutdown, Some(announce)).await {
                    eprintln!("local API: {err}");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // hardware
            commands::hardware::hardware_detect,
            // runtime
            commands::runtime::runtime_select,
            commands::runtime::runtime_install,
            // models
            commands::models::models_search,
            commands::models::models_files,
            commands::models::models_download,
            commands::models::models_installed,
            commands::models::models_remove,
            commands::models::models_fit,
            // services
            commands::services::services_postgres_ensure,
            commands::services::services_postgres_stop,
            commands::services::services_provision_app_db,
            commands::services::services_apply_migrations,
            // instances
            commands::instances::instances_list,
            commands::instances::instances_acquire,
            commands::instances::instances_release,
            commands::instances::instances_params_for,
            commands::instances::instances_stop_all,
            // api
            commands::api::api_token,
            // apps
            commands::apps::apps_validate,
            commands::apps::apps_clone,
            commands::apps::apps_build,
            commands::apps::apps_start,
            commands::apps::apps_stop,
            commands::apps::apps_index,
            commands::apps::apps_installed,
            commands::apps::apps_subscribe,
            commands::apps::apps_remove,
            commands::apps::apps_logs,
            commands::apps::apps_missing_models,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the AI App Store");

    app.run(|_handle, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => stop_everything(),
        _ => {}
    });
}

/// Stop every process this app owns, in the order their dependencies allow.
///
/// Apps first, because each one holds leases; then the instances those leases
/// kept alive; then the cluster the apps were talking to. Nothing here is
/// allowed to fail the exit, so every error is reported and stepped over.
fn stop_everything() {
    if STOPPED.swap(true, Ordering::SeqCst) {
        return;
    }

    // The API answers out of this process, so it goes first: a request that
    // arrives while the apps are being stopped would be answered by a platform
    // that is halfway gone.
    if let Some(stop) = API_SHUTDOWN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
    {
        let _ = stop.send(());
    }

    tauri::async_runtime::block_on(async {
        for app in apps::running() {
            if let Err(err) = apps::stop(&app.name).await {
                eprintln!("stopping app {}: {err}", app.name);
            }
        }
        if let Err(err) = instances::stop_all().await {
            eprintln!("stopping instances: {err}");
        }
        let pg = services::postgres_at(&paths::data_dir());
        if let Err(err) = services::stop_postgres(&pg).await {
            eprintln!("stopping postgres: {err}");
        }
    });
}
