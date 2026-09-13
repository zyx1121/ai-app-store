//! Commands over `aias_core::instances`.

use aias_core::hardware::DeviceProfile;
use aias_core::instances::{self, Instance, InstanceUrl, Params};
use aias_core::models::ModelRef;

use crate::error::CmdResult;

#[tauri::command]
pub fn instances_list() -> CmdResult<Vec<Instance>> {
    Ok(instances::list())
}

#[tauri::command]
pub async fn instances_acquire(app: String, model: ModelRef) -> CmdResult<InstanceUrl> {
    Ok(instances::acquire(&app, &model).await?)
}

#[tauri::command]
pub async fn instances_release(app: String, model: ModelRef) -> CmdResult<()> {
    Ok(instances::release(&app, &model).await?)
}

#[tauri::command]
pub fn instances_params_for(profile: DeviceProfile, model_size_bytes: u64) -> CmdResult<Params> {
    // The page asks about a model it has not downloaded, so there is no header
    // to read and the coarse cache estimate is the honest answer.
    Ok(instances::params_for(&profile, model_size_bytes, None))
}

#[tauri::command]
pub async fn instances_stop_all() -> CmdResult<()> {
    Ok(instances::stop_all().await?)
}
