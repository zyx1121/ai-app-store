//! Commands over `aias_core::runtime`.

use aias_core::hardware::DeviceProfile;
use aias_core::paths;
use aias_core::runtime::{self, Backend, RuntimeInstall};

use crate::error::CmdResult;

#[tauri::command]
pub fn runtime_select(profile: DeviceProfile, opt_in_npu: bool) -> CmdResult<Backend> {
    Ok(runtime::select_with_npu(&profile, opt_in_npu))
}

#[tauri::command]
pub async fn runtime_install(backend: Backend) -> CmdResult<RuntimeInstall> {
    Ok(runtime::install(backend, &paths::data_dir()).await?)
}
