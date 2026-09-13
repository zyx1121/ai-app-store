//! Commands over `aias_core::hardware`.

use aias_core::hardware::{self, DeviceProfile};

use crate::error::CmdResult;

#[tauri::command]
pub fn hardware_detect() -> CmdResult<DeviceProfile> {
    Ok(hardware::detect()?)
}
