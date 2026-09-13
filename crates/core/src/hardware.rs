//! Device detection. Produces the one profile every other module reads.

use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// GPU vendor as far as backend selection cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    /// A GPU we do not have a backend rule for.
    Other,
    /// No usable GPU at all.
    None,
}

/// How GPU memory relates to system memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryModel {
    /// Discrete GPU with its own VRAM.
    Dedicated,
    /// Shared or unified memory, for example Strix Halo or Panther Lake.
    Unified,
}

/// Everything the platform knows about this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceProfile {
    pub gpu_vendor: Vendor,
    pub gpu_name: String,
    /// Dedicated VRAM, or the Variable Graphics Memory value on unified devices.
    pub vram_mb: Option<u64>,
    pub total_ram_mb: u64,
    pub memory_model: MemoryModel,
    pub has_npu: bool,
}

impl DeviceProfile {
    /// Memory the model manager may plan with, in MB.
    ///
    /// Dedicated GPUs report their VRAM. Unified devices report the readable
    /// graphics memory value, or half of system RAM when it is unknown, as in
    /// PLAN.md section 5.
    pub fn effective_memory_mb(&self) -> u64 {
        match self.memory_model {
            MemoryModel::Dedicated => self.vram_mb.unwrap_or(0),
            MemoryModel::Unified => self.vram_mb.unwrap_or(self.total_ram_mb / 2),
        }
    }

    /// 90% of the budget, the ceiling every planning decision uses.
    pub fn usable_memory_mb(&self) -> u64 {
        self.effective_memory_mb() * 9 / 10
    }

    /// True when there is a GPU worth offloading layers to.
    pub fn has_gpu(&self) -> bool {
        !matches!(self.gpu_vendor, Vendor::None)
    }
}

/// Detect the device profile of the machine this runs on.
///
/// Windows is the only target of version 1, so it is the only place a real GPU
/// is looked for: one PowerShell probe returns the video controllers, the
/// registry VRAM value, `nvidia-smi` when it exists, the physical memory and
/// whether an NPU is present. Everywhere else, a Mac developing against
/// `AIAS_LLAMA_SERVER` included, the platform reports a CPU only profile with
/// the real system memory read through sysinfo, so the memory budget and
/// `instances::params_for` answer with usable numbers rather than zero.
pub fn detect() -> Result<DeviceProfile> {
    if cfg!(windows) {
        parse_probe(&run_windows_probe()?)
    } else {
        Ok(cpu_only_profile())
    }
}

/// One PowerShell script, no interpolation, passed as a single argv element.
///
/// It emits exactly one JSON object on stdout. Everything the probe cannot read
/// comes back as `null` rather than as a failure, because a missing value is a
/// weaker signal than a missing machine.
const WINDOWS_PROBE: &str = r#"
$ErrorActionPreference = 'SilentlyContinue'
$classKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$keys = @(Get-ChildItem -LiteralPath $classKey | Where-Object { $_.PSChildName -match '^[0-9]{4}$' })
$gpus = @()
foreach ($c in @(Get-CimInstance -ClassName Win32_VideoController)) {
  $qw = $null
  $pnp = ''
  if ($c.PNPDeviceID) { $pnp = $c.PNPDeviceID.ToLower() }
  foreach ($k in $keys) {
    $p = Get-ItemProperty -LiteralPath $k.PSPath
    if (-not $p.MatchingDeviceId) { continue }
    if ($pnp -and $pnp.StartsWith($p.MatchingDeviceId.ToLower())) {
      $size = $p.'HardwareInformation.qwMemorySize'
      if ($null -ne $size) { $qw = [uint64]$size }
      break
    }
  }
  $gpus += [pscustomobject]@{
    name = $c.Name
    adapterCompatibility = $c.AdapterCompatibility
    pnpDeviceId = $c.PNPDeviceID
    qwMemorySize = $qw
  }
}
$nvidiaVramMb = $null
if (Get-Command nvidia-smi) {
  $smi = @(& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  if ($smi.Count -gt 0 -and $smi[0].Trim() -match '^[0-9]+$') { $nvidiaVramMb = [uint64]$smi[0].Trim() }
}
$hasNpu = $false
# Case sensitive with word boundaries: a plain 'NPU' match also hits every
# 'USB Input Device', and only the accelerators spell it in capitals.
$npu = @(Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -cmatch '\bNPU\b|AI Boost|\bNeural\b' })
if ($npu.Count -gt 0) { $hasNpu = $true }
[pscustomobject]@{
  gpus = $gpus
  nvidiaVramMb = $nvidiaVramMb
  totalRamBytes = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory
  hasNpu = $hasNpu
} | ConvertTo-Json -Depth 4 -Compress
"#;

/// Raw shape of [`WINDOWS_PROBE`] output.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Probe {
    #[serde(default)]
    gpus: Option<OneOrMany<ProbeGpu>>,
    #[serde(default)]
    nvidia_vram_mb: Option<u64>,
    #[serde(default)]
    total_ram_bytes: Option<u64>,
    #[serde(default)]
    has_npu: bool,
}

/// One `Win32_VideoController` plus its registry memory size.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProbeGpu {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    adapter_compatibility: Option<String>,
    #[serde(default)]
    pnp_device_id: Option<String>,
    /// `HardwareInformation.qwMemorySize`, absent on most integrated GPUs.
    #[serde(default)]
    qw_memory_size: Option<u64>,
}

/// `ConvertTo-Json` collapses a one element array into a bare object.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(item) => vec![item],
            OneOrMany::Many(items) => items,
        }
    }
}

fn run_windows_probe() -> Result<String> {
    // Every value reaching the process is an argv element; the script is a
    // constant with nothing interpolated into it.
    let output = Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            WINDOWS_PROBE,
        ])
        .output()?;

    if !output.status.success() {
        return Err(Error::Process(format!(
            "hardware probe exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_probe(json: &str) -> Result<DeviceProfile> {
    let probe: Probe = serde_json::from_str(json.trim())?;
    let gpus = probe.gpus.map(OneOrMany::into_vec).unwrap_or_default();
    let total_ram_mb = probe.total_ram_bytes.unwrap_or(0) / (1024 * 1024);

    // NVIDIA first, then AMD, then Intel, then anything else that reported.
    let primary = gpus
        .iter()
        .max_by_key(|gpu| vendor_rank(vendor_of(gpu)))
        .filter(|gpu| vendor_of(gpu) != Vendor::None);

    let Some(gpu) = primary else {
        return Ok(DeviceProfile {
            gpu_vendor: Vendor::None,
            gpu_name: "none".into(),
            vram_mb: None,
            total_ram_mb,
            memory_model: MemoryModel::Unified,
            has_npu: probe.has_npu,
        });
    };

    let gpu_vendor = vendor_of(gpu);
    let gpu_name = gpu.name.clone().unwrap_or_else(|| "unknown".into());
    let (memory_model, vram_mb) = memory_of(
        gpu_vendor,
        &gpu_name,
        gpu.qw_memory_size,
        probe.nvidia_vram_mb,
    );

    Ok(DeviceProfile {
        gpu_vendor,
        gpu_name,
        vram_mb,
        total_ram_mb,
        memory_model,
        has_npu: probe.has_npu,
    })
}

fn vendor_of(gpu: &ProbeGpu) -> Vendor {
    let haystack = format!(
        "{} {} {}",
        gpu.adapter_compatibility.as_deref().unwrap_or(""),
        gpu.name.as_deref().unwrap_or(""),
        gpu.pnp_device_id.as_deref().unwrap_or("")
    )
    .to_ascii_lowercase();

    if haystack.contains("nvidia") || haystack.contains("ven_10de") {
        Vendor::Nvidia
    } else if haystack.contains("advanced micro devices")
        || haystack.contains("amd")
        || haystack.contains("ati ")
        || haystack.contains("ven_1002")
    {
        Vendor::Amd
    } else if haystack.contains("intel") || haystack.contains("ven_8086") {
        Vendor::Intel
    } else if haystack.contains("apple") {
        Vendor::Apple
    } else if gpu.name.is_some() {
        Vendor::Other
    } else {
        Vendor::None
    }
}

fn vendor_rank(vendor: Vendor) -> u8 {
    match vendor {
        Vendor::Nvidia => 4,
        Vendor::Amd => 3,
        Vendor::Intel => 2,
        Vendor::Apple | Vendor::Other => 1,
        Vendor::None => 0,
    }
}

/// Decide the memory model and the VRAM figure to plan with.
///
/// `AdapterRAM` is useless here: it is 32 bit and integrated adapters report a
/// carve out rather than what they can actually use. So NVIDIA takes the
/// `nvidia-smi` number, a discrete adapter takes the registry `qwMemorySize`,
/// and anything integrated reports unified memory with no VRAM figure, which
/// sends [`DeviceProfile::effective_memory_mb`] to the system RAM rule.
fn memory_of(
    vendor: Vendor,
    name: &str,
    qw_memory_size: Option<u64>,
    nvidia_vram_mb: Option<u64>,
) -> (MemoryModel, Option<u64>) {
    let registry_mb = qw_memory_size.map(|bytes| bytes / (1024 * 1024));

    match vendor {
        Vendor::Nvidia => (MemoryModel::Dedicated, nvidia_vram_mb.or(registry_mb)),
        Vendor::Intel | Vendor::Amd | Vendor::Apple | Vendor::Other | Vendor::None => {
            match registry_mb {
                Some(mb) if !is_integrated(vendor, name) => (MemoryModel::Dedicated, Some(mb)),
                _ => (MemoryModel::Unified, None),
            }
        }
    }
}

/// Integrated by name, for the cases where the registry alone cannot tell.
///
/// Intel ships discrete silicon only as Arc A and B series, so any other Arc
/// name is a tile on the CPU package. AMD integrated parts carry a three or
/// four digit model number ending in `M` or `S` (890M, 8060S) or say Strix.
fn is_integrated(vendor: Vendor, name: &str) -> bool {
    match vendor {
        Vendor::Intel => !is_intel_discrete_model(name),
        // `RX` marks every discrete Radeon, integrated parts never carry it.
        Vendor::Amd => {
            let lower = name.to_ascii_lowercase();
            !tokens(name).any(|token| token == "RX")
                && (lower.contains("strix") || has_mobile_model_suffix(name))
        }
        _ => false,
    }
}

/// True for an Arc A or B series model number such as `A770` or `B580`.
fn is_intel_discrete_model(name: &str) -> bool {
    tokens(name).any(|token| {
        token.len() == 4
            && matches!(token.as_bytes()[0], b'A' | b'B')
            && token[1..].bytes().all(|b| b.is_ascii_digit())
    })
}

/// True for a model number such as `890M` or `8060S`.
fn has_mobile_model_suffix(name: &str) -> bool {
    tokens(name).any(|token| {
        let (digits, suffix) = token.split_at(token.len() - 1);
        (3..=4).contains(&digits.len())
            && matches!(suffix, "M" | "S")
            && digits.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Alphanumeric runs of a device name, so `Arc(TM) B580` yields `B580`.
fn tokens(name: &str) -> impl Iterator<Item = &str> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
}

/// What every non Windows machine reports: no GPU backend, real memory.
fn cpu_only_profile() -> DeviceProfile {
    let mut system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    system.refresh_memory();

    DeviceProfile {
        gpu_vendor: Vendor::None,
        gpu_name: "cpu".into(),
        vram_mb: None,
        total_ram_mb: system.total_memory() / (1024 * 1024),
        memory_model: MemoryModel::Unified,
        has_npu: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(
        memory_model: MemoryModel,
        vram_mb: Option<u64>,
        total_ram_mb: u64,
    ) -> DeviceProfile {
        DeviceProfile {
            gpu_vendor: Vendor::Amd,
            gpu_name: "test".into(),
            vram_mb,
            total_ram_mb,
            memory_model,
            has_npu: false,
        }
    }

    /// RTX 3080 desktop: nvidia-smi answers, the registry agrees, no NPU.
    const NVIDIA_DESKTOP: &str = r#"{
        "gpus": [
            {
                "name": "NVIDIA GeForce RTX 3080",
                "adapterCompatibility": "NVIDIA",
                "pnpDeviceId": "PCI\\VEN_10DE&DEV_2206&SUBSYS_38851462&REV_A1\\4&2D9E4F9B&0&0008",
                "qwMemorySize": 10737418240
            }
        ],
        "nvidiaVramMb": 10240,
        "totalRamBytes": 68437897216,
        "hasNpu": false
    }"#;

    /// Panther Lake laptop: one integrated Arc tile, no registry size, an NPU.
    /// ConvertTo-Json collapsed the one element array into a bare object.
    const INTEL_PANTHER_LAKE: &str = r#"{
        "gpus": {
            "name": "Intel(R) Arc(TM) Graphics",
            "adapterCompatibility": "Intel Corporation",
            "pnpDeviceId": "PCI\\VEN_8086&DEV_B080&SUBSYS_00000000&REV_00\\3&11583659&0&10",
            "qwMemorySize": null
        },
        "nvidiaVramMb": null,
        "totalRamBytes": 34359738368,
        "hasNpu": true
    }"#;

    /// Strix Halo: the registry reports a carve out that is not the budget, and
    /// a second basic display adapter must not win the primary slot.
    const AMD_STRIX_HALO: &str = r#"{
        "gpus": [
            {
                "name": "Microsoft Basic Display Adapter",
                "adapterCompatibility": "(Standard display types)",
                "pnpDeviceId": "ROOT\\BasicDisplay\\0000",
                "qwMemorySize": null
            },
            {
                "name": "AMD Radeon(TM) 8060S Graphics",
                "adapterCompatibility": "Advanced Micro Devices, Inc.",
                "pnpDeviceId": "PCI\\VEN_1002&DEV_1586&SUBSYS_12341462&REV_C1\\4&1A2B3C4D&0&0041",
                "qwMemorySize": 536870912
            }
        ],
        "nvidiaVramMb": null,
        "totalRamBytes": 137438953472,
        "hasNpu": true
    }"#;

    #[test]
    fn dedicated_uses_vram() {
        let p = profile(MemoryModel::Dedicated, Some(16384), 65536);
        assert_eq!(p.effective_memory_mb(), 16384);
    }

    #[test]
    fn unified_without_a_reading_takes_half_of_ram() {
        let p = profile(MemoryModel::Unified, None, 131_072);
        assert_eq!(p.effective_memory_mb(), 65536);
    }

    #[test]
    fn unified_prefers_the_readable_value() {
        let p = profile(MemoryModel::Unified, Some(98304), 131_072);
        assert_eq!(p.effective_memory_mb(), 98304);
    }

    #[test]
    fn nvidia_desktop_takes_the_nvidia_smi_number() {
        let p = parse_probe(NVIDIA_DESKTOP).unwrap();
        assert_eq!(p.gpu_vendor, Vendor::Nvidia);
        assert_eq!(p.gpu_name, "NVIDIA GeForce RTX 3080");
        assert_eq!(p.vram_mb, Some(10240));
        assert_eq!(p.memory_model, MemoryModel::Dedicated);
        assert_eq!(p.total_ram_mb, 65267);
        assert!(!p.has_npu);
        assert_eq!(p.effective_memory_mb(), 10240);
    }

    #[test]
    fn intel_igpu_is_unified_and_reports_the_npu() {
        let p = parse_probe(INTEL_PANTHER_LAKE).unwrap();
        assert_eq!(p.gpu_vendor, Vendor::Intel);
        assert_eq!(p.memory_model, MemoryModel::Unified);
        assert_eq!(p.vram_mb, None);
        assert!(p.has_npu);
        // Half of 32 GB, because no readable graphics memory value exists.
        assert_eq!(p.effective_memory_mb(), 16384);
    }

    #[test]
    fn strix_halo_ignores_the_registry_carve_out() {
        let p = parse_probe(AMD_STRIX_HALO).unwrap();
        assert_eq!(p.gpu_vendor, Vendor::Amd);
        assert_eq!(p.gpu_name, "AMD Radeon(TM) 8060S Graphics");
        assert_eq!(p.memory_model, MemoryModel::Unified);
        assert_eq!(p.vram_mb, None);
        assert_eq!(p.effective_memory_mb(), 65536);
    }

    #[test]
    fn a_discrete_arc_keeps_its_registry_vram() {
        let (model, vram) = memory_of(
            Vendor::Intel,
            "Intel(R) Arc(TM) A770 Graphics",
            Some(17_179_869_184),
            None,
        );
        assert_eq!(model, MemoryModel::Dedicated);
        assert_eq!(vram, Some(16384));
    }

    #[test]
    fn a_discrete_radeon_keeps_its_registry_vram() {
        let (model, vram) = memory_of(
            Vendor::Amd,
            "AMD Radeon RX 7900 XTX",
            Some(25_769_803_776),
            None,
        );
        assert_eq!(model, MemoryModel::Dedicated);
        assert_eq!(vram, Some(24576));
    }

    #[test]
    fn a_machine_without_a_video_controller_has_no_gpu() {
        let p = parse_probe(r#"{"gpus":[],"totalRamBytes":8589934592,"hasNpu":false}"#).unwrap();
        assert_eq!(p.gpu_vendor, Vendor::None);
        assert!(!p.has_gpu());
        assert_eq!(p.total_ram_mb, 8192);
    }

    #[test]
    fn detect_reports_memory_on_this_machine() {
        let p = detect().unwrap();
        assert!(p.total_ram_mb > 0);
    }

    /// A developer machine is not a target machine, but every module has to
    /// keep working on one: a profile with no memory plans no context at all.
    #[test]
    #[cfg(not(windows))]
    fn a_developer_machine_gets_a_usable_cpu_profile() {
        let profile = detect().expect("detection never fails off Windows");
        assert_eq!(profile.gpu_vendor, Vendor::None);
        assert!(profile.total_ram_mb > 0, "{profile:?}");
        assert!(profile.effective_memory_mb() > 0, "{profile:?}");
        assert!(profile.usable_memory_mb() > 0, "{profile:?}");
    }
}
