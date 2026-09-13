import type { DeviceProfile } from '@/lib/api/hardware'

/**
 * Memory the model manager may plan with, in MB.
 *
 * Mirrors `DeviceProfile::effective_memory_mb` in `crates/core/src/hardware.rs`:
 * dedicated GPUs report VRAM, unified devices report the readable graphics
 * memory value or half of system RAM when it is unknown.
 */
export function effectiveMemoryMb(profile: DeviceProfile): number {
  if (profile.memoryModel === 'dedicated') return profile.vramMb ?? 0
  return profile.vramMb ?? Math.floor(profile.totalRamMb / 2)
}

/** 90% of the budget, the ceiling every planning decision uses. */
export function usableMemoryMb(profile: DeviceProfile): number {
  return Math.floor((effectiveMemoryMb(profile) * 9) / 10)
}
