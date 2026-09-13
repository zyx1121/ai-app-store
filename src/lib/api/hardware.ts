import { invoke } from '@/lib/api/invoke'

export type Vendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'other' | 'none'

export type MemoryModel = 'dedicated' | 'unified'

export interface DeviceProfile {
  gpuVendor: Vendor
  gpuName: string
  vramMb: number | null
  totalRamMb: number
  memoryModel: MemoryModel
  hasNpu: boolean
}

/** Detect the GPU, memory model and NPU of this machine. */
export function detect(): Promise<DeviceProfile> {
  return invoke<DeviceProfile>('hardware_detect')
}
