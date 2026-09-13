import { invoke } from '@/lib/api/invoke'

import type { DeviceProfile } from '@/lib/api/hardware'

export type Backend = 'cuda' | 'vulkan' | 'openVino' | 'cpu'

export interface RuntimeInstall {
  backend: Backend
  /** llama.cpp release tag the artifact came from. */
  tag: string
  /** Directory the artifact was unpacked into. */
  dir: string
}

/** Apply the PLAN section 2.1 table to a device profile. */
export function select(profile: DeviceProfile, optInNpu = false): Promise<Backend> {
  return invoke<Backend>('runtime_select', { profile, optInNpu })
}

/** Download, verify and unpack the llama-server artifact for a backend. */
export function install(backend: Backend): Promise<RuntimeInstall> {
  return invoke<RuntimeInstall>('runtime_install', { backend })
}
