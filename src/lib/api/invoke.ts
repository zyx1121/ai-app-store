import { invoke as tauriInvoke } from '@tauri-apps/api/core'

import { mockInvoke } from '@/lib/api/mock'

/** Arguments of a Tauri command, camelCase keys as the command files declare them. */
export type InvokeArgs = Record<string, unknown>

/** True when the app runs against the fixtures instead of the Rust core. */
export const isMock = import.meta.env.VITE_AIAS_MOCK === '1'

/**
 * The one `invoke` every wrapper in this folder calls.
 *
 * With `VITE_AIAS_MOCK=1` it answers from `@/lib/api/mock`, so the pages can be
 * built and reviewed while the core commands are still stubs. Every other build
 * goes straight to Tauri and gets the real `CmdError` back on failure.
 */
export function invoke<T>(command: string, args?: InvokeArgs): Promise<T> {
  if (isMock) return mockInvoke<T>(command, args)
  return tauriInvoke<T>(command, args)
}

/** What the mock seam below is hung on, mock builds only. */
interface MockWindow {
  __aiasMock?: typeof mockInvoke
}

if (isMock && typeof window !== 'undefined') {
  // An end to end test changes fixture state the way an agent changes real
  // state: through the command layer, not through the UI. Without it every
  // change on the page comes from a click, and a page that only refetches on a
  // click would pass. `isMock` is a build time constant, so nothing of this is
  // in a real build.
  ;(window as MockWindow).__aiasMock = mockInvoke
}
