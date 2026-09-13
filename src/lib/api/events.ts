import { isMock } from '@/lib/api/invoke'

/** What a change was about, mirrors `api::Changed` in `crates/core/src/api.rs`. */
export type ChangedKind = 'apps' | 'models' | 'instances'

/** Payload of {@link CHANGED_EVENT}, mirrors `api::ChangedEvent`. */
export interface Changed {
  kind: ChangedKind
}

/** Tauri event the local API announces a change on, mirrors `api::CHANGED_EVENT`. */
export const CHANGED_EVENT = 'aias://changed'

/**
 * Listen for a state change something other than this window made.
 *
 * The local API is hosted by this app, so an agent calling `apps/start` or
 * `apps/stop` changes state in this process; the handler emits this event and a
 * page that listens refetches without the user navigating away and back.
 *
 * There is no Tauri event API behind a plain dev server, so in mock mode this
 * resolves with a no-op unsubscribe and the poll in `useLiveRefresh` is the
 * only thing keeping the page current. Resolves with the unsubscribe.
 */
export async function onChanged(handler: (changed: Changed) => void): Promise<() => void> {
  if (isMock) return () => undefined
  const { listen } = await import('@tauri-apps/api/event')
  return await listen<Changed>(CHANGED_EVENT, (event) => handler(event.payload))
}
