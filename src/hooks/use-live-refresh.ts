import { useEffect, useRef } from 'react'

import { onChanged } from '@/lib/api/events'

/** How often a visible page refetches when no event arrives. */
export const LIVE_POLL_MS = 5000

/**
 * Keep a page current while something other than this window changes state.
 *
 * Two sources, because neither covers the other. The `aias://changed` event is
 * immediate but only fires for a change the local API handled, so an agent
 * calling `apps/stop` shows up at once. The poll covers everything else: an app
 * that died on its own, and the idle reaper stopping an instance, neither of
 * which any handler is in the middle of. Polling stops while the tab is hidden,
 * which is a tab nobody is reading, and one refetch happens when it comes back.
 *
 * `refetch` is read from a ref, so a page can pass a fresh closure every render
 * without restarting the listener.
 */
export function useLiveRefresh(refetch: () => void): void {
  const latest = useRef(refetch)

  useEffect(() => {
    latest.current = refetch
  })

  useEffect(() => {
    let live = true
    let stop: (() => void) | null = null

    void onChanged(() => {
      if (live) latest.current()
    }).then((unlisten) => {
      if (live) stop = unlisten
      else unlisten()
    })

    function tick() {
      if (document.visibilityState === 'visible') latest.current()
    }

    const handle = window.setInterval(tick, LIVE_POLL_MS)
    document.addEventListener('visibilitychange', tick)

    return () => {
      live = false
      stop?.()
      window.clearInterval(handle)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [])
}
