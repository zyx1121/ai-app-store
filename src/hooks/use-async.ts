import { useCallback, useEffect, useRef, useState } from 'react'
import type { DependencyList } from 'react'

import { toCmdError, type CmdError } from '@/lib/api/error'

/** The three states every command call can be in. */
export type Async<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; error: CmdError }

export interface AsyncHandle<T> {
  state: Async<T>
  /** Run the command again, for the retry button of an error state. */
  reload: () => void
  /**
   * Run the command again without emptying the page first.
   *
   * What a background refresh needs: `reload` drops back to `loading`, which
   * would blank a page every few seconds, and a failed poll must leave what is
   * on screen rather than replace it with an error.
   */
  refresh: () => void
  /** Replace the loaded value after a mutation, without a round trip. */
  replace: (data: T) => void
}

/** Run one command on mount and whenever `deps` change. */
export function useAsync<T>(run: () => Promise<T>, deps: DependencyList): AsyncHandle<T> {
  const [state, setState] = useState<Async<T>>({ status: 'loading' })
  const [nonce, setNonce] = useState(0)
  /** The latest `run`, so `refresh` never restarts the caller's timers. */
  const latest = useRef(run)
  const mounted = useRef(true)

  useEffect(() => {
    latest.current = run
  })

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    let live = true
    setState({ status: 'loading' })
    run()
      .then((data) => {
        if (live) setState({ status: 'ready', data })
      })
      .catch((error: unknown) => {
        if (live) setState({ status: 'error', error: toCmdError(error) })
      })
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps])

  const reload = useCallback(() => setNonce((value) => value + 1), [])
  const refresh = useCallback(() => {
    void latest
      .current()
      .then((data) => {
        if (mounted.current) setState({ status: 'ready', data })
      })
      .catch(() => {
        // A poll that failed is not news: the page keeps what it had, and the
        // next one either works or the user reloads and sees the error.
      })
  }, [])
  const replace = useCallback((data: T) => setState({ status: 'ready', data }), [])

  return { state, reload, refresh, replace }
}
