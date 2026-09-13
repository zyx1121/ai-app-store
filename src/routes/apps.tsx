import { useEffect, useRef, useState } from 'react'
import { Boxes, ExternalLink, Play, ScrollText, Square } from 'lucide-react'

import { Page } from '@/components/page'
import { EmptyState, ErrorState, LoadingState } from '@/components/state'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { useAsync } from '@/hooks/use-async'
import { useLiveRefresh } from '@/hooks/use-live-refresh'
import * as apps from '@/lib/api/apps'
import type { InstalledApp } from '@/lib/api/apps'
import { toCmdError, type CmdError } from '@/lib/api/error'
import * as instances from '@/lib/api/instances'
import type { Instance } from '@/lib/api/instances'
import { maskSecrets } from '@/lib/format'
import { openExternal } from '@/lib/open'

interface AppsData {
  subscribed: InstalledApp[]
  running: Instance[]
}

const LOG_POLL_MS = 1500
/** Lines of the tail the page asks for, the height of the box allows for it. */
const LOG_TAIL_LINES = 200

/** Subscribed apps: build, lease, run, stop, PLAN section 2.3. */
export default function Apps() {
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<CmdError | null>(null)
  const [openLogs, setOpenLogs] = useState<string | null>(null)

  const data = useAsync<AppsData>(async () => {
    const [subscribed, running] = await Promise.all([apps.installed(), instances.list()])
    return { subscribed, running }
  }, [])

  // An agent starts and stops the same apps this page shows, and the idle
  // reaper stops the instances the table below lists. Both land here without a
  // navigation (issue #16).
  useLiveRefresh(data.refresh)

  async function toggle(app: InstalledApp) {
    setBusy(app.name)
    setActionError(null)
    try {
      if (app.process === null) await apps.start(app.name)
      else await apps.stop(app.name)
      setOpenLogs(app.name)
      data.reload()
    } catch (caught: unknown) {
      setActionError(toCmdError(caught))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Page
      title="Apps"
      description="Subscribed apps with what they declare, the port they listen on and the instances their leases keep alive."
    >
      {data.state.status === 'loading' ? <LoadingState label="Reading the app folder" /> : null}

      {data.state.status === 'error' ? (
        <ErrorState error={data.state.error} what="The app list" onRetry={data.reload} />
      ) : null}

      {data.state.status === 'ready' ? (
        <div className="flex min-w-0 flex-col gap-4">
          {data.state.data.subscribed.length === 0 ? (
            <EmptyState
              icon={Boxes}
              title="No app subscribed"
              description="Subscribe to an app in the Store and its clone lands here."
            />
          ) : null}

          {data.state.data.subscribed.map((app) => (
            <AppCard
              key={app.name}
              app={app}
              busy={busy === app.name}
              logsOpen={openLogs === app.name}
              onToggleLogs={() => setOpenLogs(openLogs === app.name ? null : app.name)}
              onToggle={() => void toggle(app)}
            />
          ))}

          {actionError !== null ? <ErrorState error={actionError} what="The app action" /> : null}

          <RunningInstances running={data.state.data.running} />
        </div>
      ) : null}
    </Page>
  )
}

function AppCard({
  app,
  busy,
  logsOpen,
  onToggle,
  onToggleLogs,
}: {
  app: InstalledApp
  busy: boolean
  logsOpen: boolean
  onToggle: () => void
  onToggleLogs: () => void
}) {
  const running = app.process !== null
  const postgres = app.manifest.services.postgres

  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
          {app.name}
          <Badge variant="outline">v{app.manifest.version}</Badge>
          <Badge variant="outline">{app.manifest.runtime}</Badge>
          <Badge variant={running ? 'default' : 'outline'}>{running ? 'Running' : 'Stopped'}</Badge>
        </CardTitle>
        <CardDescription>{app.manifest.description}</CardDescription>
      </CardHeader>
      <CardContent className="flex min-w-0 flex-col gap-3">
        <div className="grid gap-1 text-sm">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-muted-foreground">Models</span>
            <span className="min-w-0 break-words">
              {app.manifest.models.length === 0
                ? 'none declared'
                : app.manifest.models
                    .map(
                      (model) =>
                        `${model.alias}: ${model.repo} ${model.quant.join(' or ')}${
                          model.fallback === null ? '' : ` (fallback ${model.fallback})`
                        }`,
                    )
                    .join(' / ')}
            </span>
          </div>
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-muted-foreground">Postgres</span>
            <span>
              {postgres === null
                ? 'not declared'
                : `one database, migrations from ${postgres.migrations}`}
            </span>
          </div>
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-muted-foreground">Health</span>
            <span className="font-mono text-xs">{app.manifest.health ?? 'none'}</span>
          </div>
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-muted-foreground">Source</span>
            <span className="min-w-0 font-mono text-xs break-all">
              {app.repoUrl ?? 'put here by hand'}
              {app.gitRef === null ? '' : ` @ ${app.gitRef}`}
              {app.sha === null ? '' : ` (${app.sha.slice(0, 12)})`}
            </span>
          </div>
          {running && app.process !== null ? (
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-muted-foreground">Port</span>
              <span className="font-mono text-xs">
                {app.process.url}, pid {app.process.pid ?? 'unknown'}
              </span>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant={running ? 'outline' : 'default'} disabled={busy} onClick={onToggle}>
            {running ? <Square /> : <Play />}
            {running ? 'Stop' : 'Run'}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            disabled={!running}
            onClick={() => {
              if (app.process !== null) void openExternal(app.process.url)
            }}
          >
            <ExternalLink />
            Open
          </Button>
          <Button size="sm" variant="ghost" onClick={onToggleLogs}>
            <ScrollText />
            {logsOpen ? 'Hide logs' : 'Logs'}
          </Button>
        </div>

        {logsOpen ? <LogTail appName={app.name} /> : null}
      </CardContent>
    </Card>
  )
}

/** Tail of one app log, polled from `apps_logs`. */
function LogTail({ appName }: { appName: string }) {
  const [kind, setKind] = useState<apps.LogKind>('run')
  const [tail, setTail] = useState('')
  const [error, setError] = useState<CmdError | null>(null)
  const bottom = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let live = true
    setTail('')
    setError(null)

    async function poll() {
      try {
        const text = await apps.logs(appName, kind, LOG_TAIL_LINES)
        if (!live) return
        setTail(text)
      } catch (caught: unknown) {
        if (!live) return
        setError(toCmdError(caught))
        window.clearInterval(handle)
      }
    }

    void poll()
    const handle = window.setInterval(() => void poll(), LOG_POLL_MS)
    return () => {
      live = false
      window.clearInterval(handle)
    }
  }, [appName, kind])

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [tail])

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {(['run', 'build'] as const).map((value) => (
          <Button
            key={value}
            size="sm"
            variant={kind === value ? 'secondary' : 'ghost'}
            onClick={() => setKind(value)}
          >
            {value === 'run' ? 'Run log' : 'Build log'}
          </Button>
        ))}
      </div>

      {error !== null ? (
        <ErrorState error={error} what="The log tail" />
      ) : (
        <div className="bg-muted/40 max-h-56 min-w-0 overflow-auto rounded-lg border p-3">
          {tail.length === 0 ? (
            <p className="text-muted-foreground text-xs">Waiting for output.</p>
          ) : (
            <pre className="font-mono text-xs leading-5 whitespace-pre-wrap">
              {maskSecrets(tail)}
            </pre>
          )}
          <div ref={bottom} />
        </div>
      )}
    </div>
  )
}

function RunningInstances({ running }: { running: Instance[] }) {
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle className="text-sm">Running instances</CardTitle>
        <CardDescription>
          One llama-server process per model. An instance stops when its last lease is dropped.
        </CardDescription>
      </CardHeader>
      <CardContent className="min-w-0">
        {running.length === 0 ? (
          <p className="text-muted-foreground text-sm">No instance is running.</p>
        ) : (
          <div className="min-w-0 overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Port</TableHead>
                  <TableHead>Leases</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {running.map((instance) => (
                  <TableRow key={`${instance.model.repo}-${instance.model.quant}`}>
                    <TableCell className="font-mono text-xs">
                      {instance.model.repo} {instance.model.quant}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{instance.port}</TableCell>
                    <TableCell className="text-xs">
                      {instance.leases.length === 0 ? 'none, kept warm' : instance.leases.join(', ')}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
