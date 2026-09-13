import { useEffect, useState } from 'react'
import { HardDrive, MessageSquare, Play, Square } from 'lucide-react'

import { ChatDrawer, UI_LEASE } from '@/components/chat-drawer'
import { DownloadBar } from '@/components/download-bar'
import { Page } from '@/components/page'
import { EmptyState, ErrorState, LoadingState } from '@/components/state'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAsync } from '@/hooks/use-async'
import { useLiveRefresh } from '@/hooks/use-live-refresh'
import { toCmdError, type CmdError } from '@/lib/api/error'
import * as instances from '@/lib/api/instances'
import type { Instance } from '@/lib/api/instances'
import * as models from '@/lib/api/models'
import type { DownloadProgress, InstalledModel, ModelRef } from '@/lib/api/models'
import { formatBytes } from '@/lib/format'

interface ModelsData {
  installed: InstalledModel[]
  running: Instance[]
}

function sameModel(a: ModelRef, b: ModelRef): boolean {
  return a.repo === b.repo && a.quant === b.quant
}

/** Downloaded models and the instances serving them, PLAN sections 2.2 and 2.4. */
export default function Models() {
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<CmdError | null>(null)
  const [chat, setChat] = useState<ModelRef | null>(null)
  const [downloads, setDownloads] = useState<Record<string, DownloadProgress>>({})

  // A download started in the Store keeps reporting here, so the page that owns
  // the model folder shows what is landing in it.
  useEffect(() => {
    let stop: (() => void) | null = null
    let live = true
    void models
      .onDownloadProgress((progress) => {
        setDownloads((current) => ({
          ...current,
          [`${progress.repo}-${progress.quant}`]: progress,
        }))
        const finished =
          progress.totalBytes !== null && progress.downloadedBytes >= progress.totalBytes
        if (finished) data.reload()
      })
      .then((unlisten) => {
        if (live) stop = unlisten
        else unlisten()
      })
    return () => {
      live = false
      stop?.()
    }
    // `data.reload` is stable for the life of the hook.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const data = useAsync<ModelsData>(async () => {
    const [installed, running] = await Promise.all([models.installed(), instances.list()])
    return { installed, running }
  }, [])

  // A pull an agent started and the idle reaper stopping an instance both
  // change this page while nobody clicks anything (issue #16).
  useLiveRefresh(data.refresh)

  // Below `data`, which the filter reads: at the top of the component it was a
  // use before declaration that threw as soon as one download was in flight.
  const landing = Object.entries(downloads).filter(
    ([key]) =>
      data.state.status !== 'ready' ||
      !data.state.data.installed.some((item) => `${item.model.repo}-${item.model.quant}` === key),
  )

  async function toggle(model: ModelRef, running: boolean) {
    setBusy(`${model.repo}-${model.quant}`)
    setActionError(null)
    try {
      if (running) await instances.release(UI_LEASE, model)
      else await instances.acquire(UI_LEASE, model)
      data.reload()
    } catch (caught: unknown) {
      setActionError(toCmdError(caught))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Page
      title="Models"
      description="Downloaded GGUF files, the instance serving each one and the apps holding a lease on it."
    >
      {landing.length > 0 ? (
        <Card className="min-w-0">
          <CardHeader>
            <CardTitle className="text-sm">Downloading</CardTitle>
            <CardDescription>
              Files still streaming from Hugging Face. They appear below once the download finishes.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex min-w-0 flex-col gap-3">
            {landing.map(([key, progress]) => (
              <div key={key} className="flex min-w-0 flex-col gap-1">
                <p className="font-mono text-xs break-all">
                  {progress.repo} {progress.quant}
                </p>
                <DownloadBar progress={progress} />
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {data.state.status === 'loading' ? <LoadingState label="Reading the model folder" /> : null}

      {data.state.status === 'error' ? (
        <ErrorState error={data.state.error} what="The model list" onRetry={data.reload} />
      ) : null}

      {data.state.status === 'ready' && data.state.data.installed.length === 0 ? (
        <EmptyState
          icon={HardDrive}
          title="No model downloaded"
          description="Subscribe to a quant in the Store and it lands here."
        />
      ) : null}

      {data.state.status === 'ready' ? (
        <div className="flex min-w-0 flex-col gap-3">
          {data.state.data.installed.map((item) => {
            const instance = data.state.status === 'ready'
              ? data.state.data.running.find((candidate) => sameModel(candidate.model, item.model))
              : undefined
            const key = `${item.model.repo}-${item.model.quant}`
            return (
              <Card key={key} className="min-w-0">
                <CardHeader>
                  <CardTitle className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
                    <span className="font-mono break-all">{item.model.repo}</span>
                    <Badge variant="secondary">{item.model.quant}</Badge>
                    {item.mmproj !== null ? <Badge variant="outline">mmproj</Badge> : null}
                    <Badge variant={instance === undefined ? 'outline' : 'default'}>
                      {instance === undefined ? 'Stopped' : 'Running'}
                    </Badge>
                  </CardTitle>
                  <CardDescription className="flex flex-wrap items-center gap-x-4 gap-y-1">
                    <span>{formatBytes(item.sizeBytes)}</span>
                    {instance !== undefined ? (
                      <>
                        <span className="font-mono">127.0.0.1:{instance.port}</span>
                        <span>
                          ctx {instance.params.ctx}, {instance.params.nParallel}{' '}
                          {instance.params.nParallel === 1 ? 'slot' : 'slots'}, ngl{' '}
                          {instance.params.ngl}
                        </span>
                        <span>
                          {instance.leases.length === 0
                            ? `idle, stops in ${instances.stopsIn(instance) ?? 0} s`
                            : `leases: ${instance.leases.join(', ')}`}
                        </span>
                      </>
                    ) : null}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex min-w-0 flex-col gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      variant={instance === undefined ? 'default' : 'outline'}
                      disabled={busy === key}
                      onClick={() => void toggle(item.model, instance !== undefined)}
                    >
                      {instance === undefined ? <Play /> : <Square />}
                      {instance === undefined ? 'Run' : 'Stop'}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setChat(item.model)}>
                      <MessageSquare />
                      Chat
                    </Button>
                  </div>
                  <p className="text-muted-foreground font-mono text-xs break-all">{item.path}</p>
                  {instance !== undefined && instance.leases.some((lease) => lease !== UI_LEASE) ? (
                    <p className="text-muted-foreground text-xs">
                      Stop drops this window's lease. The instance keeps running while an app holds
                      one.
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            )
          })}
        </div>
      ) : null}

      {actionError !== null ? <ErrorState error={actionError} what="The lease change" /> : null}

      <ChatDrawer
        model={chat}
        onClose={() => {
          setChat(null)
          data.reload()
        }}
      />
    </Page>
  )
}
