import { useCallback, useEffect, useRef, useState } from 'react'
import { Boxes, Download, ExternalLink, Heart, PackageSearch, Search } from 'lucide-react'

import { DownloadBar } from '@/components/download-bar'
import { Page } from '@/components/page'
import { EmptyState, ErrorState, LoadingState } from '@/components/state'
import { FitBadge } from '@/components/fit-badge'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAsync } from '@/hooks/use-async'
import * as apps from '@/lib/api/apps'
import { toCmdError, type CmdError } from '@/lib/api/error'
import * as hardware from '@/lib/api/hardware'
import * as models from '@/lib/api/models'
import type { DownloadProgress, Fit, ModelFile, ModelSummary } from '@/lib/api/models'
import { formatBytes, formatCount } from '@/lib/format'
import { effectiveMemoryMb } from '@/lib/memory'
import { openExternal } from '@/lib/open'

type KindFilter = 'all' | 'llm' | 'vlm'

const KIND_LABEL: Record<KindFilter, string> = { all: 'All', llm: 'LLM', vlm: 'VLM' }

interface QuantRow {
  quant: string
  file: ModelFile
  fit: Fit | null
}

/** A search hit with its files, their fit and the quant the store recommends. */
interface StoreModel {
  summary: ModelSummary
  rows: QuantRow[]
  kind: 'llm' | 'vlm'
  /** Fit of the smallest quant, the best this device can do with the repo. */
  best: QuantRow | null
}

function quantOf(summary: ModelSummary, file: ModelFile): string {
  const hit = summary.quants.find((quant) => file.filename.includes(quant))
  return hit ?? file.filename
}

/**
 * Attach the files of one repo and badge them.
 *
 * The best badge is the smallest quant that fits: file size is monotonic in the
 * badge, so the smallest file is the repo's best case on this device.
 */
async function enrich(summary: ModelSummary, budgetMb: number | null): Promise<StoreModel> {
  let files: ModelFile[] = []
  try {
    files = await models.files(summary.repo)
  } catch {
    files = []
  }

  const sorted = [...files].sort((a, b) => a.sizeBytes - b.sizeBytes)
  const rows: QuantRow[] = await Promise.all(
    sorted.map(async (file) => ({
      quant: quantOf(summary, file),
      file,
      fit: budgetMb === null ? null : await models.fit(file.sizeBytes, budgetMb),
    })),
  )

  const fitting = rows.filter((row) => row.fit !== null && row.fit !== 'incompatible')
  const best = fitting[0] ?? rows[0] ?? null

  return {
    summary,
    rows,
    kind: files.some((file) => file.mmproj !== null) ? 'vlm' : 'llm',
    best,
  }
}

interface SearchState {
  items: StoreModel[]
  cursor: string | null
  done: boolean
  loading: boolean
  error: CmdError | null
}

const EMPTY_SEARCH: SearchState = {
  items: [],
  cursor: null,
  done: false,
  loading: true,
  error: null,
}

/** The app index, PLAN section 2.3, plus the GGUF catalogue of section 2.2. */
export default function Store() {
  return (
    <Page
      title="Store"
      description="Search GGUF repos on Hugging Face with a fit badge against this device, and browse the apps published to the index."
    >
      <Tabs defaultValue="models" className="min-w-0 gap-4">
        <TabsList>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="apps">Apps</TabsTrigger>
        </TabsList>
        <TabsContent value="models">
          <ModelsTab />
        </TabsContent>
        <TabsContent value="apps">
          <AppsTab />
        </TabsContent>
      </Tabs>
    </Page>
  )
}

function ModelsTab() {
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [kind, setKind] = useState<KindFilter>('all')
  const [search, setSearch] = useState<SearchState>(EMPTY_SEARCH)
  const [picked, setPicked] = useState<StoreModel | null>(null)
  const sentinel = useRef<HTMLDivElement | null>(null)

  const device = useAsync(() => hardware.detect(), [])
  const budgetMb =
    device.state.status === 'ready' ? effectiveMemoryMb(device.state.data) : null
  const budgetKnown = device.state.status !== 'loading'

  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(query), 300)
    return () => window.clearTimeout(handle)
  }, [query])

  const load = useCallback(
    async (cursor: string | null, reset: boolean) => {
      setSearch((current) =>
        reset ? { ...EMPTY_SEARCH } : { ...current, loading: true, error: null },
      )
      try {
        const page = await models.search(debounced, cursor ?? undefined)
        const enriched = await Promise.all(page.items.map((item) => enrich(item, budgetMb)))
        setSearch((current) => ({
          items: reset ? enriched : [...current.items, ...enriched],
          cursor: page.nextCursor,
          done: page.nextCursor === null,
          loading: false,
          error: null,
        }))
      } catch (error: unknown) {
        setSearch((current) => ({
          ...current,
          items: reset ? [] : current.items,
          loading: false,
          done: true,
          error: toCmdError(error),
        }))
      }
    },
    [debounced, budgetMb],
  )

  // Wait for the device profile: the badges are computed against its budget.
  useEffect(() => {
    if (!budgetKnown) return
    void load(null, true)
  }, [load, budgetKnown])

  // The sentinel is keyed on the item count below, so this re-attaches to the
  // fresh node every time a page lands and can fire again for the next one.
  useEffect(() => {
    const node = sentinel.current
    if (node === null || search.done || search.loading) return
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting === true) void load(search.cursor, false)
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [load, search.cursor, search.done, search.loading, search.items.length])

  const visible = search.items.filter((item) => kind === 'all' || item.kind === kind)

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-48 flex-1">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search GGUF repos"
            className="pl-8"
            aria-label="Search GGUF repos"
          />
        </div>
        <Select value={kind} onValueChange={(value: unknown) => setKind(value as KindFilter)}>
          <SelectTrigger className="w-32" aria-label="Filter by kind">
            <SelectValue>{(value: unknown) => KIND_LABEL[value as KindFilter]}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {(['all', 'llm', 'vlm'] as const).map((value) => (
              <SelectItem key={value} value={value}>
                {KIND_LABEL[value]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {device.state.status === 'error' ? (
        <p className="text-muted-foreground text-xs">
          Fit badges need the device profile, and hardware_detect answered {device.state.error.code}.
          The list below is unbadged.
        </p>
      ) : null}

      {search.error !== null && search.items.length === 0 ? (
        <ErrorState
          error={search.error}
          what="Model search"
          onRetry={() => void load(null, true)}
        />
      ) : null}

      {search.error === null && search.loading && search.items.length === 0 ? (
        <LoadingState label="Searching Hugging Face" />
      ) : null}

      {search.error === null && !search.loading && visible.length === 0 ? (
        <EmptyState
          icon={PackageSearch}
          title="No models match"
          description={
            kind === 'all'
              ? 'No GGUF repo on Hugging Face matched this query.'
              : `No ${kind.toUpperCase()} repo matched this query.`
          }
        />
      ) : null}

      <div className="grid min-w-0 gap-3">
        {visible.map((item) => (
          <ModelCard key={item.summary.repo} item={item} onSubscribe={() => setPicked(item)} />
        ))}
      </div>

      {search.loading && search.items.length > 0 ? (
        <div className="text-muted-foreground flex items-center justify-center gap-2 py-2 text-xs">
          <Spinner className="size-3" />
          Loading more
        </div>
      ) : null}

      {search.done && search.items.length > 0 ? (
        <p className="text-muted-foreground py-2 text-center text-xs">
          End of results, {search.items.length} repos
        </p>
      ) : (
        <div key={search.items.length} ref={sentinel} className="h-8" aria-hidden />
      )}

      <QuantSheet item={picked} onClose={() => setPicked(null)} />
    </div>
  )
}

function ModelCard({ item, onSubscribe }: { item: StoreModel; onSubscribe: () => void }) {
  const { summary, best } = item
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
          <span className="font-mono break-all">{summary.repo}</span>
          <Badge variant="outline">{item.kind.toUpperCase()}</Badge>
          {best?.fit != null ? <FitBadge fit={best.fit} /> : null}
        </CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="inline-flex items-center gap-1">
            <Download className="size-3" />
            {formatCount(summary.downloads)}
          </span>
          <span className="inline-flex items-center gap-1">
            <Heart className="size-3" />
            {formatCount(summary.likes)}
          </span>
          <span>
            {summary.quants.length} quants
            {best !== null ? `, best ${best.quant} at ${formatBytes(best.file.sizeBytes)}` : ''}
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={onSubscribe}>
          Subscribe
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void openExternal(`https://huggingface.co/${summary.repo}`)}
        >
          <ExternalLink />
          Hugging Face
        </Button>
      </CardContent>
    </Card>
  )
}

function QuantSheet({ item, onClose }: { item: StoreModel | null; onClose: () => void }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [done, setDone] = useState<Record<string, string>>({})
  const [error, setError] = useState<CmdError | null>(null)
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({})

  // One listener for the sheet, not one per row: the payload names the quant.
  useEffect(() => {
    let stop: (() => void) | null = null
    let live = true
    void models
      .onDownloadProgress((event) => {
        if (item !== null && event.repo !== item.summary.repo) return
        setProgress((current) => ({ ...current, [event.quant]: event }))
      })
      .then((unlisten) => {
        if (live) stop = unlisten
        else unlisten()
      })
    return () => {
      live = false
      stop?.()
    }
  }, [item])

  async function subscribe(quant: string) {
    if (item === null) return
    setBusy(quant)
    setError(null)
    try {
      const path = await models.download({ repo: item.summary.repo, quant })
      setDone((current) => ({ ...current, [quant]: path }))
    } catch (caught: unknown) {
      setError(toCmdError(caught))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Sheet open={item !== null} onOpenChange={(open) => (open ? null : onClose())}>
      <SheetContent className="w-full gap-0 overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="font-mono text-sm break-all">
            {item?.summary.repo ?? ''}
          </SheetTitle>
          <SheetDescription>
            Pick the quant to download. Subscribing pulls the file, and the mmproj next to it for a
            vision model.
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-2 p-4">
          {item?.rows.length === 0 ? (
            <p className="text-muted-foreground text-sm">
              The repo file list is unavailable, so there is nothing to pick.
            </p>
          ) : null}

          {item?.rows.map((row) => (
            <div
              key={row.quant}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3"
            >
              <div className="min-w-0">
                <p className="font-mono text-xs break-all">{row.file.filename}</p>
                <p className="text-muted-foreground text-xs">
                  {formatBytes(row.file.sizeBytes)}
                  {row.file.mmproj !== null ? ` plus ${row.file.mmproj}` : ''}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {row.fit !== null ? <FitBadge fit={row.fit} /> : null}
                <Button
                  size="sm"
                  variant={row.fit === 'incompatible' ? 'outline' : 'default'}
                  disabled={busy !== null}
                  onClick={() => void subscribe(row.quant)}
                >
                  {done[row.quant] !== undefined
                    ? 'Downloaded'
                    : busy === row.quant
                      ? 'Downloading'
                      : 'Subscribe'}
                </Button>
              </div>
              {busy === row.quant && progress[row.quant] !== undefined ? (
                <DownloadBar progress={progress[row.quant]} />
              ) : null}
            </div>
          ))}

          {error !== null ? <ErrorState error={error} what="Download" /> : null}
        </div>
      </SheetContent>
    </Sheet>
  )
}

/**
 * What the user is asked to agree to before anything from a repo runs.
 *
 * A subscribe is two steps for a reason: the clone is inert, the build is not.
 * `build` and `start` are argv the repo author wrote and the platform runs on
 * this machine, so they are shown verbatim, with the models the app will pull
 * and whether it asks for a database, and nothing runs until Confirm.
 */
function ConsentDialog({
  pending,
  busy,
  onConfirm,
  onCancel,
}: {
  pending: { entry: apps.IndexEntry; app: apps.SubscribedApp } | null
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  if (pending === null) return null
  const { entry, app } = pending
  const manifest = app.manifest
  const argv = (command: string[]) => (command.length === 0 ? 'nothing' : command.join(' '))

  return (
    <Dialog open onOpenChange={(open) => (open ? null : onCancel())}>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Build {manifest.name}?</DialogTitle>
          <DialogDescription>
            The clone is on disk and its manifest is valid. Nothing from it has run yet. These are
            the commands this machine will run.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 px-4 text-sm">
          <Detail label="Repository" value={`${entry.repo} @ ${app.gitRef ?? entry.ref}`} mono />
          {app.sha === null ? null : <Detail label="Commit" value={app.sha} mono />}
          <Detail label="Build" value={argv(manifest.build)} mono />
          <Detail label="Start" value={argv(manifest.start)} mono />
          <Detail
            label="Models"
            value={
              manifest.models.length === 0
                ? 'none declared'
                : manifest.models
                    .map((model) => `${model.alias}: ${model.repo} ${model.quant.join(' or ')}`)
                    .join(', ')
            }
          />
          <Detail
            label="Postgres"
            value={
              manifest.services.postgres === null
                ? 'not declared'
                : `one database, migrations from ${manifest.services.postgres.migrations}`
            }
          />
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" disabled={busy} onClick={onCancel}>
            Cancel
          </Button>
          <Button size="sm" disabled={busy} onClick={onConfirm}>
            {busy ? 'Building' : 'Confirm and build'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Detail({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid gap-0.5">
      <span className="text-muted-foreground text-xs">{label}</span>
      <span className={mono ? 'min-w-0 font-mono text-xs break-all' : 'min-w-0 break-words'}>
        {value}
      </span>
    </div>
  )
}

function AppsTab() {
  const index = useAsync(() => apps.index(), [])
  const [cloned, setCloned] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<string | null>(null)
  const [pending, setPending] = useState<{
    entry: apps.IndexEntry
    app: apps.SubscribedApp
  } | null>(null)
  const [error, setError] = useState<CmdError | null>(null)

  // Step one: clone and validate. This runs nothing from the repo, so no
  // consent is needed for it, and it is what produces the manifest to show.
  async function clone(entry: apps.IndexEntry) {
    setBusy(entry.name)
    setError(null)
    try {
      setPending({ entry, app: await apps.subscribe(entry.repo, entry.ref) })
    } catch (caught: unknown) {
      setError(toCmdError(caught))
    } finally {
      setBusy(null)
    }
  }

  // Step two, and only after Confirm: run the `build` argv the dialog showed.
  async function build() {
    if (pending === null) return
    const { entry, app } = pending
    setBusy(entry.name)
    setError(null)
    try {
      await apps.build(app.name)
      setCloned((current) => ({ ...current, [entry.name]: app.dir }))
      setPending(null)
    } catch (caught: unknown) {
      setError(toCmdError(caught))
      setPending(null)
    } finally {
      setBusy(null)
    }
  }

  if (index.state.status === 'loading') return <LoadingState label="Reading the store index" />
  if (index.state.status === 'error') {
    return <ErrorState error={index.state.error} what="The store index" onRetry={index.reload} />
  }

  const entries = index.state.data
  if (entries.length === 0) {
    return (
      <EmptyState
        icon={Boxes}
        title="The index is empty"
        description="No app is published to zyx1121/aias-index yet."
      />
    )
  }

  return (
    <div className="flex min-w-0 flex-col gap-3">
      {entries.map((entry) => (
        <Card key={entry.name} className="min-w-0">
          <CardHeader>
            <CardTitle className="text-sm">{entry.name}</CardTitle>
            <CardDescription>{entry.description}</CardDescription>
          </CardHeader>
          <CardContent className="flex min-w-0 flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              {entry.tags.map((tag) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
              <span className="text-muted-foreground font-mono text-xs break-all">
                {entry.repo} @ {entry.ref}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" disabled={busy !== null} onClick={() => void clone(entry)}>
                {cloned[entry.name] !== undefined
                  ? 'Subscribed'
                  : busy === entry.name
                    ? 'Cloning'
                    : 'Subscribe'}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => void openExternal(entry.repo)}>
                <ExternalLink />
                Repository
              </Button>
            </div>
            {cloned[entry.name] !== undefined ? (
              <p className="text-muted-foreground font-mono text-xs break-all">
                {cloned[entry.name]}
              </p>
            ) : null}
          </CardContent>
        </Card>
      ))}
      <ConsentDialog
        pending={pending}
        busy={busy !== null}
        onConfirm={() => void build()}
        onCancel={() => setPending(null)}
      />
      {error !== null ? <ErrorState error={error} what="Subscribe" /> : null}
    </div>
  )
}

