import { invoke, isMock } from '@/lib/api/invoke'
import { onMockDownloadProgress } from '@/lib/api/mock'

export interface ModelRef {
  repo: string
  quant: string
}

/** What a manifest declares a model for, and the HF pipeline tag it filters on. */
export type ModelKind = 'llm' | 'vlm'

export interface ModelFile {
  filename: string
  /** Quant name parsed from the filename, unsloth `UD-` prefix kept. */
  quant: string
  sizeBytes: number
  /** The mmproj file that belongs with this quant, for vision models. */
  mmproj: string | null
}

export interface ModelSummary {
  repo: string
  downloads: number
  likes: number
  /** Empty for a search hit; `files()` is the authority. */
  quants: string[]
  pipelineTag: string | null
  /** True when the repo needs an accepted licence before it serves files. */
  gated: boolean
}

export interface SearchPage {
  items: ModelSummary[]
  nextCursor: string | null
}

/** A downloaded model on disk. */
export interface InstalledModel {
  model: ModelRef
  /** Absolute path of the weights file. */
  path: string
  sizeBytes: number
  /** The mmproj file downloaded with the weights, for vision models. */
  mmproj: string | null
}

export type Fit = 'ready' | 'maybe' | 'incompatible'

/**
 * Search Hugging Face for GGUF repos this runtime can serve.
 *
 * `kind` narrows to one HF pipeline tag, leave it out for both.
 */
export function search(
  query: string,
  cursor?: string,
  kind?: ModelKind,
): Promise<SearchPage> {
  return invoke<SearchPage>('models_search', { query, cursor, kind })
}

/** List the GGUF files of one repo. */
export function files(repo: string): Promise<ModelFile[]> {
  return invoke<ModelFile[]>('models_files', { repo })
}

/** Progress of one download, as `models_download` emits it while it streams. */
export interface DownloadProgress {
  repo: string
  quant: string
  downloadedBytes: number
  /** Null when the server sends no content length. */
  totalBytes: number | null
}

/** Tauri event `models_download` emits on, mirrors `DOWNLOAD_PROGRESS_EVENT`. */
export const DOWNLOAD_PROGRESS_EVENT = 'models://progress'

/**
 * Listen to download progress. Resolves with the function that stops listening.
 *
 * One event per callback of `models::download`, which is floored at 200 ms, so
 * a page can render every one of them without throttling.
 */
export async function onDownloadProgress(
  handler: (progress: DownloadProgress) => void,
): Promise<() => void> {
  if (isMock) return onMockDownloadProgress(handler)
  const { listen } = await import('@tauri-apps/api/event')
  return await listen<DownloadProgress>(DOWNLOAD_PROGRESS_EVENT, (event) =>
    handler(event.payload),
  )
}

/** Download one quant. Resolves with the path of the downloaded file. */
export function download(model: ModelRef): Promise<string> {
  return invoke<string>('models_download', { model })
}

/** The models this machine has downloaded. */
export function installed(): Promise<InstalledModel[]> {
  return invoke<InstalledModel[]>('models_installed')
}

/** Delete one downloaded model, weights and sidecars together. */
export function remove(model: ModelRef): Promise<void> {
  return invoke<void>('models_remove', { model })
}

/** Badge a file size against the device memory budget. */
export function fit(sizeBytes: number, budgetMb: number): Promise<Fit> {
  return invoke<Fit>('models_fit', { sizeBytes, budgetMb })
}
