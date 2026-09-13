import { invoke } from '@/lib/api/invoke'

import type { DeviceProfile } from '@/lib/api/hardware'
import type { ModelRef } from '@/lib/api/models'

export interface Params {
  ctx: number
  nParallel: number
  ngl: number
}

/**
 * One running llama-server, as instances_list reports it. The bearer token is
 * deliberately absent: it reaches the renderer through InstanceUrl alone.
 */
export interface Instance {
  model: ModelRef
  port: number
  pid: number | null
  /** Names of the apps currently holding a lease. */
  leases: string[]
  params: Params
  /** Unix seconds of the last acquire or release, the LRU key for eviction. */
  lastUsed: number
  /** Memory this instance is planned to hold, weights plus KV cache. */
  estimateMb: number
  /** Unix seconds the last lease was dropped, null while one is held. */
  idleSince: number | null
}

/**
 * The warm window of `instances::DEFAULT_IDLE_SECS`.
 *
 * `AIAS_IDLE_SECS` overrides it in the core, which a developer sets to make a
 * test finish in seconds; the window this page counts down is the default one.
 */
export const IDLE_WINDOW_SECS = 600

/**
 * Seconds a zero lease instance has left before the idle reaper stops it, or
 * null while an app still holds a lease on it.
 */
export function stopsIn(instance: Instance): number | null {
  if (instance.idleSince === null) return null
  const elapsed = Math.floor(Date.now() / 1000) - instance.idleSince
  return Math.max(0, IDLE_WINDOW_SECS - elapsed)
}

export interface InstanceUrl {
  /** OpenAI compatible base URL, ends in /v1. */
  baseUrl: string
  /** Value to put in the model field of the request body. */
  modelId: string
  port: number
  /** Bearer token for this instance, an app reads it as AIAS_MODEL_<ALIAS>_KEY. */
  apiKey: string
}

/** Snapshot of the running llama-server processes. */
export function list(): Promise<Instance[]> {
  return invoke<Instance[]>('instances_list')
}

/** Start the instance if needed and take a lease for an app. */
export function acquire(app: string, model: ModelRef): Promise<InstanceUrl> {
  return invoke<InstanceUrl>('instances_acquire', { app, model })
}

/** Drop an app's lease on a model. */
export function release(app: string, model: ModelRef): Promise<void> {
  return invoke<void>('instances_release', { app, model })
}

/** The context, slots and GPU layers the platform would pick for a model. */
export function paramsFor(profile: DeviceProfile, modelSizeBytes: number): Promise<Params> {
  return invoke<Params>('instances_params_for', { profile, modelSizeBytes })
}

/** Stop every running llama-server. The app calls this on exit. */
export function stopAll(): Promise<void> {
  return invoke<void>('instances_stop_all')
}
