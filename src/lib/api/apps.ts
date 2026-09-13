import { invoke } from '@/lib/api/invoke'

import type { ModelRef } from '@/lib/api/models'

export type AppRuntime = 'node' | 'python'

export type ModelKind = 'llm' | 'vlm'

export interface ModelDecl {
  alias: string
  kind: ModelKind
  repo: string
  /** Ordered preference, the platform picks the first that fits. */
  quant: string[]
  fallback: string | null
}

export interface PostgresService {
  migrations: string
}

export interface Services {
  postgres: PostgresService | null
}

export interface Secret {
  name: string
  description: string | null
  required: boolean
}

export interface Manifest {
  name: string
  description: string
  version: string
  license: string | null
  homepage: string | null
  tags: string[]
  runtime: AppRuntime
  build: string[]
  start: string[]
  health: string | null
  models: ModelDecl[]
  services: Services
  env: Record<string, string>
  secrets: Secret[]
}

export interface AppProcess {
  name: string
  port: number
  url: string
  pid: number | null
}

/**
 * Check a subscribed app against the PLAN section 3 rules.
 *
 * Apps are named, never pointed at: the directory is resolved in Rust under the
 * data directory, so this cannot reach a path of the renderer's choosing.
 */
export function validate(appName: string): Promise<Manifest> {
  return invoke<Manifest>('apps_validate', { appName })
}

/** Clone an app repo at a branch or tag. Resolves with the local directory. */
export function clone(repoUrl: string, gitRef: string | null = null): Promise<string> {
  return invoke<string>('apps_clone', { repoUrl, gitRef })
}

/** Run the manifest build commands in the app directory. */
export function build(appName: string): Promise<void> {
  return invoke<void>('apps_build', { appName })
}

/** Start the app with its injected environment and wait for health. */
export function start(appName: string): Promise<AppProcess> {
  return invoke<AppProcess>('apps_start', { appName })
}

/** Stop the app and release every lease it holds. */
export function stop(appName: string): Promise<void> {
  return invoke<void>('apps_stop', { appName })
}

/** One entry of the store index, `apps.yaml` in `zyx1121/aias-index`. */
export interface IndexEntry {
  name: string
  /** HTTPS git URL the machine clones and builds locally. */
  repo: string
  /** Branch or tag to clone. */
  ref: string
  description: string
  tags: string[]
  homepage: string | null
}

/** A subscribed app: the clone on disk with its manifest and its origin. */
export interface SubscribedApp {
  /** Manifest name, not the directory name. */
  name: string
  dir: string
  manifest: Manifest
  /** Where the clone came from, null for a directory put there by hand. */
  repoUrl: string | null
  /** Branch or tag the index asked for, null when it asked for nothing. */
  gitRef: string | null
  /** Commit the working tree is at. This is what actually ran. */
  sha: string | null
}

/** A subscribed app plus the process, when it is running. */
export interface InstalledApp extends SubscribedApp {
  process: AppProcess | null
}

/** Which of an app's two logs to read. */
export type LogKind = 'build' | 'run'

/** The published apps in the store index. */
export function index(): Promise<IndexEntry[]> {
  return invoke<IndexEntry[]>('apps_index')
}

/** The apps this machine is subscribed to. */
export function installed(): Promise<InstalledApp[]> {
  return invoke<InstalledApp[]>('apps_installed')
}

/**
 * Clone and validate one app repo. Nothing from the repo is run.
 *
 * The build is a separate call on purpose: `build` and `start` are argv the
 * repo author wrote, so the store shows them and asks before `build` runs.
 */
export function subscribe(repoUrl: string, gitRef: string | null = null): Promise<SubscribedApp> {
  return invoke<SubscribedApp>('apps_subscribe', { repoUrl, gitRef })
}

/** Delete a subscribed app's clone. It must not be running. */
export function remove(appName: string): Promise<void> {
  return invoke<void>('apps_remove', { appName })
}

/** The last `tailLines` lines of an app's build or run log. */
export function logs(appName: string, kind: LogKind, tailLines: number): Promise<string> {
  return invoke<string>('apps_logs', { appName, kind, tailLines })
}

/** The models an app declares that are not downloaded yet. */
export function missingModels(appName: string): Promise<ModelRef[]> {
  return invoke<ModelRef[]>('apps_missing_models', { appName })
}
