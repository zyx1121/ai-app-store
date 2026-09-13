/**
 * Fixtures for `VITE_AIAS_MOCK=1`.
 *
 * The Rust commands are real, but they need a Windows box with a GPU, a
 * runtime and downloaded weights. This module answers the same shapes from one
 * such machine instead: an RTX 3080 box with a CUDA runtime, two downloaded
 * models, one running instance with two leases and the `example-chat` entry of
 * the store index, so the pages can be built and reviewed anywhere.
 *
 * Every value here mirrors a struct in `crates/core/src/`. It is a fixture, not
 * a second implementation: the arithmetic it repeats (`fit`, `params_for`) is
 * copied from the core so the pages can be reviewed against real numbers.
 */
import type { IndexEntry, InstalledApp, LogKind, Manifest } from '@/lib/api/apps'
import type { CmdError } from '@/lib/api/error'
import type { DeviceProfile } from '@/lib/api/hardware'
import type { Instance, InstanceUrl, Params } from '@/lib/api/instances'
import type { DownloadProgress, Fit, InstalledModel, ModelFile, ModelRef, ModelSummary, SearchPage } from '@/lib/api/models'
import type { Backend, RuntimeInstall } from '@/lib/api/runtime'
import type { InvokeArgs } from '@/lib/api/invoke'

const PROFILE: DeviceProfile = {
  gpuVendor: 'nvidia',
  gpuName: 'NVIDIA GeForce RTX 3080',
  vramMb: 10240,
  totalRamMb: 63488,
  memoryModel: 'dedicated',
  hasNpu: false,
}

const RUNTIME_INSTALL: RuntimeInstall = {
  backend: 'cuda',
  tag: 'b10905',
  dir: 'C:\\Users\\demo\\AppData\\Local\\aias\\runtime\\b10905',
}

/** Bits per weight of each quant, used to size the fixture files. */
const BITS_PER_WEIGHT: Record<string, number> = {
  Q2_K: 3.35,
  Q3_K_M: 3.91,
  Q4_K_M: 4.83,
  Q5_K_M: 5.67,
  Q6_K: 6.56,
  Q8_0: 8.5,
  F16: 16,
}

interface RepoFixture {
  repo: string
  downloads: number
  likes: number
  quants: string[]
  /** Billions of parameters, the file size follows from it. */
  params: number
  /** Vision models carry an mmproj file next to every quant. */
  mmproj: string | null
}

const REPOS: RepoFixture[] = [
  { repo: 'Qwen/Qwen3-8B-GGUF', downloads: 412_338, likes: 1_204, quants: ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'], params: 8.19, mmproj: null },
  { repo: 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', downloads: 1_284_907, likes: 2_871, quants: ['Q3_K_M', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'], params: 8.03, mmproj: null },
  { repo: 'Qwen/Qwen3-4B-GGUF', downloads: 238_115, likes: 612, quants: ['Q4_K_M', 'Q5_K_M', 'Q8_0'], params: 4.02, mmproj: null },
  { repo: 'ggml-org/gemma-3-4b-it-GGUF', downloads: 96_442, likes: 389, quants: ['Q4_K_M', 'Q8_0'], params: 4.3, mmproj: 'mmproj-gemma-3-4b-it-f16.gguf' },
  { repo: 'Qwen/Qwen2.5-VL-7B-Instruct-GGUF', downloads: 141_770, likes: 733, quants: ['Q4_K_M', 'Q5_K_M', 'Q8_0'], params: 7.62, mmproj: 'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf' },
  { repo: 'bartowski/Mistral-7B-Instruct-v0.3-GGUF', downloads: 874_201, likes: 1_542, quants: ['Q3_K_M', 'Q4_K_M', 'Q5_K_M', 'Q6_K'], params: 7.25, mmproj: null },
  { repo: 'Qwen/Qwen3-14B-GGUF', downloads: 187_904, likes: 841, quants: ['Q4_K_M', 'Q5_K_M', 'Q6_K'], params: 14.8, mmproj: null },
  { repo: 'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF', downloads: 322_558, likes: 967, quants: ['Q4_K_M', 'Q5_K_M', 'Q8_0'], params: 7.62, mmproj: null },
  { repo: 'unsloth/gemma-3-12b-it-GGUF', downloads: 154_330, likes: 508, quants: ['Q3_K_M', 'Q4_K_M', 'Q5_K_M'], params: 12.2, mmproj: 'mmproj-gemma-3-12b-it-f16.gguf' },
  { repo: 'unsloth/Llama-3.2-3B-Instruct-GGUF', downloads: 511_066, likes: 1_118, quants: ['Q4_K_M', 'Q6_K', 'Q8_0'], params: 3.21, mmproj: null },
  { repo: 'Qwen/Qwen3-32B-GGUF', downloads: 98_771, likes: 726, quants: ['Q4_K_M', 'Q5_K_M', 'Q6_K'], params: 32.8, mmproj: null },
  { repo: 'Qwen/Qwen3-0.6B-GGUF', downloads: 205_419, likes: 344, quants: ['Q4_K_M', 'Q8_0'], params: 0.75, mmproj: null },
]

const PAGE_SIZE = 6

function shortName(repo: string): string {
  const tail = repo.split('/')[1] ?? repo
  return tail.replace(/-GGUF$/, '')
}

function fileSizeBytes(fixture: RepoFixture, quant: string): number {
  const bpw = BITS_PER_WEIGHT[quant] ?? 4.83
  return Math.round((fixture.params * 1e9 * bpw) / 8)
}

function filesOf(fixture: RepoFixture): ModelFile[] {
  return fixture.quants.map((quant) => ({
    filename: `${shortName(fixture.repo)}-${quant}.gguf`,
    quant,
    sizeBytes: fileSizeBytes(fixture, quant),
    mmproj: fixture.mmproj,
  }))
}

function summaryOf(fixture: RepoFixture): ModelSummary {
  return {
    repo: fixture.repo,
    downloads: fixture.downloads,
    likes: fixture.likes,
    quants: fixture.quants,
    pipelineTag: fixture.mmproj === null ? 'text-generation' : 'image-text-to-text',
    gated: false,
  }
}

/** The ladder in `models::PLAN_LADDER`, most generous rung first. */
const PLAN_LADDER: [number, number][] = [
  [32768, 4],
  [32768, 2],
  [16384, 2],
  [16384, 1],
  [8192, 1],
  [4096, 1],
  [2048, 1],
]

/**
 * Same arithmetic as `models::estimate_mb`, with no GGUF header to read.
 *
 * Weights are the file size plus a tenth, the cache is 18 MiB per thousand
 * tokens per billion parameters, and a billion parameters is taken as 0.6 GB on
 * disk.
 */
function estimateMb(sizeBytes: number, ctx: number, nParallel: number): number {
  const weights = Math.floor((sizeBytes * 11) / 10)
  const tokens = ctx * Math.max(1, nParallel)
  const cache = Math.floor((tokens * sizeBytes * 18 * 1024 * 1024) / (1000 * 600 * 1024 * 1024))
  return Math.ceil((weights + cache) / (1024 * 1024))
}

/** Same ladder walk as `models::plan_context`. */
function planContext(sizeBytes: number, budgetMb: number): [number, number] {
  const ceiling = Math.floor((budgetMb * 9) / 10)
  for (const [ctx, nParallel] of PLAN_LADDER) {
    if (estimateMb(sizeBytes, ctx, nParallel) <= ceiling) return [ctx, nParallel]
  }
  return PLAN_LADDER[PLAN_LADDER.length - 1]
}

/** Same arithmetic as `models::fit` in `crates/core/src/models.rs`. */
function fit(sizeBytes: number, budgetMb: number): Fit {
  const [ctx, nParallel] = planContext(sizeBytes, budgetMb)
  const estimate = estimateMb(sizeBytes, ctx, nParallel)
  // Ready is the rung the planner stops on being a usable one: `models::FIT_MIN_CTX`
  // tokens over one slot, inside the 90% ceiling.
  if (ctx >= 4096 && nParallel >= 1 && estimate <= Math.floor((budgetMb * 9) / 10)) return 'ready'
  if (estimate <= budgetMb) return 'maybe'
  return 'incompatible'
}

/** Same plan as `instances::params_for` in `crates/core/src/instances.rs`. */
function paramsFor(profile: DeviceProfile, modelSizeBytes: number): Params {
  const budgetMb =
    profile.memoryModel === 'dedicated'
      ? (profile.vramMb ?? 0)
      : (profile.vramMb ?? Math.floor(profile.totalRamMb / 2))
  const [ctx, nParallel] = planContext(modelSizeBytes, budgetMb)
  return { ctx, nParallel, ngl: profile.gpuVendor === 'none' ? 0 : 999 }
}

const CHAT_MODEL: ModelRef = { repo: 'Qwen/Qwen3-8B-GGUF', quant: 'Q4_K_M' }
const VISION_MODEL: ModelRef = { repo: 'Qwen/Qwen2.5-VL-7B-Instruct-GGUF', quant: 'Q4_K_M' }

const INSTALLED_MODELS: InstalledModel[] = [
  {
    model: CHAT_MODEL,
    path: 'C:\\Users\\demo\\AppData\\Local\\aias\\models\\Qwen_Qwen3-8B-GGUF-Q4_K_M\\Qwen3-8B-Q4_K_M.gguf',
    sizeBytes: 4_944_000_000,
    mmproj: null,
  },
  {
    model: VISION_MODEL,
    path: 'C:\\Users\\demo\\AppData\\Local\\aias\\models\\Qwen_Qwen2.5-VL-7B-Instruct-GGUF-Q4_K_M\\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf',
    sizeBytes: 4_600_000_000,
    mmproj: 'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf',
  },
]

const EXAMPLE_CHAT_MANIFEST: Manifest = {
  name: 'example-chat',
  description: 'Chat with one local model, history in Postgres.',
  version: '0.1.0',
  license: 'MIT',
  homepage: 'https://github.com/zyx1121/aias-example-chat',
  tags: ['chat', 'example'],
  runtime: 'node',
  build: ['bun', 'run', 'aias:build'],
  start: ['node', '.next/standalone/server.js'],
  health: '/api/health',
  models: [
    {
      alias: 'chat',
      kind: 'llm',
      repo: 'Qwen/Qwen3-8B-GGUF',
      quant: ['Q4_K_M'],
      fallback: 'Qwen/Qwen3-4B-GGUF',
    },
  ],
  services: { postgres: { migrations: 'db/migrations' } },
  env: { NEXT_TELEMETRY_DISABLED: '1', HOSTNAME: '127.0.0.1' },
  secrets: [],
}

const INDEX: IndexEntry[] = [
  {
    name: 'example-chat',
    repo: 'https://github.com/zyx1121/aias-example-chat',
    ref: 'main',
    description: 'Chat with one local model, history in Postgres',
    tags: ['chat', 'example'],
    homepage: null,
  },
]

const APP_LOG: string[] = [
  '[build] bun run aias:build',
  '[build] $ next build && node scripts/standalone.mjs',
  '[build]   ▲ Next.js 15.5.4',
  '[build] Creating an optimized production build ...',
  '[build] Compiled successfully in 21.4s',
  '[build] copied .next/static and public into .next/standalone',
  '[start] PORT=41800 HOSTNAME=127.0.0.1',
  '[start] AIAS_MODEL_CHAT_URL=http://127.0.0.1:41337/v1',
  '[start] AIAS_MODEL_CHAT_ID=Qwen_Qwen3-8B-GGUF-Q4_K_M',
  '[start] DATABASE_URL=postgresql://example-chat:Hn3kQ2vTpLsd8XwR@127.0.0.1:41999/example-chat',
  '[start] migrations: applied 2 files from db/migrations',
  '[start] ready on http://127.0.0.1:41800',
  '[health] GET /api/health 200 in 41ms',
]

/** Stand in for the per instance bearer token the real manager generates. */
const MOCK_API_KEY = '0'.repeat(64)

/** Everything the fixtures let the user change from the UI. */
interface MockState {
  runtime: RuntimeInstall | null
  instances: Instance[]
  apps: InstalledApp[]
}

const state: MockState = {
  runtime: RUNTIME_INSTALL,
  instances: [
    {
      model: CHAT_MODEL,
      port: 41337,
      pid: 21480,
      leases: ['example-chat', 'contract-review'],
      params: paramsFor(PROFILE, 4_944_000_000),
      lastUsed: Math.floor(Date.now() / 1000),
      estimateMb: 5657,
      idleSince: null,
    },
  ],
  apps: [
    {
      name: 'example-chat',
      dir: 'C:\\Users\\demo\\AppData\\Local\\aias\\apps\\example-chat',
      repoUrl: 'https://github.com/zyx1121/aias-example-chat',
      gitRef: 'main',
      sha: '4c1de9a80f37b526ad1e0c94f8b2735ea6d10c8f',
      manifest: EXAMPLE_CHAT_MANIFEST,
      process: {
        name: 'example-chat',
        port: 41800,
        url: 'http://127.0.0.1:41800',
        pid: 33012,
      },
    },
  ],
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function fail(code: string, message: string): CmdError {
  return { code, message }
}

function arg<T>(args: InvokeArgs | undefined, key: string): T {
  return (args ?? {})[key] as T
}

function searchPage(query: string, cursor: string | undefined): SearchPage {
  const needle = query.trim().toLowerCase()
  const hits = needle.length === 0 ? REPOS : REPOS.filter((r) => r.repo.toLowerCase().includes(needle))
  const start = cursor === undefined || cursor === null ? 0 : Number.parseInt(cursor, 10)
  const page = hits.slice(start, start + PAGE_SIZE)
  const next = start + PAGE_SIZE
  return {
    items: page.map(summaryOf),
    nextCursor: next < hits.length ? String(next) : null,
  }
}

function instanceFor(model: ModelRef): Instance | undefined {
  return state.instances.find((i) => i.model.repo === model.repo && i.model.quant === model.quant)
}

function portFor(model: ModelRef): number {
  let hash = 2166136261
  const slug = `${model.repo.replace(/\//g, '_')}-${model.quant}`
  for (const char of slug) {
    hash ^= char.charCodeAt(0)
    hash = Math.imul(hash, 16777619) >>> 0
  }
  return 41000 + (hash % 800)
}

function acquire(app: string, model: ModelRef): InstanceUrl {
  let instance = instanceFor(model)
  if (instance === undefined) {
    const installed = INSTALLED_MODELS.find((m) => m.model.repo === model.repo && m.model.quant === model.quant)
    if (installed === undefined) throw fail('not_found', `model not downloaded: ${model.repo} ${model.quant}`)
    instance = {
      model,
      port: portFor(model),
      pid: 20000 + (portFor(model) % 9000),
      leases: [],
      params: paramsFor(PROFILE, installed.sizeBytes),
      lastUsed: Math.floor(Date.now() / 1000),
      estimateMb: Math.floor((installed.sizeBytes / (1024 * 1024)) * 1.2),
      idleSince: null,
    }
    state.instances = [...state.instances, instance]
  }
  if (!instance.leases.includes(app)) instance.leases = [...instance.leases, app]
  instance.idleSince = null
  return {
    baseUrl: `http://127.0.0.1:${instance.port}/v1`,
    modelId: `${model.repo}:${model.quant}`,
    port: instance.port,
    apiKey: MOCK_API_KEY,
  }
}

function release(app: string, model: ModelRef): void {
  const instance = instanceFor(model)
  if (instance === undefined) return
  instance.leases = instance.leases.filter((lease) => lease !== app)
  // Same as `instances::release`: a zero lease instance is kept warm and
  // stamped, and the idle reaper is what stops it later.
  if (instance.leases.length === 0 && instance.idleSince === null) {
    instance.idleSince = Math.floor(Date.now() / 1000)
  }
}

/** Callbacks of the fixture stand in for the `models://progress` Tauri event. */
const progressHandlers = new Set<(progress: DownloadProgress) => void>()

/** Steps one fixture download reports before it resolves. */
const PROGRESS_STEPS = 8

/** `models.onDownloadProgress` in mock mode. Returns the unsubscribe. */
export function onMockDownloadProgress(
  handler: (progress: DownloadProgress) => void,
): () => void {
  progressHandlers.add(handler)
  return () => {
    progressHandlers.delete(handler)
  }
}

function emitProgress(progress: DownloadProgress): void {
  for (const handler of progressHandlers) handler(progress)
}

/** The fixture answer for one command, or a rejection shaped like `CmdError`. */
export async function mockInvoke<T>(command: string, args?: InvokeArgs): Promise<T> {
  await delay(120)

  switch (command) {
    case 'hardware_detect':
      return PROFILE as T

    case 'runtime_select': {
      const profile = arg<DeviceProfile>(args, 'profile')
      const optInNpu = arg<boolean>(args, 'optInNpu')
      let backend: Backend = 'cpu'
      if (profile.gpuVendor === 'nvidia') backend = 'cuda'
      else if (profile.gpuVendor === 'amd') backend = 'vulkan'
      else if (profile.gpuVendor === 'intel') backend = optInNpu && profile.hasNpu ? 'openVino' : 'vulkan'
      return backend as T
    }

    case 'runtime_install': {
      await delay(1800)
      state.runtime = { ...RUNTIME_INSTALL, backend: arg<Backend>(args, 'backend') }
      return state.runtime as T
    }

    case 'models_search':
      return searchPage(arg<string>(args, 'query') ?? '', arg<string | undefined>(args, 'cursor')) as T

    case 'models_files': {
      const repo = arg<string>(args, 'repo')
      const fixture = REPOS.find((r) => r.repo === repo)
      if (fixture === undefined) throw fail('not_found', `repo not found: ${repo}`)
      return filesOf(fixture) as T
    }

    case 'models_fit':
      return fit(arg<number>(args, 'sizeBytes'), arg<number>(args, 'budgetMb')) as T

    case 'models_installed':
      return INSTALLED_MODELS as T

    case 'models_remove':
      return undefined as T

    case 'models_download': {
      const model = arg<ModelRef>(args, 'model')
      const fixture = REPOS.find((repo) => repo.repo === model.repo)
      const total =
        fixture === undefined ? 4_000_000_000 : fileSizeBytes(fixture, model.quant)
      // Same shape and cadence as the Tauri event, so the pages that render it
      // are reviewed against a moving bar and not a single jump to 100%.
      for (let step = 1; step <= PROGRESS_STEPS; step += 1) {
        await delay(200)
        emitProgress({
          repo: model.repo,
          quant: model.quant,
          downloadedBytes: Math.round((total * step) / PROGRESS_STEPS),
          totalBytes: total,
        })
      }
      return `C:\\Users\\demo\\AppData\\Local\\aias\\models\\${model.repo.replace(/\//g, '_')}-${model.quant}` as T
    }

    case 'instances_list':
      return state.instances.map((i) => ({ ...i })) as T

    case 'instances_acquire':
      return acquire(arg<string>(args, 'app'), arg<ModelRef>(args, 'model')) as T

    case 'instances_release':
      release(arg<string>(args, 'app'), arg<ModelRef>(args, 'model'))
      return undefined as T

    case 'instances_params_for':
      return paramsFor(arg<DeviceProfile>(args, 'profile'), arg<number>(args, 'modelSizeBytes')) as T

    case 'apps_index':
      return INDEX as T

    case 'apps_installed':
      return state.apps.map((a) => ({ ...a })) as T

    case 'apps_clone': {
      await delay(700)
      const repoUrl = arg<string>(args, 'repoUrl')
      const name = repoUrl.split('/').pop() ?? 'app'
      return `C:\\Users\\demo\\AppData\\Local\\aias\\apps\\${name.replace(/^aias-/, '')}` as T
    }

    case 'apps_validate':
      return EXAMPLE_CHAT_MANIFEST as T

    case 'apps_build':
      // The real one runs the manifest `build` argv, which is why the store
      // asks first. Here it only takes as long as one would.
      await delay(1200)
      return undefined as T

    case 'apps_logs': {
      const kind = arg<LogKind>(args, 'kind')
      const tailLines = arg<number>(args, 'tailLines') ?? 200
      const lines = APP_LOG.filter((line) =>
        kind === 'build' ? line.startsWith('[build]') : !line.startsWith('[build]'),
      )
      return lines.slice(-tailLines).join('\n') as T
    }

    case 'apps_subscribe': {
      // Clone and validate only: the build is `apps_build`, after consent.
      await delay(700)
      const repoUrl = arg<string>(args, 'repoUrl')
      const name = (repoUrl.split('/').pop() ?? 'app').replace(/^aias-/, '').replace(/\.git$/, '')
      const existing = state.apps.find((app) => app.name === name)
      if (existing !== undefined) return existing as T
      const subscribed: InstalledApp = {
        name,
        dir: `C:\\Users\\demo\\AppData\\Local\\aias\\apps\\${name}`,
        repoUrl,
        gitRef: arg<string | null>(args, 'gitRef') ?? 'main',
        sha: '9f3b1c4d2e6a70815c93ad4fe2b0c7d81a5e6f42',
        process: null,
        manifest: { ...EXAMPLE_CHAT_MANIFEST, name },
      }
      state.apps = [...state.apps, subscribed]
      return subscribed as T
    }

    case 'apps_remove': {
      const name = arg<string>(args, 'appName')
      const app = state.apps.find((candidate) => candidate.name === name)
      if (app === undefined) throw fail('not_found', `app not subscribed: ${name}`)
      if (app.process !== null) throw fail('process', `app ${name} is running, stop it first`)
      state.apps = state.apps.filter((candidate) => candidate !== app)
      return undefined as T
    }

    case 'apps_missing_models': {
      const name = arg<string>(args, 'appName')
      const app = state.apps.find((candidate) => candidate.name === name)
      if (app === undefined) throw fail('not_found', `app not subscribed: ${name}`)
      const missing = app.manifest.models
        .map((declared) => ({ repo: declared.repo, quant: declared.quant[0] }))
        .filter(
          (model) =>
            !INSTALLED_MODELS.some(
              (installed) =>
                installed.model.repo === model.repo && installed.model.quant === model.quant,
            ),
        )
      return missing as T
    }

    case 'api_token':
      return { url: 'http://127.0.0.1:40999/v1', token: 'f'.repeat(64) } as T

    case 'services_postgres_stop':
      return { dir: 'C:\\Users\\demo\\AppData\\Local\\aias\\postgres', port: 41500 } as T

    case 'instances_stop_all':
      state.instances = []
      return undefined as T

    case 'apps_start': {
      await delay(900)
      const name = arg<string>(args, 'appName')
      const app = state.apps.find((a) => a.name === name)
      if (app === undefined) throw fail('not_found', `app not subscribed: ${name}`)
      for (const declared of app.manifest.models) {
        acquire(app.name, { repo: declared.repo, quant: declared.quant[0] })
      }
      app.process = { name: app.name, port: 41800, url: 'http://127.0.0.1:41800', pid: 33012 }
      return app.process as T
    }

    case 'apps_stop': {
      const name = arg<string>(args, 'appName')
      const app = state.apps.find((a) => a.name === name)
      if (app === undefined) throw fail('not_found', `app not subscribed: ${name}`)
      for (const declared of app.manifest.models) {
        release(app.name, { repo: declared.repo, quant: declared.quant[0] })
      }
      app.process = null
      return undefined as T
    }

    default:
      throw fail('not_implemented', `no fixture for ${command}`)
  }
}
