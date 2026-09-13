/**
 * A stand in for the app's local API, so `aias mcp` can be driven end to end
 * without a GPU, a model or a desktop session.
 *
 * It answers the routes of the contract in PLAN.md section 6.1 with canned
 * data, checks the bearer token the way the real server does, and finishes a
 * job on its second poll so `app_run` walks the whole build path in a second.
 * Nothing here is a second implementation of the platform: it exists to prove
 * the bridge shapes its requests and reads its answers correctly.
 *
 * Run with `bun scripts/mock-api.ts`, then point a client at it:
 *
 *   AIAS_API_URL=http://127.0.0.1:40999/v1 AIAS_API_TOKEN=mock-token aias mcp
 */
const PORT = Number(process.env.AIAS_API_PORT ?? 40999)
const TOKEN = process.env.AIAS_API_TOKEN ?? 'mock-token'
/** An app with this name reports a model that is not downloaded. */
const APP_MISSING_MODEL = 'needs-model'
/** How many polls a job takes before it is done. */
const JOB_POLLS = 2

const polls = new Map<string, number>()

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const fail = (code: string, message: string, status: number) =>
  json({ error: { code, message } }, status)

/** Manifest name of the target, from `{name}` or from the tail of `{dir}`. */
function targetName(body: Record<string, unknown>): string {
  if (typeof body.name === 'string' && body.name.length > 0) return body.name
  if (typeof body.dir === 'string') {
    const tail = body.dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? ''
    if (/^[a-z][a-z0-9-]*$/.test(tail)) return tail
  }
  return 'demo-chat'
}

function manifest(name: string) {
  return {
    name,
    description: 'A demo app that answers with a local model.',
    version: '0.1.0',
    license: 'MIT',
    homepage: null,
    tags: ['demo'],
    runtime: 'node',
    build: ['bun', 'install', '--frozen-lockfile'],
    start: ['node', 'server.js'],
    health: '/api/health',
    models: [
      { alias: 'chat', kind: 'llm', repo: 'Qwen/Qwen3-8B-GGUF', quant: ['Q4_K_M'], fallback: null },
    ],
    services: { postgres: null },
    env: {},
    secrets: [],
  }
}

const INDEX = [
  {
    name: 'example-chat',
    repo: 'https://github.com/zyx1121/aias-example-chat',
    ref: 'main',
    description: 'A minimal chat app on one local model.',
    tags: ['demo', 'chat'],
  },
  {
    name: 'contract-review',
    repo: 'https://github.com/example-firm/contract-review',
    ref: 'main',
    description: "Review a contract against a firm's playbook and flag deviations.",
    tags: ['legal', 'documents'],
  },
]

async function route(method: string, path: string, body: Record<string, unknown>) {
  switch (`${method} ${path}`) {
    case 'GET /health':
      return json({ ok: true, version: '0.1.0', sha: 'mock' })
    case 'GET /hardware':
      return json({
        gpuVendor: 'nvidia',
        gpuName: 'NVIDIA GeForce RTX 3080',
        vramMb: 10240,
        ramMb: 63488,
        npu: false,
        effectiveMemoryMb: 10240,
      })
    case 'GET /runtime':
      return json({ backend: 'cuda', tag: 'b10905', dir: 'C:/aias/runtime/b10905' })
    case 'GET /models/search':
      return json({
        items: [
          {
            repo: 'Qwen/Qwen3-8B-GGUF',
            kind: 'llm',
            files: [{ quant: 'Q4_K_M', sizeBytes: 4_900_000_000 }],
            fit: 'ready',
          },
          {
            repo: 'Qwen/Qwen3-14B-GGUF',
            kind: 'llm',
            files: [{ quant: 'Q4_K_M', sizeBytes: 9_000_000_000 }],
            fit: 'maybe',
          },
        ],
        cursor: null,
      })
    case 'POST /models/files':
      return json([{ quant: 'Q4_K_M', sizeBytes: 4_900_000_000, mmproj: null }])
    case 'POST /models/pull':
      return json({ jobId: `pull-${body.quant ?? 'q'}` })
    case 'GET /models/installed':
      return json([
        {
          repo: 'Qwen/Qwen3-8B-GGUF',
          quant: 'Q4_K_M',
          path: 'C:/aias/models/qwen3-8b-q4_k_m.gguf',
          sizeBytes: 4_900_000_000,
        },
      ])
    case 'GET /instances':
      return json([
        { repo: 'Qwen/Qwen3-8B-GGUF', quant: 'Q4_K_M', port: 41444, leases: ['demo-chat'] },
      ])
    case 'GET /apps/index':
      return json(INDEX)
    case 'GET /apps/installed':
      return json([
        {
          name: 'demo-chat',
          dir: 'C:/aias/apps/demo-chat',
          manifest: manifest('demo-chat'),
          repoUrl: 'https://github.com/zyx1121/aias-example-chat',
          gitRef: 'main',
          sha: 'a1b2c3d',
        },
      ])
    case 'POST /apps/clone':
      return json({
        name: 'demo-chat',
        dir: 'C:/aias/apps/demo-chat',
        manifest: manifest('demo-chat'),
        repoUrl: body.url,
        gitRef: body.ref ?? 'main',
        sha: 'a1b2c3d',
      })
    case 'POST /apps/init':
      return json({ dir: body.dir, manifest: manifest(String(body.name ?? 'demo-chat')) })
    case 'POST /apps/validate':
      return json(manifest(targetName(body)))
    case 'POST /apps/build':
      return json({ jobId: `build-${targetName(body)}` })
    case 'POST /apps/start': {
      const name = targetName(body)
      return json({ name, port: 41501, url: 'http://127.0.0.1:41501', pid: 4242 })
    }
    case 'POST /apps/stop':
      return json({})
    case 'POST /apps/missing': {
      if (targetName(body) === APP_MISSING_MODEL) {
        return json([{ repo: 'Qwen/Qwen3-8B-GGUF', quant: 'Q4_K_M' }])
      }
      return json([])
    }
    case 'POST /apps/fork':
      return json({ dir: body.dir, manifest: manifest(String(body.name ?? 'demo-fork')) })
  }

  const logs = path.match(/^\/apps\/([a-z][a-z0-9-]*)\/logs$/)
  if (method === 'GET' && logs) {
    return json({ text: `[mock] ${logs[1]} started and answered its health check\n` })
  }

  const job = path.match(/^\/jobs\/(.+)$/)
  if (method === 'GET' && job) {
    const id = job[1]
    const seen = (polls.get(id) ?? 0) + 1
    polls.set(id, seen)
    if (seen < JOB_POLLS) {
      return json({
        id,
        kind: id.startsWith('pull-') ? 'pull' : 'build',
        state: 'running',
        progress: { downloadedBytes: 1_000_000, totalBytes: 4_900_000_000 },
      })
    }
    return json({ id, kind: id.startsWith('pull-') ? 'pull' : 'build', state: 'done', result: {} })
  }

  return fail('not_found', `no route for ${method} ${path}`, 404)
}

const server = Bun.serve({
  hostname: '127.0.0.1',
  port: PORT,
  async fetch(request) {
    const url = new URL(request.url)
    if (!url.pathname.startsWith('/v1')) {
      return fail('not_found', `no route for ${url.pathname}`, 404)
    }
    if (request.headers.get('authorization') !== `Bearer ${TOKEN}`) {
      return fail('unauthorized', 'the bearer token is missing or wrong', 401)
    }
    const path = url.pathname.slice('/v1'.length) || '/'
    const body =
      request.method === 'POST' ? ((await request.json().catch(() => ({}))) as Record<string, unknown>) : {}
    const response = await route(request.method, path, body)
    console.error(`${request.method} ${path} -> ${response.status}`)
    return response
  },
})

console.error(`mock API on http://127.0.0.1:${server.port}/v1, token ${TOKEN}`)
