# AI App Store plan

> One runtime contract for local AI apps on Windows AI PCs. The app declares which models it needs. The platform picks the backend for the hardware, starts the model server, hands the app a port.

This is the single living document for the product. It replaces a design doc, an architecture doc and a roadmap. When a decision changes, this file changes. Status: draft v0.1, 2026-09-11.

The previous PoC (`zyx1121/ai-app-store-poc`, Tauri + WSL2 + Ollama + HF Spaces) is retired. This plan starts from its conclusions, not its code.

## 1. Positioning

### 1.1 The problem

Device vendors hear the same complaint from every ISV: a local AI app is feasible when it targets one CUDA GPU. Supporting AMD, Intel, unified memory and NPUs on top of that means owning one machine per vendor and one build per backend. Nobody ships that.

### 1.2 What it is

A desktop application preinstalled on the AI PC. It has three roles:

- **Runtime.** At install time it detects the hardware and installs the matching `llama-server` build. It owns every model process on the machine.
- **Store.** It browses GGUF models on Hugging Face that llama.cpp can run, and browses apps published to the store. The user subscribes to both.
- **SDK.** A manifest, an OpenAI compatible model API, a shared Postgres, and an agent plugin (skill plus MCP) so a developer can build, fork and publish an app from Claude Code without knowing which GPU the user has.

### 1.3 One sentence

The app says `I need model X`; the platform says `it is at this port`.

### 1.4 Who it is for

- ISVs writing local AI apps who want one build for every AI PC.
- End users of an AI PC who want to install an AI app the way they install any app.
- Device vendors who want a story for `what runs on this hardware` that is not `bring your own CUDA`.
- Organizations (a law firm, a clinic) who want to fork a published app and make it theirs.

### 1.5 What it is not

- Not a fleet PaaS. One machine is one user. No remote control plane.
- Not a Hugging Face mirror. Models are pulled from HF on demand, only GGUF, only what llama.cpp runs.
- Not a training or fine tuning tool.
- Not a container platform. Version 1 runs apps as native processes, see decision D3.

## 2. Object model

There are five objects. New capability is added as a field on one of these, never as a sixth object.

| Object | What it is | Backed by | Lifetime |
|--------|-----------|-----------|----------|
| Runtime | The `llama-server` build chosen for this hardware | ggml-org release artifact | Installed once, updated with the store |
| Model | A GGUF file plus optional mmproj the user subscribed to | HF repo + quant, file on disk | Until unsubscribed |
| App | A git repo with an `aias.yaml` the user subscribed to | store index + local clone | Until unsubscribed |
| Instance | A running `llama-server` serving one Model on one port | native process | While any App holds a lease |
| Service | A platform provided dependency an App can declare | one native Postgres | Always running |

### 2.1 Runtime

Exactly one `llama-server` binary is installed. Selection happens once at install and can be re-run from Settings. Every Instance is this one binary with a different `--model`.

| Hardware | Backend | Release artifact | Note |
|----------|---------|------------------|------|
| NVIDIA GPU | CUDA | `win-cuda-13.3-x64` (driver 5xx and newer; a `cuda-12.4` artifact exists for older drivers) | Fastest path, no surprises |
| AMD GPU / Strix Halo | Vulkan | `win-vulkan-x64` | Decode 13 to 25% faster than HIP on gfx1151; HIP has no official Windows artifact |
| Intel Arc / Panther Lake | Vulkan | `win-vulkan-x64` | SYCL artifact exists and is faster on large models but has open TDR crash issues |
| Intel NPU (opt-in) | OpenVINO | `win-openvino-2026.3.1-x64` | Experimental: stateless only, single sequence, Q4_0 centric, manual context cap |
| No usable GPU | CPU | `win-cpu-x64` | Fallback, store marks large models incompatible |

AMD XDNA NPU has no llama.cpp backend. It is out of scope until one exists.

Second backends (whisper.cpp for speech to text, stable-diffusion.cpp for images) are the same family, the same OpenAI shaped HTTP, and the same Vulkan build story. They are version 2 and enter as a new `kind` on Model, not a new object.

### 2.2 Model

A Model is identified by `(hf_repo, quant)`. Two Apps that declare the same pair share one Instance. The store shows every GGUF repo on HF, filtered to what the Runtime can run, and marks each `ready`, `maybe` or `incompatible` against the device memory budget.

Rules:

1. GGUF only. No Ollama modelfiles, no safetensors, no OpenVINO IR.
2. Vision models pull the mmproj file with the weights.
3. Subscribing downloads. Unsubscribing deletes the file unless another App still declares it.

### 2.3 App

An App is a git repository. The store index points at repos, not images. The user's machine clones and builds locally. This is what makes fork possible: a second law firm forks the first one's repo, edits it with their agent, publishes under their own name.

An App is a frontend plus a manifest. It never bundles a model runtime, never talks to the GPU, never opens a port the platform did not give it.

### 2.4 Instance

An Instance is one `llama-server` process. The platform owns every parameter: port, context size, parallel slots, KV cache type, GPU layers, flash attention. An App cannot set any of them. This is the price of sharing; an App that needs a private configuration is not a store App.

Lifecycle is lease based: starting an App acquires a lease on each of its Models, stopping releases it. An Instance with zero leases is kept warm for an idle window then stopped. When the memory budget is exceeded the least recently used zero lease Instance is stopped first; if that is not enough, the user is asked which running App to stop.

### 2.5 Service

Version 1 has one Service: Postgres. One native `postgres.exe` runs for the whole machine. Each App that declares `services.postgres` gets one database and one role that owns only that database. The App receives a `DATABASE_URL` and nothing else. Migrations in the App's declared folder are applied by the platform, in filename order, as that role, on every start.

No other Service in version 1. Redis, object storage and vector stores are not needed when the tenancy model is `one database per app on one machine`; Postgres extensions cover vectors.

## 3. The manifest

One file, `aias.yaml`, at the repo root. Normative schema lives in [`spec/aias.schema.json`](spec/aias.schema.json) once written; this is the readable version. The full example is [`spec/aias.example.yaml`](spec/aias.example.yaml).

```yaml
# aias.yaml
name: contract-review
description: Review a contract against a firm's playbook and flag deviations.
version: 0.3.1
license: MIT
tags: [legal, documents]

runtime: node                # node | python
start: ["node", "server.js"] # run from the build output directory
build: ["bun", "install", "--frozen-lockfile", "&&", "bun", "run", "build"]
health: /api/health          # GET must return 2xx within 60 s of start

models:
  - alias: reviewer
    kind: llm
    repo: Qwen/Qwen3-14B-GGUF
    quant: [Q4_K_M, Q5_K_M]  # ordered preference, platform picks first that fits
    fallback: Qwen/Qwen3-8B-GGUF   # optional, used when nothing above fits the memory budget
  - alias: reader
    kind: vlm
    repo: Qwen/Qwen2.5-VL-7B-Instruct-GGUF
    quant: [Q4_K_M]

services:
  postgres:
    migrations: db/migrations   # *.sql applied in filename order

env:                            # static, non secret, visible in the store
  NEXT_TELEMETRY_DISABLED: "1"

secrets:                        # user is asked once on install, see section 10 for where they are kept
  - name: DOCUSIGN_API_KEY
    description: Optional. Enables sending the reviewed contract for signature.
```

What the platform injects at start, and the only way an App reaches anything:

| Variable | Value |
|----------|-------|
| `PORT` | The port the App must listen on, loopback only |
| `HOST` | `127.0.0.1`. The App must bind this address; the platform reads the listening socket back after the health check and stops an App that bound anything else |
| `HOSTNAME` | `127.0.0.1`. The same value under the name Node and Next.js read |
| `AIAS_MODEL_<ALIAS>_URL` | Base URL of the Instance, ends in `/v1`, OpenAI compatible |
| `AIAS_MODEL_<ALIAS>_ID` | The model name to put in the request body |
| `AIAS_MODEL_<ALIAS>_KEY` | Bearer token of that Instance. `llama-server` refuses a request without it, so loopback alone is not a permission |
| `DATABASE_URL` | Postgres DSN scoped to this App's database, present only when declared |
| Declared `env` and `secrets` | As written |

Rules the platform enforces on every manifest:

1. `name`, `description`, `version`, `runtime`, `start` are required.
2. `models[].alias` is unique and matches `^[a-z][a-z0-9_]*$`; it becomes the env var suffix in upper case.
3. `quant` is a list of GGUF quant names that exist in the repo at publish time; the store rejects a manifest whose files it cannot find.
4. No inference parameters. `ctx`, `n_parallel`, `temperature` defaults and similar keys are rejected at validate time. Temperature and sampling belong in the request body.
5. Dockerfile is required in the repo and must expose `$PORT`. Version 1 does not run it; it is the portability contract for Linux hosts later and is validated by a build in CI on publish.
6. Every string reaching a shell or a process argv is passed as an argv element, never interpolated.

## 4. Decisions

Each decision states what was chosen and the one reason that decided it. Rejected alternatives are listed so they are not re-proposed.

### D1. Unify at the inference API, not at the binary

The contract is an OpenAI compatible HTTP API on a loopback port. Which binary serves it is the platform's business. Rejected: shipping one runtime binary per vendor inside each App; a custom RPC.

### D2. llama.cpp is the only LLM runtime

One family of binaries, one model format, official Windows artifacts for every backend we target. Rejected: Ollama (extra layer, hides parameters, fights over the port), vLLM (Linux, one GPU vendor at a time), LM Studio and Lemonade (closed or vendor specific).

### D3. Apps run as native processes, not containers

The previous PoC's operational pain came almost entirely from WSL2: a 25 GB `vmmemWSL`, a 311 GB virtual disk that filled `C:`, port relay bound only to `[::1]`, background processes killed with the session, installers stuck on interactive prompts. On a single user AI PC, isolation guards against a buggy App, not a hostile one; per app directory, port, database and role are enough.

The platform ships `bun` and `uv`, both single executables. `runtime: node` starts under `bun` or `node`; `runtime: python` starts under `uv run`. Dockerfile stays mandatory as the portability contract and is exercised by CI at publish time. A container runtime is a future second rung, added only when a Linux host target appears.

### D4. One Postgres, one database and one role per App

Separate Postgres per App buys version independence and nothing else, at the cost of N processes, N backups and N upgrades. Database level isolation already hides data between Apps. The platform holds the superuser; Apps never see it.

### D5. The store indexes repos, builds happen locally

Fork is a product requirement, and fork needs source. Local build is slower than pulling an image but it is what makes `edit it with your agent and republish` true. The store index is a git repo of manifests pointing at git repos; publishing is a PR to the index.

### D6. The platform owns every inference parameter

Sharing one Instance across Apps is impossible if any App can set context size or slot count. Device profile decides: on a 96 GB unified Strix Halo a 14B model gets a large context and several slots; on a 16 GB Panther Lake it gets less. Apps declare the model and a quant preference, nothing more.

### D7. GGUF only

llama.cpp, whisper.cpp, stable-diffusion.cpp and the OpenVINO backend all consume GGUF. One format means one subscribe flow, one disk layout, one compatibility check.

### D8. Backend selection is a fixed table, not a probe

Table in section 2.1. Vulkan is the default for every non NVIDIA GPU; SYCL and HIP are not offered until their open crash issues close. Re-selection is a Settings action, not automatic.

### D9. Tauri 2 over a Rust core, with a CLI as the second shell

All logic lives in one Rust library crate, `aias-core`. The Tauri app and a headless `aias` CLI are thin shells over it. Rust because version 1 is mostly Windows process control, a single binary with no runtime to install, and the same code has to drive `llama-server`, Postgres, `bun` and `uv`. The CLI because every module can then be verified over SSH on the target machines without the GUI, and because it is the surface the MCP server in section 6.1 grows out of.

The frontend is Vite, React 19, TypeScript, Tailwind v4 and shadcn on the base-nova style with the neutral palette. Bun is the package manager. Rejected: Electron (a second runtime to ship and update), a web UI over a local daemon (two processes and a port to defend for no gain on a single user machine).

### D10. One process owns the model manager

The running Tauri app is the only owner of Instances, leases and app processes. The `aias` CLI keeps its own in-process registry and is a development aid: it can start, verify and stop things inside one invocation, but it does not see what the app started. Nothing writes shared registry files. When the developer plugin in section 6.1 arrives, it talks to the running app over a loopback MCP endpoint, so there is still exactly one owner. Rejected: a registry file on disk shared by several processes (stale pids, two owners racing over the memory budget).

### D11. The agent reaches the platform through a stdio bridge, not a second owner

`aias mcp` is a stateless stdio MCP server that forwards to the running app's loopback API with a bearer token. Rejected: an MCP endpoint served directly over HTTP from the app (a token in every agent's config file, and a CSRF surface on a loopback port), and an in-process MCP server inside the CLI (a second model manager, which D10 forbids).

## 5. Model manager

The model manager is the one stateful component. Everything else is a thin client of it.

- **Port allocation.** The local API takes 40999. A fixed range, for example 41000 to 41999, split between Instances and Apps. Ports are stable per Model across restarts so URLs in App logs stay meaningful.
- **Port split.** Instances take 41000 to 41499, the one Postgres takes 41500, and Apps start at 41501.
- **Leases.** `acquire(app, model)` starts the Instance if needed and returns its URL. `release(app, model)` decrements. A crashed App releases on process exit.
- **Idle window.** A zero lease Instance is kept warm for `AIAS_IDLE_SECS` seconds, 600 by default, and one reaper task stops it once the window passes. `AIAS_IDLE_SECS=0` turns the reaper off: a zero lease Instance then stays warm until an eviction, an explicit stop or shutdown takes it, which is what a developer keeping one model loaded across runs wants. A value that is not a whole number of seconds is a typo and not a policy, so it is reported once and the 600 s default stands.
- **Budget.** `effective_memory_mb` comes from the device profile: dedicated VRAM on discrete GPUs, half of system RAM on unified memory by default, the Adrenalin Variable Graphics Memory value on Strix Halo when readable. Each Instance has an estimate: the file size plus a tenth, plus the KV cache at the chosen context, `ctx x n_parallel x 2 x n_layers x n_kv_heads x head_dim x 2 bytes`, with the layer and head counts read out of the GGUF header. Before a file is downloaded there is no header, so the cache falls back to 18 MiB per thousand tokens per billion parameters, the rate Qwen3-8B keeps at 147,456 bytes a token. One estimator serves the fit badge, the admission check and the parameter planner, and the planner steps the context and the slot count down until the estimate fits. Sum must stay under 90% of budget.
- **Eviction.** LRU among zero lease Instances first. Then ask the user. Never kill a leased Instance silently.
- **Health.** Poll `/health` on every Instance; restart once on failure, then mark the App degraded and surface it.
- **Fail fast.** If `llama-server` exits with an out of memory message during load, stop and tell the user which model and how much it needed. Do not retry into a wall.

## 6. Developer surface

### 6.1 The plugin

A Claude Code plugin `aias`, shipped from this repo under `plugin/` and listed in `.claude-plugin/marketplace.json` at the repo root, with one skill and one MCP server. The skill explains the manifest, the injected variables, the rules in section 3 and the two templates. The MCP server is `aias mcp`, a stdio process started by the agent's client; it holds no state and forwards every call to the running app over the local API below. If the app is not running, every tool returns one error that says to start AI App Store.

**Local API.** The running app listens on `http://127.0.0.1:40999/v1`, loopback only, bearer token generated on first launch and stored at `<data_dir>/api.token` (0600). The routes mirror the Tauri commands one to one, so the GUI, the CLI and the agent see the same state (D10). Long operations (`models/pull`, `apps/build`) return a job id; `GET /v1/jobs/<id>` reports progress and the result. The CLI can host the same API without the GUI (`aias api serve`), which is how headless machines and CI run it.

MCP tools, version 1:

| Tool | Does |
|------|------|
| `store_search_models` | HF GGUF search filtered to this Runtime, with fit against this device |
| `store_search_apps` | Search the app index |
| `model_pull` | Download a model, returns a job id |
| `job_status` | Progress and result of a pull or build job |
| `app_init` | Scaffold a repo from the Next.js or FastAPI template with a valid `aias.yaml` |
| `app_validate` | Run the section 3 rules against a local repo |
| `app_run` | Clone if needed, build (job), acquire leases, apply migrations, start the App, return its URL |
| `app_logs` | Tail the App's build or run log |
| `app_stop` | Release and stop |
| `app_fork` | Copy a published or installed App into a new repo with `name` changed and a fresh git history |
| `app_publish` | Check the repo is pushed and clean, validate, and open the PR against the index (through `gh` when present, otherwise return the entry and the URL) |

### 6.2 Templates

Two templates, both minimal, both already wired to `AIAS_MODEL_*_URL` with the OpenAI SDK and to `DATABASE_URL` with a migrations folder:

- Next.js standalone, `runtime: node`
- FastAPI plus a static frontend, `runtime: python`

Templates are the default the agent reaches for, not a requirement. Any repo that satisfies section 3 is an App.

## 7. Install flow

1. Detect: GPU vendor and model, VRAM or unified memory, NPU presence, RAM, free disk.
2. Pick backend from the section 2.1 table, show the choice, allow override.
3. Download the `llama-server` artifact, verify SHA-256 against the release, unpack to `%LOCALAPPDATA%\aias\runtime\<tag>`.
4. Download and unpack Postgres, `bun`, `uv` to sibling folders. Initialize the Postgres cluster bound to loopback with a generated superuser password in Credential Manager.
5. Register the platform as a user level autostart. No admin rights after this point.
6. On Strix Halo, show the Variable Graphics Memory value read from the registry and link to how to raise it.

No WSL, no Docker, no reboot.

## 8. Milestones

Each milestone closes when its acceptance sentence is true on both target machines.

| # | Name | Acceptance | Status |
|---|------|-----------|--------|
| M1 | Runtime | A fresh Windows machine installs, picks the right backend, and `llama-server` answers `/v1/chat/completions` for one downloaded GGUF | Verified 2026-09-12 on the NVIDIA reference box (RTX 3080, 10 GB VRAM, 62 GB RAM): detection reads NVIDIA, the table picks CUDA, `aias runtime install` unpacks `win-cuda-13.3-x64` from llama.cpp `b10905`, and Qwen3-8B-GGUF Q4_K_M serves at 122.5 tok/s prompt eval and 119.1 tok/s generation with `ctx 8192`, 1 slot, `ngl 999`. Not run yet on Strix Halo or Panther Lake |
| M2 | Models | The Models page lists HF GGUF repos with fit badges, subscribes, and shows the running Instance with memory used | Verified 2026-09-12 on the same box: Store search returns HF GGUF repos with `ready`, `maybe` and `incompatible` badges, `models pull` downloads Qwen3-8B Q4_K_M and SmolVLM-500M Q8_0 with its mmproj, and the Models page shows the Instance on `127.0.0.1:41444` with its parameters and its lease. Memory used is not on the page yet, only the file size |
| M3 | Apps | A template App from the store index clones, builds, gets its lease and `DATABASE_URL`, runs, opens in the browser; two Apps declaring the same Model share one Instance | Verified 2026-09-12 on the same box: `aias apps subscribe` clones and builds `aias-example-chat`, the platform provisions `app_example_chat`, applies `0001_init.sql`, starts the App on 41501 and it streams a Qwen3-8B answer; a fork named `example-chat-2` runs beside it on 41502 with its own database and both hold a lease on one Instance. The App was subscribed from its git URL, so the store index path is still untested |
| M4 | SDK | From a bare Claude Code with the plugin, an agent runs `app.init`, edits, `app.run`, `app.publish`, and the App appears in the store on another machine | Verified 2026-09-12 on the Linux dev VM (Ubuntu 26.04, CPU only, 8 GB), from a bare Claude Code with `aias@ai-app-store` installed from the marketplace and `aias mcp` found on PATH: three headless agent runs drove the platform through the MCP tools. Run 1 called `app_init`, edited `app/page.tsx` and `aias.yaml`, `app_validate`, `app_run`, then `model_pull` and `job_status` for the model `app_run` reported missing, then `app_run` again, and `GET /api/health` on the port it returned answered 200. Run 2 called `app_publish`, which refused the uncommitted tree and then the missing `origin`; the agent committed and reported what a human has to do. The VM has no `gh`, so the degraded publish path was checked separately on a pushed HTTPS clone and it returns the index entry YAML plus the URL to paste it into. The store hop is verified as far as a build on 2026-09-13: a fourth agent run on the same VM called `app_init`, edited `aias.yaml` and `app/page.tsx`, `app_validate`, pushed `zyx1121/aias-agent-demo` with `gh`, and `app_publish` opened zyx1121/aias-index#1, which passed the index CI and was squash merged, after which the NVIDIA reference box read `agent-demo` out of `aias apps index`, cloned and validated it with `aias apps subscribe`, built it with `aias apps build`, and `aias apps missing` named the one model left to download. The published app was never started on that machine, the model was still missing there, and only the first publish of a name was exercised: publishing the same app twice pushed the same branch and git refused it as a non fast forward until 2026-09-13, and that fix has a regression test but has not been rerun through an agent |
| M5 | Fork | A second user forks a published App through the agent, changes the prompt and the schema, publishes, and both versions run side by side | Partially verified 2026-09-13: the fork through the agent and the side by side run are verified on the Linux dev VM (Ubuntu 26.04, CPU only, 8 GB), with the same account and a heading change only. A fifth agent run found the published `agent-demo` through `store_search_apps`, ran it from the store entry after `model_pull` of Qwen3-0.6B-GGUF Q8_0, forked it into `agent-demo-b`, changed the heading and ran that too, and both answered `/api/health` with 200 on 41501 and 41502 while `GET /v1/instances` showed one `llama-server` on 41029 holding the two leases. The fork's own publish, a schema change and a second GitHub account were not exercised, and the fork path of `publish`, the one a user who does not own the index takes, has only a shim test. Run 3 called `store_search_apps`, then `app_fork` on the index URL of `example-chat` into `law-firm-b`, rewrote the system prompt and the title, `app_validate` and `app_run`: it came up on 41502 beside the first App on 41501, and one Qwen3-0.6B-GGUF Q8_0 Instance held both leases |

Target machines: one AMD Strix Halo box (128 GB unified), one Intel Panther Lake box (32 GB shared), plus one NVIDIA reference box.

## 9. Known hardware notes

Kept here so they land in the installer, not in someone's memory.

- Strix Halo on Windows: usable GPU memory is set by the Adrenalin Variable Graphics Memory slider, up to about 96 GB. BIOS GPU allocation left on Auto makes some tools misreport VRAM.
- Intel Arc Vulkan: `VK_KHR_cooperative_matrix` triggers GPU timeouts on recent drivers with quantized models; the installer should disable it by default and expose a toggle.
- Intel Arc iGPU: memory accounting bugs in the Vulkan backend can under report OOM. Budget conservatively, verify by loading.
- Throughput numbers from the web vary too much to plan on. Run `llama-bench` on both machines during M1 and record the numbers in this file.
- Native services bound to a port trigger the Windows Firewall prompt unless bound to `127.0.0.1`. Every process the platform starts binds loopback.

## 10. Open questions

- Store index hosting: a public GitHub repo of manifests is enough for M4; who owns it after handoff is not decided.
- Secrets storage: today a JSON file per App under the platform data directory, `secrets/<app>.json`, owner only (`0600`, and an ACL with inheritance broken on Windows). Windows Credential Manager is the target for version 1 and is not built yet. Whether the store ever syncs secrets between machines is open and defaults to no.
- Signing: Apps are source, so signing is a git signature on the index commit, not on a binary. Whether the store requires it is open.
- Model updates: when an HF repo republishes a quant under the same filename, do subscribed users update automatically. Default no, badge yes.
