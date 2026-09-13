---
name: aias-app
description: Build, run, fork or publish an app for the AI App Store (aias). Use when the user mentions "AI App Store", "aias", "aias.yaml", a "local AI app", an app that "needs a model on this machine", or asks to scaffold, validate, run, fork or publish one. Covers the manifest, the injected environment variables, the two templates and the aias MCP tools.
---

# AI App Store apps

The AI App Store is a desktop platform for local AI apps on an AI PC. An app is a git repo with an
`aias.yaml` manifest that declares which models it needs. The platform picks the llama.cpp backend
for the hardware, downloads the GGUF weights, runs one `llama-server` per model and hands the app a
loopback port plus an OpenAI compatible base URL. Two apps that declare the same model share one
server. The app never touches the GPU, never picks a quant at runtime and never opens a port it was
not given.

Every action goes through the `aias` MCP tools in this plugin. They forward to the running desktop
app, so the platform stays the single owner of models and processes.

## Workflow

1. `app_init` with a name, a template and a directory. This writes a valid manifest, a Dockerfile
   and working model wiring.
2. Edit the code and the manifest. Run `app_validate` after every manifest edit.
3. `app_run`. It builds, starts and returns the URL. If it answers with missing models, call
   `model_pull` for each one, poll `job_status` until every job is `done`, then call `app_run` again.
4. Test the app in the browser at the URL it returned. Read `app_logs` when something is wrong.
5. Commit and push the repo to GitHub over HTTPS, then `app_publish` with the directory. It opens a
   pull request against the store index.

## The manifest

`aias.yaml` at the repo root is the whole contract. Every key that is not marked optional is
required.

```yaml
name: contract-review        # ^[a-z][a-z0-9-]*$, becomes a directory and a database name
description: Review a contract against a firm's playbook and flag deviations.
version: 0.3.1               # SemVer
license: MIT                 # optional, SPDX id
homepage: https://github.com/example-firm/contract-review   # optional
tags: [legal, documents]     # optional, for humans searching the store

runtime: node                # node | python
build: ["bun", "install", "--frozen-lockfile", "&&", "bun", "run", "build"]   # optional, argv
start: ["node", ".next/standalone/server.js"]                                 # argv
health: /api/health          # optional, GET on http://127.0.0.1:$PORT<health> must be 2xx in 60 s

models:                      # optional, but this is the point of the platform
  - alias: reviewer          # ^[a-z][a-z0-9_]*$, becomes AIAS_MODEL_REVIEWER_URL
    kind: llm                # llm | vlm
    repo: Qwen/Qwen3-14B-GGUF          # a Hugging Face GGUF repo
    quant: [Q4_K_M, Q5_K_M]            # ordered preference, first that fits the device wins
    fallback: Qwen/Qwen3-8B-GGUF       # optional, tried when nothing above fits
  - alias: reader
    kind: vlm                # the mmproj file is pulled with the weights
    repo: Qwen/Qwen2.5-VL-7B-Instruct-GGUF
    quant: [Q4_K_M]

services:                    # optional
  postgres:
    migrations: db/migrations   # *.sql applied in filename order on every start

env:                         # optional, static, non secret, visible in the store
  NEXT_TELEMETRY_DISABLED: "1"

secrets:                     # optional, asked once at install, kept in an owner only JSON file today
  - name: DOCUSIGN_API_KEY
    description: Optional. Enables sending the reviewed contract for signature.
    required: false
```

A declared secret is kept in `secrets/<app>.json` under the platform data directory, readable by
the owner only: mode `0600` on macOS and Linux, and on Windows an ACL with inheritance broken and
one grant to this user. The Windows Credential Manager is the target for version 1 and is not
built yet, so treat the file as the only copy.

### Rejected keys

`app_validate` fails when the manifest carries any of these, at the top level or inside a model
entry: `ctx`, `ctx_size`, `n_parallel`, `parallel`, `gpu_layers`, `n_gpu_layers`, `ngl`, `threads`,
`batch_size`, `kv_cache_type`, `flash_attn`, `temperature`, `top_p`, `top_k`, `seed`, `port`,
`host`. Sampling belongs in the request body of each call. Everything else belongs to the platform,
which sizes the context and the slots against the device.

## Injected variables

These are the only way an app reaches anything. Read them from the process environment at start.

| Variable | Value |
|---|---|
| `PORT` | The port the app must listen on |
| `HOST` | `127.0.0.1`. The app must bind this address |
| `HOSTNAME` | `127.0.0.1`, the same value under the name Node and Next.js read |
| `AIAS_MODEL_<ALIAS>_URL` | Base URL of the model server, ends in `/v1`, OpenAI compatible |
| `AIAS_MODEL_<ALIAS>_ID` | The model name to put in the request body |
| `AIAS_MODEL_<ALIAS>_KEY` | Bearer token of that model server, required on every request |
| `DATABASE_URL` | Postgres DSN for this app, only when `services.postgres` is declared |
| Declared `env` | As written in the manifest |
| Declared `secrets` | Each `name` as the user entered it, or absent when not required |

`<ALIAS>` is the manifest alias in upper case: alias `reviewer` gives `AIAS_MODEL_REVIEWER_URL`.

```ts
const client = new OpenAI({
  baseURL: process.env.AIAS_MODEL_REVIEWER_URL,
  apiKey: process.env.AIAS_MODEL_REVIEWER_KEY,
})
const answer = await client.chat.completions.create({
  model: process.env.AIAS_MODEL_REVIEWER_ID!,
  messages: [{ role: 'user', content: prompt }],
  temperature: 0.2,
})
```

## Templates

`app_init` takes `template: "next"` or `template: "fastapi"`.

- `next`: Next.js standalone, `runtime: node`. Pick it for a normal web UI, for streaming chat, and
  whenever the user says nothing about the stack.
- `fastapi`: FastAPI plus a static frontend, `runtime: python`. Pick it when the app needs the
  Python ecosystem, for example documents, PDFs, pandas or an existing Python library.

Both templates already read `AIAS_MODEL_*` with the OpenAI SDK, ship a Dockerfile and carry a
`db/migrations` folder. A template is a starting point, not a requirement: any repo that satisfies
this file is an app.

## Fork

Forking is how a second organization takes a published app and makes it theirs.

1. `store_search_apps` to find the app, or use an installed name.
2. `app_fork` with `source` (an installed name or an HTTPS git URL), the new `name` and a `dir`.
   This copies the repo with a fresh git history and the new name in the manifest.
3. Edit what differs: the prompt, the schema, the branding. Keep the model aliases unless the fork
   really needs a different model.
4. `app_run`, test, then create a GitHub repo of your own, push, and `app_publish`.

Both versions run side by side: each has its own name, port, database and clone.

## Rules

1. Bind `127.0.0.1` on `PORT` and nothing else. The platform reads the listening socket back after
   the health check and stops an app that bound a routable address.
2. Never start a model server, never download weights, never read the GPU. Declare the model and
   let the platform serve it.
3. No inference parameters in the manifest. Temperature, top_p and seed go in the request body.
4. A Dockerfile at the repo root is required and must expose `$PORT`. Version 1 does not run it; it
   is the portability contract and it is built in CI when the app is published.
5. `build` and `start` are argv lists, never shell strings. The one exception is the literal `&&`
   element inside `build`, which splits it into two commands.
6. `name` matches `^[a-z][a-z0-9-]*$` and `models[].alias` matches `^[a-z][a-z0-9_]*$`. An alias is
   unique inside one manifest.
7. Secrets are declared, never committed. Read them from the environment.
8. The app writes only inside its own directory and its own database.

## Common errors

**A model is not downloaded.** `app_run` answers with `started: false` and a `missingModels` list
instead of starting. Call `model_pull` with the `repo` and one `quant` from the list, poll
`job_status` with the returned `jobId` every few seconds until `state` is `done`, then call
`app_run` again. A first pull of a 14B model is several gigabytes, so tell the user it will take
minutes.

**"AI App Store is not running."** Every tool answers with that one sentence when the desktop app is
not listening on the local API. Ask the user to start AI App Store, or run `aias api serve` on a
headless machine, then retry. Nothing else in this skill works until then.

**The app bound the wrong address.** The run log says the platform stopped it for listening on
something other than `127.0.0.1`. Bind `process.env.HOST` (Node) or `host="127.0.0.1"` (uvicorn),
never `0.0.0.0`.

**The build failed.** `app_run` returns `build_failed`. Read `app_logs` with `kind: "build"` to see
which command failed, fix it, then run again.

**The app started but answers nothing.** Read `app_logs` with `kind: "run"`. A missing
`AIAS_MODEL_<ALIAS>_KEY` on the request is the usual cause: `llama-server` refuses a request without
the bearer token even on loopback.

**`app_validate` rejects the manifest.** The message names the rule. An inference parameter, a
duplicate alias, a missing Dockerfile and an empty `quant` list are the four that come up most.

**`app_publish` refuses.** It requires a clean working tree, an HTTPS GitHub origin, and a commit
that is already on the remote branch. Commit, push, then publish. Without `gh` it returns the entry
and the index URL, so paste the entry into `apps.yaml` in a pull request by hand.

## Tools

| Tool | Use it for |
|---|---|
| `store_search_models` | Find a GGUF repo and quant this machine can run |
| `store_search_apps` | Search the published apps in the store index |
| `model_pull` | Download one quant of one repo, returns a `jobId` |
| `job_status` | Poll a pull or a build until `done` or `failed` |
| `app_init` | Scaffold a repo from the `next` or `fastapi` template |
| `app_validate` | Check a repo against the manifest rules |
| `app_run` | Clone if needed, build, start, return the URL |
| `app_logs` | Tail the `build` or `run` log of an app |
| `app_stop` | Stop an app and release its model leases |
| `app_fork` | Copy an app into a new repo under a new name |
| `app_publish` | Validate, check the repo is pushed, open the index pull request |
