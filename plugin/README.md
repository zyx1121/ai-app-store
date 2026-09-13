# aias

The Claude Code plugin for the [AI App Store](https://github.com/zyx1121/ai-app-store): build, run,
fork and publish a local AI app from an agent session.

It ships one skill and one MCP server:

- `aias-app` explains the `aias.yaml` manifest, the injected environment variables, the two
  templates and the rules the platform enforces.
- `aias` is `aias mcp`, a stdio server that forwards to the running desktop app over its loopback
  API. The desktop app stays the single owner of models and processes.

## Install

```sh
claude plugin marketplace add zyx1121/ai-app-store
claude plugin install aias@ai-app-store
```

The MCP server is the `aias` binary. Install AI App Store, or build the CLI from this repo with
`cargo build --release -p aias-core`, and make sure `aias` is on `PATH`.

`.mcp.json` names the command as `aias` rather than an absolute path, because the path differs per
user and per machine and no one value works for everybody. The installer is what puts `aias` on
`PATH`. One consequence is worth knowing on Windows: a process there searches the current directory
before `PATH`, so an `aias.exe` sitting in the directory a session was started from would be run
instead. Start sessions in a directory you trust, which is the same rule that already applies to
every other tool an agent runs.

## Requirements

The desktop app must be running, because every tool forwards to its local API on
`http://127.0.0.1:40999/v1`. On a headless machine run `aias api serve` instead. When neither is
up, every tool answers with one sentence saying to start it.

`AIAS_API_URL` and `AIAS_API_TOKEN` override the base URL and the bearer token, which is how the
bridge is pointed at a test server.

## Tools

| Tool | Does |
|---|---|
| `store_search_models` | Hugging Face GGUF search filtered to this runtime, with fit against this device |
| `store_search_apps` | Search the store index |
| `model_pull` | Download a model, returns a job id |
| `job_status` | Progress and result of a pull or a build |
| `app_init` | Scaffold a repo from the Next.js or FastAPI template |
| `app_validate` | Run the manifest rules against a local repo |
| `app_run` | Clone if needed, build, start, return the URL |
| `app_logs` | Tail the build or run log |
| `app_stop` | Release the leases and stop the app |
| `app_fork` | Copy an app into a new repo under a new name |
| `app_publish` | Check the repo is pushed and clean, then open the store index pull request |

## Publishing

`app_publish` needs a clean working tree, an HTTPS GitHub origin and a commit that is already on the
remote. With `gh` installed and logged in it forks `zyx1121/aias-index`, pushes `add-<name>` and
opens the pull request. Without `gh` it returns the index entry and the URL to paste it into.
