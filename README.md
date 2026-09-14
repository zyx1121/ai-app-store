# AI App Store

> The app says `I need model X`, the platform says `it is at this port`.

[![CI](https://github.com/zyx1121/ai-app-store/actions/workflows/ci.yml/badge.svg)](https://github.com/zyx1121/ai-app-store/actions)

A local AI app is easy to ship when it targets one CUDA GPU. The moment AMD, Intel, unified memory and NPUs enter the room, every ISV needs one machine per vendor and one build per backend, so nobody ships it. This desktop application takes that problem off the app: it detects the hardware, installs the matching `llama-server`, owns every model process on the machine, and hands each app a loopback port that speaks the OpenAI API.

## Features

- **Install one runtime**: hardware detection picks CUDA, Vulkan, OpenVINO or CPU from a fixed table and installs a single `llama-server` build.
- **Share one instance**: two apps that declare the same model and quant get the same process, with the platform owning context size, slots and GPU layers.
- **Run apps from source**: an app is a git repo with an `aias.yaml`, cloned and built locally, so forking a published app is an edit and a push.

## Plan

[PLAN.md](PLAN.md) is the single living document: positioning, object model, manifest, decisions, milestones. Read it before changing anything here. The manifest example is [spec/aias.example.yaml](spec/aias.example.yaml).

## Tech stack

| Layer | Choice |
|-------|--------|
| Desktop shell | Tauri 2 |
| Core logic | Rust, one crate `aias-core` |
| Headless surface | `aias` CLI over the same crate |
| Frontend | Vite, React 19, TypeScript, Tailwind v4, shadcn |
| Package manager | Bun |

## Getting started

```bash
git clone https://github.com/zyx1121/ai-app-store && cd ai-app-store
bun install
bun run tauri dev
```

## Layout

| Path | Holds |
|------|-------|
| `crates/core` | `aias-core`, every module of the object model plus the `aias` CLI |
| `src-tauri` | The desktop shell, one command file per core module |
| `src` | React routes and one API wrapper file per core module |
| `spec` | The manifest contract apps are validated against |

## Commands

```bash
bun run dev            # frontend only
bun run tauri dev      # desktop shell
bun run build          # typecheck and build the frontend
cargo test --workspace # core tests
cargo run --bin aias -- apps validate <dir>   # headless, no GUI needed
```

Every module is reachable from the CLI, so a machine reached over SSH verifies the same code the desktop app runs.

## Releasing

[Release Please](https://github.com/googleapis/release-please) owns the version. Nobody edits it by hand: the pull request title decides the next number, so the title is written as a [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) and squash merging turns it into the commit message Release Please reads.

```
feat: share one llama-server between apps
fix(store): stop the search box eating the first keystroke
docs: explain the lease model
```

CI rejects a title that is not one of `feat`, `fix`, `perf`, `refactor`, `docs`, `ci`, `chore`, `test`, `build` or `revert`, or whose subject starts with a capital letter. Only `feat`, `fix` and `perf` reach the changelog.

This project is below 1.0, so the bump rules are the pre-1.0 ones:

| Title | Bump | 0.1.1 becomes |
|-------|------|---------------|
| `fix:`, `perf:` | patch | 0.1.2 |
| `feat:` | minor | 0.2.0 |
| `feat!:` or a `BREAKING CHANGE:` footer | minor | 0.2.0 |
| everything else, alone | nothing, no release pull request opens | 0.1.1 |

A break bumping the minor rather than the major is the one pre-1.0 rule here: until 1.0 a break is allowed to be cheap, so it lands in the same place a feature does. That rule drops away at 1.0, where a break starts bumping the major.

The last row is per batch, not per commit: a `docs` commit merged alongside a `fix` ships with it and is simply left out of the changelog. Only a batch with nothing in `feat`, `fix` or `perf` produces no release at all.

Releasing is then a merge, not a command:

1. Merge work into `main`. Release Please keeps one open pull request titled `chore(main): release <version>`, holding the next version in `package.json`, `src-tauri/tauri.conf.json`, `Cargo.toml` and the workspace entries in `Cargo.lock`, plus the `CHANGELOG.md` entry. It rewrites that pull request on every push.
2. Merge the release pull request. That tags `v<version>`, creates the GitHub release from the changelog entry, and the `release-please` workflow builds the Windows installer from the release commit and uploads it to that release.

CI runs `bun run bump --check` on every pull request, so the four files are proved to agree before anything ships.

`bun run bump X.Y.Z` still exists as an escape hatch for the case where Release Please cannot run. It warns, and the next release pull request overwrites whatever it wrote.

No secret is needed. GitHub starts no workflow for a pull request opened by `GITHUB_TOKEN`, so `release-please.yml` dispatches `ci.yml` by name on the release branch instead, which is one of the two documented exceptions to that rule, and the run reports against the branch head the pull request shows.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

MIT. The GPU is somebody else's problem now.
