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

```bash
bun run bump 0.2.0   # the only way to change the version
```

It rewrites the version in `package.json`, `src-tauri/tauri.conf.json`, `Cargo.toml` and the workspace entries in `Cargo.lock`; CI fails when they disagree. Open a pull request with that change, and merging it into `main` makes the release workflow tag `v0.2.0` and publish the Windows installer built from the merge commit. A merge that does not bump the version publishes nothing.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

MIT. The GPU is somebody else's problem now.
