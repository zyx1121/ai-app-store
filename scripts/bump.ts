/**
 * Keep one version number across every file that carries it.
 *
 *   bun run bump 0.2.0     rewrite every version to 0.2.0
 *   bun run bump --check   exit 1 when the files disagree, used by CI
 *
 * The version lives in four files: `package.json` for the frontend,
 * `src-tauri/tauri.conf.json` for the installer, `[workspace.package]` in
 * `Cargo.toml` for both crates, and the `aias-core` and `aias-app` entries in
 * `Cargo.lock`, which cargo would otherwise only refresh on the next build.
 * The lock is edited by text so no other dependency moves.
 *
 * This is the only way to change the version: merging a bump into main makes
 * release.yml tag `v<version>` and publish the installer.
 */
import { readFile, writeFile } from 'node:fs/promises'

/** Major.minor.patch with an optional prerelease, no leading `v`. */
const SEMVER = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/

/** The `version` of the first JSON object, never a dependency range. */
const JSON_VERSION = /("version":\s*")[^"]+(")/

/** The `version` of the `[workspace.package]` table, not a dependency one. */
const WORKSPACE_VERSION = /(\[workspace\.package\][^[]*?\bversion\s*=\s*")[^"]+(")/

/** The `version` line of one crate's entry in the lock file. */
function lockVersion(crate: string): RegExp {
  return new RegExp(`(name = "${crate}"\\nversion = ")[^"]+(")`)
}

type Source = {
  /** Label in the output, the file path plus the crate for lock entries. */
  label: string
  path: string
  pattern: RegExp
}

const sources: Source[] = [
  { label: 'package.json', path: 'package.json', pattern: JSON_VERSION },
  {
    label: 'src-tauri/tauri.conf.json',
    path: 'src-tauri/tauri.conf.json',
    pattern: JSON_VERSION,
  },
  { label: 'Cargo.toml [workspace.package]', path: 'Cargo.toml', pattern: WORKSPACE_VERSION },
  { label: 'Cargo.lock aias-core', path: 'Cargo.lock', pattern: lockVersion('aias-core') },
  { label: 'Cargo.lock aias-app', path: 'Cargo.lock', pattern: lockVersion('aias-app') },
]

function read(source: Source, text: string): string {
  const found = text.match(source.pattern)
  if (found === null) {
    console.error(`no version to read in ${source.label}`)
    process.exit(3)
  }
  return found[0].slice(found[1].length, -found[2].length)
}

const target = process.argv[2]

if (target === undefined || target === '--check') {
  const versions = new Map<string, string>()
  for (const source of sources) {
    versions.set(source.label, read(source, await readFile(source.path, 'utf8')))
  }
  for (const [label, version] of versions) console.log(`${version}\t${label}`)

  const distinct = new Set(versions.values())
  if (distinct.size !== 1) {
    console.error('\nversion files disagree; run `bun run bump <version>`')
    process.exit(1)
  }
  console.log(`\nversion ${[...distinct][0]}`)
  process.exit(0)
}

if (!SEMVER.test(target)) {
  console.error(`not a SemVer version: ${target}`)
  console.error('expected X.Y.Z or X.Y.Z-prerelease, without a leading `v`')
  process.exit(2)
}

// One file holds two entries, so rewrite in sequence and re-read every time.
for (const source of sources) {
  const before = await readFile(source.path, 'utf8')
  const from = read(source, before)
  const after = before.replace(source.pattern, `$1${target}$2`)
  if (read(source, after) !== target) {
    console.error(`failed to rewrite ${source.label}`)
    process.exit(3)
  }
  await writeFile(source.path, after)
  console.log(`${from} -> ${target}\t${source.label}`)
}

console.log(`\nversion ${target}`)
