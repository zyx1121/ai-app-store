/**
 * Fail when the frontend and the Tauri shell disagree on the command list.
 *
 * The `invoke` boundary is the one contract with no compiler behind it: a
 * command the frontend calls but nobody registers only fails when a user clicks
 * it, and a registered command nobody calls is dead weight. Both directions are
 * errors here, and so is a third: an argument the frontend sends that the
 * command does not take, or one it takes and the frontend never sends. That is
 * what lets a command quietly keep accepting a path or a DSN from the web
 * context after it was supposed to stop.
 *
 * Run with `bun run check:commands`.
 */
import { readdir } from 'node:fs/promises'
import { join } from 'node:path'

const API_DIR = 'src/lib/api'
const COMMAND_DIR = 'src-tauri/src/commands'
const HANDLER_FILE = 'src-tauri/src/lib.rs'

/** Parameters Tauri injects itself, matched on the type, never on the name. */
const INJECTED_TYPES = /^(tauri::|AppHandle|Window|WebviewWindow|State<|tauri::State)/

/** Every `invoke<T>('name'` and `invoke("name"` in the API wrappers. */
async function calledCommands(): Promise<Map<string, string[]>> {
  const found = new Map<string, string[]>()
  const files = (await readdir(API_DIR)).filter((name) => name.endsWith('.ts')).sort()

  for (const name of files) {
    const path = join(API_DIR, name)
    const source = await Bun.file(path).text()
    // The generic argument is optional, the quote style is not fixed.
    for (const match of source.matchAll(/\binvoke\s*(?:<[^>]*>)?\s*\(\s*['"`]([^'"`]+)['"`]/g)) {
      const command = match[1]
      found.set(command, [...(found.get(command) ?? []), path])
    }
  }
  return found
}

/** The argument object of every `invoke` call, by command name. */
async function calledArguments(): Promise<Map<string, string[]>> {
  const found = new Map<string, string[]>()
  const files = (await readdir(API_DIR)).filter((name) => name.endsWith('.ts')).sort()

  for (const name of files) {
    const source = await Bun.file(join(API_DIR, name)).text()
    const call =
      /\binvoke\s*(?:<[^>]*>)?\s*\(\s*['"`]([^'"`]+)['"`]\s*(?:,\s*\{([^}]*)\})?\s*\)/g
    for (const match of source.matchAll(call)) {
      const keys = (match[2] ?? '')
        .split(',')
        .map((part) => part.split(':')[0].trim())
        .filter((part) => part.length > 0)
      found.set(match[1], keys.sort())
    }
  }
  return found
}

/** The parameter list of every `#[tauri::command]`, by command name. */
async function declaredArguments(): Promise<Map<string, string[]>> {
  const found = new Map<string, string[]>()
  const files = (await readdir(COMMAND_DIR)).filter((name) => name.endsWith('.rs')).sort()

  for (const name of files) {
    const source = await Bun.file(join(COMMAND_DIR, name)).text()
    const declaration = /#\[tauri::command\]\s*pub\s+(?:async\s+)?fn\s+(\w+)\s*\(([\s\S]*?)\)\s*->/g
    for (const match of source.matchAll(declaration)) {
      const params = splitParams(match[2])
        .filter((param) => !INJECTED_TYPES.test(param.type))
        .map((param) => snakeToCamel(param.name))
        .sort()
      found.set(match[1], params)
    }
  }
  return found
}

/** Split a Rust parameter list on the commas that are not inside a generic. */
function splitParams(text: string): { name: string; type: string }[] {
  const params: { name: string; type: string }[] = []
  let depth = 0
  let current = ''
  for (const char of `${text},`) {
    if (char === '<' || char === '(' || char === '[') depth += 1
    if (char === '>' || char === ')' || char === ']') depth -= 1
    if (char === ',' && depth === 0) {
      const at = current.indexOf(':')
      if (at !== -1) {
        params.push({ name: current.slice(0, at).trim(), type: current.slice(at + 1).trim() })
      }
      current = ''
      continue
    }
    current += char
  }
  return params
}

function snakeToCamel(name: string): string {
  return name.replace(/_(\w)/g, (_, letter: string) => letter.toUpperCase())
}

/** Every command inside `tauri::generate_handler!` in the shell. */
async function registeredCommands(): Promise<string[]> {
  const source = await Bun.file(HANDLER_FILE).text()
  const block = source.match(/generate_handler!\s*\[([\s\S]*?)\]/)
  if (block === null) {
    throw new Error(`no generate_handler! block in ${HANDLER_FILE}`)
  }
  return [...block[1].matchAll(/commands::\w+::(\w+)/g)].map((match) => match[1])
}

function report(title: string, items: string[]): void {
  console.error(`\n${title}`)
  for (const item of items) console.error(`  ${item}`)
}

const called = await calledCommands()
const registered = await registeredCommands()
const registeredSet = new Set(registered)

const missing = [...called.keys()]
  .filter((command) => !registeredSet.has(command))
  .sort()
  .map((command) => `${command}  called from ${[...new Set(called.get(command))].join(', ')}`)

const unused = registered.filter((command) => !called.has(command)).sort()

const duplicates = registered.filter((command, at) => registered.indexOf(command) !== at).sort()

const sent = await calledArguments()
const declared = await declaredArguments()
const mismatched: string[] = []
for (const [command, keys] of sent) {
  const params = declared.get(command)
  if (params === undefined) continue
  if (keys.join(',') !== params.join(',')) {
    mismatched.push(
      `${command}  frontend sends {${keys.join(', ')}}, the command takes (${params.join(', ')})`,
    )
  }
}
mismatched.sort()

if (missing.length > 0) {
  report(`Called by the frontend but not in ${HANDLER_FILE}:`, missing)
}
if (unused.length > 0) {
  report(`Registered in ${HANDLER_FILE} but never called from ${API_DIR}:`, unused)
}
if (duplicates.length > 0) {
  report(`Registered twice in ${HANDLER_FILE}:`, duplicates)
}
if (mismatched.length > 0) {
  report('Arguments that do not match the command signature:', mismatched)
}

if (missing.length > 0 || unused.length > 0 || duplicates.length > 0 || mismatched.length > 0) {
  console.error('\ncheck:commands failed')
  process.exit(1)
}

console.log(
  `check:commands: ${registered.length} commands and their arguments match across the invoke boundary`,
)
