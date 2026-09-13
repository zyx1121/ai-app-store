/**
 * What a failing command sends over the `invoke` boundary.
 *
 * Mirrors `CmdError` in `src-tauri/src/error.rs`. `code` is the stable tag from
 * `aias_core::Error::code`, for example `not_implemented` or `not_found`.
 */
export interface CmdError {
  code: string
  message: string
}

function isCmdError(value: unknown): value is CmdError {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.code === 'string' && typeof candidate.message === 'string'
}

/** Narrow anything a rejected `invoke` throws into a `CmdError`. */
export function toCmdError(value: unknown): CmdError {
  if (isCmdError(value)) return value
  if (value instanceof Error) return { code: 'unknown', message: value.message }
  return { code: 'unknown', message: String(value) }
}

/** True when the command exists but has no implementation behind it yet. */
export function isNotImplemented(error: CmdError): boolean {
  return error.code === 'not_implemented'
}
