/** Bytes as GB or MB with one decimal, the unit the store lists file sizes in. */
export function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

/** Megabytes as GB with one decimal, the unit the device profile reports in. */
export function formatMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

/** Download and like counts, 12345 becomes 12.3k. */
export function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return String(value)
}

/**
 * Blank the platform's secrets in a block of text: DSN passwords and API keys.
 *
 * `instances::mask_secrets` in `crates/core/src/instances.rs` already does this
 * before a log tail leaves the core, and the CLI depends on that. This is the
 * second pass, for text the mock produces and for anything that reaches a screen
 * by another route: a screenshot of a log should carry neither the generated
 * Postgres password nor the bearer token that gates an Instance.
 */
export function maskSecrets(text: string): string {
  return text
    .replace(/\b(postgres(?:ql)?:\/\/[^\s:@/]+):[^\s@]*@/gi, (_match, prefix: string) => `${prefix}:***@`)
    .replace(/(\w*_KEY=)\S+/g, (_match, prefix: string) => `${prefix}***`)
}
