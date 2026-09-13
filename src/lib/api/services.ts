import { invoke } from '@/lib/api/invoke'

export interface Postgres {
  /** Cluster directory, binaries and data. */
  dir: string
  /** Loopback port the cluster listens on. */
  port: number
}

/** Start the one Postgres cluster, initializing it on first call. */
export function postgresEnsure(): Promise<Postgres> {
  return invoke<Postgres>('services_postgres_ensure')
}

/** Stop the cluster. A cluster already down resolves without doing anything. */
export function postgresStop(): Promise<Postgres> {
  return invoke<Postgres>('services_postgres_stop')
}

/**
 * Create the database and role for one subscribed app, and resolve with its
 * DATABASE_URL. The cluster is derived in Rust; the frontend names the app.
 */
export function provisionAppDb(appName: string): Promise<string> {
  return invoke<string>('services_provision_app_db', { appName })
}

/**
 * Apply one subscribed app's declared migrations to its own database.
 *
 * The folder comes from the manifest of the clone on disk, not from here, so
 * there is no path to pass and none to get wrong.
 */
export function applyMigrations(appName: string): Promise<number> {
  return invoke<number>('services_apply_migrations', { appName })
}
