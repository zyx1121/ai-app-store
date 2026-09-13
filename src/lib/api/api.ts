import { invoke } from '@/lib/api/invoke'

/** How an agent reaches this running app, PLAN section 6.1. */
export interface ApiToken {
  /** Base URL of the local API, ends in `/v1`. */
  url: string
  /** Bearer token every request carries. */
  token: string
}

/** The local API URL and its bearer token, generated on first launch. */
export function token(): Promise<ApiToken> {
  return invoke<ApiToken>('api_token')
}
