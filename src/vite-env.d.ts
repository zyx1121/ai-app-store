/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** `1` swaps every `invoke` for the fixtures in `@/lib/api/mock`. */
  readonly VITE_AIAS_MOCK?: string
  /** Short git sha stamped at build time, `dev` when it is absent. */
  readonly VITE_GIT_SHA?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
