import packageJson from '../../package.json?raw'

const parsed = JSON.parse(packageJson) as { version: string }

/** Version of the desktop shell, from package.json. */
export const appVersion = parsed.version

/** Short git sha of the build, `dev` outside CI. */
export const gitSha = import.meta.env.VITE_GIT_SHA ?? 'dev'

/** The line shown at the bottom of Setup, for example `v0.1.0 dev`. */
export const versionLine = `v${appVersion} ${gitSha}`
