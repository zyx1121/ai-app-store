import { isMock } from '@/lib/api/invoke'

/**
 * Open a loopback URL in the user's browser.
 *
 * The shell registers `tauri_plugin_opener` and the capability allows
 * `opener:allow-open-url` with a scope: the plugin denies every URL that no
 * scope entry matches. The scope is an app's own loopback port, `github.com`
 * and `huggingface.co`, which is every link this window offers. The fallback
 * stays for the browser dev server, where the plugin import resolves to
 * nothing.
 */
export async function openExternal(url: string): Promise<void> {
  if (isMock) {
    window.open(url, '_blank', 'noopener,noreferrer')
    return
  }
  try {
    const { openUrl } = await import('@tauri-apps/plugin-opener')
    await openUrl(url)
  } catch (error) {
    // A denied scope reads exactly like a missing plugin here, and the window
    // fallback is a no op inside the webview, so say what happened.
    console.error(`opening ${url} failed`, error)
    window.open(url, '_blank', 'noopener,noreferrer')
  }
}
