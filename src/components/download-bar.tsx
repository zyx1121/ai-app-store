import { Progress, ProgressLabel, ProgressValue } from '@/components/ui/progress'
import type { DownloadProgress } from '@/lib/api/models'
import { formatBytes } from '@/lib/format'

/**
 * One download in flight, as the `models://progress` event reports it.
 *
 * A payload with no total leaves the bar indeterminate rather than guessing:
 * Hugging Face sends no content length for some files.
 */
export function DownloadBar({ progress }: { progress: DownloadProgress }) {
  const { downloadedBytes, totalBytes } = progress
  const value =
    totalBytes === null || totalBytes === 0
      ? null
      : Math.min(100, Math.round((downloadedBytes / totalBytes) * 100))

  return (
    <Progress value={value} className="w-full">
      <ProgressLabel className="text-muted-foreground text-xs">
        {formatBytes(downloadedBytes)}
        {totalBytes === null ? '' : ` of ${formatBytes(totalBytes)}`}
      </ProgressLabel>
      <ProgressValue className="text-xs">
        {() => (value === null ? '' : `${value}%`)}
      </ProgressValue>
    </Progress>
  )
}
