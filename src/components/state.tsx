import { AlertTriangle, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@/components/ui/empty'
import { Spinner } from '@/components/ui/spinner'
import { isNotImplemented, type CmdError } from '@/lib/api/error'

/** Placeholder while a command is in flight. */
export function LoadingState({ label }: { label: string }) {
  return (
    <Empty className="border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <Spinner />
        </EmptyMedia>
        <EmptyTitle>{label}</EmptyTitle>
      </EmptyHeader>
    </Empty>
  )
}

/**
 * What a failed command looks like.
 *
 * `not_implemented` is the common case today: the command exists, the core
 * behind it does not. It is named as such so a reader knows the page is fine and
 * the backend is not.
 */
export function ErrorState({
  error,
  what,
  onRetry,
}: {
  error: CmdError
  what: string
  onRetry?: () => void
}) {
  const pending = isNotImplemented(error)
  return (
    <Empty className="border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <AlertTriangle />
        </EmptyMedia>
        <EmptyTitle>{pending ? `${what} is not implemented yet` : `${what} failed`}</EmptyTitle>
        <EmptyDescription>
          <span className="font-mono text-xs">{error.code}</span>
          <span className="block">{error.message}</span>
        </EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        {pending ? (
          <p className="text-muted-foreground text-xs">
            Start the app with VITE_AIAS_MOCK=1 to see this page against fixtures.
          </p>
        ) : null}
        {onRetry ? (
          <Button variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        ) : null}
      </EmptyContent>
    </Empty>
  )
}

/** A list that loaded and holds nothing. */
export function EmptyState({
  icon: Icon,
  title,
  description,
  children,
}: {
  icon: LucideIcon
  title: string
  description: string
  children?: ReactNode
}) {
  return (
    <Empty className="border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <Icon />
        </EmptyMedia>
        <EmptyTitle>{title}</EmptyTitle>
        <EmptyDescription>{description}</EmptyDescription>
      </EmptyHeader>
      {children ? <EmptyContent>{children}</EmptyContent> : null}
    </Empty>
  )
}
