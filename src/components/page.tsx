import type { ReactNode } from 'react'

import { SidebarTrigger } from '@/components/ui/sidebar'

/** Shared frame of every route: a header with the sidebar trigger and a body. */
export function Page({
  title,
  description,
  actions,
  children,
}: {
  title: string
  description: string
  actions?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="flex min-h-svh w-full min-w-0 flex-col">
      <header className="bg-background sticky top-0 z-10 flex h-14 items-center gap-2 border-b px-4">
        <SidebarTrigger />
        <h1 className="truncate text-sm font-medium">{title}</h1>
        {actions ? <div className="ml-auto flex items-center gap-2">{actions}</div> : null}
      </header>
      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4 sm:p-6">
        <p className="text-muted-foreground text-sm">{description}</p>
        {children}
      </main>
    </div>
  )
}
