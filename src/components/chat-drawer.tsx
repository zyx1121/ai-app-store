import { useEffect, useRef, useState } from 'react'
import { SendHorizontal } from 'lucide-react'

import { ErrorState } from '@/components/state'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Textarea } from '@/components/ui/textarea'
import { toCmdError, type CmdError } from '@/lib/api/error'
import * as instances from '@/lib/api/instances'
import type { InstanceUrl } from '@/lib/api/instances'
import type { ModelRef } from '@/lib/api/models'
import { streamChat, type ChatMessage } from '@/lib/chat'

/** The lease holder name the desktop shell uses when the user tries a model. */
export const UI_LEASE = 'aias-ui'

/**
 * Try a model without an app.
 *
 * The drawer acquires a lease the same way an app does, then talks to the
 * instance over its OpenAI compatible port. Closing it releases the lease.
 */
export function ChatDrawer({ model, onClose }: { model: ModelRef | null; onClose: () => void }) {
  const [target, setTarget] = useState<InstanceUrl | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<CmdError | null>(null)
  const abort = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (model === null) return
    let live = true
    setMessages([])
    setError(null)
    setTarget(null)
    instances
      .acquire(UI_LEASE, model)
      .then((url) => {
        if (live) setTarget(url)
      })
      .catch((caught: unknown) => {
        if (live) setError(toCmdError(caught))
      })
    return () => {
      live = false
      abort.current?.abort()
      void instances.release(UI_LEASE, model).catch(() => undefined)
    }
  }, [model])

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  async function send() {
    const text = draft.trim()
    if (text.length === 0 || target === null || streaming) return
    const next: ChatMessage[] = [...messages, { role: 'user', content: text }]
    setMessages([...next, { role: 'assistant', content: '' }])
    setDraft('')
    setStreaming(true)
    setError(null)

    const controller = new AbortController()
    abort.current = controller
    try {
      for await (const delta of streamChat(target, next, controller.signal)) {
        setMessages((current) => {
          const head = current.slice(0, -1)
          const tail = current[current.length - 1]
          return [...head, { role: 'assistant', content: tail.content + delta }]
        })
      }
    } catch (caught: unknown) {
      setError(toCmdError(caught))
    } finally {
      setStreaming(false)
      abort.current = null
    }
  }

  return (
    <Sheet open={model !== null} onOpenChange={(open) => (open ? null : onClose())}>
      <SheetContent className="flex w-full flex-col gap-0 sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="font-mono text-sm break-all">
            {model?.repo ?? ''} {model?.quant ?? ''}
          </SheetTitle>
          <SheetDescription>
            {target === null
              ? 'Acquiring a lease on the instance'
              : `Streaming from ${target.baseUrl}/chat/completions`}
          </SheetDescription>
        </SheetHeader>

        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4">
          {messages.length === 0 && error === null ? (
            <p className="text-muted-foreground text-sm">
              Send a message to check the instance answers before any app depends on it.
            </p>
          ) : null}
          {messages.map((message, position) => (
            <div
              key={position}
              className={
                message.role === 'user'
                  ? 'bg-muted ml-auto max-w-[85%] rounded-xl px-3 py-2 text-sm whitespace-pre-wrap'
                  : 'max-w-[95%] rounded-xl border px-3 py-2 text-sm whitespace-pre-wrap'
              }
            >
              {message.content.length === 0 ? '...' : message.content}
            </div>
          ))}
          {error !== null ? <ErrorState error={error} what="The chat request" /> : null}
          <div ref={bottom} />
        </div>

        <div className="flex items-end gap-2 border-t p-4">
          <Textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void send()
              }
            }}
            placeholder="Message the model"
            className="min-h-16 flex-1 text-sm"
            aria-label="Message the model"
          />
          <Button
            size="icon"
            onClick={() => void send()}
            disabled={target === null || streaming || draft.trim().length === 0}
            aria-label="Send"
          >
            <SendHorizontal />
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}
