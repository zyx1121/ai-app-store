import { isMock } from '@/lib/api/invoke'
import type { InstanceUrl } from '@/lib/api/instances'

/** One turn of an OpenAI compatible conversation. */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface ChatChunk {
  choices: { delta: { content?: string } }[]
}

const MOCK_REPLY =
  'The instance is serving this reply over the OpenAI compatible endpoint on its loopback port. ' +
  'An app reaches the same endpoint through AIAS_MODEL_<ALIAS>_URL, so what you read here is what it would read.'

async function* mockStream(signal: AbortSignal): AsyncGenerator<string, void, void> {
  for (const word of MOCK_REPLY.split(' ')) {
    if (signal.aborted) return
    await new Promise((resolve) => window.setTimeout(resolve, 24))
    yield `${word} `
  }
}

/**
 * Stream one completion from a running instance.
 *
 * This talks to `llama-server` directly over loopback, not through a command:
 * the instance is an OpenAI compatible server and the drawer is just another
 * client of it, the same way an app is.
 */
export async function* streamChat(
  target: InstanceUrl,
  messages: ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<string, void, void> {
  if (isMock) {
    yield* mockStream(signal)
    return
  }

  const response = await fetch(`${target.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      // llama-server is started with --api-key, so loopback alone is not enough.
      Authorization: `Bearer ${target.apiKey}`,
    },
    body: JSON.stringify({ model: target.modelId, messages, stream: true }),
    signal,
  })

  if (!response.ok || response.body === null) {
    throw new Error(`instance answered ${response.status} ${response.statusText}`)
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += value

    // Server sent events: one event per blank line, payload after `data: `.
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const event = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')

      for (const line of event.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') return
        const chunk = JSON.parse(payload) as ChatChunk
        const delta = chunk.choices[0]?.delta.content
        if (delta !== undefined && delta.length > 0) yield delta
      }
    }
  }
}
