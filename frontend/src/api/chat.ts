import client from './client'
import type { ChatMessage, ChatResponse, HealthResponse } from '../types'

export async function sendMessage(
  message: string,
  history: ChatMessage[],
  opts: { requestId?: string; signal?: AbortSignal } = {},
): Promise<ChatResponse> {
  const { data } = await client.post<ChatResponse>(
    '/chat',
    { message, history, request_id: opts.requestId },
    { signal: opts.signal },
  )
  return data
}

/** Ask the backend to stop generating the answer for a request started with sendMessage. */
export async function cancelMessage(requestId: string): Promise<void> {
  await client.post('/chat/cancel', { request_id: requestId })
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health')
  return data
}
