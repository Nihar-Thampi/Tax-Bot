import { useCallback } from 'react'
import { cancelMessage, sendMessage } from '../api/chat'
import { newId, useChatStore } from '../store/chatStore'

export const CANCELLED_TEXT = 'You cancelled this request.'
export const ERROR_TEXT = 'Sorry, there was an issue getting an answer. Please try again.'

/** In-flight requests by conversation id, so the stop button can cancel them. */
const inFlight = new Map<string, { requestId: string; controller: AbortController }>()

export function useChat() {
  const activeId = useChatStore((s) => s.activeId)
  const active = useChatStore((s) => (s.activeId ? s.conversations[s.activeId] : undefined))
  const sending = useChatStore((s) => (s.activeId ? !!s.pending[s.activeId] : false))

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return

    const store = useChatStore.getState()
    // Pin the request to the chat it was asked in, so switching chats mid-answer is safe.
    const id = store.activeId && store.conversations[store.activeId] ? store.activeId : store.newChat()
    const history = useChatStore.getState().conversations[id]?.history ?? []

    const requestId = newId()
    const controller = new AbortController()
    inFlight.set(id, { requestId, controller })

    store.addMessage(id, { role: 'user', content: trimmed })
    store.setPending(id, true)
    try {
      const { reply, history: newHistory, points, sources } = await sendMessage(trimmed, history, {
        requestId,
        signal: controller.signal,
      })
      useChatStore.getState().addMessage(id, { role: 'assistant', content: reply, points, sources })
      useChatStore.getState().setHistory(id, newHistory)
    } catch (e: unknown) {
      // A stopped question stays out of the history sent to the model on the next turn.
      // Details go to the browser console (and the backend log); the chat shows a plain message.
      if (!controller.signal.aborted) console.error('Chat request failed:', e)
      const content = controller.signal.aborted ? CANCELLED_TEXT : ERROR_TEXT
      useChatStore.getState().addMessage(id, { role: 'assistant', content })
    } finally {
      if (inFlight.get(id)?.requestId === requestId) inFlight.delete(id)
      useChatStore.getState().setPending(id, false)
    }
  }, [])

  const stop = useCallback(() => {
    const id = useChatStore.getState().activeId
    const req = id ? inFlight.get(id) : undefined
    if (!req) return
    req.controller.abort()
    // The backend keeps generating unless told to stop; best effort.
    cancelMessage(req.requestId).catch(() => {})
  }, [])

  return { activeId, messages: active?.messages ?? [], sending, send, stop }
}
