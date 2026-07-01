import { useCallback, useState } from 'react'
import { sendMessage } from '../api/chat'
import { useChatStore } from '../store/chatStore'

export function useChat() {
  const { messages, history, addMessage, setHistory } = useChatStore()
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return

    addMessage({ role: 'user', content: trimmed })
    setSending(true)
    setError(null)

    try {
      const { reply, history: newHistory } = await sendMessage(trimmed, history)
      addMessage({ role: 'assistant', content: reply })
      setHistory(newHistory)
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Request failed'
      setError(message)
      addMessage({ role: 'assistant', content: `Error: ${message}` })
    } finally {
      setSending(false)
    }
  }, [history, addMessage, setHistory])

  return { messages, sending, error, send }
}
