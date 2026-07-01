import { create } from 'zustand'
import type { ChatMessage } from '../types'

const WELCOME_MESSAGE: ChatMessage = {
  role: 'assistant',
  content:
    'Ask a question about the South African Income Tax Act No. 58 of 1962. I will answer from the Act and cite sections where relevant.',
}

interface ChatState {
  messages: ChatMessage[]
  history: ChatMessage[]
  addMessage: (message: ChatMessage) => void
  setHistory: (history: ChatMessage[]) => void
  reset: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [WELCOME_MESSAGE],
  history: [],
  addMessage: (message) => set((s) => ({ messages: [...s.messages, message] })),
  setHistory: (history) => set({ history }),
  reset: () => set({ messages: [WELCOME_MESSAGE], history: [] }),
}))
