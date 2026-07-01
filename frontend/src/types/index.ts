export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatResponse {
  reply: string
  history: ChatMessage[]
}

export interface HealthResponse {
  status: string
  use_finetuned: boolean
}
