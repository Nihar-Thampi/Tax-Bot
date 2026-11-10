export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

/** Where in the Act a point came from. */
export interface Source {
  id: number
  section: string
  line_start: number
  line_end: number
  page: number | null
  excerpt: string
}

/** One paragraph or list item of an answer, with its matched source (if any). */
export interface AnswerPoint {
  text: string
  source_id: number | null
  score: number | null
  /** summary: the opening answer / closing summary around the cited bullets (no badge of its own). */
  kind: 'cited' | 'unsupported' | 'summary' | 'disclaimer'
}

/** A message as shown in the UI; assistant replies may carry citations. */
export interface DisplayMessage extends ChatMessage {
  points?: AnswerPoint[]
  sources?: Source[]
}

export interface ChatResponse {
  reply: string
  history: ChatMessage[]
  points: AnswerPoint[]
  sources: Source[]
}

export interface HealthResponse {
  status: string
  use_finetuned: boolean
}
