import client from './client'
import type { ChatMessage, ChatResponse, HealthResponse } from '../types'

export async function sendMessage(message: string, history: ChatMessage[]): Promise<ChatResponse> {
  const { data } = await client.post<ChatResponse>('/chat', { message, history })
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health')
  return data
}
