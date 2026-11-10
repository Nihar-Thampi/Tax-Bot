import client from './client'
import type { Conversation } from '../store/chatStore'

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await client.get<{ conversations: Conversation[] }>('/conversations')
  return data.conversations
}

export async function saveConversation(conversation: Conversation): Promise<void> {
  await client.put(`/conversations/${encodeURIComponent(conversation.id)}`, conversation)
}

export async function deleteConversation(id: string): Promise<void> {
  await client.delete(`/conversations/${encodeURIComponent(id)}`)
}
