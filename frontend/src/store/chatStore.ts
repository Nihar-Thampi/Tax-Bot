import { create } from 'zustand'
import { deleteConversation, listConversations, saveConversation } from '../api/conversations'
import type { ChatMessage, DisplayMessage } from '../types'
import { safeStorage } from './safeStorage'

export const WELCOME_TEXT =
  'Ask a question about the South African Income Tax Act No. 58 of 1962. Each point of my answer is linked to the passage of the Act it comes from; points that cannot be tied to the Act are left out.'

export interface Conversation {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  /** What the UI shows, including citations. */
  messages: DisplayMessage[]
  /** Plain role/content history the backend expects back on the next turn. */
  history: ChatMessage[]
}

interface ChatState {
  conversations: Record<string, Conversation>
  activeId: string | null
  /** Conversations with a request in flight (not saved). */
  pending: Record<string, boolean>
  newChat: () => string
  selectChat: (id: string) => void
  deleteChat: (id: string) => void
  renameChat: (id: string, title: string) => void
  addMessage: (id: string, message: DisplayMessage) => void
  setHistory: (id: string, history: ChatMessage[]) => void
  setPending: (id: string, pending: boolean) => void
}

const TITLE_MAX = 48

function makeTitle(text: string) {
  const oneLine = text.replace(/\s+/g, ' ').trim()
  return oneLine.length > TITLE_MAX ? `${oneLine.slice(0, TITLE_MAX - 1)}…` : oneLine || 'New chat'
}

export function newId() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

// Chats are saved on the backend (backend/data/conversations/, synced by OneDrive),
// so every device that runs the backend sees the same history. See the sync code below.
export const useChatStore = create<ChatState>()((set, get) => ({
  conversations: {},
  activeId: null,
  pending: {},

  newChat: () => {
    // Reuse the current chat if nothing has been asked in it yet.
    const { activeId, conversations } = get()
    if (activeId && conversations[activeId]?.messages.length === 0) return activeId
    const id = newId()
    const now = Date.now()
    set((s) => ({
      conversations: {
        ...s.conversations,
        [id]: { id, title: 'New chat', createdAt: now, updatedAt: now, messages: [], history: [] },
      },
      activeId: id,
    }))
    return id
  },

  selectChat: (id) => set({ activeId: id }),

  deleteChat: (id) =>
    set((s) => {
      const conversations = { ...s.conversations }
      delete conversations[id]
      let activeId = s.activeId
      if (activeId === id) {
        const next = Object.values(conversations).sort((a, b) => b.updatedAt - a.updatedAt)[0]
        activeId = next?.id ?? null
      }
      return { conversations, activeId }
    }),

  renameChat: (id, title) =>
    set((s) => {
      const c = s.conversations[id]
      if (!c) return s
      return { conversations: { ...s.conversations, [id]: { ...c, title: makeTitle(title) } } }
    }),

  addMessage: (id, message) =>
    set((s) => {
      const c = s.conversations[id]
      if (!c) return s
      const isFirstQuestion = message.role === 'user' && !c.messages.some((m) => m.role === 'user')
      return {
        conversations: {
          ...s.conversations,
          [id]: {
            ...c,
            title: isFirstQuestion && c.title === 'New chat' ? makeTitle(message.content) : c.title,
            messages: [...c.messages, message],
            updatedAt: Date.now(),
          },
        },
      }
    }),

  setHistory: (id, history) =>
    set((s) => {
      const c = s.conversations[id]
      if (!c) return s
      return { conversations: { ...s.conversations, [id]: { ...c, history } } }
    }),

  setPending: (id, pending) =>
    set((s) => {
      const next = { ...s.pending }
      if (pending) next[id] = true
      else delete next[id]
      return { pending: next }
    }),
}))

// ---- Sync with the backend ----

/** Whose chats are loaded; null while logged out or switching users. */
let currentUser: string | null = null
/** Set while applying server state, so it isn't echoed straight back. */
let applyingRemote = false
/** Chats the server has a copy of (a missing one was then deleted on another device). */
const syncedIds = new Set<string>()
/** Chats changed or deleted locally and not yet written to the server. */
const dirty = new Set<string>()
let flushing: Promise<void> | null = null

function setRemote(partial: Partial<ChatState>) {
  applyingRemote = true
  try {
    useChatStore.setState(partial)
  } finally {
    applyingRemote = false
  }
}

/** Write dirty chats to the server one at a time, so saves of the same chat can't land out of order. */
function flush(): Promise<void> {
  if (flushing) return flushing
  flushing = (async () => {
    while (dirty.size && currentUser) {
      const [id] = dirty
      dirty.delete(id)
      const c = useChatStore.getState().conversations[id]
      try {
        if (c) {
          await saveConversation(c)
          syncedIds.add(id)
        } else if (syncedIds.has(id)) {
          await deleteConversation(id)
          syncedIds.delete(id)
        }
      } catch (e) {
        console.warn(`Could not sync chat ${id}`, e)
      }
    }
  })().finally(() => {
    flushing = null
  })
  return flushing
}

useChatStore.subscribe((state, prev) => {
  if (!currentUser) return
  if (state.activeId !== prev.activeId) safeStorage.setItem(activeKey(currentUser), state.activeId ?? '')
  if (applyingRemote || state.conversations === prev.conversations) return
  for (const [id, c] of Object.entries(state.conversations)) {
    // Empty "New chat"s stay local until something is asked in them.
    if (c !== prev.conversations[id] && c.messages.length > 0) dirty.add(id)
  }
  for (const id of Object.keys(prev.conversations)) {
    if (!state.conversations[id]) dirty.add(id)
  }
  if (dirty.size) void flush()
})

function activeKey(username: string) {
  return `sa-tax-bot-active:${username.toLowerCase()}`
}

/** Pull chats saved by other devices (via OneDrive) and merge them in. Local unsaved edits win. */
export async function refreshChats() {
  const user = currentUser
  if (!user) return
  const server = await listConversations()
  if (currentUser !== user) return

  const { conversations, pending, activeId } = useChatStore.getState()
  const next: Record<string, Conversation> = {}
  const onServer = new Set<string>()
  for (const remote of server) {
    onServer.add(remote.id)
    const local = conversations[remote.id]
    const keepLocal = local && (pending[remote.id] || dirty.has(remote.id) || local.updatedAt > remote.updatedAt)
    next[remote.id] = keepLocal ? local : remote
  }
  for (const [id, local] of Object.entries(conversations)) {
    if (onServer.has(id)) continue
    // Previously synced but gone from the server: deleted on another device.
    const deletedElsewhere = syncedIds.has(id) && !pending[id] && !dirty.has(id)
    if (!deletedElsewhere) next[id] = local
  }
  syncedIds.clear()
  onServer.forEach((id) => syncedIds.add(id))

  const stillActive = activeId && next[activeId] ? activeId : null
  const newest = Object.values(next).sort((a, b) => b.updatedAt - a.updatedAt)[0]
  setRemote({ conversations: next, activeId: stillActive ?? newest?.id ?? null })
}

/** Chats saved in this browser before they moved to the backend, for one-time upload. */
function readLocalChats(username: string): { keys: string[]; conversations: Conversation[] } {
  const keys = [`sa-tax-bot-chats:${username.toLowerCase()}`, 'sa-tax-bot-chats']
  const conversations: Conversation[] = []
  for (const key of keys) {
    try {
      const parsed = JSON.parse(safeStorage.getItem(key) as string | null ?? 'null')
      conversations.push(...Object.values<Conversation>(parsed?.state?.conversations ?? {}))
    } catch {
      /* ignore unreadable data */
    }
  }
  return { keys, conversations }
}

/** Swap the in-memory chats for this user's chats from the backend (or none, when logged out). */
export async function switchChatUser(username: string | null) {
  currentUser = null
  dirty.clear()
  syncedIds.clear()
  setRemote({ conversations: {}, activeId: null, pending: {} })
  if (!username) return
  currentUser = username

  try {
    await refreshChats()
  } catch (e) {
    console.warn('Could not load chats from the backend', e)
  }
  if (currentUser !== username) return

  const remembered = safeStorage.getItem(activeKey(username)) as string | null
  if (remembered && useChatStore.getState().conversations[remembered]) setRemote({ activeId: remembered })

  // Upload chats left in this browser's localStorage from before, then drop the local copies.
  const local = readLocalChats(username)
  const missing = local.conversations.filter(
    (c) => c?.id && c.messages?.length && !useChatStore.getState().conversations[c.id],
  )
  if (missing.length) {
    useChatStore.setState((s) => {
      const conversations = { ...s.conversations }
      for (const c of missing) conversations[c.id] = c
      return { conversations }
    })
    await flush()
  }
  if (missing.every((c) => syncedIds.has(c.id))) local.keys.forEach((k) => safeStorage.removeItem(k))
}

// Pick up chats from the other device when coming back to the tab, and periodically while it's open.
if (typeof window !== 'undefined') {
  const refresh = () => {
    if (currentUser && document.visibilityState === 'visible') refreshChats().catch(() => {})
  }
  window.addEventListener('focus', refresh)
  document.addEventListener('visibilitychange', refresh)
  setInterval(refresh, 30_000)
}
