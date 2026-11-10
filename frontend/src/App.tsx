import { useEffect, useState } from 'react'
import { Center, Loader } from '@mantine/core'
import { getMe } from './api/auth'
import { useAuthStore } from './store/authStore'
import { switchChatUser } from './store/chatStore'
import ChatPage from './pages/ChatPage'
import LoginPage from './pages/LoginPage'

function App() {
  const token = useAuthStore((s) => s.token)
  const username = useAuthStore((s) => s.username)
  const sessionUser = token ? username : null
  // Which user's chats are loaded; ChatPage waits until it matches the session.
  const [chatsFor, setChatsFor] = useState<string | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    switchChatUser(sessionUser).then(() => {
      if (!cancelled) setChatsFor(sessionUser)
    })
    return () => {
      cancelled = true
    }
  }, [sessionUser])

  // Check a remembered session is still valid; a 401 logs out via the API client.
  useEffect(() => {
    if (token) getMe().catch(() => {})
  }, [token])

  if (!sessionUser) return <LoginPage />
  if (chatsFor !== sessionUser) {
    return (
      <Center mih="100dvh">
        <Loader />
      </Center>
    )
  }
  return <ChatPage />
}

export default App
