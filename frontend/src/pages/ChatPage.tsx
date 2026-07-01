import { useEffect, useRef, useState } from 'react'
import { AppShell, Badge, Button, Container, Group, Paper, ScrollArea, Stack, Text, Textarea } from '@mantine/core'
import { getHealth } from '../api/chat'
import { useChat } from '../hooks/useChat'
import MessageBubble from '../components/chat/MessageBubble'

export default function ChatPage() {
  const { messages, sending, error, send } = useChat()
  const [input, setInput] = useState('')
  const [useFinetuned, setUseFinetuned] = useState(false)
  const viewport = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getHealth()
      .then((h) => setUseFinetuned(h.use_finetuned))
      .catch(() => {})
  }, [])

  useEffect(() => {
    viewport.current?.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || sending) return
    send(input)
    setInput('')
  }

  return (
    <AppShell header={{ height: 56 }} padding={0}>
      <AppShell.Header style={{ display: 'flex', alignItems: 'center', padding: '0 1.5rem' }}>
        <Group justify="space-between" w="100%">
          <Text fw={600}>SA Income Tax Act Q&A</Text>
          {useFinetuned ? (
            <Badge color="blue">Fine-tuned model</Badge>
          ) : (
            <Text size="xs" c="dimmed">Set TAX_MODEL_ADAPTER for fine-tuned mode.</Text>
          )}
        </Group>
      </AppShell.Header>

      <AppShell.Main style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <ScrollArea style={{ flex: 1 }} viewportRef={viewport}>
          <Container size="sm" py="md">
            <Stack gap="md">
              {messages.map((m, i) => (
                <MessageBubble key={i} role={m.role} content={m.content} />
              ))}
            </Stack>
          </Container>
        </ScrollArea>

        <Paper p="md" withBorder radius={0}>
          <Container size="sm">
            <Group align="flex-end" gap="sm">
              <Textarea
                style={{ flex: 1 }}
                placeholder="Type your question..."
                autosize
                minRows={1}
                maxRows={5}
                value={input}
                onChange={(e) => setInput(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
              />
              <Button onClick={handleSend} loading={sending} disabled={!input.trim()}>
                Send
              </Button>
            </Group>
            {error && (
              <Text size="xs" c="red" mt="xs">
                {error}
              </Text>
            )}
          </Container>
        </Paper>
      </AppShell.Main>
    </AppShell>
  )
}
