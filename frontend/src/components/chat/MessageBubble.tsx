import { Paper, Text } from '@mantine/core'
import type { ChatMessage } from '../../types'

export default function MessageBubble({ role, content }: ChatMessage) {
  const isUser = role === 'user'
  return (
    <Paper
      p="sm"
      radius="md"
      withBorder
      style={{
        marginLeft: isUser ? '10%' : 0,
        marginRight: isUser ? 0 : '10%',
        borderLeft: `3px solid var(--mantine-color-${isUser ? 'blue-5' : 'green-6'})`,
      }}
    >
      <Text size="xs" tt="uppercase" c="dimmed" mb={4}>
        {isUser ? 'You' : 'Assistant'}
      </Text>
      <Text style={{ whiteSpace: 'pre-wrap' }}>{content}</Text>
    </Paper>
  )
}
