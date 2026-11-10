import { useMemo, useState } from 'react'
import {
  ActionIcon,
  Button,
  Group,
  Highlight,
  Loader,
  Menu,
  Modal,
  ScrollArea,
  Stack,
  Text,
  TextInput,
  UnstyledButton,
} from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { useChatStore, type Conversation } from '../../store/chatStore'
import { IconDots, IconPencil, IconPlus, IconSearch, IconTrash } from './icons'
import classes from './ChatSidebar.module.css'

const DAY = 24 * 60 * 60 * 1000

function dateGroup(ts: number, now: Date) {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  if (ts >= startOfToday) return 'Today'
  if (ts >= startOfToday - DAY) return 'Yesterday'
  if (ts >= startOfToday - 7 * DAY) return 'Previous 7 days'
  if (ts >= startOfToday - 30 * DAY) return 'Previous 30 days'
  return 'Older'
}

/** Text around the first match in a chat's messages, for search results. */
function matchSnippet(c: Conversation, q: string) {
  for (const m of c.messages) {
    const idx = m.content.toLowerCase().indexOf(q)
    if (idx >= 0) {
      const start = Math.max(0, idx - 30)
      const text = m.content.slice(start, idx + q.length + 50).replace(/\s+/g, ' ')
      return `${start > 0 ? '…' : ''}${text}…`
    }
  }
  return null
}

interface ChatSidebarProps {
  onNavigate?: () => void
}

export default function ChatSidebar({ onNavigate }: ChatSidebarProps) {
  const conversations = useChatStore((s) => s.conversations)
  const activeId = useChatStore((s) => s.activeId)
  const pending = useChatStore((s) => s.pending)
  const { newChat, selectChat, deleteChat, renameChat } = useChatStore.getState()

  const [query, setQuery] = useState('')
  const [debounced] = useDebouncedValue(query.trim().toLowerCase(), 150)
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<Conversation | null>(null)

  const results = useMemo(() => {
    // Hide empty "New chat" drafts from the list; they appear once a question is asked.
    const all = Object.values(conversations)
      .filter((c) => c.messages.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt)
    if (!debounced) return all.map((c) => ({ c, snippet: null as string | null }))
    return all
      .map((c) => {
        const inTitle = c.title.toLowerCase().includes(debounced)
        const snippet = matchSnippet(c, debounced)
        return inTitle || snippet ? { c, snippet: inTitle ? null : snippet } : null
      })
      .filter((r): r is { c: Conversation; snippet: string | null } => r !== null)
  }, [conversations, debounced])

  const groups = useMemo(() => {
    const now = new Date()
    const out: { label: string; items: typeof results }[] = []
    for (const r of results) {
      const label = debounced ? 'Search results' : dateGroup(r.c.updatedAt, now)
      const last = out[out.length - 1]
      if (last?.label === label) last.items.push(r)
      else out.push({ label, items: [r] })
    }
    return out
  }, [results, debounced])

  const open = (id: string) => {
    selectChat(id)
    onNavigate?.()
  }

  return (
    <Stack h="100%" gap="sm" p="sm">
      <Button
        variant="default"
        size="md"
        radius="md"
        leftSection={<IconPlus size={18} />}
        justify="flex-start"
        onClick={() => {
          newChat()
          setQuery('')
          onNavigate?.()
        }}
      >
        New chat
      </Button>

      <TextInput
        placeholder="Search chats…"
        size="md"
        radius="md"
        leftSection={<IconSearch size={18} />}
        value={query}
        onChange={(e) => setQuery(e.currentTarget.value)}
        aria-label="Search chats"
      />

      <ScrollArea style={{ flex: 1 }} type="auto" offsetScrollbars>
        <Stack gap="lg">
          {groups.length === 0 && (
            <Text size="md" c="dimmed" ta="center" mt="lg">
              {debounced ? `No chats match “${query.trim()}”` : 'No chats yet. Ask a question to start one.'}
            </Text>
          )}
          {groups.map((g) => (
            <div key={g.label}>
              <Text size="xs" fw={600} c="dimmed" px={12} mb={6}>
                {g.label}
              </Text>
              <Stack gap={2}>
                {g.items.map(({ c, snippet }) => (
                  <div key={c.id} className={classes.item} data-active={c.id === activeId || undefined}>
                    <UnstyledButton className={classes.itemButton} onClick={() => open(c.id)} px={12} py={10}>
                      <Group gap={8} wrap="nowrap">
                        {pending[c.id] && <Loader size={12} />}
                        <Highlight highlight={debounced} className={classes.title} truncate="end" style={{ flex: 1, minWidth: 0 }}>
                          {c.title}
                        </Highlight>
                      </Group>
                      {snippet && (
                        <Highlight highlight={debounced} size="sm" c="dimmed" lineClamp={2} mt={2}>
                          {snippet}
                        </Highlight>
                      )}
                    </UnstyledButton>
                    <Menu position="bottom-end" withinPortal>
                      <Menu.Target>
                        <ActionIcon
                          className={classes.itemMenu}
                          variant="subtle"
                          color="gray"
                          size="lg"
                          aria-label={`Options for ${c.title}`}
                        >
                          <IconDots size={18} />
                        </ActionIcon>
                      </Menu.Target>
                      <Menu.Dropdown>
                        <Menu.Item leftSection={<IconPencil size={14} />} onClick={() => setRenaming({ id: c.id, title: c.title })}>
                          Rename
                        </Menu.Item>
                        <Menu.Item color="red" leftSection={<IconTrash size={14} />} onClick={() => setConfirmDelete(c)}>
                          Delete
                        </Menu.Item>
                      </Menu.Dropdown>
                    </Menu>
                  </div>
                ))}
              </Stack>
            </div>
          ))}
        </Stack>
      </ScrollArea>

      <Text size="sm" c="dimmed" px={12}>
        Chats are saved in this browser only.
      </Text>

      <Modal opened={!!renaming} onClose={() => setRenaming(null)} title="Rename chat" centered>
        <form
          aria-label="Rename chat"
          onSubmit={(e) => {
            e.preventDefault()
            if (renaming && renaming.title.trim()) renameChat(renaming.id, renaming.title)
            setRenaming(null)
          }}
        >
          <TextInput
            data-autofocus
            label="Chat name"
            value={renaming?.title ?? ''}
            onChange={(e) => {
              const title = e.currentTarget.value
              setRenaming((r) => (r ? { ...r, title } : r))
            }}
          />
          <Group justify="flex-end" mt="md">
            <Button variant="default" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button type="submit">Save</Button>
          </Group>
        </form>
      </Modal>

      <Modal opened={!!confirmDelete} onClose={() => setConfirmDelete(null)} title="Delete chat?" centered>
        <Text size="sm">
          This permanently deletes <b>{confirmDelete?.title}</b> from this browser.
        </Text>
        <Group justify="flex-end" mt="md">
          <Button variant="default" onClick={() => setConfirmDelete(null)}>
            Cancel
          </Button>
          <Button
            color="red"
            onClick={() => {
              if (confirmDelete) deleteChat(confirmDelete.id)
              setConfirmDelete(null)
            }}
          >
            Delete
          </Button>
        </Group>
      </Modal>
    </Stack>
  )
}
