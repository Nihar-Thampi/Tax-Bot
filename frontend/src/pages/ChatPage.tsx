import { useEffect, useRef, useState } from 'react'
import {
  ActionIcon,
  AppShell,
  Avatar,
  Badge,
  Burger,
  Container,
  Group,
  Menu,
  ScrollArea,
  Stack,
  Text,
  Textarea,
  Title,
  Tooltip,
  UnstyledButton,
  useComputedColorScheme,
  useMantineColorScheme,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { getHealth } from '../api/chat'
import { useChat } from '../hooks/useChat'
import { useAuthStore } from '../store/authStore'
import { WELCOME_TEXT } from '../store/chatStore'
import MessageBubble from '../components/chat/MessageBubble'
import ChatSidebar from '../components/sidebar/ChatSidebar'
import {
  IconArrowUp,
  IconLogout,
  IconMoon,
  IconStop,
  IconSun,
  LogoMark,
} from '../components/sidebar/icons'
import classes from './ChatPage.module.css'

const EXAMPLE_QUESTIONS = [
  'What expenses can I deduct from income from my trade?',
  'How are assessed losses set off against income?',
  'What income is exempt from tax?',
  'Can medical expenses be deducted?',
]

export default function ChatPage() {
  const { activeId, messages, sending, send, stop } = useChat()
  const username = useAuthStore((s) => s.username)
  const logout = useAuthStore((s) => s.logout)
  const [input, setInput] = useState('')
  const [useFinetuned, setUseFinetuned] = useState(false)
  const [navOpened, { toggle: toggleNav, close: closeNav }] = useDisclosure(false)
  const { setColorScheme } = useMantineColorScheme()
  const scheme = useComputedColorScheme('light')
  const viewport = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    getHealth()
      .then((h) => setUseFinetuned(h.use_finetuned))
      .catch(() => {})
  }, [])

  useEffect(() => {
    viewport.current?.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' })
  }, [messages, sending])

  // Fresh composer when switching chats.
  const [inputFor, setInputFor] = useState(activeId)
  if (inputFor !== activeId) {
    setInputFor(activeId)
    setInput('')
  }
  useEffect(() => {
    inputRef.current?.focus()
  }, [activeId])

  const handleSend = (text = input) => {
    if (!text.trim() || sending) return
    send(text)
    setInput('')
  }

  return (
    <AppShell
      header={{ height: 60 }}
      navbar={{ width: 300, breakpoint: 'sm', collapsed: { mobile: !navOpened } }}
      padding={0}
    >
      <AppShell.Header className={classes.header}>
        <Group justify="space-between" w="100%" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <Burger opened={navOpened} onClick={toggleNav} hiddenFrom="sm" size="sm" aria-label="Toggle chat list" />
            <LogoMark size={28} />
            <Text className={classes.brand} component="h1" m={0}>
              Tax Act Q&amp;A
            </Text>
          </Group>
          <Group gap="xs" wrap="nowrap">
            <Badge
              className={classes.modeBadge}
              variant="light"
              color={useFinetuned ? 'ledger' : 'gray'}
              radius="sm"
              visibleFrom="xs"
            >
              {useFinetuned ? 'Fine-tuned model' : 'Answers from the Act'}
            </Badge>
            <Tooltip label={scheme === 'dark' ? 'Light mode' : 'Dark mode'} withArrow>
              <ActionIcon
                variant="subtle"
                color="gray"
                size="lg"
                onClick={() => setColorScheme(scheme === 'dark' ? 'light' : 'dark')}
                aria-label={scheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {scheme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
              </ActionIcon>
            </Tooltip>
            <Menu position="bottom-end" withinPortal>
              <Menu.Target>
                <UnstyledButton aria-label="Account menu" style={{ borderRadius: '50%' }}>
                  <Avatar size={34} radius="xl" color="ledger" variant="light">
                    {username?.slice(0, 2).toUpperCase()}
                  </Avatar>
                </UnstyledButton>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Label>Signed in as {username}</Menu.Label>
                <Menu.Item leftSection={<IconLogout size={14} />} onClick={logout}>
                  Log out
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar className={classes.navbar}>
        <ChatSidebar onNavigate={closeNav} />
      </AppShell.Navbar>

      <AppShell.Main className={classes.main}>
        <ScrollArea style={{ flex: 1 }} viewportRef={viewport}>
          <Container size={760} py="lg">
            {messages.length === 0 ? (
              <div className={classes.welcome}>
                <Text className={classes.welcomeAct}>Income Tax Act No. 58 of 1962</Text>
                <Title order={2} className={classes.welcomeTitle}>
                  Ask the Act a question.
                </Title>
                <Text c="dimmed" className={classes.welcomeText}>
                  {WELCOME_TEXT}
                </Text>
                <Text size="sm" fw={600} mt={36} mb={8} id="try-asking">
                  Try asking
                </Text>
                <ul className={classes.suggestions} aria-labelledby="try-asking">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <li key={q}>
                      <button type="button" className={classes.suggestion} onClick={() => handleSend(q)}>
                        <span className={classes.suggestionMark} aria-hidden="true">
                          §
                        </span>
                        {q}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <Stack gap="xl">
                {messages.map((m, i) => (
                  <MessageBubble key={i} {...m} />
                ))}
                {sending && (
                  <div className={classes.thinking} role="status" aria-live="polite">
                    <span className={classes.dots} aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                    Searching the Act and writing an answer…
                  </div>
                )}
              </Stack>
            )}
          </Container>
        </ScrollArea>

        <div className={classes.composerDock}>
          <Container size={760} px={0}>
            <div className={classes.composer}>
              <Textarea
                ref={inputRef}
                className={classes.composerInput}
                variant="unstyled"
                placeholder="Ask about the Income Tax Act…"
                aria-label="Your question"
                autosize
                minRows={1}
                maxRows={6}
                value={input}
                onChange={(e) => setInput(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
              />
              {sending ? (
                <ActionIcon size={38} radius="xl" color="red" onClick={stop} aria-label="Stop generating">
                  <IconStop size={16} />
                </ActionIcon>
              ) : (
                <ActionIcon
                  size={38}
                  radius="xl"
                  onClick={() => handleSend()}
                  disabled={!input.trim()}
                  aria-label="Send question"
                >
                  <IconArrowUp size={18} />
                </ActionIcon>
              )}
            </div>
            <Text size="xs" c="dimmed" mt={8} ta="center" className={classes.footnote}>
              Based on the Act as published in 1962; later amendments are not included. Informational only, not
              professional tax advice.
            </Text>
          </Container>
        </div>
      </AppShell.Main>
    </AppShell>
  )
}
