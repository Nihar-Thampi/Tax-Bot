import { useEffect, useId, useRef, useState } from 'react'
import { CloseButton, Group, Paper, Stack, Text, UnstyledButton } from '@mantine/core'
import { LogoMark } from '../sidebar/icons'
import type { DisplayMessage, Source } from '../../types'
import classes from './MessageBubble.module.css'

// Same rules as the backend (citation_service.DISCLAIMER_MAX_CHARS, chat_service.NO_CITED_ANSWER).
const DISCLAIMER_MAX_CHARS = 250
const NO_CITED_ANSWER =
  "I couldn't find a passage in the Income Tax Act that supports an answer to this. Try rephrasing the question or asking about a specific section."

function sourceLocation(s: Source) {
  const lines = s.line_start === s.line_end ? `line ${s.line_start}` : `lines ${s.line_start}–${s.line_end}`
  return s.page ? `Gazette page ${s.page}, ${lines}` : lines
}

/** "Section 11(1)" → "§ 11(1)"; schedule paragraphs keep their own wording. */
function shortSection(section: string) {
  return section.replace(/^Section\s+/i, '§\u00a0')
}

export default function MessageBubble({ role, content, points, sources }: DisplayMessage) {
  const isUser = role === 'user'
  const panelId = `citation-${useId().replace(/:/g, '')}`
  // Index (in shownPoints) of the point whose citation is open; one at a time per message.
  const [openPoint, setOpenPoint] = useState<number | null>(null)
  const sourceById = new Map((sources ?? []).map((s) => [s.id, s]))
  // Points with no matching passage are not shown. Chats saved before that rule may still
  // contain them, including long paragraphs once mislabelled as disclaimers (never checked).
  const shownPoints = points?.filter(
    (p) =>
      p.kind === 'cited' || p.kind === 'summary' || (p.kind === 'disclaimer' && p.text.length <= DISCLAIMER_MAX_CHARS),
  )
  const nothingCited = !!points && !shownPoints?.some((p) => p.kind === 'cited')

  const openSourceId = openPoint != null ? shownPoints?.[openPoint]?.source_id : null
  const openSource = openSourceId != null ? sourceById.get(openSourceId) : undefined
  const panelRef = useRef<HTMLDivElement>(null)

  // On a long answer the panel can be below the fold; bring it into view when a citation opens.
  useEffect(() => {
    if (openPoint != null) panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [openPoint])

  if (isUser) {
    return (
      <div className={classes.userRow}>
        <div className={classes.userBubble}>{content}</div>
      </div>
    )
  }

  return (
    <div className={classes.assistantRow}>
      <div className={classes.assistantMark}>
        <LogoMark size={30} />
      </div>
      <div className={classes.assistantBody}>
        {nothingCited ? (
          <Text className={classes.point}>{NO_CITED_ANSWER}</Text>
        ) : shownPoints && shownPoints.length > 0 ? (
          <Stack gap={10}>
            {shownPoints.map((p, i) => {
              const source = p.kind === 'cited' && p.source_id != null ? sourceById.get(p.source_id) : undefined
              const isOpen = openPoint === i
              return (
                <Text
                  key={i}
                  c={p.kind === 'disclaimer' ? 'dimmed' : undefined}
                  size={p.kind === 'disclaimer' ? 'sm' : undefined}
                  className={`${classes.point} ${isOpen ? classes.activePoint : ''}`}
                >
                  {p.text}
                  {source && (
                    <UnstyledButton
                      component="span"
                      role="button"
                      tabIndex={0}
                      className={classes.citation}
                      data-active={isOpen || undefined}
                      aria-expanded={isOpen}
                      aria-controls={panelId}
                      aria-label={`Show citation: ${source.section}`}
                      title={source.section}
                      onClick={() => setOpenPoint(isOpen ? null : i)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setOpenPoint(isOpen ? null : i)
                        }
                      }}
                    >
                      {shortSection(source.section)}
                    </UnstyledButton>
                  )}
                </Text>
              )
            })}
          </Stack>
        ) : (
          <Text className={classes.point}>{content}</Text>
        )}

        {openSource && (
          <Paper ref={panelRef} id={panelId} mt="md" p="md" radius="md" className={classes.panel} aria-live="polite">
            <Group justify="space-between" align="flex-start" wrap="nowrap" gap="xs">
              <div>
                <Text className={classes.panelSection}>{openSource.section}</Text>
                <Text size="xs" c="dimmed">
                  Income Tax Act No. 58 of 1962, {sourceLocation(openSource)}
                </Text>
              </div>
              <CloseButton size="sm" aria-label="Close citation" onClick={() => setOpenPoint(null)} />
            </Group>
            <Text mt="sm" className={classes.excerpt}>
              {openSource.excerpt}
            </Text>
          </Paper>
        )}
      </div>
    </div>
  )
}
