import { useState } from 'react'
import { Alert, Button, Group, PasswordInput, Stack, Text, TextInput, Title } from '@mantine/core'
import { login } from '../api/auth'
import { useAuthStore } from '../store/authStore'
import { LogoMark } from '../components/sidebar/icons'
import classes from './LoginPage.module.css'

export default function LoginPage() {
  const setSession = useAuthStore((s) => s.setSession)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) {
      setError('Enter your username and password.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const { token, user } = await login(username.trim(), password)
      setSession(token, user.username)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed. Check your username and password.')
      setPassword('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={classes.shell}>
      <aside className={classes.statute} aria-hidden="true">
        <div className={classes.statuteBrand}>
          <LogoMark size={30} />
          Tax Act Q&amp;A
        </div>
        <figure className={classes.quote}>
          <div className={classes.quoteSection}>Section 11(a)</div>
          <blockquote className={classes.quoteText}>
            “expenditure and losses actually incurred in the production of the income, provided such expenditure
            and losses are not of a capital nature”
          </blockquote>
        </figure>
        <p className={classes.statuteNote}>
          Every answer links each point to the passage of the Income Tax Act it comes from, so you can read the
          law for yourself.
        </p>
      </aside>

      <main className={classes.formSide}>
        <form onSubmit={handleSubmit} noValidate className={classes.form}>
          <Stack gap="md">
            <Group gap="xs" className={classes.mobileBrand}>
              <LogoMark size={30} />
              <Text fw={650}>Tax Act Q&amp;A</Text>
            </Group>
            <div>
              <Title order={1} size="h2">
                Log in
              </Title>
              <Text c="dimmed" size="sm" mt={6}>
                Ask questions about the South African Income Tax Act.
              </Text>
            </div>
            {error && (
              <Alert color="red" variant="light" role="alert">
                {error}
              </Alert>
            )}
            <TextInput
              label="Username"
              name="username"
              autoComplete="username"
              spellCheck={false}
              autoCapitalize="none"
              autoFocus
              size="md"
              value={username}
              onChange={(e) => setUsername(e.currentTarget.value)}
            />
            <PasswordInput
              label="Password"
              name="password"
              autoComplete="current-password"
              size="md"
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
            />
            <Button type="submit" loading={loading} fullWidth size="md" mt={4}>
              Log in
            </Button>
          </Stack>
        </form>
      </main>
    </div>
  )
}
