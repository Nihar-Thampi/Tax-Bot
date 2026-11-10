import client from './client'

export interface LoginResponse {
  token: string
  user: { username: string }
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const { data } = await client.post<LoginResponse>('/auth/login', { username, password })
  return data
}

export async function getMe(): Promise<{ user: { username: string } }> {
  const { data } = await client.get<{ user: { username: string } }>('/auth/me')
  return data
}
