import axios from 'axios'
import { useAuthStore } from '../store/authStore'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

client.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

client.interceptors.response.use(
  (res) => res,
  (err) => {
    // An expired or invalid session sends the user back to the login page.
    // (A failed login attempt is also a 401, but there is no session to clear then.)
    if (err.response?.status === 401 && useAuthStore.getState().token) {
      useAuthStore.getState().logout()
    }
    const message = err.response?.data?.error ?? err.message ?? 'Unknown error'
    return Promise.reject(new Error(message))
  }
)

export default client
