import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'

const localDirectory = new URL('../.pose-local/', import.meta.url)
let session

function installationID(directory) {
  mkdirSync(directory, { recursive: true, mode: 0o700 })
  const file = new URL('gateway-device-id', directory)
  try {
    writeFileSync(file, randomUUID(), { flag: 'wx', mode: 0o600 })
  } catch (error) {
    if (error.code !== 'EEXIST') throw error
  }
  const id = readFileSync(file, 'utf8').trim()
  if (!/^[\da-f]{8}-(?:[\da-f]{4}-){3}[\da-f]{12}$/i.test(id)) throw Error('Invalid local device ID')
  return id
}

export async function gatewayAccessToken(base, signal, directory = localDirectory) {
  const existing = process.env.AGENT_GATEWAY_TOKEN?.trim()
  if (existing) return existing
  const device = installationID(directory)
  if (session?.base === base && session.device === device && session.expires > Date.now() + 60_000) return session.token

  const response = await fetch(base + '/api/auth/guest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ app_id: 'photo-pose', device_id: device, platform: 'web' }),
    signal,
    redirect: 'error',
  })
  const result = response.ok ? await response.json().catch(() => null) : null
  if (!response.ok) await response.body?.cancel()
  const token = result?.data?.access_token
  if (result?.code !== 200 || typeof token !== 'string' || !token.trim()) {
    throw Object.assign(Error('网关游客登录失败，请检查本机网关的登录服务。'), { status: 502 })
  }
  // The gateway validates the JWT; its expiry here only controls reuse.
  let expires = 0
  try { expires = Number(JSON.parse(Buffer.from(token.split('.')[1], 'base64url')).exp) * 1000 } catch {}
  session = { base, device, token, expires }
  return token
}
