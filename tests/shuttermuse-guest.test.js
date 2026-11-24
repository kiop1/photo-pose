import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'
import { gatewayAccessToken } from '../tools/shuttermuse-guest.mjs'

test('a fresh install gets a guest session without provider credentials and reuses its device ID', async t => {
  const prior = process.env.AGENT_GATEWAY_TOKEN
  delete process.env.AGENT_GATEWAY_TOKEN
  t.after(() => { if (prior !== undefined) process.env.AGENT_GATEWAY_TOKEN = prior })
  const dir = await mkdtemp(join(tmpdir(), 'pose-guest-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  const directory = pathToFileURL(dir + '/')
  const base = 'http://127.0.0.1:8080'
  const requests = []
  let token
  const payload = Buffer.from(JSON.stringify({ exp: Date.now() / 1000 + 3600 })).toString('base64url')
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, base + '/api/auth/guest')
    assert.equal(options.headers.Authorization, undefined)
    requests.push(JSON.parse(options.body))
    return Response.json({ code: 200, data: { access_token: token } })
  })
  await assert.rejects(gatewayAccessToken(base, undefined, directory), { status: 502 })
  token = 'header.' + payload + '.signature'
  assert.equal(await gatewayAccessToken(base, undefined, directory), token)
  assert.equal(await gatewayAccessToken(base, undefined, directory), token)
  assert.equal(requests.length, 2)
  assert.deepEqual(requests[0], requests[1])
  assert.equal(requests[0].app_id, 'photo-pose')
  assert.equal(requests[0].platform, 'web')
  assert.match(requests[0].device_id, /^[a-f0-9-]{36}$/)
})
