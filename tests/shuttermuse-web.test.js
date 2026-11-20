import assert from 'node:assert/strict'
import { Readable, Writable } from 'node:stream'
import test from 'node:test'
import { poseApi } from '../tools/shuttermuse-web.mjs'

async function send(data, predict, headers = {}) {
  const req = Readable.from([Buffer.from(JSON.stringify(data))])
  req.headers = { host: '127.0.0.1:4174', 'content-type': 'application/json', ...headers }
  req.method = 'POST'
  const chunks = []
  const res = new Writable({ write(chunk, _, done) { chunks.push(chunk); done() } })
  res.setHeader = () => {}
  await poseApi(req, res, predict)
  return { status: res.statusCode || 200, json: JSON.parse(Buffer.concat(chunks).toString()) }
}

test('forwards only validated local image bytes to inference and returns model output', async () => {
  const png = Buffer.from('89504e470d0a1a0a','hex')
  const photo = { image: png.toString('base64'), mime: 'image/png' }
  const answer = { landmarks: Array(17).fill({ x: 0.5, y: 0.25, visibility: 1 }) }
  let calls = 0
  const predict = async ({ image, mime, task }) => {
    calls++
    assert.deepEqual(image, png)
    assert.equal(mime, photo.mime)
    assert.equal(task, 'pose')
    return answer
  }
  assert.equal((await send(photo, predict, { origin: 'https://example.org' })).status, 403)
  assert.equal((await send({ ...photo, mime: 'image/jpeg' }, predict)).status, 415)
  assert.equal((await send({ ...photo, image: '!invalid' }, predict)).status, 400)
  assert.equal((await send(photo, predict, { 'content-length': '12000000' })).status, 413)
  assert.deepEqual(await send(photo, predict), { status: 200, json: answer })
  assert.equal(calls, 1)
  assert.equal((await send({ ...photo, task: 'unknown' }, predict)).status, 400)
  const composition = await send({ ...photo, task: 'composition' }, async ({ task }) => {
    assert.equal(task, 'composition')
    return { decision: 'keep' }
  })
  assert.deepEqual(composition, { status: 200, json: { decision: 'keep' } })
  const advice = await send({ ...photo, task: 'advice' }, async ({ image, mime, task }) => {
    assert.deepEqual(image, png)
    assert.equal(mime, photo.mime)
    assert.equal(task, 'advice')
    return { suggestion: '可以保留山脉上方的留白。' }
  })
  assert.deepEqual(advice, { status: 200, json: { suggestion: '可以保留山脉上方的留白。' } })
})
