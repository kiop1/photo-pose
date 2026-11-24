import assert from 'node:assert/strict'
import { Readable, Writable } from 'node:stream'
import test from 'node:test'
import { poseApi } from '../tools/shuttermuse-web.mjs'

async function send(data, predict, headers = {}, url = '/api/shuttermuse') {
  const req = Readable.from([Buffer.from(JSON.stringify(data))])
  req.url = url
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

test('accepts a valid previous suggestion when requesting another pose', async () => {
  const photo = { image: 'iVBORw0KGgo=', mime: 'image/png' }
  const suggestion = '可以换另一条腿承重。'
  const predict = async ({ previousSuggestion }) => ({ previousSuggestion })
  for (const task of ['pose', 'scene-pose', 'joint']) {
    const result = await send({ ...photo, task, previousSuggestion: ' ' + suggestion + ' ' }, predict)
    assert.deepEqual(result, { status: 200, json: { previousSuggestion: suggestion } })
    for (const previousSuggestion of [null, ' ', 'x'.repeat(501)]) {
      assert.equal((await send({ ...photo, task, previousSuggestion }, predict)).status, 400)
    }
  }
})

test('accepts photo-specific direction for planning and for grounding a chosen pose', async () => {
  const photo = { image: 'iVBORw0KGgo=', mime: 'image/png' }
  const predict = async ({ poseChoices, instruction, selectedSuggestion, retryGrounding }) => ({ poseChoices, instruction, selectedSuggestion, retryGrounding })
  const instruction = '保持站立，手低于肩膀'
  const selectedSuggestion = '双脚错开，一手轻扶衣角'
  for (const task of ['pose', 'scene-pose', 'joint']) {
    for (const options of [{ poseChoices: true }, { selectedSuggestion }, { selectedSuggestion, retryGrounding: true }]) {
      const result = await send({ ...photo, task, instruction: ' ' + instruction, ...options }, predict)
      assert.deepEqual(result, { status: 200, json: { instruction, ...options } })
    }
  }
  for (const invalid of [{ poseChoices: 'true' }, { poseChoices: true, selectedSuggestion },
    { instruction: 5 }, { instruction: '长'.repeat(301) }, { selectedSuggestion: ' ' }, { selectedSuggestion: '长'.repeat(501) },
    { retryGrounding: true }, { selectedSuggestion, retryGrounding: 1 }, { task: 'composition', instruction }]) {
    assert.equal((await send({ ...photo, ...invalid }, predict)).status, 400)
  }
})

test('validates crop intent and original dimensions and forwards scene tasks', async () => {
  const photo = { image: 'iVBORw0KGgo=', mime: 'image/png' }
  const predict = async ({ task, intent, width, height }) => ({ task, intent, width, height })
  const crop = { ...photo, task: 'crop', intent: '4:3', width: 1920, height: 1080 }
  for (const fields of [{ intent: 'wide' }, { width: undefined }, { height: '1080' }, { width: 3 }, { height: 2.5 }]) {
    assert.equal((await send({ ...crop, ...fields }, predict)).status, 400)
  }
  assert.deepEqual(await send(crop, predict), { status: 200, json: { task: 'crop', intent: '4:3', width: 1920, height: 1080 } })
  assert.deepEqual(await send({ ...photo, task: 'crop', intent: 'half-body' }, predict), {
    status: 200, json: { task: 'crop', intent: 'half-body' },
  })
  for (const task of ['scene-pose', 'joint']) {
    assert.deepEqual(await send({ ...photo, task }, predict), { status: 200, json: { task } })
  }
})

test('preview requires a local photo and its suggestion before calling the editor', async () => {
  const original = Buffer.from('ffd8ff', 'hex')
  const guide = Buffer.from('ffd8ff2d2d2dffd9', 'hex')
  const photo = { mime: 'image/jpeg', image: original.toString('base64') }
  const suggestion = '可以试试双臂向两侧伸展。'
  const inputs = []
  const edit = async (input) => {
    inputs.push({ poseGuide: input.poseGuide, landmarks: input.landmarks })
    assert.deepEqual(input.image, original)
    assert.equal(input.suggestion, suggestion)
    assert.ok(input.signal instanceof AbortSignal)
    return { url: 'https://cdn.example.test/demo.png' }
  }
  const preview = (data, headers) => send(data, edit, headers, '/api/shuttermuse/preview')
  assert.equal((await preview(photo)).status, 400)
  assert.equal((await preview({ ...photo, suggestion: 'a'.repeat(501) })).status, 400)
  const data = { ...photo, suggestion }
  assert.equal((await preview(data, { origin: 'https://example.org' })).status, 403)
  assert.equal((await preview({ ...data, mime: 'image/png' })).status, 415)
  for (const poseGuide of ['', null, '!!!', 'data:image/jpeg;base64,' + guide.toString('base64')]) {
    assert.equal((await preview({ ...data, poseGuide })).status, 400)
  }
  assert.equal((await preview({ ...data, poseGuide: 'iVBORw0KGgo=' })).status, 415)
  assert.equal((await preview({ ...data, poseGuide: 'A'.repeat(2 * 1024 * 1024) })).status, 413)
  assert.equal((await preview(data, { 'content-length': String(2 * 1024 * 1024 + 1) })).status, 413)
  assert.deepEqual(await preview(data), { status: 200, json: { url: 'https://cdn.example.test/demo.png' } })
  assert.deepEqual(await preview({ ...data, poseGuide: guide.toString('base64') }),
    { status: 200, json: { url: 'https://cdn.example.test/demo.png' } })
  const landmarks = Array.from({ length: 17 }, (_, index) => ({
    x: index === 5 ? 0 : 1, y: 0.4, visibility: [5, 7].includes(index) ? 1 : 0,
  }))
  const guided = { ...data, poseGuide: guide.toString('base64') }
  const targetPoseGuide = guide.toString('base64')
  assert.equal((await preview({ ...data, targetPoseGuide })).status, 400)
  assert.equal((await preview({ ...data, landmarks, targetPoseGuide: 'iVBORw0KGgo=' })).status, 415)
  assert.equal((await preview({ ...data, landmarks, targetPoseGuide: '!!!' })).status, 400)
  assert.equal((await preview({ ...data, landmarks, instruction: ' ' })).status, 400)
  const withGuide = await send({ ...data, landmarks, targetPoseGuide, instruction: ' 保持站姿 ' }, async input => {
    assert.deepEqual(input.targetPoseGuide, guide)
    assert.equal(input.instruction, '保持站姿')
    return { ok: true }
  }, {}, '/api/shuttermuse/preview')
  assert.equal(withGuide.status, 200)
  for (const malformed of [null, landmarks.slice(1),
    landmarks.map(point => ({ ...point, x: '0.25' })),
    landmarks.map(point => ({ ...point, y: -0.01 })),
    landmarks.map(point => ({ ...point, visibility: 0.5 })),
    landmarks.map(point => ({ ...point, visibility: -1 })),
    landmarks.map(point => ({ ...point, x: 1 })),
    landmarks.map((point, index) => ({ ...point, visibility: [5, 10].includes(index) ? 1 : 0 })),
  ]) {
    assert.equal((await preview({ ...data, landmarks: malformed })).status, 400)
  }
  assert.deepEqual(await preview({ ...data, landmarks }),
    { status: 200, json: { url: 'https://cdn.example.test/demo.png' } })
  assert.deepEqual(await preview({ ...guided, landmarks }),
    { status: 200, json: { url: 'https://cdn.example.test/demo.png' } })
  assert.deepEqual(inputs, [
    { poseGuide: undefined, landmarks: undefined }, { poseGuide: undefined, landmarks: undefined },
    { poseGuide: undefined, landmarks }, { poseGuide: undefined, landmarks },
  ])

  for (const previewMode of ['scene-pose', 'joint']) {
    assert.equal((await preview({ ...guided, previewMode })).status, 400)
    assert.deepEqual(await send({ ...data, previewMode, landmarks }, async input => ({ previewMode: input.previewMode }),
      {}, '/api/shuttermuse/preview'), { status: 200, json: { previewMode } })
  }
  assert.equal((await preview({ ...guided, landmarks, previewMode: 'crop' })).status, 400)
})
