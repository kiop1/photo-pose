import assert from 'node:assert/strict'
import test, { beforeEach } from 'node:test'
import { gatewayBodyLimit, generatePreview } from '../tools/shuttermuse-preview.mjs'

const source = {
  image: Buffer.from('89504e470d0a1a0a', 'hex'),
  mime: 'image/png',
  suggestion: '可以把左肩稍微转向镜头。',
}
const guide = Buffer.from('ffd8ff2d2d2dffd9', 'hex')
const landmarks = Array.from({ length: 17 }, (_, index) => ({
  x: index === 5 ? 0.25 : index === 9 ? 0.75 : 0.5,
  y: index === 7 ? 0.625 : 0.5,
  visibility: index === 1 ? -1 : [5, 7, 9].includes(index) ? 1 : 0,
}))
const output = 'https://cdn.example.com/preview.jpg'
const success = data => Response.json({ code: 200, data })
const complete = (id, url = output) => ({ id, status: 2, output_url: url })

beforeEach(t => {
  const previousToken = process.env.AGENT_GATEWAY_TOKEN
  const previousPort = process.env.AGENT_GATEWAY_PORT
  process.env.AGENT_GATEWAY_TOKEN = 'fake-token-for-preview-test'
  delete process.env.AGENT_GATEWAY_PORT
  t.after(() => {
    if (previousToken === undefined) delete process.env.AGENT_GATEWAY_TOKEN
    else process.env.AGENT_GATEWAY_TOKEN = previousToken
    if (previousPort === undefined) delete process.env.AGENT_GATEWAY_PORT
    else process.env.AGENT_GATEWAY_PORT = previousPort
  })
})

test('submits one grounded edit and polls the exact managed record on the configured port', async t => {
  process.env.AGENT_GATEWAY_PORT = '8180'
  const responses = [
    success({ record_id: 42, data: [{ url: '' }] }),
    success({ items: [complete('43', 'https://cdn.example.com/unrelated.jpg'), { id: '42', status: 1 }] }),
    success({ items: [complete(42)] }),
  ]
  const fetch = t.mock.method(globalThis, 'fetch', async () => responses.shift())
  assert.deepEqual(await generatePreview(source), { url: output })
  assert.equal(fetch.mock.callCount(), 3)

  const [url, options] = fetch.mock.calls[0].arguments
  assert.equal(url, 'http://127.0.0.1:8180/v1/images/edits')
  assert.equal(options.method, 'POST')
  assert.equal(options.redirect, 'error')
  assert.deepEqual(options.headers, {
    Authorization: 'Bearer fake-token-for-preview-test', 'Content-Type': 'application/json',
  })
  const { prompt, ...body } = JSON.parse(options.body)
  assert.ok(prompt.includes(source.suggestion))
  assert.match(prompt, /以原照片为底图.*保留人物身份.*构图保持原样/)
  assert.deepEqual(body, { model: 'group:image_edit', n: 1, image_data_uri: 'data:image/png;base64,' + source.image.toString('base64') })

  for (const call of fetch.mock.calls.slice(1)) {
    const [recordUrl, recordOptions] = call.arguments
    assert.equal(recordUrl, 'http://127.0.0.1:8180/v1/images/records?record_id=42')
    assert.deepEqual(recordOptions, {
      method: 'GET', redirect: 'error', signal: options.signal,
      headers: { Authorization: 'Bearer fake-token-for-preview-test' },
    })
  }
})

test('preserves a uint64 record ID when neighboring records round to the same JS number', async t => {
  const recordId = '9007199254740993'
  const responses = [
    new Response('{"code":200,"data":{"record_id":9007199254740993}}'),
    new Response('{"code":200,"data":{"items":[' +
      '{"id":9007199254740992,"status":2,"output_url":"https://cdn.example.com/unrelated.jpg"},' +
      '{"id":"9007199254740993","status":2,"output_url":"https://cdn.example.com/preview.jpg"}]}}'),
  ]
  const fetch = t.mock.method(globalThis, 'fetch', async () => responses.shift())
  assert.deepEqual(await generatePreview(source), { url: output })
  assert.equal(fetch.mock.calls[1].arguments[0], 'http://127.0.0.1:8080/v1/images/records?record_id=' + recordId)
})

test('submits only the clean photo and visible target joints, even with a legacy guide', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method }) =>
    method === 'POST' ? success({ record_id: 42 }) : success({ items: [complete(42)] }))
  const input = { ...source, poseGuide: guide, landmarks }
  assert.deepEqual(await generatePreview(input), { url: output })
  const [, options] = fetch.mock.calls[0].arguments
  const { prompt, ...body } = JSON.parse(options.body)
  assert.deepEqual(body, {
    model: 'group:image_edit',
    n: 1,
    image_data_uri: 'data:image/png;base64,' + source.image.toString('base64'),
  })
  assert.match(prompt, /让输入照片中的人物改成目标姿势.*保留原照片的人物身份/)
  assert.match(prompt, /目标关节坐标.*左上角为原点.*x 向右、y 向下.*左肩\[0\.25,0\.5\] 左肘\[0\.5,0\.625\] 左腕\[0\.75,0\.5\]/)
  assert.match(prompt, /让手腕落在对应目标.*只输出一张写实照片.*不要出现骨架线、关节点/)
  assert.doesNotMatch(prompt, /第二张图|左眼\[|右肩\[/)
  assert.ok(prompt.includes(source.suggestion))
  assert.doesNotMatch(prompt, /最小调整/)

  const remaining = gatewayBodyLimit - Buffer.byteLength(options.body)
  const image = Buffer.alloc(9 + Math.floor(remaining / 4) * 3)
  source.image.copy(image)
  const suggestion = source.suggestion + 'x'.repeat(remaining % 4)
  assert.deepEqual(await generatePreview({ ...input, image, suggestion }), { url: output })
  assert.equal(Buffer.byteLength(fetch.mock.calls[2].arguments[1].body), gatewayBodyLimit)
  await assert.rejects(generatePreview({ ...input, image, suggestion: suggestion + 'x' }), { status: 413 })
  assert.equal(fetch.mock.callCount(), 4)
})

test('places one person in the supplied scene using joints relative to the submitted frame', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method }) =>
    method === 'POST' ? success({ record_id: 42 }) : success({ items: [complete(42)] }))
  const input = { ...source, landmarks }
  for (const previewMode of ['scene-pose', 'joint']) {
    const callIndex = fetch.mock.callCount()
    assert.deepEqual(await generatePreview({ ...input, previewMode }), { url: output })
    const { prompt, ...body } = JSON.parse(fetch.mock.calls[callIndex].arguments[1].body)
    assert.match(prompt, previewMode === 'joint' ? /裁切后的空场景照片/ : /空场景原照片/)
    assert.match(prompt, /生成恰好一位人物.*保留所提交照片的画幅、取景、背景和光线/)
    assert.match(prompt, /左肩\[0\.25,0\.5\] 左肘\[0\.5,0\.625\] 左腕\[0\.75,0\.5\].*让手腕落在对应目标/)
    assert.match(prompt, /只输出一张写实照片.*不要出现骨架线、关节点/)
    assert.doesNotMatch(prompt, /保留原照片的人物身份/)
    assert.deepEqual(body, {
      model: 'group:image_edit',
      n: 1,
      image_data_uri: 'data:image/png;base64,' + source.image.toString('base64'),
    })
  }
  await generatePreview({ ...input, previewMode: 'pose' })
  assert.match(JSON.parse(fetch.mock.calls[4].arguments[1].body).prompt, /保留原照片的人物身份/)
})

test('sends the clean photo first and the target skeleton second for each guided edit', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method }) =>
    method === 'POST' ? success({ record_id: 42 }) : success({ items: [complete(42)] }))
  const input = { ...source, landmarks, targetPoseGuide: guide }
  for (const previewMode of ['pose', 'scene-pose', 'joint']) {
    const callIndex = fetch.mock.callCount()
    assert.deepEqual(await generatePreview({ ...input, previewMode }), { url: output })
    const { prompt, ...body } = JSON.parse(fetch.mock.calls[callIndex].arguments[1].body)
    assert.deepEqual(body, {
      model: 'group:image_edit',
      n: 1,
      images: [
        'data:image/png;base64,' + source.image.toString('base64'),
        'data:image/jpeg;base64,' + guide.toString('base64'),
      ],
    })
    assert.match(prompt, /图一是干净的照片底图.*图二是与图一同画幅的目标骨架.*身体朝向、倾斜和支撑方式/)
    assert.match(prompt, /左肩\[0\.25,0\.5\] 左肘\[0\.5,0\.625\] 左腕\[0\.75,0\.5\].*拍照建议补充坐标未描述的细节/)
    assert.ok(prompt.includes(source.suggestion))
    assert.match(prompt, /只输出一张写实照片.*不要出现骨架线、关节点/)
  }

  const body = fetch.mock.calls[0].arguments[1].body
  const remaining = gatewayBodyLimit - Buffer.byteLength(body)
  const targetPoseGuide = Buffer.alloc(guide.length + Math.floor(remaining / 4) * 3)
  guide.copy(targetPoseGuide)
  const suggestion = source.suggestion + 'x'.repeat(remaining % 4)
  assert.deepEqual(await generatePreview({ ...input, targetPoseGuide, suggestion }), { url: output })
  assert.equal(Buffer.byteLength(fetch.mock.calls[6].arguments[1].body), gatewayBodyLimit)
  await assert.rejects(generatePreview({ ...input, targetPoseGuide, suggestion: suggestion + 'x' }), { status: 413 })
  assert.equal(fetch.mock.callCount(), 8)
})

test('keeps the requested action and the selected pose in the edit prompt', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method }) =>
    method === 'POST' ? success({ record_id: 42 }) : success({ items: [complete(42)] }))
  const instruction = '请直立，双手扶在腰间。'
  assert.deepEqual(await generatePreview({ ...source, landmarks, targetPoseGuide: guide, instruction: ` ${instruction} ` }), { url: output })
  const { prompt } = JSON.parse(fetch.mock.calls[0].arguments[1].body)
  assert.ok(prompt.includes(JSON.stringify(instruction)))
  assert.ok(prompt.includes(source.suggestion))
  assert.match(prompt, /动作以用户指令为准.*拍照建议/)
})

test('rejects invalid input and oversized serialized gateway requests before fetch', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Unexpected request') })
  await assert.rejects(generatePreview({ ...source, mime: 'image/jpeg' }), { status: 415 })
  await assert.rejects(generatePreview({ ...source, suggestion: ' ' }), { status: 400 })
  await assert.rejects(generatePreview({ ...source, instruction: ' ' }), { status: 400 })
  await assert.rejects(generatePreview({ ...source, instruction: 'x'.repeat(301) }), { status: 400 })
  await assert.rejects(generatePreview({ ...source, landmarks: landmarks.slice(1) }), { status: 400 })
  await assert.rejects(generatePreview({ ...source, landmarks, targetPoseGuide: source.image }), { status: 415 })
  await assert.rejects(generatePreview({ ...source, landmarks, targetPoseGuide: 'not a Buffer' }), { status: 415 })
  await assert.rejects(generatePreview({ ...source, targetPoseGuide: guide }), { status: 400 })
  await assert.rejects(generatePreview({ ...source, previewMode: 'framing' }), { status: 400 })
  for (const previewMode of ['scene-pose', 'joint']) {
    await assert.rejects(generatePreview({ ...source, previewMode }), { status: 400 })
    await assert.rejects(generatePreview({ ...source, previewMode, poseGuide: guide }), { status: 400 })
  }
  const large = Buffer.alloc((2 * 1024 * 1024 / 4) * 3 - 6)
  source.image.copy(large)
  await assert.rejects(generatePreview({ ...source, image: large }), { status: 413 })
  process.env.AGENT_GATEWAY_PORT = '8080@example.com'
  await assert.rejects(generatePreview(source), { status: 503 })
  assert.equal(fetch.mock.callCount(), 0)
})

test('rejects malformed responses and untrustworthy results without retrying a submission', async t => {
  let responses
  const fetch = t.mock.method(globalThis, 'fetch', async () => responses.shift())
  const oversized = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(2 * 1024 * 1024))
      controller.close()
    },
  })
  for (responses of [
    [success({ record_id: '1&records=another' })],
    [success({ record_id: '18446744073709551616' })],
    [success({ record_id: '0' })],
    [success({ data: { record_id: 42 } })],
    [new Response(oversized)],
    [new Response('not json')],
    [success({ record_id: 42 }), success({ items: {} })],
    [success({ record_id: 42 }), success({ items: [{ id: 42, status: 4 }] })],
    [success({ record_id: 42 }), success({ items: [complete(42, 'http://cdn.example.com/image.jpg')] })],
    [success({ record_id: 42 }), success({ items: [complete(42, 'https://user:pass@cdn.example.com/image.jpg')] })],
    [success({ record_id: 42 }), success({ items: [complete(42, 'data:image/png;base64,AAAA')] })],
    [success({ record_id: 42 }), success({ items: [complete(42, '')] })],
  ]) {
    const expectedCalls = responses.length
    const start = fetch.mock.callCount()
    await assert.rejects(generatePreview(source), { status: 502, message: '图片服务未返回可用的示意图。' })
    assert.equal(fetch.mock.callCount() - start, expectedCalls)
  }
})

test('stops on a failed record and keeps upstream details out of errors', async t => {
  let response = success({ items: [{ id: 42, status: 3, output_url: output, error: 'private upstream detail' }] })
  const fetch = t.mock.method(globalThis, 'fetch', async (_, options) => {
    return options.method === 'POST' ? success({ record_id: '42' }) : response
  })
  await assert.rejects(generatePreview(source), { status: 502, message: '图片服务未能生成示意图，请重试。' })

  for (const status of [401, 403, 429, 500, 504]) {
    response = new Response('private upstream detail', { status })
    await assert.rejects(generatePreview(source), error => {
      assert.equal(error.status, [429, 504].includes(status) ? status : 502)
      if (status === 401 || status === 403) assert.match(error.message, /未授权/)
      assert.doesNotMatch(error.message, /private upstream detail/)
      return true
    })
  }
  assert.deepEqual(fetch.mock.calls.map(call => call.arguments[1].method), Array(6).fill(['POST', 'GET']).flat())
})

test('aborts an in-flight submission and enforces the same deadline while polling', async t => {
  const controller = new AbortController()
  const deadline = new AbortController()
  const timeout = t.mock.method(AbortSignal, 'timeout', () => deadline.signal)
  let notifyRequest
  let nextRequest = new Promise(resolve => { notifyRequest = resolve })
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method, signal }) => {
    if (signal === deadline.signal && method === 'POST') return success({ record_id: 42 })
    notifyRequest(method)
    return new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
  })
  const cancelled = generatePreview({ ...source, signal: controller.signal })
  assert.equal(await nextRequest, 'POST')
  controller.abort()
  await assert.rejects(cancelled, { status: 499 })

  nextRequest = new Promise(resolve => { notifyRequest = resolve })
  const timedOut = generatePreview(source)
  assert.equal(await nextRequest, 'GET')
  deadline.abort()
  await assert.rejects(timedOut, { status: 504, message: /网关任务可能仍在处理/ })
  assert.deepEqual(timeout.mock.calls[0].arguments, [180_000])
  assert.deepEqual(fetch.mock.calls.map(call => call.arguments[1].method), ['POST', 'POST', 'GET'])
})

test('cancels the polling delay without another request', async t => {
  const controller = new AbortController()
  const fetch = t.mock.method(globalThis, 'fetch', async (_, { method }) => {
    if (method === 'POST') return success({ record_id: 42 })
    setImmediate(() => controller.abort())
    return success({ items: [{ id: 42, status: 1 }] })
  })
  await assert.rejects(generatePreview({ ...source, signal: controller.signal }), { status: 499 })
  assert.deepEqual(fetch.mock.calls.map(call => call.arguments[1].method), ['POST', 'GET'])
})
