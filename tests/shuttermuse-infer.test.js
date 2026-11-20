import assert from 'node:assert/strict'
import { test } from 'node:test'
import { advicePrompt, adviceSystemPrompt, compositionPrompt, inferPose, parseComposition, parsePose, subjectPrompt } from '../tools/shuttermuse-infer.mjs'

const keypoints = [[0.51, 0.22], [0.52, 0.21], [0.5, 0.21], [0.54, 0.22], [0.48, 0.22],
  [0.56, 0.29], [0.46, 0.29], [0.57, 0.39], [0.43, 0.39], [0.64, 0.37], [0.41, 0.44],
  [0.56, 0.52], [0.47, 0.52], [0.59, 0.75], [0.43, 0.78], [-1, -1], [0.39, 0.98]]
const visibility = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1]
const prediction = {
  instance_info: [{ keypoints_xyn: keypoints, visibility, reason: '左手搭在栏杆上，身体稍微侧向相机。' }],
}

test('normalizes an upstream pose with visible, occluded, and outside-frame landmarks', () => {
  const pose = parsePose(`\`\`\`json\n${JSON.stringify(prediction)}\n\`\`\``)
  assert.equal(pose.landmarks.length, 17)
  assert.deepEqual(pose.landmarks[0], { x: 0.51, y: 0.22, visibility: 1 })
  assert.deepEqual(pose.landmarks[4], { x: 0.48, y: 0.22, visibility: 0 })
  assert.deepEqual(pose.landmarks[15], { x: 0, y: 0, visibility: -1 })
  assert.equal(pose.suggestion, prediction.instance_info[0].reason)
})

test('supports the upstream flat keypoint variant and keeps model text optional', () => {
  const pose = parsePose(JSON.stringify({ keypoints_xyn: keypoints.flat(), visibility }))
  assert.deepEqual(pose.landmarks.at(-1), { x: 0.39, y: 0.98, visibility: 1 })
  assert.equal('suggestion' in pose, false)
})

test('rejects incomplete and out-of-range visible model output', () => {
  assert.throws(() => parsePose('No suggestion'), /invalid pose JSON/)
  assert.throws(() => parsePose('{"keypoints_xyn": [[0.5,0.2]], "visibility": [1]}'), /17 ordered COCO/)
  assert.throws(() => parsePose(JSON.stringify({ keypoints_xyn: keypoints, visibility: visibility.map(() => 1) })), /17 ordered COCO/)
})

test('distinguishes a full frame, a normalized crop, and an explicit rejection', () => {
  assert.deepEqual(parseComposition('(0,0),(1000,1000)'), { decision: 'keep' })
  assert.deepEqual(parseComposition('(0,0),(1,1)'), { decision: 'keep' })
  assert.deepEqual(parseComposition('<think>(0,0),(1000,1000)</think>裁掉上方的留白。具体裁剪区域为：(50,200),(950,1000)'), {
    decision: 'refine', box: [0.05, 0.2, 0.95, 1], suggestion: '裁掉上方的留白。',
  })
  assert.deepEqual(parseComposition('主体无法突出。\n<non>'), {
    decision: 'reject', suggestion: '主体无法突出。',
  })
  const rejection = '旗杆和红旗在画面中所占比例过小，导致主体不突出，且大量留白的天空使画面重心明显偏低，视觉上缺乏稳定感。因此，无法通过裁剪来优化构图，原图整体效果不佳。'
  assert.deepEqual(parseComposition(rejection), { decision: 'reject', suggestion: rejection })
})

test('reads structured composition coordinates and validates crops', () => {
  assert.deepEqual(parseComposition('```json\n{"instance_info":[{"composition_xy":[0.1,0.2,1,0.9],"reason":" 保留山脉。 "}]}\n```'), {
    decision: 'refine', box: [0.1, 0.2, 1, 0.9], suggestion: '保留山脉。',
  })
  assert.deepEqual(parseComposition('{"decision":"reject","box":null}'), { decision: 'reject' })
  for (const invalid of [
    '', '裁剪可以优化构图。', '无法通过裁剪来优化构图。\n(100,200),(600,',
    '(-1,100),(900,900)', '(100,100),(1001,1000)', '(0,0),(1,1.2)',
    '(100,100),(100,900)', '(100,900),(900,100)', '{"decision":"refine"}',
    '{"decision":"keep","bbox":[100,100,900,900]}', '{"bbox":[0,0,1,"1"]}',
  ]) {
    assert.throws(() => parseComposition(invalid), /valid composition decision/)
  }
})

test('discovers the ShutterMuse model and sends the scene with the pose coaching prompt', async t => {
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options })
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'other' }, { id: 'ShutterMuse-Q4' }] })
    return Response.json({ choices: [{ message: { content: JSON.stringify(prediction) } }] })
  })
  const previousModel = process.env.SHUTTERMUSE_MODEL
  delete process.env.SHUTTERMUSE_MODEL
  t.after(() => {
    if (previousModel !== undefined) process.env.SHUTTERMUSE_MODEL = previousModel
  })

  const image = Buffer.from('scene')
  const result = await inferPose({ image, mime: 'image/jpeg' })
  assert.equal(result.landmarks.length, 17)
  assert.deepEqual(requests.map(request => request.url), [
    'http://127.0.0.1:8081/v1/models', 'http://127.0.0.1:8081/v1/chat/completions',
  ])
  const body = JSON.parse(requests[1].options.body)
  assert.equal(body.model, 'ShutterMuse-Q4')
  assert.deepEqual(body.messages[1].content, [
    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${image.toString('base64')}` } },
    { type: 'text', text: subjectPrompt },
  ])
})

test('sends the photographer prompt and normalizes its crop result when composition is requested', async t => {
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    requests.push(JSON.parse(options.body))
    return Response.json({ choices: [{ message: { content: '(0,100),(1000,900)' } }] })
  })
  const image = Buffer.from('scene')
  assert.deepEqual(await inferPose({ image, mime: 'image/png', task: 'composition' }), {
    decision: 'refine', box: [0, 0.1, 1, 0.9],
  })
  assert.deepEqual(requests[0].messages, [{ role: 'user', content: [
    { type: 'image_url', image_url: { url: `data:image/png;base64,${image.toString('base64')}` } },
    { type: 'text', text: compositionPrompt },
  ] }])
  await assert.rejects(inferPose({ image, mime: 'image/png', task: 'other' }), /Expected pose, composition, or advice/)
})

test('uses the same plain-text advice prompt for each photo in one completion', async t => {
  const requests = []
  const suggestion = '可以保留人物身后这条柔和的光影交界，让轮廓更清楚。'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    requests.push({ url, body: JSON.parse(options.body) })
    return Response.json({ choices: [{ message: { content: `<think>内部分析</think> <answer>${suggestion}</answer>` } }] })
  })
  for (const scene of ['portrait', 'landscape']) {
    const image = Buffer.from(scene)
    assert.deepEqual(await inferPose({ image, mime: 'image/jpeg', task: 'advice' }), { suggestion })
    assert.equal(requests.length, scene === 'portrait' ? 1 : 2)
    const { url, body } = requests.at(-1)
    assert.equal(url, 'http://127.0.0.1:8081/v1/chat/completions')
    assert.deepEqual(body.messages, [{ role: 'system', content: adviceSystemPrompt }, { role: 'user', content: [
      { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${image.toString('base64')}` } },
      { type: 'text', text: advicePrompt },
    ] }])
  }
})

test('rejects missing, structured, or incomplete advice', async t => {
  let choice
  t.mock.method(globalThis, 'fetch', async url => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    return Response.json({ choices: [choice] })
  })
  const photo = { image: Buffer.from('scene'), mime: 'image/png', task: 'advice' }
  for (const content of [undefined, ' ', '<think>内部分析</think>', '{"suggestion":"可以保留柔光。"}', '(0,0),(1000,1000)']) {
    choice = { message: { content } }
    await assert.rejects(inferPose(photo), /no usable photography advice/)
  }
  choice = { finish_reason: 'length', message: { content: '可以保留柔和的' } }
  await assert.rejects(inferPose(photo), /photography advice was incomplete/)
})

test('surfaces a disconnected model server without returning a pose', async t => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Connection refused') })
  await assert.rejects(inferPose({ image: Buffer.from('scene'), mime: 'image/jpeg' }), /Cannot reach the local ShutterMuse server/)
})

test('requires an explicit override for an unrelated model on the shared port', async t => {
  const original = process.env.SHUTTERMUSE_MODEL
  delete process.env.SHUTTERMUSE_MODEL
  t.after(() => { if (original !== undefined) process.env.SHUTTERMUSE_MODEL = original })
  t.mock.method(globalThis, 'fetch', async url => {
    assert.ok(url.endsWith('/models'))
    return Response.json({ data: [{ id: 'qwen-general' }] })
  })
  await assert.rejects(inferPose({ image: Buffer.from('scene'), mime: 'image/jpeg' }), /identify/)
})
