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
  const tilt = '因此无法通过简单的裁剪来优化构图。建议对准远处的雪山与天空重拍。'
  assert.deepEqual(parseComposition(tilt), { decision: 'reject', suggestion: tilt })
})

test('reads structured composition coordinates and validates crops', () => {
  assert.deepEqual(parseComposition('```json\n{"instance_info":[{"composition_xy":[0.1,0.2,1,0.9],"reason":" 保留山脉。 "}]}\n```'), {
    decision: 'refine', box: [0.1, 0.2, 1, 0.9], suggestion: '保留山脉。',
  })
  assert.deepEqual(parseComposition('{"decision":"reject","box":null}'), { decision: 'reject' })
  assert.deepEqual(parseComposition('{"composition_xy":[],"reason":"退后一步，保留更多场景。"}'), {
    decision: 'reject', suggestion: '退后一步，保留更多场景。',
  })
  for (const invalid of [
    '', '裁剪可以优化构图。', '无法通过裁剪来优化构图。\n(100,200),(600,',
    '(-1,100),(900,900)', '(100,100),(1001,1000)', '(0,0),(1,1.2)',
    '(100,100),(100,900)', '(100,900),(900,100)', '{"decision":"refine"}',
    '{"decision":"keep","bbox":[100,100,900,900]}', '{"bbox":[0,0,1,"1"]}',
  ]) {
    assert.throws(() => parseComposition(invalid), /valid composition decision/)
  }
})

test('discovers the model and uses the photo-specific coaching text to request posture landmarks', async t => {
  const requests = []
  const brief = '把肩膀转向栏杆，左手自然搭上去。'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options })
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'other' }, { id: 'ShutterMuse-Q4' }] })
    const { max_tokens: budget } = JSON.parse(options.body)
    const content = budget === 256 ? brief : JSON.stringify({ ...prediction.instance_info[0], reason: brief })
    return Response.json({ choices: [{ message: { content } }] })
  })
  const previousModel = process.env.SHUTTERMUSE_MODEL
  delete process.env.SHUTTERMUSE_MODEL
  t.after(() => {
    if (previousModel !== undefined) process.env.SHUTTERMUSE_MODEL = previousModel
  })

  const image = Buffer.from('scene')
  const result = await inferPose({ image, mime: 'image/jpeg' })
  assert.equal(result.landmarks.length, 17)
  assert.equal(result.suggestion, brief)
  assert.deepEqual(requests.map(request => request.url), [
    'http://127.0.0.1:8081/v1/models', 'http://127.0.0.1:8081/v1/chat/completions', 'http://127.0.0.1:8081/v1/chat/completions',
  ])
  const [planning, coordinates] = requests.slice(1).map(request => JSON.parse(request.options.body))
  assert.equal(planning.model, 'ShutterMuse-Q4')
  assert.equal(coordinates.model, planning.model)
  assert.equal(requests[1].options.signal, requests[2].options.signal)
  assert.match(planning.messages[0].content[1].text, /根据当前景别、可见的身体部位/)
  assert.match(subjectPrompt, /手腕坐标应落在物体可触及的表面/)
  assert.deepEqual(coordinates.messages[0].content, [
    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${image.toString('base64')}` } },
    { type: 'text', text: subjectPrompt + '\n摄影方案：' + JSON.stringify(brief) },
  ])
  assert.deepEqual(planning.messages[0].content[0], coordinates.messages[0].content[0])
})

test('an alternative treats the previous action as history and requests landmarks for the new brief', async t => {
  const requests = []
  const brief = '右脚向前迈半步，让两臂自然摆动。'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    requests.push(body)
    const joint = { ...prediction.instance_info[0], reason: brief, composition_xy: [0.25, 0.05, 0.8, 1] }
    const content = body.max_tokens === 256 ? brief : JSON.stringify(joint)
    return Response.json({ choices: [{ message: { content } }] })
  })
  const previousSuggestion = '可以试试把手轻轻搭在肩上。'
  for (const task of ['pose', 'scene-pose', 'joint']) {
    const result = await inferPose({ image: Buffer.from('scene'), mime: 'image/png', task, previousSuggestion })
    const [planning, coordinates] = requests.splice(0)
    assert.deepEqual(planning.messages[1], { role: 'assistant', content: previousSuggestion })
    assert.equal(planning.messages[2].role, 'user')
    if (task !== 'pose') assert.match(planning.messages[2].content, /换一种身体支撑方式或重心/)
    assert.ok(planning.temperature > 0)
    assert.equal(coordinates.temperature, 0)
    assert.ok(coordinates.messages[0].content[1].text.includes(brief))
    assert.equal(result.landmarks.length, 17)
  }
})

test('retries an identical alternative brief once and stops before generating duplicate landmarks', async t => {
  const requests = []
  const previousSuggestion = '把肩膀转向栏杆。'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    requests.push(JSON.parse(options.body))
    return Response.json({ choices: [{ message: { content: previousSuggestion } }] })
  })
  const photo = { image: Buffer.from('scene'), mime: 'image/png', task: 'scene-pose' }
  await assert.rejects(inferPose({ ...photo, previousSuggestion }), { status: 422, message: /仍显示上一种姿势/ })
  assert.equal(requests.length, 2)
  assert.equal(requests[0].max_tokens, requests[1].max_tokens)
  assert.notEqual(requests[0].messages.at(-1).content, requests[1].messages.at(-1).content)
})

test('plans several distinct choices together and honors the photographer’s request', async t => {
  const previousSuggestion = '把右手搭在墙面上。'
  const instruction = '保持坐姿，手势活泼一点。'
  const suggestions = ['抬起右脚，膝盖弯曲，左脚支撑在地面上，身体向画面右侧倾斜，左手扶在墙沿上，右手插在裤袋里，头转向画面左侧。', '把右肘往外打开。', '让双手交叠在膝盖上。']
  const sameAction = '将身体完全靠在墙面上，右脚抬起，膝盖弯曲，左脚支撑在地面上，左手扶在墙沿上，右手插在裤袋里，头转向画面左侧。'
  const lines = [`1. ${previousSuggestion.replace('右手', '左手')}`, `2. ${suggestions[0]}`, `3. ${sameAction}`, ...suggestions.slice(1)]
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    requests.push(body)
    const content = body.max_tokens === 384 ? `<answer>\n${lines.join('\n')}\n</answer>` : JSON.stringify(prediction)
    return Response.json({ choices: [{ message: { content } }] })
  })

  const photo = { image: Buffer.from('photo'), mime: 'image/jpeg' }
  const pose = await inferPose({ ...photo, poseChoices: true, previousSuggestion, instruction })
  assert.deepEqual(pose.suggestions, suggestions)
  assert.equal(pose.suggestion, suggestions[0])
  assert.equal(requests.length, 2)
  assert.deepEqual(requests[0].messages[1], { role: 'assistant', content: previousSuggestion })
  assert.match(requests[0].messages.at(-1).content, /三个不同的备选方案/)
  assert.equal(requests[0].response_format.json_schema.schema.properties.suggestions.minItems, 3)
  assert.ok(requests[0].messages[0].content[1].text.includes(instruction))
  assert.ok(requests[1].messages[0].content[1].text.includes(instruction))
  assert.ok(requests[1].messages[0].content[1].text.includes(suggestions[0]))

  const selected = await inferPose({ ...photo, selectedSuggestion: suggestions[2], instruction })
  assert.equal(selected.suggestion, suggestions[2])
  assert.equal(requests.length, 3)
  const selectedPrompt = requests[2].messages[0].content[1].text
  assert.ok(selectedPrompt.includes(suggestions[2]))
  assert.ok(selectedPrompt.includes(instruction))

  const retried = await inferPose({ ...photo, selectedSuggestion: suggestions[2], instruction, retryGrounding: true })
  assert.equal(retried.suggestion, suggestions[2])
  assert.equal(requests.length, 4)
  assert.ok(requests[3].temperature > requests[2].temperature)
  assert.match(requests[3].messages[0].content[1].text, /四肢轮廓和另一方案太接近/)
})

test('retries if the requested support hand is above the ground, then keeps the selected action', async t => {
  const selectedSuggestion = '蹲低，左手撑在身前的路面上。'
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    requests.push(body)
    const corrected = keypoints.map((point, index) => [9, 10].includes(index) ? [point[0], 0.96] : point)
    const content = JSON.stringify({ ...prediction.instance_info[0], keypoints_xyn: requests.length === 2 ? corrected : keypoints })
    return Response.json({ choices: [{ message: { content } }] })
  })
  const photo = { image: Buffer.from('photo'), mime: 'image/jpeg', selectedSuggestion }
  const pose = await inferPose(photo)
  assert.equal(pose.suggestion, selectedSuggestion)
  assert.equal(pose.landmarks[9].y, 0.96)
  assert.equal(requests.length, 2)
  assert.match(requests[1].messages[0].content[1].text, /撑地的手腕应接近脚下的地面/)

  await assert.rejects(inferPose(photo), { status: 422, message: /模型未能画出这条姿势/ })
  assert.equal(requests.length, 4)

  const negated = '不要用右手撑地，保持平衡。'
  assert.equal((await inferPose({ ...photo, selectedSuggestion: negated })).suggestion, negated)
  assert.equal(requests.length, 5)
  await assert.rejects(inferPose({ ...photo, selectedSuggestion: '左手扶在额头前。' }), {
    status: 422, message: /手腕应靠近头部高度/,
  })
  assert.equal(requests.length, 7)
})

test('offers the next choices when the first suggestion cannot be drawn', async t => {
  const first = '蹲下，右手撑地。'
  const second = '右肘往外打开。'
  const third = '双手虚放在胸前。'
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    requests.push(body)
    const content = body.max_tokens === 384 ? JSON.stringify({ suggestions: [first, second, third] }) : JSON.stringify(prediction)
    return Response.json({ choices: [{ message: { content } }] })
  })
  const result = await inferPose({ image: Buffer.from('photo'), mime: 'image/png', poseChoices: true })
  assert.deepEqual(result.suggestions, [second, third])
  assert.equal(result.suggestion, second)
  assert.equal(requests.length, 4)
  assert.ok(requests[3].messages[0].content[1].text.includes(second))
})

test('falls back to unconstrained suggestions when the model server rejects JSON Schema', async t => {
  const requests = []
  const suggestion = '把右肘往外打开。'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    requests.push(body)
    if (body.response_format) return new Response('', { status: 400 })
    const content = body.max_tokens === 384 ? suggestion : JSON.stringify(prediction)
    return Response.json({ choices: [{ message: { content } }] })
  })
  const result = await inferPose({ image: Buffer.from('photo'), mime: 'image/png', poseChoices: true })
  assert.deepEqual(result.suggestions, [suggestion])
  assert.equal(requests.length, 3)
  assert.equal(requests[1].response_format, undefined)
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
  await assert.rejects(inferPose({ image, mime: 'image/png', task: 'other' }), /Expected pose, composition, advice, crop, scene-pose, or joint/)
})

test('applies the selected crop intent and fits exactly 4:3 in original pixels', async t => {
  const prompts = []
  let answer = '保留主体和背景。\n(100,100),(900,1000)'
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    prompts.push(JSON.parse(options.body).messages.at(-1).content[1].text)
    return Response.json({ choices: [{ message: { content: answer } }] })
  })
  const photo = { image: Buffer.from('scene'), mime: 'image/png', task: 'crop' }
  const dimensions = { width: 1979, height: 1081 }
  await assert.rejects(inferPose({ ...photo, intent: '4:3', ...dimensions, height: '1081' }), /有效的原图宽高/)
  await assert.rejects(inferPose({ ...photo, intent: 'wide', ...dimensions }), /有效的裁切意图/)
  const framed = await inferPose({ ...photo, intent: '4:3', ...dimensions })
  assert.equal(framed.decision, 'refine')
  assert.equal(framed.suggestion, '保留主体和背景。')
  assert.ok(framed.box.every(value => value >= 0 && value <= 1))
  const [left, top, right, bottom] = framed.box.map((value, index) => Math.round(value * (index % 2 ? dimensions.height : dimensions.width)))
  assert.equal(3 * (right - left), 4 * (bottom - top))
  assert.match(prompts[0], /横向 4:3/)

  answer = '{"instance_info":[{"task_type":"crop","reason":"保留头顶，裁至腰部。","bbox_xy":"(200,100),(850,700)"}]}'
  assert.deepEqual(await inferPose({ ...photo, intent: 'half-body' }), {
    decision: 'refine', box: [0.2, 0.1, 0.85, 0.7], suggestion: '保留头顶，裁至腰部。',
  })
  assert.match(prompts[1], /半身人像/)
  answer = '画面没有适合的人物。\n<non>'
  assert.deepEqual(await inferPose({ ...photo, intent: 'half-body' }), {
    decision: 'reject', suggestion: '画面没有适合的人物。',
  })
})

test('plans an empty-scene pose and validates a single jointly generated original-frame crop', async t => {
  const prompts = []
  let answer = { ...prediction.instance_info[0], reason: '站在路边的空地上，双臂自然展开。' }
  let rawAnswer
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/models')) return Response.json({ data: [{ id: 'ShutterMuse-Q4' }] })
    const body = JSON.parse(options.body)
    prompts.push(body.messages.at(-1).content[1].text)
    const content = body.max_tokens === 256 ? '在路边空地上把两臂向外展开。' : rawAnswer ?? JSON.stringify(answer)
    return Response.json({ choices: [{ message: { content } }] })
  })
  const photo = { image: Buffer.from('scene'), mime: 'image/png' }
  const pose = await inferPose({ ...photo, task: 'scene-pose' })
  assert.equal(pose.suggestion, answer.reason)
  assert.equal(pose.landmarks.length, 17)
  assert.match(prompts[0], /尚未入镜/)
  assert.match(prompts[1], /让承担体重的部位落在画面中可见、稳固的表面上/)
  assert.match(prompts[1], /关节坐标要清楚体现文字里的支撑点和身体动作/)

  const box = [0.3, 0.05, 0.8, 1]
  answer = { ...answer, composition_xy: box }
  assert.deepEqual(await inferPose({ ...photo, task: 'joint' }), { ...pose, box })
  assert.equal(prompts.length, 4)
  assert.match(prompts[3], /姿势坐标也必须相对原图/)

  const wrapped = JSON.stringify({ instance_info: [answer] })
  rawAnswer = wrapped.replace('"composition_xy":[0.3,0.05,0.8,1]', '"composition_xy": [0.3,0.05,0.8,1]]')
  assert.deepEqual(await inferPose({ ...photo, task: 'joint' }), { ...pose, box })

  rawAnswer = wrapped.replace('"composition_xy":[0.3,0.05,0.8,1]', '"composition_xy": [0.3966,0.3816], [0.5766,0.5366]]')
  const fitted = await inferPose({ ...photo, task: 'joint' })
  assert.deepEqual(fitted.landmarks, pose.landmarks)
  const [left, top, right, bottom] = fitted.box
  assert.ok(left >= 0 && top >= 0 && right <= 1 && bottom <= 1)
  assert.ok(pose.landmarks.filter(point => point.visibility === 1).every(({ x, y }) =>
    x >= left && y >= top && x <= right && y <= bottom))
  rawAnswer = undefined
  answer = { ...answer, composition_xy: [0.6, 0.1, 0.45, 0.9] }
  await assert.rejects(inferPose({ ...photo, task: 'joint' }), /valid composition decision/)
  answer = { ...answer, visibility: Array(17).fill(0) }
  await assert.rejects(inferPose({ ...photo, task: 'scene-pose' }), /visible body pose/)
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
