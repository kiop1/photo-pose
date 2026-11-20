import { readFile } from 'node:fs/promises'
import { extname } from 'node:path'
import { pathToFileURL } from 'node:url'

const endpoint = 'http://127.0.0.1:8081/v1'
const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png' }

export const subjectPrompt = '你是一个人像摄影摆姿分析专家。请结合这张照片，为模特推荐一个能在当前景别内看到的姿势微调。沿用人物现有的姿势和道具。已有的动作合适时，直接说可以保持什么。看不清的部位、道具和环境不要猜。'
  + '\n' + '只输出姿势预测的 JSON。instance_info 中的 reason 是相机屏幕上的短提示，限18个汉字，一句话，一个动作，以句号结束，不使用逗号。合适就只说可以保持的可见细节；需要微调就只说一个可选的小动作。'
  + '\n' + 'keypoints_xyn 给出推荐姿势的17对归一化坐标 [x,y]，左上角为原点。保留未调整关节的位置和人物大小。visibility 依次标记可见为1、遮挡为0、画外为-1。关节顺序：鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝。'

export const compositionPrompt = '请判断当前照片的构图。如果构图已经很好，保留原图并输出(0,0),(1000,1000)；'
  + '如果裁剪能明显改善构图，找出最好的区域并输出左上角和右下角的坐标(x1,y1),(x2,y2)，坐标以整张图片归一化到0–1000；'
  + '如果无法通过裁剪得到好的构图，输出<non>。请先用一句中文说明理由，最后一行只输出坐标或<non>。'

export const advicePrompt = '基于眼前这张照片，你会给模特哪一句可选建议？觉得现在就好，也可以说保持。根据当前景别里看得到的身体部位或道具，只说一个细节。'
export const adviceSystemPrompt = '你是提供可选拍照灵感的助手。尊重模特自己的喜好。回答只有一行，以“可以”开头，最多25个汉字。只提一个具体可见的细节或动作。'

function stripThinking(content) {
  return content.replace(/^\s*<think>[\s\S]*?<\/think>/, '').trim()
}

function jsonBlock(answer) {
  return answer.match(/```(?:json)?\s*({[\s\S]*?})\s*```/i)?.[1]
    ?? answer.match(/{[\s\S]*}/)?.[0]
}

export function parsePose(content) {
  if (typeof content !== 'string') throw new Error('ShutterMuse returned no pose text.')

  let prediction
  try {
    prediction = JSON.parse(jsonBlock(stripThinking(content)))
  } catch {
    throw new Error('ShutterMuse returned invalid pose JSON.')
  }

  const pose = prediction.instance_info?.[0] ?? prediction
  const { keypoints_xyn: keypoints, visibility } = pose
  const malformed = new Error('ShutterMuse must return 17 ordered COCO keypoints with visibility states.')
  if (!Array.isArray(keypoints) || !Array.isArray(visibility) || visibility.length !== 17) throw malformed

  const pairs = keypoints.length === 34
    ? Array.from({ length: 17 }, (_, index) => keypoints.slice(2 * index, 2 * index + 2))
    : keypoints
  if (pairs.length !== 17) throw malformed

  const landmarks = pairs.map((pair, index) => {
    const state = visibility[index]
    // The upstream renderer treats both 0 and -1 as hidden.
    const numericState = typeof state === 'string' && !state.trim() ? NaN : Number(state)
    if (state === null || typeof state === 'object' || !Number.isFinite(numericState)) throw malformed
    const visible = Math.sign(Math.trunc(numericState)) || 0
    if (!Array.isArray(pair) || pair.length !== 2 || !pair.every(Number.isFinite)) throw malformed
    const [x, y] = pair
    if (visible > 0 && [x, y].some(value => value < 0 || value > 1)) throw malformed
    return { x: Math.max(0, Math.min(x, 1)), y: Math.max(0, Math.min(y, 1)), visibility: visible }
  })

  const suggestion = pose.reason ?? prediction.reason
  return typeof suggestion === 'string' && suggestion.trim()
    ? { landmarks, suggestion: suggestion.trim() }
    : { landmarks }
}

const number = String.raw`[+-]?(?:\d+(?:\.\d+)?|\.\d+)`
const pair = String.raw`\(\s*${number}\s*,\s*${number}\s*\)`
const flatBox = String.raw`\[\s*${number}(?:\s*,\s*${number}){3}\s*\]`
const boxPattern = new RegExp(String.raw`${pair}\s*,\s*${pair}|${flatBox}`, 'g')
const rejectMarker = /<non>|^\s*REJECT\s*[。.]?\s*$/im
const rejectsCropping = /(?:无法|不能)(?:通过|用)裁剪(?:来)?(?:改善|优化|挽救)构图/
const coordinateIntro = /(?:具体)?(?:裁剪区域|裁剪范围|裁剪坐标)(?:为)?[：:]\s*$/u

export function parseComposition(content) {
  const malformed = new Error('ShutterMuse must return a valid composition decision and an in-frame box for refinement.')
  if (typeof content !== 'string') throw malformed
  const answer = stripThinking(content).replace(/<\/?answer>|```(?:json|text)?/gi, '').trim()
  const json = jsonBlock(answer)
  let decision, box, suggestion
  if (json) {
    let prediction
    try {
      prediction = JSON.parse(json)
    } catch {
      throw malformed
    }
    const composition = prediction.instance_info?.[0] ?? prediction
    const rawDecision = composition.decision ?? prediction.decision
    decision = typeof rawDecision === 'string' ? rawDecision.trim().toLowerCase() : rawDecision
    box = composition.composition_xy ?? composition.bbox ?? composition.composition_bbox ?? composition.box
    suggestion = composition.reason ?? composition.suggestion ?? prediction.reason ?? prediction.suggestion
  } else {
    const crop = answer.match(boxPattern)?.at(-1)
    const rejected = answer.match(rejectMarker)?.[0]
    if (rejected) decision = 'reject'
    box = crop
    const token = rejected ?? crop
    if (token) {
      const index = answer.lastIndexOf(token)
      const before = answer.slice(0, index).replace(coordinateIntro, '')
      suggestion = (before + answer.slice(index + token.length)).trim()
    } else if (rejectsCropping.test(answer) && !/[([]\s*[+-]?[\d.]+\s*[,，]/.test(answer)) {
      decision = 'reject'
      suggestion = answer
    }
  }

  if (typeof box === 'string') {
    if (rejectMarker.test(box)) {
      decision ??= 'reject'
      box = undefined
    } else {
      box = box.match(boxPattern)?.at(-1)?.match(new RegExp(number, 'g')).map(Number)
      if (!box) throw malformed
    }
  }
  if (box !== undefined && box !== null) {
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) throw malformed
    // ShutterMuse emits both unit coordinates and its legacy 0–1000 coordinates.
    const scale = Math.max(...box) <= 1.5 ? 1 : 1000
    const [x1, y1, x2, y2] = box.map(value => value / scale)
    if (x1 < 0 || y1 < 0 || x2 > 1 || y2 > 1 || x2 <= x1 || y2 <= y1) throw malformed
    const inferred = x1 === 0 && y1 === 0 && x2 === 1 && y2 === 1 ? 'keep' : 'refine'
    if (decision !== undefined && decision !== inferred) throw malformed
    decision = inferred
    box = [x1, y1, x2, y2]
  }
  if (decision !== 'keep' && decision !== 'reject' && !(decision === 'refine' && box)) throw malformed
  const result = { decision }
  if (decision === 'refine') result.box = box
  if (typeof suggestion === 'string' && suggestion.trim()) result.suggestion = suggestion.trim()
  return result
}

function parseAdvice(content) {
  const suggestion = typeof content === 'string'
    ? stripThinking(content).replace(/^<answer>\s*|\s*<\/answer>$/gi, '').trim()
    : ''
  if (!/\p{Script=Han}/u.test(suggestion) || /[\r\n<>{}\[\]]/.test(suggestion) || suggestion.match(boxPattern)) {
    throw new Error('ShutterMuse returned no usable photography advice.')
  }
  return { suggestion }
}

const tasks = new Map([
  ['pose', { prompt: subjectPrompt, parse: parsePose }],
  ['composition', { prompt: compositionPrompt, parse: parseComposition }],
  ['advice', { prompt: advicePrompt, parse: parseAdvice }],
])

async function request(path, options) {
  let response
  try {
    response = await fetch(endpoint + path, { ...options, redirect: 'error' })
  } catch (error) {
    if (options.signal.aborted) throw options.signal.reason
    throw new Error(`Cannot reach the local ShutterMuse server at ${endpoint}.`, { cause: error })
  }
  if (!response.ok) throw new Error(`Local ShutterMuse ${path} returned HTTP ${response.status}.`)
  try {
    return await response.json()
  } catch (error) {
    if (options.signal.aborted) throw options.signal.reason
    throw new Error(`Local ShutterMuse ${path} returned invalid JSON.`, { cause: error })
  }
}

export async function inferPose({ image, mime, signal, task = 'pose' }) {
  if (!(image instanceof Uint8Array) || !image.byteLength || !Object.values(mimeTypes).includes(mime)) {
    throw new TypeError('Expected nonempty JPEG or PNG image bytes.')
  }
  const config = tasks.get(task)
  if (!config) throw new TypeError('Expected pose, composition, or advice task.')
  const timeout = AbortSignal.timeout(240_000)
  signal = signal ? AbortSignal.any([signal, timeout]) : timeout

  let model = process.env.SHUTTERMUSE_MODEL?.trim()
  if (!model) {
    const data = (await request('/models', { signal }))?.data
    const models = Array.isArray(data) ? data.map(entry => entry?.id).filter(id => typeof id === 'string' && id) : []
    const named = models.filter(id => /shuttermuse/i.test(id))
    model = named.length === 1 ? named[0] : undefined
    if (!model) throw new Error('Could not identify the local ShutterMuse model. Set SHUTTERMUSE_MODEL to its served model ID.')
  }

  const messages = []
  if (task === 'pose') messages.push({ role: 'system', content: '你是相机取景器上的姿势教练。reason 只有一行空间，最多18个汉字。提到这张照片中一个具体可见细节，只说一句“可以保持…”或“可以试试…”。说完就结束，禁止叠加第二个动作。按用户要求返回 JSON。' })
  if (task === 'advice') messages.push({ role: 'system', content: adviceSystemPrompt })
  messages.push({ role: 'user', content: [
    { type: 'image_url', image_url: { url: `data:${mime};base64,${Buffer.from(image).toString('base64')}` } },
    { type: 'text', text: config.prompt },
  ] })
  const result = await request('/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      model,
      temperature: 0,
      max_tokens: 1024,
      messages,
    }),
  })
  const choice = result?.choices?.[0]
  if (task === 'advice' && choice?.finish_reason === 'length') throw new Error('ShutterMuse photography advice was incomplete.')
  return config.parse(choice?.message?.content)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const [filename, task] = process.argv.slice(2)
    if (!filename) throw new Error('Usage: node tools/shuttermuse-infer.mjs <image.jpg|image.png> [pose|composition|advice]')
    const mime = mimeTypes[extname(filename).toLowerCase()]
    const result = await inferPose({ image: await readFile(filename), mime, task })
    console.log(JSON.stringify(result, null, 2))
  } catch (error) {
    console.error(error.message)
    process.exitCode = 1
  }
}
