import { readFile } from 'node:fs/promises'
import { extname } from 'node:path'
import { pathToFileURL } from 'node:url'

const endpoint = 'http://127.0.0.1:8081/v1'
const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png' }

const jointOrder = '鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝。'
const poseCoordinates = 'keypoints_xyn 给出目标姿势的 17 对归一化坐标 [x,y]，以整张图片左上角为原点。visibility 标记可见为1、遮挡为0、画外为-1；关节顺序：' + jointOrder
const poseSchema = '只返回一个 JSON 对象，含 instance_info 数组中的一个元素，字段 task_type 为 pose，reason 用一句中文描述入镜位置和动作，keypoints_xyn 为17对归一化坐标，visibility 为每点的可见状态。关节顺序：'
  + jointOrder + '坐标以原图左上角为原点，x 向右，y 向下，范围为0到1。可见标1、遮挡标0、画外标-1。'

const visibleProps = '现场没有看见的道具不要让人物拿取或依靠。'
const subjectBrief = '为照片里已入镜的人设计下一张照片的姿势。根据当前景别、可见的身体部位、活动空间和附近可触及的物体决定动作。先辨清人物原本在坐、站还是倚靠，再设计同一景别里能看出轮廓变化的新动作。保持身体平衡，仅在画面里的新支撑面能稳定承重时才改变支撑。' + visibleProps
const sceneBrief = '为这张空场景中的一位尚未入镜的人设计人像姿势。先确认画面中容得下人物的位置、可触及的物体和可以承重的表面。' + visibleProps
const shortPose = '每个方案只说一个在当前景别能看清的主要动作，用中文说清哪个关节移到哪里，方便照着做。改变手臂轮廓、重心或支撑比仅改变表情和目光更容易在骨架中读出。'
const poseOptions = '给出三个不同的备选方案，分开填入 JSON 的 suggestions 数组。每项只用一句简短中文说清一个主要动作。没有指定动作时，第一项在当前支撑上改变手臂轮廓，第二项换一种可行的身体重心，第三项改变前两项都没有用过的可见关节。保持原来的景别。换左右、换视线或重复相同肢体动作不算新方案。'
const poseOptionsFormat = {
  type: 'json_schema',
  json_schema: {
    name: 'pose_suggestions',
    strict: true,
    schema: {
      type: 'object',
      properties: { suggestions: { type: 'array', items: { type: 'string' }, minItems: 3, maxItems: 3 } },
      required: ['suggestions'],
      additionalProperties: false,
    },
  },
}
const subjectAlternative = '另选一种当前景别中可见的身体动作，让参与动作的关节明显变化。只返回一句中文，描述这个新动作。'
const sceneAlternative = '请针对同一张照片想一个明显不同的动作。重新观察画面，换一种身体支撑方式或重心；若无法安全改变支撑，就选择不同的肢体动作。只返回一句中文，描述新的入镜位置和动作。'

const poseGrounding = '按照摄影方案标注，关节坐标要清楚体现文字里的支撑点和身体动作。先定位方案移动的手腕、肘、膝或踝，再确定余下关节；写出坐标前核对抬高的关节是否确实有更小的 y 值。让 reason 与实际标注一致。手接触现场物体时，手腕坐标应落在物体可触及的表面。文字中的左右以人物自己为准。'
export const subjectPrompt = '给照片里已入镜的人标注做出摄影方案所述动作之后的姿势。保留原照片的景别，无关的关节和支撑位置尽量留在原位；人物已经坐着时保持坐姿，除非方案明确需要起身。保持身体平衡，支撑位置要与可见的环境相符。' + poseGrounding + poseSchema

const sceneInstruction = '给这张空场景中尚未入镜的人标注目标姿势。让承担体重的部位落在画面中可见、稳固的表面上。' + poseGrounding
const scenePosePrompt = sceneInstruction + poseSchema
const jointPrompt = sceneInstruction + '给摄影师推荐一个保留整个人物及适量背景的裁切框。画框四边留出空间，必须包含所有可见关节。'
  + '只输出一个 JSON 对象，字段为 reason、keypoints_xyn、visibility、composition_xy。reason 用一句中文说明入镜位置和摆姿势的方法。'
  + poseCoordinates + 'composition_xy 用一组四个数字表示 [左,上,右,下]，以整张图片左上角为原点，归一化到 [0,1]。姿势坐标也必须相对原图。'

export const compositionPrompt = '请判断当前照片的构图，留意地平线是否倾斜、画面边缘是否有干扰物。如果构图已经很好，保留原图并输出(0,0),(1000,1000)；'
  + '如果裁剪能明显改善构图，找出最好的区域并输出左上角和右下角的坐标(x1,y1),(x2,y2)，坐标以整张图片归一化到0–1000；'
  + '如果无法通过裁剪得到好的构图，输出<non>，并给一条重拍时可尝试的取景建议，例如需要更多场景时退后一步。'
  + '请先用一句中文说明理由，最后一行只输出坐标或<non>。'

export const advicePrompt = '基于眼前这张照片，你会给模特哪一句可选建议？觉得现在就好，也可以说保持。根据当前景别里看得到的身体部位或道具，只说一个细节。'
export const adviceSystemPrompt = '你是提供可选拍照灵感的助手。尊重模特自己的喜好。回答只有一行，以“可以”开头，最多25个汉字。只提一个具体可见的细节或动作。'

const cropPrompts = new Map([
  ['4:3', '请找出照片中构图最好的区域，按横向 4:3 的实像素比例输出裁切框，保留主体及关键环境。即使主体很小，也选一个符合 4:3 的画框。'],
  ['half-body', '请找出照片中适合裁成半身人像的区域，请按照(x1,y1),(x2,y2)的格式返回一个 bounding box，其中(x1,y1)是左上角的顶点，(x2,y2)是右下角的顶点。画面上方留出抬起的手与头顶，裁切框下沿紧贴人物的腰部，框内不要有下半身。坐标按原图归一化到0–1000。只返回裁切坐标；若画面完全没有人物，只输出<non>。'],
])

export function cropRequestError({ intent, width, height }) {
  if (!cropPrompts.has(intent)) return '请选择有效的裁切意图。'
  if (intent === '4:3' && (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 4 || height < 3)) {
    return '4:3 裁切需要有效的原图宽高。'
  }
}

function cropPrompt({ intent, width, height }) {
  const instruction = cropPrompts.get(intent)
  if (intent === 'half-body') return instruction
  return instruction + `原图宽 ${width} 像素，高 ${height} 像素。`
    + '坐标按原图归一化到 0–1000，(x1,y1) 是左上角，(x2,y2) 是右下角。用一句中文说明取景，最后一行只输出 (x1,y1),(x2,y2)。'
}

function stripThinking(content) {
  return content.replace(/^\s*<think>[\s\S]*?<\/think>/, '').trim()
}

function plainAnswer(content) {
  return typeof content === 'string'
    ? stripThinking(content).replace(/^<answer>\s*|\s*<\/answer>$/gi, '').trim()
    : ''
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

  const pose = prediction?.instance_info?.[0] ?? prediction
  const { keypoints_xyn: keypoints, visibility } = pose ?? {}
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
const rejectsCropping = /(?:无法|不能)(?:(?:通过|用)(?:简单(?:的)?)?裁剪(?:来)?(?:改善|优化|挽救)构图|从(?:现有)?(?:的)?(?:原图|照片)中?裁出[^。！？\n]+)/
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
    const composition = prediction?.instance_info?.[0] ?? prediction
    if (!composition || typeof composition !== 'object' || Array.isArray(composition)) throw malformed
    const rawDecision = composition.decision ?? prediction.decision
    decision = typeof rawDecision === 'string' ? rawDecision.trim().toLowerCase() : rawDecision
    box = Object.hasOwn(composition, 'composition_xy')
      ? composition.composition_xy
      : composition.bbox_xy ?? composition.bbox ?? composition.composition_bbox ?? composition.box
    suggestion = composition.reason ?? composition.suggestion ?? prediction.reason ?? prediction.suggestion
    if (Object.hasOwn(composition, 'composition_xy') && (box === null || (Array.isArray(box) && !box.length))) {
      if (decision !== undefined && decision !== 'reject') throw malformed
      decision = 'reject'
      box = undefined
    }
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

const bodyEdges = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

function parseGuidedPose(content) {
  const pose = parsePose(content)
  const hasVisibleBody = bodyEdges.some(([a, b]) => {
    const [start, end] = [pose.landmarks[a], pose.landmarks[b]]
    return start.visibility === 1 && end.visibility === 1 && (start.x !== end.x || start.y !== end.y)
  })
  if (!hasVisibleBody || !pose.suggestion || pose.suggestion.length > 500 || !/\p{L}/u.test(pose.suggestion)) {
    throw new Error('ShutterMuse must return a visible body pose and actionable guidance.')
  }
  return pose
}

const jsonNumber = String.raw`-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?`
const cornerList = new RegExp(String.raw`("composition_xy"\s*:\s*)\[?\s*\[\s*(${jsonNumber})\s*,\s*(${jsonNumber})\s*\]\s*,\s*\[\s*(${jsonNumber})\s*,\s*(${jsonNumber})\s*\]\s*\]`, 'g')
const extraBoxBracket = new RegExp(String.raw`("composition_xy"\s*:\s*${flatBox})\s*\](?=\s*[,}])`, 'g')

function parseJoint(content) {
  // The model occasionally mismatches brackets around the composition field.
  const answer = typeof content === 'string'
    ? content.replace(cornerList, '$1[$2,$3,$4,$5]').replace(extraBoxBracket, '$1')
    : content
  const pose = parseGuidedPose(answer)
  const { box } = parseComposition(answer)
  if (!box) throw new Error('ShutterMuse must return a crop for the pose in the original image.')

  const [x1, y1, x2, y2] = box
  const visible = pose.landmarks.filter(point => point.visibility === 1)
  if (visible.every(({ x, y }) => x >= x1 && x <= x2 && y >= y1 && y <= y2)) return { ...pose, box }

  const xs = visible.map(point => point.x)
  const ys = visible.map(point => point.y)
  const margin = 0.03
  const fitted = [
    Math.max(0, Math.min(x1, Math.min(...xs) - margin)),
    Math.max(0, Math.min(y1, Math.min(...ys) - margin)),
    Math.min(1, Math.max(x2, Math.max(...xs) + margin)),
    Math.min(1, Math.max(y2, Math.max(...ys) + margin)),
  ]
  return { ...pose, box: fitted }
}

function parseCrop(content, { intent, width, height }) {
  const result = parseComposition(content)
  if (result.decision === 'reject') {
    return { ...result, suggestion: result.suggestion ?? '这张照片暂时无法按所选意图裁切。' }
  }
  if (intent === 'half-body' || (result.decision === 'keep' && width / 4 === height / 3)) {
    return { ...result, suggestion: result.suggestion ?? (result.decision === 'keep' ? '原画面已符合所选意图。' : '可以参考框中的半身取景。') }
  }

  const [x1, y1, x2, y2] = result.box ?? [0, 0, 1, 1]
  const scale = Math.floor(Math.min((x2 - x1) * width / 4, (y2 - y1) * height / 3))
  if (scale < 1) throw new Error('ShutterMuse returned a crop too small to fit 4:3.')
  const cropWidth = scale * 4
  const cropHeight = scale * 3
  const x = Math.max(0, Math.min(width - cropWidth, Math.round(((x1 + x2) * width - cropWidth) / 2)))
  const y = Math.max(0, Math.min(height - cropHeight, Math.round(((y1 + y2) * height - cropHeight) / 2)))
  return {
    decision: 'refine',
    box: [x / width, y / height, (x + cropWidth) / width, (y + cropHeight) / height],
    suggestion: (result.decision === 'refine' && result.suggestion) || '可以参考框中的横向 4:3 取景。',
  }
}

function parseAdvice(content) {
  const suggestion = plainAnswer(content)
  if (!/\p{Script=Han}/u.test(suggestion) || /[\r\n<>{}\[\]]/.test(suggestion) || suggestion.match(boxPattern)) {
    throw new Error('ShutterMuse returned no usable photography advice.')
  }
  return { suggestion }
}

function parsePoseBrief(choice) {
  const brief = plainAnswer(choice?.message?.content)
  if (choice?.finish_reason === 'length' || brief.length > 500 || /[\r\n<>{}\[\]]/.test(brief) || !/\p{Script=Han}/u.test(brief)) {
    throw new Error('ShutterMuse must return a short photography brief before predicting a pose.')
  }
  return brief
}

const poseWords = suggestion => suggestion.replace(/[\s\p{P}]/gu, '').replace(/[左右]/gu, '侧')

function nearlySamePoseWords(first, second) {
  if (first === second) return true
  if (Math.min(first.length, second.length) < 20) return false
  const pairs = text => new Set(Array.from({ length: text.length - 1 }, (_, index) => text.slice(index, index + 2)))
  const firstPairs = pairs(first)
  const secondPairs = pairs(second)
  const shared = [...firstPairs].filter(pair => secondPairs.has(pair)).length
  return 2 * shared / (firstPairs.size + secondPairs.size) >= 0.78
}

function parsePoseOptions(choice, previous) {
  const text = plainAnswer(choice?.message?.content)
  const block = jsonBlock(text)
  let lines
  if (block) {
    try {
      const suggestions = JSON.parse(block)?.suggestions
      if (Array.isArray(suggestions)) lines = suggestions
    } catch {
      // Unconstrained backends may use plain text instead.
    }
  }
  if (!lines) {
    lines = text.split(/\r?\n/u)
    if (choice?.finish_reason === 'length' && !/[。！？!?]$/u.test(text)) lines.pop()
  }
  const options = new Map()
  const previousWords = poseWords(previous)
  for (const line of lines) {
    if (typeof line !== 'string') continue
    const brief = line.trim().replace(/^(?:[-*]\s+|(?:方案\s*)?[1-3一二三][.)、．：:）]\s*)/u, '')
    if (brief.length <= 500 && !/[\r\n<>{}\[\]：:]|^(?:以下|这里)/u.test(brief) && /\p{Script=Han}/u.test(brief)) {
      const key = poseWords(brief)
      if (![previousWords, ...options.keys()].some(existing => nearlySamePoseWords(existing, key))) options.set(key, brief)
    }
  }
  if (!options.size) throw Object.assign(new Error('这次没有生成可用的新姿势，可以换个拍摄要求再试。'), { status: 422 })
  return [...options.values()].slice(0, 3)
}

function poseConflict(suggestion, points) {
  const visibleY = index => points[index].visibility === 1 ? points[index].y : undefined
  const nose = visibleY(0)
  const ankles = [visibleY(15), visibleY(16)].filter(y => y !== undefined)
  const knees = [visibleY(13), visibleY(14)].filter(y => y !== undefined)
  const ground = ankles.length ? Math.max(...ankles) - 0.07 : knees.length ? Math.max(...knees) : undefined
  // Only check explicit relations supported by the visible joints in this frame.
  const negation = /不|别|避免/u
  const groundContact = /(?:撑|按|扶|触|贴|着|放)[到在住向着于]?(?:(?:身前|身旁|身后|脚边|旁边|前方)的)?(?:地面|地板|地上|地毯|路面|地$)/u
  const raisedToHead = /(?:举|抬|伸|越|放在|扶在)(?:到|至|过|向|在)?(?:头顶(?:上方)?|头部上方|额头(?:前|上方)?)/u

  for (const clause of suggestion.split(/[，,。；;、！？]/u)) {
    const hand = /(?:(左|右|双|两)(?:只|侧)?)?(?:手(?:掌|腕|臂)?|臂)/gu
    const mentions = [...clause.matchAll(hand)]
    for (const [index, match] of mentions.entries()) {
      const [, side] = match
      const prefix = clause.slice(index ? mentions[index - 1].index + mentions[index - 1][0].length : 0, match.index)
      if (negation.test(prefix)) continue
      const action = clause.slice(match.index + match[0].length, mentions[index + 1]?.index)
      const affirmed = pattern => {
        const match = action.match(pattern)
        return match && !negation.test(action.slice(0, match.index))
      }
      const wrists = side === '左' ? [9] : side === '右' ? [10] : [9, 10]
      const known = wrists.map(visibleY).filter(y => y !== undefined)
      if (!known.length) continue
      const both = side === '双' || side === '两'
      const misses = predicate => both ? known.some(predicate) : known.every(predicate)
      if (ground !== undefined && affirmed(groundContact) && misses(y => y < ground)) {
        return '撑地的手腕应接近脚下的地面。'
      }
      if (nose !== undefined && affirmed(raisedToHead) && misses(y => y > nose + 0.03)) {
        return '抬到头顶或额头的手腕应靠近头部高度。'
      }
    }
  }
}

const tasks = new Map([
  ['pose', { brief: subjectBrief, alternative: subjectAlternative, prompt: subjectPrompt, parse: parseGuidedPose }],
  ['composition', { prompt: compositionPrompt, parse: parseComposition }],
  ['advice', { prompt: advicePrompt, parse: parseAdvice }],
  ['crop', { prompt: cropPrompt, parse: parseCrop }],
  ['scene-pose', { brief: sceneBrief, alternative: sceneAlternative, prompt: scenePosePrompt, parse: parseGuidedPose }],
  ['joint', { brief: sceneBrief, alternative: sceneAlternative, prompt: jointPrompt, parse: parseJoint }],
])

export const supportsInferenceTask = task => tasks.has(task)
export const supportsPoseAlternative = task => tasks.get(task)?.alternative !== undefined

async function request(path, options) {
  let response
  try {
    response = await fetch(endpoint + path, { ...options, redirect: 'error' })
  } catch (error) {
    if (options.signal.aborted) throw options.signal.reason
    throw new Error(`Cannot reach the local ShutterMuse server at ${endpoint}.`, { cause: error })
  }
  if (!response.ok) throw Object.assign(new Error(`Local ShutterMuse ${path} returned HTTP ${response.status}.`), { httpStatus: response.status })
  try {
    return await response.json()
  } catch (error) {
    if (options.signal.aborted) throw options.signal.reason
    throw new Error(`Local ShutterMuse ${path} returned invalid JSON.`, { cause: error })
  }
}

export async function inferPose({ image, mime, signal, task = 'pose', previousSuggestion, poseChoices, selectedSuggestion, retryGrounding, instruction, intent, width, height }) {
  if (!(image instanceof Uint8Array) || !image.byteLength || !Object.values(mimeTypes).includes(mime)) {
    throw new TypeError('Expected nonempty JPEG or PNG image bytes.')
  }
  const config = tasks.get(task)
  if (!config) throw new TypeError('Expected pose, composition, advice, crop, scene-pose, or joint task.')
  const userDirection = typeof instruction === 'string' ? instruction.trim() : ''
  const selection = typeof selectedSuggestion === 'string' ? selectedSuggestion.trim() : ''
  if (instruction !== undefined && (typeof instruction !== 'string' || userDirection.length > 300)) throw new TypeError('拍摄要求最多 300 个字符。')
  if (selectedSuggestion !== undefined && (!selection || selection.length > 500)) throw new TypeError('请选择本轮已有的姿势。')
  const crop = { intent, width, height }
  if (task === 'crop') {
    const error = cropRequestError(crop)
    if (error) throw new TypeError(error)
  }
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

  const photo = { type: 'image_url', image_url: { url: `data:${mime};base64,${Buffer.from(image).toString('base64')}` } }
  const imageMessage = prompt => ({ role: 'user', content: [photo, { type: 'text', text: prompt }] })
  async function complete(messages, { temperature = 0, maxTokens = 1024, responseFormat } = {}) {
    const result = await request('/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({ model, temperature, max_tokens: maxTokens, messages, response_format: responseFormat }),
    })
    return result?.choices?.[0]
  }

  const prompt = typeof config.prompt === 'function' ? config.prompt(crop) : config.prompt
  if (config.brief) {
    const direction = userDirection ? '\n用户拍摄要求（限动作与取景偏好，请在现场可行的范围内优先满足）：' + JSON.stringify(userDirection) : ''
    let brief = selection
    let suggestions
    if (!brief) {
      const previous = typeof previousSuggestion === 'string' ? previousSuggestion.trim() : ''
      const planningPrompt = poseChoices === true
        ? poseOptions + config.brief + direction
        : config.brief + shortPose + '只用一句话，只返回这一句话。' + direction
      const planningMessages = [imageMessage(planningPrompt)]
      if (poseChoices === true) {
        if (previous) {
          planningMessages.push({ role: 'assistant', content: previous }, {
            role: 'user', content: '为这张照片选一组与刚才不同的姿势。' + poseOptions + direction,
          })
        }
        const settings = { temperature: 0.45, maxTokens: 384 }
        let choice
        try {
          choice = await complete(planningMessages, { ...settings, responseFormat: poseOptionsFormat })
        } catch (error) {
          if (![400, 422, 501].includes(error.httpStatus)) throw error
          choice = await complete(planningMessages, settings)
        }
        suggestions = parsePoseOptions(choice, previous)
        brief = suggestions[0]
      } else {
        const followUp = { role: 'user', content: config.alternative + direction }
        if (previous) planningMessages.push({ role: 'assistant', content: previous }, followUp)
        const settings = { temperature: previous ? 0.5 : 0, maxTokens: 256 }
        brief = parsePoseBrief(await complete(planningMessages, settings))
        if (previous && poseWords(brief) === poseWords(previous)) {
          followUp.content = '请选择与上一种不同的身体动作。仅改变描述、位置或左右不算新姿势。' + config.alternative + direction
          brief = parsePoseBrief(await complete(planningMessages, settings))
          if (poseWords(brief) === poseWords(previous)) {
            throw Object.assign(new Error('这次建议与上一种很接近，画面和文字仍显示上一种姿势。'), { status: 422 })
          }
        }
      }
    }
    const preserveBrief = Boolean(selection || suggestions || userDirection)
    const newGrounding = selection && retryGrounding === true
    async function predict(candidate, correction = '') {
      const posePrompt = prompt + '\n摄影方案：' + JSON.stringify(candidate) + direction
        + (preserveBrief ? '\n按选定方案画目标姿势，不要用别的动作替换主要动作。' : '')
        + (newGrounding ? '\n上次四肢轮廓和另一方案太接近。请重新计算这条动作实际移动的关节。' : '') + correction
      const choice = await complete([imageMessage(posePrompt)], { temperature: newGrounding ? 0.3 : 0 })
      try {
        return config.parse(choice?.message?.content, crop)
      } catch (error) {
        if (!preserveBrief) throw error
        throw Object.assign(new Error('模型没有给出可以绘制的姿势。', { cause: error }), { status: 422 })
      }
    }

    const candidates = suggestions ?? [brief]
    for (const [index, candidate] of candidates.entries()) {
      try {
        let pose = await predict(candidate)
        let conflict = poseConflict(candidate, pose.landmarks)
        if (conflict) {
          pose = await predict(candidate, '\n刚才的坐标没有体现方案：' + conflict + '重新标注同一个方案，逐一检查移动的关节。')
          conflict = poseConflict(candidate, pose.landmarks)
          if (conflict) throw Object.assign(new Error('模型未能画出这条姿势。' + conflict), { status: 422 })
        }
        if (preserveBrief) pose.suggestion = candidate
        if (suggestions) pose.suggestions = suggestions.slice(index)
        return pose
      } catch (error) {
        if (error.status !== 422 || index === candidates.length - 1) throw error
      }
    }
  }
  const messages = task === 'advice' ? [{ role: 'system', content: adviceSystemPrompt }] : []
  messages.push(imageMessage(prompt))
  const choice = await complete(messages)
  if (task === 'advice' && choice?.finish_reason === 'length') throw new Error('ShutterMuse photography advice was incomplete.')
  return config.parse(choice?.message?.content, crop)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const [filename, task, intent, width, height] = process.argv.slice(2)
    if (!filename) throw new Error('Usage: node tools/shuttermuse-infer.mjs <image.jpg|image.png> [pose|composition|advice|scene-pose|joint|crop half-body|crop 4:3 <width> <height>]')
    const mime = mimeTypes[extname(filename).toLowerCase()]
    const result = await inferPose({ image: await readFile(filename), mime, task, intent, width: Number(width), height: Number(height) })
    console.log(JSON.stringify(result, null, 2))
  } catch (error) {
    console.error(error.message)
    process.exitCode = 1
  }
}
