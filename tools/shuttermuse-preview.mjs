import { setTimeout as delay } from 'node:timers/promises'
import { gatewayAccessToken } from './shuttermuse-guest.mjs'

export const gatewayBodyLimit = 2 * 1024 * 1024
const responseBodyLimit = 1024 * 1024
const timeoutMessage = '等待示意图超时。网关任务可能仍在处理，请稍后查看生成记录。'
const jointNames = ['鼻子', '左眼', '右眼', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘',
  '左腕', '右腕', '左髋', '右髋', '左膝', '右膝', '左踝', '右踝']
const bodyEdges = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

class PreviewError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

export function imageType(image) {
  const head = image.subarray(0, 8).toString('hex')
  if (head.startsWith('ffd8ff')) return 'image/jpeg'
  if (head === '89504e470d0a1a0a') return 'image/png'
  if (image.length >= 16 && image.toString('latin1', 0, 4) === 'RIFF' &&
      /^WEBPVP8[ LX]$/.test(image.toString('latin1', 8, 16))) return 'image/webp'
}

export function validatePreviewLandmarks(landmarks) {
  if (landmarks === undefined) return
  if (!Array.isArray(landmarks) || landmarks.length !== jointNames.length || !jointNames.every((_, index) => {
    const point = landmarks[index]
    return point && [-1, 0, 1].includes(point.visibility)
      && [point.x, point.y].every(value => Number.isFinite(value) && value >= 0 && value <= 1)
  })) throw new PreviewError(400, '请提供 17 个有效的姿势坐标。')
  if (!bodyEdges.some(([a, b]) => {
    const start = landmarks[a]
    const end = landmarks[b]
    return start.visibility === 1 && end.visibility === 1 && (start.x !== end.x || start.y !== end.y)
  })) throw new PreviewError(400, '姿势坐标中缺少可见的身体骨架。')
}

export function validatePreviewMode(previewMode, landmarks) {
  if (previewMode !== undefined && !['pose', 'scene-pose', 'joint'].includes(previewMode)) {
    throw new PreviewError(400, '无效的示意图任务。')
  }
  if ((previewMode === 'scene-pose' || previewMode === 'joint') && !landmarks) {
    throw new PreviewError(400, '生成入镜示意图需要姿势坐标。')
  }
}

function recordId(value) {
  if (typeof value === 'number' && Number.isSafeInteger(value)) value = String(value)
  if (typeof value === 'string' && /^[1-9]\d{0,19}$/.test(value) && BigInt(value) <= 18446744073709551615n) return value
}

function invalidResult() {
  return new PreviewError(502, '图片服务未返回可用的示意图。')
}

async function readGatewayData(response) {
  if (!response.ok || Number(response.headers.get('content-length')) > responseBodyLimit) {
    await response.body?.cancel().catch(() => {})
    if (response.status === 401) {
      throw new PreviewError(502, '图片服务未授权，请检查本机网关访问令牌。')
    }
    if (response.status === 403) throw new PreviewError(502, '图片服务未授权，请检查网关访问权限或余额。')
    if (response.status === 429) throw new PreviewError(429, '图片服务请求过多，请稍后重试。')
    if (response.status === 504) throw new PreviewError(504, timeoutMessage)
    if (!response.ok) throw new PreviewError(502, '图片服务暂时无法生成示意图，请重试。')
    throw invalidResult()
  }
  if (!response.body) throw invalidResult()
  const chunks = []
  let size = 0
  for await (const chunk of response.body) {
    size += chunk.length
    if (size > responseBodyLimit) throw invalidResult()
    chunks.push(chunk)
  }
  let result
  try {
    result = JSON.parse(Buffer.concat(chunks).toString(), (key, value, context) => {
      // The gateway serializes uint64 IDs as JSON numbers, which can exceed JS integer precision.
      if ((key === 'record_id' || key === 'id') && typeof value === 'number') return context?.source ?? value
      return value
    })
  } catch { throw invalidResult() }
  if (result?.code !== 200) throw invalidResult()
  return result.data
}

export async function generatePreview({ image, mime, suggestion, instruction, landmarks, previewMode, targetPoseGuide, signal }) {
  if (!Buffer.isBuffer(image) || imageType(image) !== mime || !mime) {
    throw new PreviewError(415, '请提供 JPEG、PNG 或 WebP 原图。')
  }
  if (typeof suggestion !== 'string' || !suggestion.trim() || suggestion.length > 500) {
    throw new PreviewError(400, '缺少有效的拍照建议。')
  }
  if (instruction !== undefined && (typeof instruction !== 'string' || !instruction.trim() || instruction.trim().length > 300)) {
    throw new PreviewError(400, '请提供有效的姿势指令（最多 300 字）。')
  }
  validatePreviewLandmarks(landmarks)
  validatePreviewMode(previewMode, landmarks)
  if (targetPoseGuide !== undefined) {
    if (!Buffer.isBuffer(targetPoseGuide) || imageType(targetPoseGuide) !== 'image/jpeg') {
      throw new PreviewError(415, '姿势参考图必须是 JPEG 图片。')
    }
    if (!landmarks) throw new PreviewError(400, '姿势参考图需要对应的姿势坐标。')
  }
  const tooLarge = new PreviewError(413, '照片太大，无法生成示意图。')
  const encodedImageBytes = 4 * (Math.ceil(image.length / 3) + Math.ceil((targetPoseGuide?.length ?? 0) / 3))
  if (encodedImageBytes >= gatewayBodyLimit) throw tooLarge

  const port = process.env.AGENT_GATEWAY_PORT ?? '8080'
  if (!/^\d{1,5}$/.test(port) || Number(port) < 1 || Number(port) > 65535) {
    throw new PreviewError(503, '本机图片网关端口配置无效。')
  }

  let prompt
  if (previewMode === 'scene-pose' || previewMode === 'joint') {
    const framing = previewMode === 'joint' ? '输入为按建议画框裁切后的空场景照片。' : '输入为空场景原照片。'
    prompt = framing + '在场景中生成恰好一位人物。让人物比例、透视、脚下接触和光影与场景自然衔接。'
      + '保留所提交照片的画幅、取景、背景和光线。'
  } else if (landmarks) {
    prompt = '让输入照片中的人物改成目标姿势。保留原照片的人物身份、五官、服装、道具、画幅、取景、背景和光线。'
  } else {
    prompt = '以原照片为底图，真实地展示以下一条拍照建议，只做这条建议所必需的最小调整。'
      + '以原照片可见的人物和环境为准，保留人物身份、五官、服装、道具、背景和光线。'
      + '未涉及的姿势、视角和构图保持原样。建议保持某个细节时原样保留。'
      + '不要添加文字、标注或原图之外的内容。'
  }
  const guideInstructions = targetPoseGuide ? '图一是干净的照片底图，决定最终画面的场景和取景。'
    + '图二是与图一同画幅的目标骨架，决定人物姿势和在画面中的位置。将图二的关节和肢体方向逐一映射到图一，'
    + '让人物的头、躯干、手臂和双腿沿对应骨架摆放，保持骨架所示的身体朝向、倾斜和支撑方式。' : ''
  const coordinates = landmarks ? '以下是所提交画幅内的目标关节坐标 [x,y]，以左上角为原点，'
    + 'x 向右、y 向下，按画幅宽高归一化到 [0,1]，左右以人物自身为准。'
    + landmarks.flatMap(({ x, y, visibility }, index) => visibility === 1 ? [`${jointNames[index]}[${x},${y}]`] : []).join(' ')
    + '。优先按坐标摆出实际人体的姿势，匹配各可见关节在画面中的位置以及躯干和四肢的方向。'
    + '按每侧可见肩、肘、腕的位置摆放手臂，让手腕落在对应目标，手掌与前臂自然衔接。'
    + '拍照建议补充坐标未描述的细节。' : ''
  const finish = landmarks ? '。只输出一张写实照片，画面中不要出现骨架线、关节点、坐标、文字或其他标注。'
    + (targetPoseGuide ? '以图一的背景为准。' : '') : ''
  const userInstruction = instruction ? `用户的姿势指令：${JSON.stringify(instruction.trim())}。动作以用户指令为准。`
    + (landmarks ? '用参考关节位置落实动作在画面中的位置、比例和方向。' : '') : ''
  const imageData = `data:${mime};base64,${image.toString('base64')}`
  const body = JSON.stringify({
    model: 'group:image_edit',
    ...(targetPoseGuide
      ? { images: [imageData, `data:image/jpeg;base64,${targetPoseGuide.toString('base64')}`] }
      : { image_data_uri: imageData }),
    prompt: `${prompt}${guideInstructions}${coordinates}${userInstruction}拍照建议：${JSON.stringify(suggestion.trim())}${finish}`,
    n: 1,
  })
  if (Buffer.byteLength(body) > gatewayBodyLimit) throw tooLarge

  const timeout = AbortSignal.timeout(180_000)
  const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout
  const base = `http://127.0.0.1:${Number(port)}`
  try {
    requestSignal.throwIfAborted()
    const token = await gatewayAccessToken(base, requestSignal).catch(error => {
      if (error?.status === 502) throw new PreviewError(502, error.message)
      throw error
    })
    requestSignal.throwIfAborted()
    const authorization = { Authorization: `Bearer ${token}` }
    const response = await fetch(`${base}/v1/images/edits`, {
      method: 'POST',
      headers: { ...authorization, 'Content-Type': 'application/json' },
      body,
      redirect: 'error',
      signal: requestSignal,
    })
    const id = recordId((await readGatewayData(response))?.record_id)
    if (!id) throw invalidResult()
    while (true) {
      requestSignal.throwIfAborted()
      const records = await fetch(`${base}/v1/images/records?record_id=${id}`, {
        method: 'GET',
        headers: authorization,
        redirect: 'error',
        signal: requestSignal,
      })
      const { items } = (await readGatewayData(records)) ?? {}
      requestSignal.throwIfAborted()
      if (!Array.isArray(items)) throw invalidResult()
      const record = items.find(item => recordId(item?.id) === id)
      if (record?.status === 3) throw new PreviewError(502, '图片服务未能生成示意图，请重试。')
      if (record?.status === 2) {
        if (typeof record.output_url !== 'string' || !URL.canParse(record.output_url)) throw invalidResult()
        const url = new URL(record.output_url)
        if (url.protocol !== 'https:' || url.username || url.password) throw invalidResult()
        return { url: url.href }
      }
      if (record && record.status !== 1) throw invalidResult()
      await delay(1000, undefined, { signal: requestSignal })
    }
  } catch (error) {
    if (signal?.aborted) throw new PreviewError(499, '已取消等待。')
    if (timeout.aborted) throw new PreviewError(504, timeoutMessage)
    if (error instanceof PreviewError) throw error
    throw new PreviewError(502, '暂时无法连接本机图片网关，请检查网关后重试。')
  }
}
