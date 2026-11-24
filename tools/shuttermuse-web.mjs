import { spawn } from 'node:child_process'
import { createServer } from 'vite'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { cropRequestError, inferPose, supportsInferenceTask, supportsPoseAlternative } from './shuttermuse-infer.mjs'
import { gatewayBodyLimit, generatePreview, imageType, validatePreviewLandmarks, validatePreviewMode } from './shuttermuse-preview.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const maxImage = 8 * 1024 * 1024
const maxBody = 4 * Math.ceil(maxImage / 3) + 4096
const previewPath = '/api/shuttermuse/preview'

function failure(status, message) {
  return Object.assign(new Error(message), { status })
}

function readInstruction(value) {
  const instruction = typeof value === 'string' ? value.trim() : ''
  if (!instruction || instruction.length > 300) throw failure(400, '拍摄要求须为 1 至 300 字。')
  return instruction
}

function readGuide(encoded) {
  if (typeof encoded !== 'string' || !encoded) throw failure(400, '缺少有效的姿势参考图数据。')
  const guide = Buffer.from(encoded, 'base64')
  if (guide.toString('base64') !== encoded) throw failure(400, '姿势参考图编码无效。')
  if (imageType(guide) !== 'image/jpeg') throw failure(415, '姿势参考图必须是 JPEG 图片。')
  return guide
}

async function readPhoto(req, preview) {
  if (req.headers['content-type']?.split(';')[0] !== 'application/json') throw failure(415, '请发送 JSON 格式的照片。')
  const bodyLimit = preview ? gatewayBodyLimit : maxBody
  const sizeMessage = preview ? '照片或姿势参考图太大，无法生成示意图。' : '照片超过 8 MiB。'
  if (Number(req.headers['content-length']) > bodyLimit) throw failure(413, sizeMessage)
  let size = 0
  const chunks = []
  for await (const chunk of req) {
    size += chunk.length
    if (size > bodyLimit) throw failure(413, sizeMessage)
    chunks.push(chunk)
  }
  let data
  try { data = JSON.parse(Buffer.concat(chunks).toString()) }
  catch { throw failure(400, '无法读取照片请求。') }

  if (typeof data?.image !== 'string' || !data.image.length) throw failure(400, '缺少图片数据。')
  const task = data.task === undefined ? 'pose' : data.task
  if (!supportsInferenceTask(task)) throw failure(400, '无效的推理任务。')
  const image = Buffer.from(data.image, 'base64')
  if (image.length > maxImage) throw failure(413, '照片超过 8 MiB。')
  if (image.toString('base64') !== data.image) throw failure(400, '照片编码无效。')
  const mime = data.mime
  if (!['image/jpeg', 'image/png'].includes(mime) || imageType(image) !== mime) {
    throw failure(415, '推理接口接受 JPEG 或 PNG 图片。')
  }
  if (preview) {
    const suggestion = typeof data.suggestion === 'string' ? data.suggestion.trim() : ''
    if (!suggestion || suggestion.length > 500) throw failure(400, '请提供当前照片的一条拍摄建议（最多 500 字）。')
    const photo = { image, mime, suggestion }
    // Older studios may still send a guide image. Keep its validation at ingress.
    if (data.poseGuide !== undefined) readGuide(data.poseGuide)
    if (data.landmarks !== undefined) {
      validatePreviewLandmarks(data.landmarks)
      photo.landmarks = data.landmarks
    }
    if (data.previewMode !== undefined) {
      validatePreviewMode(data.previewMode, photo.landmarks)
      photo.previewMode = data.previewMode
    }
    if (data.targetPoseGuide !== undefined) {
      if (!photo.landmarks) throw failure(400, '目标骨架图需要对应的姿势坐标。')
      photo.targetPoseGuide = readGuide(data.targetPoseGuide)
    }
    if (data.instruction !== undefined) photo.instruction = readInstruction(data.instruction)
    return photo
  }
  const photo = { image, mime, task }
  if (task === 'crop') {
    const { intent, width, height } = data
    const error = cropRequestError({ intent, width, height })
    if (error) throw failure(400, error)
    photo.intent = intent
    if (intent === '4:3') Object.assign(photo, { width, height })
  }
  if (supportsPoseAlternative(task) && data.previousSuggestion !== undefined) {
    const previousSuggestion = typeof data.previousSuggestion === 'string' ? data.previousSuggestion.trim() : ''
    if (!previousSuggestion || previousSuggestion.length > 500) {
      throw failure(400, '请提供有效的上一条姿势建议（最多 500 字）。')
    }
    photo.previousSuggestion = previousSuggestion
  }
  if (data.poseChoices !== undefined || data.selectedSuggestion !== undefined || data.instruction !== undefined || data.retryGrounding !== undefined) {
    if (!supportsPoseAlternative(task)) throw failure(400, '当前任务不支持姿势方案。')
    if (data.poseChoices !== undefined) {
      if (typeof data.poseChoices !== 'boolean') throw failure(400, '姿势方案选项无效。')
      photo.poseChoices = data.poseChoices
    }
    if (data.instruction !== undefined) photo.instruction = readInstruction(data.instruction)
    if (data.selectedSuggestion !== undefined) {
      const selected = typeof data.selectedSuggestion === 'string' ? data.selectedSuggestion.trim() : ''
      if (!selected || selected.length > 500 || data.poseChoices === true) throw failure(400, '所选姿势方案无效。')
      photo.selectedSuggestion = selected
    }
    if (data.retryGrounding !== undefined) {
      if (data.retryGrounding !== true || !photo.selectedSuggestion) throw failure(400, '请先选择要重新计算的姿势。')
      photo.retryGrounding = true
    }
  }
  return photo
}

export async function poseApi(req, res, predict) {
  const preview = req.url?.split('?')[0] === previewPath
  const controller = new AbortController()
  res.once('close', () => controller.abort())
  res.setHeader('Cache-Control', 'no-store')
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  try {
    const host = req.headers.host
    if (!/^(localhost|127\.0\.0\.1):4174$/.test(host) ||
        (req.headers.origin && req.headers.origin !== 'http://' + host)) throw failure(403, '仅可从本机试拍页提交照片。')
    if (req.method !== 'POST') throw failure(405, '请用 POST 提交照片。')
    const photo = await readPhoto(req, preview)
    const run = predict ?? (preview ? generatePreview : inferPose)
    const result = await run({ ...photo, signal: controller.signal })
    res.end(JSON.stringify(result))
  } catch (err) {
    if (res.destroyed) return
    res.statusCode = err.status || 503
    const message = err.status ? err.message : preview ? '示意图生成失败，请检查本机图像网关后重试。' : '模型暂时没有给出可用的建议。请查看终端中的模型状态后重试。'
    if (!err.status) console.error('ShutterMuse:', err.message)
    res.end(JSON.stringify({ error: message }))
  }
}

async function start(backend) {
  if (!['llama', 'mlx', 'existing'].includes(backend)) throw Error('请选择后端：llama、mlx 或 existing（使用已启动的 8081 模型）。')
  const web = await createServer({
    root,
    server: { host: '127.0.0.1', port: 4174, strictPort: true },
    plugins: [{
      name: 'local-shuttermuse',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (['/api/shuttermuse', previewPath].includes(req.url?.split('?')[0])) return void poseApi(req, res)
          next()
        })
      },
    }],
  })
  try { await web.listen() }
  catch (error) { await web.close(); throw error }

  console.log('试拍页面：http://127.0.0.1:4174/shuttermuse.html')
  if (backend === 'existing') return
  const modelEnv = { ...process.env }
  delete modelEnv.AGENT_GATEWAY_TOKEN
  delete modelEnv.XAI_API_KEY
  const model = spawn('python3', ['tools/shuttermuse-server.py', backend], { cwd: root, env: modelEnv, stdio: 'inherit', detached: true })
  let stopping = false
  const stop = () => {
    stopping = true
    if (model.pid) {
      try { process.kill(-model.pid, 'SIGTERM') } catch (error) { if (error.code !== 'ESRCH') throw error }
    }
  }
  process.once('SIGINT', stop)
  process.once('SIGTERM', stop)
  try {
    const exitCode = await new Promise((ok, fail) => {
      model.once('error', fail)
      model.once('exit', (code) => ok(code ?? 1))
    })
    if (!stopping && exitCode) throw Error('模型服务提前退出，请检查上方日志。')
  } finally {
    await web.close()
    process.removeListener('SIGINT', stop)
    process.removeListener('SIGTERM', stop)
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  start(process.argv[2]).catch(error => { console.error(error.message); process.exitCode = 1 })
}
