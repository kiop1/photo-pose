import { spawn } from 'node:child_process'
import { createServer } from 'vite'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { inferPose } from './shuttermuse-infer.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const maxImage = 8 * 1024 * 1024
const maxBody = 4 * Math.ceil(maxImage / 3) + 1024

function failure(status, message) {
  return Object.assign(new Error(message), { status })
}

async function readPhoto(req) {
  if (req.headers['content-type']?.split(';')[0] !== 'application/json') throw failure(415, '请发送 JSON 格式的照片。')
  if (Number(req.headers['content-length']) > maxBody) throw failure(413, '照片超过 8 MiB。')
  let size = 0
  const chunks = []
  for await (const chunk of req) {
    size += chunk.length
    if (size > maxBody) throw failure(413, '照片超过 8 MiB。')
    chunks.push(chunk)
  }
  let data
  try { data = JSON.parse(Buffer.concat(chunks).toString()) }
  catch { throw failure(400, '无法读取照片请求。') }

  if (typeof data?.image !== 'string' || !data.image.length) throw failure(400, '缺少图片数据。')
  const task = data.task === undefined ? 'pose' : data.task
  if (!['pose', 'composition', 'advice'].includes(task)) throw failure(400, '无效的推理任务。')
  const image = Buffer.from(data.image, 'base64')
  if (image.length > maxImage) throw failure(413, '照片超过 8 MiB。')
  if (image.toString('base64') !== data.image) throw failure(400, '照片编码无效。')
  const mime = data.mime
  const head = image.subarray(0, 8).toString('hex')
  if (!(mime === 'image/jpeg' && head.startsWith('ffd8ff') || mime === 'image/png' && head === '89504e470d0a1a0a')) {
    throw failure(415, '推理接口接受 JPEG 或 PNG 图片。')
  }
  return { image, mime, task }
}

export async function poseApi(req, res, predict = inferPose) {
  const controller = new AbortController()
  res.once('close', () => controller.abort())
  res.setHeader('Cache-Control', 'no-store')
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  try {
    const host = req.headers.host
    if (!/^(localhost|127\.0\.0\.1):4174$/.test(host) ||
        (req.headers.origin && req.headers.origin !== 'http://' + host)) throw failure(403, '仅可从本机试拍页提交照片。')
    if (req.method !== 'POST') throw failure(405, '请用 POST 提交照片。')
    const photo = await readPhoto(req)
    const result = await predict({ ...photo, signal: controller.signal })
    res.end(JSON.stringify(result))
  } catch (err) {
    if (res.destroyed) return
    res.statusCode = err.status || 503
    const message = err.status ? err.message : '模型暂时没有给出可用的建议。请查看终端中的模型状态后重试。'
    if (!err.status) console.error('ShutterMuse:', err.message)
    res.end(JSON.stringify({ error: message }))
  }
}

async function start(backend) {
  if (!['llama', 'mlx'].includes(backend)) throw Error('请选择后端：npm run shuttermuse -- llama 或 npm run shuttermuse -- mlx')
  const web = await createServer({
    root,
    server: { host: '127.0.0.1', port: 4174, strictPort: true },
    plugins: [{
      name: 'local-shuttermuse',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.split('?')[0] === '/api/shuttermuse') return void poseApi(req, res)
          next()
        })
      },
    }],
  })
  try { await web.listen() }
  catch (error) { await web.close(); throw error }

  const model = spawn('python3', ['tools/shuttermuse-server.py', backend], { cwd: root, stdio: 'inherit', detached: true })
  console.log('试拍页面：http://127.0.0.1:4174/shuttermuse.html')
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
