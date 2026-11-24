export const isLocalStudio = ['http://127.0.0.1:4174', 'http://localhost:4174']
  .includes(globalThis.location?.origin);

async function requestPhoto(photo, path, fields, signal, maxEdge = 640) {
  if (!isLocalStudio) throw new Error('请从本机试拍页打开照片。');
  signal?.throwIfAborted();
  const width = photo.naturalWidth || photo.width;
  const height = photo.naturalHeight || photo.height;
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, maxEdge / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);
  const prefix = 'data:image/jpeg;base64,';
  const data = canvas.toDataURL('image/jpeg', 0.9);
  if (!data.startsWith(prefix)) throw new Error('无法读取照片，请重新拍摄或选择照片。');
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    redirect: 'error',
    signal,
    body: JSON.stringify({ ...fields, mime: 'image/jpeg', image: data.slice(prefix.length) }),
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) throw new Error(result?.error || '本机模型暂时无法给出建议。');
  return result;
}

export function requestPhotoResult(photo, task, signal, intent) {
  const fields = { task };
  if (task === 'crop') {
    fields.intent = intent;
    fields.width = photo.naturalWidth || photo.width;
    fields.height = photo.naturalHeight || photo.height;
  }
  return requestPhoto(photo, '/api/shuttermuse', fields, signal);
}

// COCO-17 connections for the on-screen overlay.
const edges = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

function poseSuggestion(value) {
  const text = typeof value === 'string' ? value.trim() : '';
  return text.length <= 500 && text.replace(/^(可以)?(试试|尝试)?[，。！：\s]*/u, '').length > 1 ? text : '';
}

export function posesDiffer(points, previousPoints, aspect = 1) {
  const common = edges.filter(([a, b]) => a >= 5 && [points, previousPoints].every(pose =>
    pose[a].visibility === 1 && pose[b].visibility === 1));
  if (!common.length) return false;
  const torso = common.filter(([a, b]) => [5, 6, 11, 12].includes(a) && [5, 6, 11, 12].includes(b));
  const reference = torso.length ? torso : common;
  const bodyScale = pose => Math.max(1e-6, ...reference.map(([a, b]) =>
    Math.hypot((pose[b].x - pose[a].x) * aspect, pose[b].y - pose[a].y)));
  const size = bodyScale(points);
  const previousSize = bodyScale(previousPoints);
  // Compare limb directions and lengths relative to the body, independent of framing.
  return common.some(([a, b]) => Math.hypot(
    ((points[b].x - points[a].x) / size - (previousPoints[b].x - previousPoints[a].x) / previousSize) * aspect,
    (points[b].y - points[a].y) / size - (previousPoints[b].y - previousPoints[a].y) / previousSize,
  ) >= 0.18);
}

export async function requestPhotoPose(photo, signal, previousPose, task = 'pose', options = {}) {
  const fields = { task };
  if (previousPose) fields.previousSuggestion = previousPose.suggestion;
  if (options.poseChoices) fields.poseChoices = true;
  if (options.selectedSuggestion) fields.selectedSuggestion = options.selectedSuggestion;
  if (options.retryGrounding) fields.retryGrounding = true;
  const instruction = options.instruction?.trim();
  if (instruction) fields.instruction = instruction;
  const result = await requestPhoto(photo, '/api/shuttermuse', fields, signal);
  const points = result?.landmarks;
  if (!Array.isArray(points) || points.length !== 17 || !points.every(p =>
    p && [0, 1, -1].includes(p.visibility) && [p.x, p.y].every(n => Number.isFinite(n) && n >= 0 && n <= 1))) {
    throw new Error('模型没有返回可用的姿势坐标，请重试。');
  }
  if (!edges.some(([a, b]) => a >= 5 && points[a].visibility === 1 && points[b].visibility === 1
    && (points[a].x !== points[b].x || points[a].y !== points[b].y))) {
    throw new Error('这张照片没有可用的身体姿势参考，可以试试让肩膀或手臂入镜。');
  }
  const suggestion = typeof result.suggestion === 'string' ? result.suggestion.trim() : '';
  const repeatedAction = suggestion && suggestion === previousPose?.suggestion;
  const aspect = (photo.naturalWidth || photo.width) / (photo.naturalHeight || photo.height);
  if (previousPose && (repeatedAction || !posesDiffer(points, previousPose.landmarks, aspect))) {
    throw new Error('这次建议与上一种很接近，画面和文字仍显示上一种姿势。');
  }
  if (options.selectedSuggestion && suggestion !== options.selectedSuggestion) {
    throw new Error('模型未返回所选姿势，请重试。');
  }
  const usable = poseSuggestion(suggestion);
  const pose = { landmarks: points, suggestion: usable || '可以参考图中的身体姿势。' };
  if (instruction) pose.instruction = instruction;
  if (options.poseChoices && usable && Array.isArray(result.suggestions)) {
    pose.suggestions = [...new Set([usable, ...result.suggestions.map(poseSuggestion).filter(Boolean)])];
  }
  if (task === 'scene-pose' || task === 'joint') pose.previewMode = task;
  if (task === 'joint') {
    validateBox(result.box);
    pose.box = result.box;
  }
  return pose;
}

export function drawTargetPose(ctx, landmarks, width, height) {
  const points = landmarks.map(p => p.visibility === 1 ? [p.x * width, p.y * height] : null);
  ctx.strokeStyle = ctx.fillStyle = '#d4fd69';
  ctx.lineWidth = Math.max(2, Math.max(width, height) * 0.005);
  ctx.lineCap = 'round';
  ctx.beginPath();
  let drawn = false;
  for (const [a, b] of edges) {
    if (!points[a] || !points[b]) continue;
    ctx.moveTo(...points[a]);
    ctx.lineTo(...points[b]);
    drawn = true;
  }
  ctx.stroke();
  for (const point of points) {
    if (!point) continue;
    ctx.beginPath();
    ctx.arc(...point, ctx.lineWidth, 0, Math.PI * 2);
    ctx.fill();
  }
  return drawn;
}

function validateBox(box) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(n => Number.isFinite(n) && n >= 0 && n <= 1)
    || box[0] >= box[2] || box[1] >= box[3]) throw new Error('模型返回的取景范围无效。');
}

export function cropPhoto(photo, box, intent) {
  validateBox(box);
  const width = photo.naturalWidth || photo.width;
  const height = photo.naturalHeight || photo.height;
  let left = Math.min(width - 1, Math.floor(box[0] * width));
  let top = Math.min(height - 1, Math.floor(box[1] * height));
  let w = Math.ceil(box[2] * width) - left;
  let h = Math.ceil(box[3] * height) - top;
  if (intent === '4:3') {
    const unit = Math.floor(Math.min(w / 4, h / 3));
    if (!unit) throw new Error('模型推荐的区域过小，无法裁出 4:3 照片。');
    left += Math.floor((w - 4 * unit) / 2);
    top += Math.floor((h - 3 * unit) / 2);
    w = 4 * unit;
    h = 3 * unit;
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.sourceBox = [left / width, top / height, (left + w) / width, (top + h) / height];
  canvas.getContext('2d').drawImage(photo, left, top, w, h, 0, 0, w, h);
  return canvas;
}

export async function requestGeneratedPreview(photo, pose, signal) {
  signal?.throwIfAborted();
  let landmarks = pose.landmarks;
  if (pose.previewMode === 'joint') {
    photo = cropPhoto(photo, pose.box);
    const [left, top, right, bottom] = photo.sourceBox;
    landmarks = landmarks.map(point => {
      const x = (point.x - left) / (right - left);
      const y = (point.y - top) / (bottom - top);
      const outside = x < 0 || x > 1 || y < 0 || y > 1;
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), visibility: outside ? -1 : point.visibility };
    });
  }
  const fields = { suggestion: pose.suggestion, landmarks, previewMode: pose.previewMode, instruction: pose.instruction };
  if (landmarks) {
    const width = photo.naturalWidth || photo.width;
    const height = photo.naturalHeight || photo.height;
    const scale = Math.min(1, 640 / Math.max(width, height));
    const guide = document.createElement('canvas');
    guide.width = Math.max(1, Math.round(width * scale));
    guide.height = Math.max(1, Math.round(height * scale));
    const ctx = guide.getContext('2d');
    ctx.fillStyle = '#151515';
    ctx.fillRect(0, 0, guide.width, guide.height);
    if (!drawTargetPose(ctx, landmarks, guide.width, guide.height)) throw new Error('当前画幅内没有可用的目标骨架。');
    const encoded = guide.toDataURL('image/jpeg', 0.85);
    const prefix = 'data:image/jpeg;base64,';
    if (!encoded.startsWith(prefix)) throw new Error('无法读取目标骨架图。');
    fields.targetPoseGuide = encoded.slice(prefix.length);
  }
  const { url } = await requestPhoto(photo, '/api/shuttermuse/preview', fields, signal, 1280);
  signal?.throwIfAborted();
  if (typeof url !== 'string' || !URL.canParse(url)) throw new Error('网关未返回可显示的示意图。');
  const remote = new URL(url);
  if (remote.protocol !== 'https:' || remote.username || remote.password) throw new Error('示意图地址无效。');
  const decoded = new Image();
  decoded.referrerPolicy = 'no-referrer';
  decoded.src = remote.href;
  await decoded.decode();
  return remote.href;
}
