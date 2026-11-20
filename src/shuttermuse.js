import './shuttermuse.css';
import { isLocalStudio, requestPhotoAdvice, requestPhotoResult } from './photo-advice.js';

const select = (id) => document.getElementById(id);
const file = select('file');
const sample = select('sample');
const compositionSample = select('composition-sample');
const pose = select('pose');
const composition = select('composition');
const retryAdvice = select('retry-advice');
const original = select('original');
const preview = select('preview');
const empty = select('empty');
const filename = select('filename');
const status = select('status');
const legend = select('target-legend');
const suggestion = select('suggestion');
const context = preview.getContext('2d');
const imageTypes = ['image/jpeg', 'image/png', 'image/webp'];
const tasks = {
  pose: {
    label: '姿势',
    samplePath: 'mountain-sunrise.jpg',
    sampleName: '山顶背影 · 示例照片',
  },
  composition: {
    label: '构图',
    samplePath: 'shuttermuse-composition.webp',
    sampleName: '旗杆与红旗 · 官方构图示例',
  },
};
const edges = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];
let image = null;
let advice = null;
let operation;

function setStatus(message, error = false) {
  status.textContent = message;
  status.toggleAttribute('data-error', error);
}

function setBusy(busy) {
  pose.disabled = composition.disabled = busy || !image || !isLocalStudio;
}

function beginOperation() {
  operation?.abort();
  operation = new AbortController();
  resetResult();
  setBusy(true);
  return operation.signal;
}

function showPhoto() {
  const { width, height } = preview;
  context.fillStyle = '#fff';
  context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);
}

function resetResult() {
  legend.hidden = retryAdvice.hidden = original.hidden = true;
  showSuggestion('');
  preview.setAttribute('aria-label', '原片');
  if (image) showPhoto();
}

function showSuggestion(text) {
  suggestion.textContent = text;
  suggestion.hidden = !text;
}

function showAdvice() {
  showSuggestion(advice || '');
  retryAdvice.hidden = !image || advice !== null;
}

function drawPose(landmarks) {
  const { width, height } = preview;
  const points = landmarks.map((point) => (
    point?.visibility > 0
    && Number.isFinite(point.x) && point.x >= 0 && point.x <= 1
    && Number.isFinite(point.y) && point.y >= 0 && point.y <= 1
      ? [point.x * width, point.y * height]
      : null
  ));
  const visible = points.filter(Boolean);
  if (!visible.length) return false;

  const lineWidth = Math.max(width, height) * 0.004;
  context.lineWidth = Math.max(2, lineWidth);
  context.lineCap = 'round';
  context.strokeStyle = context.fillStyle = '#d4fd69';
  context.shadowColor = '#16200d';
  context.shadowBlur = context.lineWidth * 1.5;
  context.beginPath();
  for (const [start, end] of edges) {
    if (points[start] && points[end]) {
      context.moveTo(...points[start]);
      context.lineTo(...points[end]);
    }
  }
  context.stroke();
  for (const [x, y] of visible) {
    context.beginPath();
    context.arc(x, y, context.lineWidth, 0, Math.PI * 2);
    context.fill();
  }
  context.shadowBlur = 0;
  return true;
}

function showComposition(result) {
  switch (result?.decision) {
    case 'keep':
      setStatus('模型倾向于保留当前构图。');
      break;
    case 'refine': {
      const { box } = result;
      if (!Array.isArray(box) || box.length !== 4
        || !box.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
        || box[0] >= box[2] || box[1] >= box[3]) {
        throw new Error('推理结果格式有误，请重试。');
      }
      const [x1, y1, x2, y2] = box;
      const { width, height } = preview;
      context.strokeStyle = '#d4fd69';
      context.lineWidth = Math.max(2, Math.max(width, height) * 0.004);
      context.strokeRect(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height);
      preview.setAttribute('aria-label', '叠加参考裁切框的照片');
      legend.textContent = '绿色框表示可尝试的裁切范围';
      legend.hidden = false;
      setStatus('模型提供了一种可尝试的裁切。');
      return;
    }
    case 'reject':
      setStatus('模型建议换个取景角度，也可以保留原片。');
      break;
    default:
      throw new Error('推理结果格式有误，请重试。');
  }
  preview.setAttribute('aria-label', '已分析的照片');
}

async function loadPhoto(getBlob, name) {
  const signal = beginOperation();
  setStatus('正在读取照片…');
  let url;
  try {
    const blob = await getBlob(signal);
    if (signal.aborted) return;
    if (!imageTypes.includes(blob.type)) throw new Error('请选择 JPEG、PNG 或 WebP 照片。');
    url = URL.createObjectURL(blob);
    const decoded = new Image();
    decoded.src = url;
    await decoded.decode().catch(() => { throw new Error('无法打开照片，请重新选择。'); });
    if (signal.aborted) return;
    const { naturalWidth: width, naturalHeight: height } = decoded;
    const scale = Math.min(1, 640 / Math.max(width, height));
    preview.width = Math.max(1, Math.round(width * scale));
    preview.height = Math.max(1, Math.round(height * scale));
    image = decoded;
    advice = null;
    showPhoto();
    preview.hidden = filename.hidden = false;
    empty.hidden = true;
    filename.textContent = name;
  } catch (error) {
    if (!signal.aborted) {
      showAdvice();
      setStatus(error.message, true);
      setBusy(false);
    }
    return;
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
  await analyze('advice', signal);
}

file.addEventListener('change', () => {
  const chosen = file.files[0];
  if (!chosen) return;
  file.value = '';
  loadPhoto(() => chosen, chosen.name);
});

function loadSample({ samplePath, sampleName }) {
  loadPhoto(async (signal) => {
    const response = await fetch(`${import.meta.env.BASE_URL}pose-images/${samplePath}`, { signal }).catch(() => null);
    if (!response?.ok) throw new Error('无法加载示例照片，请重试。');
    return response.blob();
  }, sampleName);
}

sample.addEventListener('click', () => loadSample(tasks.pose));
compositionSample.addEventListener('click', () => loadSample(tasks.composition));
pose.addEventListener('click', () => analyze('pose', beginOperation()));
composition.addEventListener('click', () => analyze('composition', beginOperation()));
retryAdvice.addEventListener('click', () => analyze('advice', beginOperation()));
original.addEventListener('click', () => {
  resetResult();
  showAdvice();
  setStatus('');
});

async function analyze(mode, signal) {
  setStatus(mode === 'advice'
    ? '正在本机生成拍照建议…'
    : `正在本机生成${tasks[mode].label}建议…`);
  try {
    if (mode === 'advice') {
      const text = await requestPhotoAdvice(preview, signal);
      if (signal.aborted) return;
      advice = text;
      showAdvice();
      setStatus('');
      return;
    }
    const result = await requestPhotoResult(preview, mode, signal);
    if (signal.aborted) return;
    if (mode === 'composition') {
      showComposition(result);
    } else {
      if (!Array.isArray(result?.landmarks) || result.landmarks.length !== 17) {
        throw new Error('推理结果格式有误，请重试。');
      }
      const drawn = drawPose(result.landmarks);
      legend.textContent = '绿色骨架表示建议姿势';
      legend.hidden = !drawn;
      preview.setAttribute('aria-label', drawn ? '叠加建议姿势骨架的照片' : '已分析的照片');
      setStatus(drawn ? '姿势建议已生成。' : '模型没有返回可绘制的姿势，请尝试另一张照片。');
    }
    showSuggestion(typeof result.suggestion === 'string' ? result.suggestion.trim() : '');
    original.hidden = false;
  } catch (error) {
    if (signal.aborted) return;
    showAdvice();
    setStatus(error instanceof TypeError ? '无法连接推理服务，请重试。' : error.message, true);
  } finally {
    if (!signal.aborted) setBusy(false);
  }
}

if (!isLocalStudio) {
  file.disabled = sample.disabled = compositionSample.disabled = true;
  setStatus('请在项目目录运行 npm run shuttermuse -- llama 或 npm run shuttermuse -- mlx，然后打开 http://127.0.0.1:4174/shuttermuse.html。');
}
