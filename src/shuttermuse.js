import './shuttermuse.css';
import fieldCases from './field-cases.json';
import { cropPhoto, drawTargetPose, isLocalStudio, posesDiffer, requestPhotoPose, requestPhotoResult } from './photo-advice.js';
import { mountPhotoPreview, photoPreviewResultMarkup, photoPreviewMarkup } from './photo-preview.js';

const select = (id) => document.getElementById(id);
const file = select('file');
const mode = select('mode');
const intent = select('intent');
const cropOptions = select('crop-options');
const example = select('example');
const sample = select('sample');
const alternatePose = select('alternate-pose');
const retryAdvice = select('retry-advice');
const poseDirection = select('pose-direction');
const instruction = select('pose-instruction');
const applyDirection = select('apply-direction');
const poseChoices = select('pose-choices');
const views = {
  original: select('original'),
  prediction: select('pose'),
  following: select('following'),
};
const preview = select('preview');
const empty = select('empty');
const filename = select('filename');
const photoCaption = select('photo-caption');
const photoKind = select('photo-kind');
const status = select('status');
const legend = select('target-legend');
const suggestion = select('suggestion');
select('generated-preview-slot').outerHTML = photoPreviewMarkup({ inlineResult: false });
select('generated-result-slot').outerHTML = photoPreviewResultMarkup;
const setPreviewPose = mountPhotoPreview(() => {
  showView('original');
  select('photo-pair').scrollIntoView({ block: 'nearest' });
});
const context = preview.getContext('2d');
const imageTypes = ['image/jpeg', 'image/png', 'image/webp'];

const examples = {
  'composition-wide': ['composition', '宽景'],
  'composition-tilt': ['composition', '倾斜视角'],
  'composition-keep': ['composition', '当前构图'],
  'crop-ratio': ['crop', '湖边人像', '4:3'],
  'crop-half': ['crop', '山景半身人像', 'half-body'],
  'scene-stairs': ['scene-pose', '台阶'],
  'scene-rail': ['scene-pose', '护栏'],
  'joint-orchard': ['joint', '果园'],
  'refine-cafe': ['pose', '咖啡馆'],
};
const fieldExamples = new Map(fieldCases.map(entry => [entry.id, entry]));
select('field-examples').replaceChildren?.(...fieldCases.map(({ id, title }) => {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = title;
  return option;
}));
const decisions = {
  keep: '模型建议保留原构图。按建议拍摄时可以参考原片。',
  refine: '模型给出了一个裁切范围。',
  reject: '模型建议重新取景。',
};
const poseTasks = new Set(['pose', 'scene-pose', 'joint']);
const poseChoiceCount = 3;
const poseWords = text => text.replace(/[\s\p{P}]/gu, '');
let image = null;
let result = null;
let followingPhoto = null;
let operation;
let choiceBatch;
let appliedInstruction = '';

function setStatus(message, error = false) {
  status.textContent = message;
  status.toggleAttribute('data-error', error);
}

function setResult(target, cropped = null) {
  result = target;
  followingPhoto = cropped;
  suggestion.textContent = target?.suggestion ?? '';
  suggestion.hidden = !suggestion.textContent;
  alternatePose.hidden = !target?.landmarks;
  retryAdvice.hidden = !image || !!target;
  views.original.hidden = views.prediction.hidden = !target;
  views.following.hidden = !cropped;
  setPreviewPose(image, target?.landmarks ? target : null);
  for (const choice of choiceBatch?.choices ?? []) {
    choice.button.setAttribute('aria-pressed', String(choice.target === target && !!target));
  }
}

function setPoseStatus(target) {
  const framing = target.box ? '可依照框内取景，再参考骨架摆姿势。' : '';
  const count = choiceBatch ? '' : '本次只返回一种姿势，可以点击“换个姿势”重试。';
  setStatus(framing + count);
}

function showPose(target) {
  setResult(target);
  showView('prediction');
  setPoseStatus(target);
}

function markChoice(choice, label) {
  choice.label.textContent = label;
  choice.label.hidden = !label;
}

function setChoiceTarget(choice, target) {
  choice.target = target;
  choice.description.textContent = target.suggestion;
  markChoice(choice, '');
  // Frame the visible joints tightly so a distant subject is readable on a small card.
  const visible = target.landmarks.filter(point => point.visibility === 1);
  const xs = visible.map(point => point.x * image.naturalWidth);
  const ys = visible.map(point => point.y * image.naturalHeight);
  const left = Math.min(...xs), right = Math.max(...xs);
  const top = Math.min(...ys), bottom = Math.max(...ys);
  const scale = 0.88 / Math.max(right - left, bottom - top, 1);
  const thumbnailPoints = target.landmarks.map(point => ({
    ...point,
    x: 0.5 + (point.x * image.naturalWidth - (left + right) / 2) * scale,
    y: 0.5 + (point.y * image.naturalHeight - (top + bottom) / 2) * scale,
  }));
  choice.canvas.width = choice.canvas.height = 160;
  drawTargetPose(choice.canvas.getContext('2d'), thumbnailPoints, 160, 160);
  choice.canvas.hidden = false;
}

function suspendChoices() {
  if (!choiceBatch) return;
  choiceBatch.controller?.abort();
  choiceBatch.wanted = null;
  for (const choice of choiceBatch.choices) {
    choice.button.disabled = true;
    if (!choice.target && !choice.failed) markChoice(choice, '等待姿势骨架');
  }
}

function setChoices(target) {
  suspendChoices();
  choiceBatch = null;
  poseChoices.replaceChildren();
  poseChoices.hidden = !Array.isArray(target?.suggestions);
  if (poseChoices.hidden) return;

  const batch = { photo: image, task: mode.value, instruction: appliedInstruction, choices: [] };
  const matchingWords = new Set();
  const texts = [target.suggestion, ...target.suggestions].filter(text => {
    const words = poseWords(text);
    if (matchingWords.has(words)) return false;
    matchingWords.add(words);
    return true;
  });
  for (let index = 0; index < poseChoiceCount; index += 1) {
    const text = texts[index];
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'secondary';
    const thumbnail = document.createElement('span');
    thumbnail.className = 'pose-choice-preview';
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    canvas.hidden = true;
    const label = document.createElement('span');
    label.className = 'pose-choice-status';
    const description = document.createElement('span');
    description.textContent = text ?? `姿势 ${index + 1}`;
    thumbnail.append(canvas, label);
    button.append(thumbnail, description);
    const choice = { button, canvas, label, description, text };
    markChoice(choice, '等待姿势骨架');
    if (text === target.suggestion) setChoiceTarget(choice, target);
    button.addEventListener('click', () => {
      if (batch !== choiceBatch || batch.controller.signal.aborted) return;
      if (choice.target) {
        if (!batch.wanted && choice.target === result) return;
        batch.wanted = null;
        showPose(choice.target);
        return;
      }
      batch.wanted = choice;
      if (choice.failed && choice.text) choice.retryGrounding = true;
      choice.failed = false;
      setPreviewPose(image, null);
      setStatus('正在准备所选姿势的骨架…');
      prepareChoices(batch);
    });
    batch.choices.push(choice);
    poseChoices.append(button);
  }
  choiceBatch = batch;
  resumeChoices();
}

function resumeChoices() {
  if (!choiceBatch) return;
  choiceBatch.controller = new AbortController();
  choiceBatch.running = false;
  for (const choice of choiceBatch.choices) choice.button.disabled = false;
  prepareChoices(choiceBatch);
}

async function prepareChoices(batch) {
  if (batch.running) return;
  batch.running = true;
  const { signal } = batch.controller;
  const aspect = batch.photo.naturalWidth / batch.photo.naturalHeight;
  try {
    while (!signal.aborted) {
      const choice = batch.wanted ?? batch.choices.find(candidate => !candidate.target && !candidate.failed);
      if (!choice) return;
      markChoice(choice, '正在生成姿势骨架…');
      try {
        const previous = choice.text ? undefined : batch.choices.findLast(candidate => candidate.target).target;
        const target = await requestPhotoPose(batch.photo, signal, previous, batch.task, {
          selectedSuggestion: choice.text, instruction: batch.instruction, retryGrounding: choice.retryGrounding,
        });
        if (signal.aborted) return;
        const words = poseWords(target.suggestion);
        if (batch.choices.some(candidate => candidate !== choice
          && poseWords(candidate.target?.suggestion ?? candidate.text ?? '') === words)) {
          throw new Error('所选动作与已有的姿势很接近，可以重试。');
        }
        if (batch.choices.some(candidate => candidate.target && !posesDiffer(target.landmarks, candidate.target.landmarks, aspect))) {
          throw new Error('所选姿势与已有的骨架很接近，可以重试。');
        }
        setChoiceTarget(choice, target);
        if (batch.wanted === choice) {
          batch.wanted = null;
          showPose(target);
        }
      } catch (error) {
        if (signal.aborted) return;
        choice.failed = true;
        markChoice(choice, '加载失败，点击重试');
        if (batch.wanted === choice) {
          batch.wanted = null;
          setPreviewPose(image, result);
          setStatus(error instanceof TypeError ? '无法连接推理服务，请重试。' : error.message, true);
        }
      }
    }
  } finally {
    if (!signal.aborted) batch.running = false;
  }
}

function showView(view) {
  if (!image) return;
  for (const [key, button] of Object.entries(views)) {
    button.setAttribute('aria-pressed', String(view === key));
  }

  const source = view === 'following' ? followingPhoto : image;
  const sourceWidth = source.naturalWidth || source.width;
  const sourceHeight = source.naturalHeight || source.height;
  const scale = Math.min(1, 640 / Math.max(sourceWidth, sourceHeight));
  const width = preview.width = Math.max(1, Math.round(sourceWidth * scale));
  const height = preview.height = Math.max(1, Math.round(sourceHeight * scale));
  context.fillStyle = '#fff';
  context.fillRect(0, 0, width, height);
  context.drawImage(source, 0, 0, width, height);

  let label = view === 'following' ? '按建议裁切的照片' : '原片';
  legend.hidden = true;
  if (view === 'prediction') {
    if (result.box) {
      const [x1, y1, x2, y2] = result.box;
      context.strokeStyle = '#d4fd69';
      context.lineWidth = Math.max(2, Math.max(width, height) * 0.004);
      context.strokeRect(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height);
      label = '叠加参考裁切框的照片';
      legend.textContent = '绿色框表示建议裁切范围';
      legend.hidden = false;
    }
    if (result.landmarks) {
      drawTargetPose(context, result.landmarks, width, height);
      label = result.box ? '叠加建议裁切框和姿势骨架的照片' : '叠加建议姿势骨架的照片';
      legend.textContent = result.box ? '绿色框和骨架表示建议构图与姿势' : '绿色骨架表示建议姿势';
      legend.hidden = false;
    }
    if (!result.box && !result.landmarks) label = '模型预测 · ' + (result.decision === 'keep' ? '保留原构图' : '重新取景');
  }
  preview.setAttribute('aria-label', label);
  photoKind.textContent = view === 'prediction' ? '模型建议' : view === 'following' ? '建议裁切' : '原片';
}

function reset() {
  setChoices(null);
  setResult(null);
  showView('original');
}

function beginOperation() {
  operation?.abort();
  operation = new AbortController();
  return operation.signal;
}

async function analyze(signal, previousPose) {
  const task = mode.value;
  const cropIntent = task === 'crop' ? intent.value : undefined;
  setStatus('正在本机分析照片…');
  try {
    if (poseTasks.has(task)) {
      setStatus(previousPose ? '正在结合现场和上一种姿势设计新动作…' : '正在看原片，设计动作并计算关节位置…');
      const target = await requestPhotoPose(image, signal, previousPose, task, {
        poseChoices: true, instruction: appliedInstruction,
      });
      if (signal.aborted) return;
      setChoices(target);
      showPose(target);
      return;
    }

    const target = await requestPhotoResult(image, task, signal, cropIntent);
    if (signal.aborted) return;
    if (!Object.hasOwn(decisions, target?.decision)) throw new Error('推理结果格式有误，请重试。');
    const needsRatioCrop = target.decision === 'keep' && cropIntent === '4:3'
      && image.naturalWidth * 3 !== image.naturalHeight * 4;
    const box = needsRatioCrop ? [0, 0, 1, 1] : target.box;
    const cropped = target.decision === 'refine' || needsRatioCrop ? cropPhoto(image, box, cropIntent) : null;
    setResult({
      decision: target.decision,
      box: cropped?.sourceBox,
      suggestion: typeof target.suggestion === 'string' ? target.suggestion.trim() : '',
    }, cropped);
    showView('prediction');
    setStatus(needsRatioCrop ? '模型倾向于保留原构图，已按指令居中裁切为横向 4:3。' : decisions[target.decision]);
  } catch (error) {
    if (signal.aborted) return;
    setResult(previousPose ?? null);
    if (previousPose) resumeChoices();
    setStatus(error instanceof TypeError ? '无法连接推理服务，请重试。' : error.message, true);
  } finally {
    if (operation?.signal === signal) operation = null;
  }
}

function readInstruction() {
  return instruction.value.trim().slice(0, 300);
}

function refreshAdvice() {
  if (!image) return;
  const signal = beginOperation();
  appliedInstruction = readInstruction();
  reset();
  retryAdvice.hidden = true;
  analyze(signal);
}

async function loadPhoto(getBlob, name) {
  const signal = beginOperation();
  appliedInstruction = readInstruction();
  image = null;
  reset();
  applyDirection.disabled = preview.hidden = photoCaption.hidden = true;
  empty.hidden = false;
  filename.textContent = filename.title = '';
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
    image = decoded;
    showView('original');
    preview.hidden = photoCaption.hidden = false;
    applyDirection.disabled = false;
    empty.hidden = true;
    filename.textContent = filename.title = name;
  } catch (error) {
    if (!signal.aborted) {
      operation = null;
      setStatus(error.message, true);
    }
    return;
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
  await analyze(signal);
}

function taskChanged() {
  operation?.abort();
  operation = null;
  cropOptions.hidden = mode.value !== 'crop';
  poseDirection.hidden = !poseTasks.has(mode.value);
  reset();
  setStatus(image ? '可以分析当前照片，或换一张照片。' : '选一张照片，或试用一个样本。');
}

mode.addEventListener('change', () => {
  example.value = Object.keys(examples).find(key => examples[key][0] === mode.value);
  taskChanged();
});
intent.addEventListener('change', taskChanged);
file.addEventListener('change', () => {
  const chosen = file.files[0];
  if (!chosen) return;
  file.value = '';
  loadPhoto(() => chosen, chosen.name);
});
function loadSample(key) {
  const field = fieldExamples.get(key);
  const sampleDetails = field ? [field.task, field.title] : examples[key];
  if (!sampleDetails) return;
  const [task, name, cropIntent] = sampleDetails;
  const path = field ? `field-cases/${key}.jpg` : `pose-examples/${key}.png`;
  const source = field ? '摄影实测原片' : '论文示例原片';
  mode.value = task;
  cropOptions.hidden = task !== 'crop';
  poseDirection.hidden = !poseTasks.has(task);
  if (cropIntent) intent.value = cropIntent;
  return loadPhoto(async (signal) => {
    const response = await fetch(`${import.meta.env.BASE_URL}${path}`, { signal }).catch(() => null);
    if (!response?.ok) throw new Error('无法加载样本照片，请重试。');
    return response.blob();
  }, `${source} · ${name}`);
}

sample.addEventListener('click', () => loadSample(example.value));
retryAdvice.addEventListener('click', refreshAdvice);
poseDirection.addEventListener('submit', event => {
  event.preventDefault();
  refreshAdvice();
});
alternatePose.addEventListener('click', () => {
  const previousPose = result;
  if (!previousPose?.landmarks) return;
  suspendChoices();
  setPreviewPose(image, null);
  analyze(beginOperation(), previousPose);
});
for (const [key, button] of Object.entries(views)) {
  button.addEventListener('click', () => showView(key));
}
window.addEventListener('pagehide', () => {
  operation?.abort();
  suspendChoices();
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  operation = null;
  if (result?.landmarks) {
    setResult(result);
    resumeChoices();
    setPoseStatus(result);
  } else if (!result) {
    retryAdvice.hidden = !image;
    setStatus(image ? '可以分析当前照片，或换一张照片。' : '选一张照片，或试用一个样本。');
  }
});

if (!isLocalStudio) {
  file.disabled = sample.disabled = mode.disabled = intent.disabled = example.disabled = instruction.disabled = true;
  setStatus('请在项目目录运行 npm run shuttermuse -- llama 或 npm run shuttermuse -- mlx，然后打开 http://127.0.0.1:4174/shuttermuse.html。');
} else {
  const fieldId = new URLSearchParams(location.search).get('field');
  if (fieldExamples.has(fieldId)) {
    example.value = fieldId;
    loadSample(fieldId);
  }
}
