const panel = document.querySelector('#pose-upload');

panel.innerHTML = `
  <div class="pose-upload-intro">
    <h2 id="pose-upload-title">用自己的照片找姿势</h2>
    <p id="pose-upload-hint">选择一张 JPG、PNG 或 WebP 场景照片。大图会在浏览器内等比缩小或压缩。</p>
  </div>
  <form class="pose-upload-form" aria-describedby="pose-upload-hint">
    <label for="pose-upload-file">照片</label>
    <input id="pose-upload-file" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" required>
    <label for="pose-upload-angle">拍摄角度</label>
    <select id="pose-upload-angle">
      <option value="auto">自动判断</option>
      <option value="high">俯拍</option>
      <option value="level">平拍</option>
      <option value="low">仰拍</option>
    </select>
    <button type="submit" disabled>检索姿势</button>
  </form>
  <p class="pose-upload-status" role="status" aria-atomic="true"></p>
  <figure class="pose-upload-preview" hidden>
    <figcaption>输入原图</figcaption>
    <img alt="你选择的场景照片">
  </figure>
  <div class="pose-upload-results"></div>
`;
panel.setAttribute('aria-labelledby', 'pose-upload-title');

const form = panel.querySelector('form');
const input = form.querySelector('input');
const angle = form.querySelector('select');
const submit = form.querySelector('button');
const status = panel.querySelector('[role="status"]');
const preview = panel.querySelector('.pose-upload-preview');
const previewImage = preview.querySelector('img');
const results = panel.querySelector('.pose-upload-results');

const mimeTypes = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' };
const maxUploadBytes = 8 * 1024 * 1024;
const maxUploadPixels = 25_000_000;
const resizeLongEdge = 2560;

let file;
let mimeType;
let previewUrl;
let activeRequest;

function resetSearch(message = '') {
  activeRequest?.abort();
  activeRequest = undefined;
  results.replaceChildren();
  preview.hidden = !previewUrl;
  status.textContent = message;
  submit.textContent = '检索姿势';
  submit.disabled = !mimeType;
}

input.addEventListener('change', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  file = input.files[0];
  const extension = file?.name.split('.').pop().toLowerCase();
  const inferredType = file?.type || mimeTypes[extension];
  mimeType = file?.size && Object.values(mimeTypes).includes(inferredType) ? inferredType : undefined;
  previewUrl = mimeType ? URL.createObjectURL(file) : undefined;
  if (previewUrl) previewImage.src = previewUrl;
  else previewImage.removeAttribute('src');
  resetSearch(file && !mimeType ? '请选择可读取的 JPG、PNG 或 WebP 图片。' : '');
});

angle.addEventListener('change', () => resetSearch());

async function isAnimated(file, signal) {
  const header = new Uint8Array(await file.slice(0, 21).arrayBuffer());
  if (header.byteLength < 21) return false;
  const text = new TextDecoder();
  if (text.decode(header.subarray(0, 4)) === 'RIFF' && text.decode(header.subarray(8, 12)) === 'WEBP') {
    return text.decode(header.subarray(12, 16)) === 'VP8X' && Boolean(header[20] & 0x02);
  }
  if (header[0] !== 0x89 || text.decode(header.subarray(1, 4)) !== 'PNG') return false;

  // APNG declares its animation before the first image data chunk.
  for (let offset = 8; offset + 8 <= file.size;) {
    signal.throwIfAborted();
    const chunk = await file.slice(offset, offset + 8).arrayBuffer();
    const type = text.decode(new Uint8Array(chunk, 4));
    if (type === 'acTL') return true;
    if (type === 'IDAT') return false;
    offset += new DataView(chunk).getUint32(0) + 12;
  }
  return false;
}

async function prepareUpload(file, signal) {
  await previewImage.decode();
  signal.throwIfAborted();
  const { naturalWidth: width, naturalHeight: height } = previewImage;
  if (!width || !height) throw new Error('Empty image');
  if (file.size <= maxUploadBytes && width * height <= maxUploadPixels) return file;
  if (await isAnimated(file, signal)) return null;
  signal.throwIfAborted();

  const scale = Math.min(1, resizeLongEdge / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  try {
    const context = canvas.getContext('2d', { alpha: false });
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingQuality = 'high';
    context.drawImage(previewImage, 0, 0, canvas.width, canvas.height);
    for (const quality of [0.9, 0.75]) {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
      signal.throwIfAborted();
      if (blob?.type === 'image/jpeg' && blob.size > 0 && blob.size <= maxUploadBytes) return blob;
    }
    throw new Error('Image compression failed');
  } finally {
    canvas.width = 0;
    canvas.height = 0;
  }
}

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function photo(caption, url, description) {
  const figure = element('figure');
  const image = element('img');
  image.alt = description;
  image.loading = 'lazy';
  image.src = url;
  figure.append(element('figcaption', caption), image);
  return figure;
}

function renderMatches(data) {
  const backgroundReferences = (data.scene_matches ?? data.excluded ?? []).filter((entry) => entry.reference_url);
  if (backgroundReferences.length) {
    const photos = element('div', '', 'photos pose-upload-background');
    for (const reference of backgroundReferences) {
      const score = Number.isFinite(reference.similarity)
        ? ` · 余弦相似度 ${reference.similarity.toFixed(3)}` : '';
      const figure = photo(`场景参考${score}`, reference.reference_url, '背景相似的场景照片');
      if (reference.reason) figure.append(element('p', `姿势筛选 · ${reference.reason}`));
      if (/^https?:\/\//i.test(reference.source || '')) {
        const source = element('a', '照片来源');
        source.href = reference.source;
        source.target = '_blank';
        source.rel = 'noopener noreferrer';
        figure.append(source);
      }
      photos.append(figure);
    }
    const references = element('article');
    references.append(element('h3', '相似场景照片'), photos);
    results.append(references);
  }

  for (const [index, match] of data.matches.entries()) {
    const heading = element('header', '', 'match-head');
    const score = Number.isFinite(match.similarity)
      ? `余弦相似度 ${match.similarity.toFixed(3)}` : '';
    heading.append(element('h3', `姿势候选 ${index + 1}`), element('span', score, 'score'));

    const points = Number.isInteger(match.estimated_count)
      ? ` · 33 点骨架 · 实心 ${match.visible_count} · 空心估计 ${match.estimated_count} · 画外 ${match.outside_count}` : '';
    const photos = element('div', '', 'photos pose-upload-photos');
    photos.append(
      photo('检索照片', data.query_url || previewUrl, '用于检索的场景照片'),
      photo('参考姿势', match.reference_url, '匹配的参考人物照片'),
      photo(`九宫格叠加${points}`, match.overlay_url, '输入照片上的三分线与投影骨架'),
    );
    const candidate = element('article');
    candidate.append(heading, photos);
    results.append(candidate);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!mimeType || activeRequest) return;

  resetSearch('正在准备照片…');
  const request = new AbortController();
  activeRequest = request;
  submit.disabled = true;
  submit.textContent = '检索中…';
  let upload;
  try {
    upload = await prepareUpload(file, request.signal);
    if (request !== activeRequest) return;
    if (!upload) {
      status.textContent = '请选择一张静态照片。';
      submit.textContent = '重试检索';
      return;
    }
    status.textContent = '正在检索相似场景和参考姿势…';
    const response = await fetch(`/api/search?angle=${encodeURIComponent(angle.value)}`, {
      method: 'POST',
      headers: { 'Content-Type': upload.type || mimeType },
      body: upload,
      signal: request.signal,
    });
    const data = await response.json();
    if (request !== activeRequest) return;
    if (!response.ok || !Array.isArray(data?.matches)) {
      status.textContent = data?.error || '检索失败，请重试。';
      submit.textContent = '重试检索';
      return;
    }

    const labels = [data.camera_label && `拍摄角度 ${data.camera_label}`,
      data.framing_label && `取景 ${data.framing_label}`].filter(Boolean);
    const outcome = data.message || (data.matches.length ? `找到 ${data.matches.length} 个候选` : '暂无合适的姿势参考');
    status.textContent = [...labels, outcome].join(' · ');
    renderMatches(data);
    preview.hidden = data.matches.length > 0;
    submit.textContent = '重新检索';
  } catch {
    if (request !== activeRequest) return;
    status.textContent = upload ? '检索失败，请重试。' : '无法处理这张照片，请换一张有效的 JPG、PNG 或 WebP，或导出较小的副本后重试。';
    submit.textContent = '重试检索';
  } finally {
    if (request === activeRequest) {
      activeRequest = undefined;
      submit.disabled = false;
    }
  }
});
