import { requestGeneratedPreview } from './photo-advice.js';
import './photo-preview.css';

const defaultDisclosure = '生成时，本机网关会记录照片和拍摄建议，并将它们发给图像服务。';

export const photoPreviewResultMarkup = `
  <figure id="preview-result" hidden>
    <figcaption>AI 示意图</figcaption>
    <div class="result-frame"><img id="generated-photo" referrerpolicy="no-referrer" alt="根据拍照建议生成的示意图"></div>
  </figure>`;

const controls = `
    <p id="generation-disclosure">${defaultDisclosure}</p>
    <button id="generate-preview" type="button" class="preview-action" aria-describedby="generation-disclosure">生成 AI 示意图</button>
    <button id="cancel-preview" type="button" class="preview-action" hidden>取消等待</button>
    <p id="generation-status" role="status"></p>`;

export function photoPreviewMarkup({ inlineResult = true } = {}) {
  return `<section id="generated-preview" aria-label="AI 拍摄示意图" hidden>${controls}${inlineResult ? photoPreviewResultMarkup : ''}</section>`;
}

export function mountPhotoPreview(onGenerated) {
  const section = document.getElementById('generated-preview');
  const disclosure = document.getElementById('generation-disclosure');
  const generate = document.getElementById('generate-preview');
  const cancel = document.getElementById('cancel-preview');
  const status = document.getElementById('generation-status');
  const result = document.getElementById('preview-result');
  const generatedPhoto = document.getElementById('generated-photo');
  let request;
  let source;
  let pose;

  function stop() {
    request?.abort();
    request = null;
    cancel.hidden = true;
    generate.disabled = false;
  }

  function setPose(photo, target) {
    if (source === photo && pose === target) return;
    stop();
    source = photo;
    pose = target;
    section.hidden = !photo || !target;
    const scenePhoto = target?.previewMode === 'joint' ? '裁切后的照片' : '原片';
    disclosure.textContent = target?.landmarks
      ? `生成时，本机网关会记录${scenePhoto}、所选骨架图、关节坐标和拍摄指令，并将它们发给图像服务。`
      : defaultDisclosure;
    result.hidden = true;
    generate.hidden = false;
    generatedPhoto.removeAttribute('src');
    generate.textContent = '生成 AI 示意图';
    status.textContent = '';
  }

  cancel.addEventListener('click', () => {
    stop();
    status.textContent = '已取消等待。';
    generate.focus();
  });

  generate.addEventListener('click', async () => {
    if (request || !source || !pose || !result.hidden) return;
    request = new AbortController();
    const { signal } = request;
    generate.disabled = true;
    cancel.hidden = false;
    status.textContent = '正在生成 AI 示意图…';
    try {
      const url = await requestGeneratedPreview(source, pose, signal);
      if (signal.aborted) return;
      generatedPhoto.src = url;
      generate.hidden = true;
      result.hidden = false;
      status.textContent = 'AI 示意图已生成。';
      onGenerated?.();
    } catch {
      if (signal.aborted) return;
      status.textContent = '暂时无法生成 AI 示意图，可以重试。';
      generate.textContent = '重试生成 AI 示意图';
    } finally {
      if (!signal.aborted) stop();
    }
  });

  window.addEventListener('pagehide', () => {
    if (!request) return;
    stop();
    status.textContent = '已取消等待。';
  });
  return setPose;
}
