export const isLocalStudio = ['http://127.0.0.1:4174', 'http://localhost:4174']
  .includes(globalThis.location?.origin);

export async function requestPhotoResult(photo, task, signal) {
  if (!isLocalStudio) throw new Error('请从本机试拍页打开照片。');
  signal?.throwIfAborted();
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 640 / Math.max(photo.width, photo.height));
  canvas.width = Math.max(1, Math.round(photo.width * scale));
  canvas.height = Math.max(1, Math.round(photo.height * scale));
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);
  const prefix = 'data:image/jpeg;base64,';
  const data = canvas.toDataURL('image/jpeg', 0.9);
  if (!data.startsWith(prefix)) throw new Error('无法读取照片，请重新拍摄或选择照片。');
  const response = await fetch('/api/shuttermuse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    redirect: 'error',
    signal,
    body: JSON.stringify({ task, mime: 'image/jpeg', image: data.slice(prefix.length) }),
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) throw new Error(result?.error || '本机模型暂时无法给出建议。');
  return result;
}

export async function requestPhotoAdvice(photo, signal) {
  const result = await requestPhotoResult(photo, 'advice', signal);
  if (typeof result.suggestion !== 'string' || !result.suggestion.trim()) {
    throw new Error('本机模型未返回拍摄建议。');
  }
  return result.suggestion.trim();
}
