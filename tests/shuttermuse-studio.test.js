import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { createServer } from 'vite';

test('studio automatically requests advice, ignores stale results, and retries on the original', async (t) => {
  const canvas = {
    getContext: () => ({ fillRect() {}, drawImage() {} }),
    setAttribute(name, value) { this[name] = value; },
    toDataURL: () => 'data:image/jpeg;base64,/9j/',
  };
  const elements = new Map([['preview', canvas]]);
  const element = (id) => {
    if (!elements.has(id)) {
      elements.set(id, Object.assign(new EventTarget(), { toggleAttribute() {} }));
    }
    return elements.get(id);
  };
  const browserGlobals = {
    document: { getElementById: element, createElement: () => ({ ...canvas }) },
    location: { origin: 'http://127.0.0.1:4174' },
    Image: class {
      naturalWidth = 100;
      naturalHeight = 100;
      async decode() {}
    },
  };
  for (const [key, value] of Object.entries(browserGlobals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else Reflect.deleteProperty(globalThis, key);
    });
  }
  const requests = [];
  t.mock.method(globalThis, 'fetch', (path, { body, signal }) => new Promise((resolve) => {
    assert.equal(path, '/api/shuttermuse');
    assert.deepEqual(JSON.parse(body), { task: 'advice', mime: 'image/jpeg', image: '/9j/' });
    requests.push({ signal, resolve });
  }));

  const vite = await createServer({
    configFile: false,
    optimizeDeps: { noDiscovery: true },
    server: { middlewareMode: true, ws: false, watch: null },
  });
  t.after(() => vite.close());
  await vite.ssrLoadModule('/src/shuttermuse.js');
  const choose = async (name) => {
    const file = element('file');
    file.files = [Object.assign(new Blob([], { type: 'image/jpeg' }), { name })];
    file.dispatchEvent(new Event('change'));
    await setImmediate();
  };

  await choose('first.jpg');
  assert.equal(requests.length, 1);
  await choose('second.jpg');
  assert.equal(requests.length, 2);
  assert.equal(requests[0].signal.aborted, true);
  requests[1].resolve(Response.json({ suggestion: '把相机放低一点。' }));
  await setImmediate();
  requests[0].resolve(Response.json({ suggestion: '向左移动一步。' }));
  await setImmediate();
  assert.equal(element('filename').textContent, 'second.jpg');
  assert.equal(element('preview')['aria-label'], '原片');
  assert.equal(element('suggestion').textContent, '把相机放低一点。');
  assert.equal(element('target-legend').hidden, true);

  await choose('third.jpg');
  assert.equal(requests.length, 3);
  requests[2].resolve(Response.json({ error: '模型暂时不可用' }, { status: 503 }));
  await setImmediate();
  assert.equal(element('suggestion').hidden, true);
  assert.equal(element('retry-advice').hidden, false);
  assert.equal(element('status').textContent, '模型暂时不可用');
  assert.equal(element('filename').textContent, 'third.jpg');
  element('retry-advice').dispatchEvent(new Event('click'));
  assert.equal(requests.length, 4);
  requests[3].resolve(Response.json({ suggestion: '往后退一步。' }));
  await setImmediate();
  assert.equal(element('suggestion').textContent, '往后退一步。');
  assert.equal(element('retry-advice').hidden, true);
});
