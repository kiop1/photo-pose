import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { createServer } from 'vite';

test('Studio compares original photos with live predictions and results across all five tasks', async (t) => {
  const encodedPhotos = [];
  const draws = [];
  const rects = [];
  let imageSize = [100, 100];
  const makeCanvas = () => {
    const points = [];
    let source;
    const context = {
      fillRect() {},
      drawImage(photo, ...coordinates) {
        source = photo;
        draws.push({ photo, coordinates });
        points.length = 0;
      },
      beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
      arc(x, y) { points.push([x, y]); },
      strokeRect(...rect) { rects.push(rect); },
    };
    return {
      points,
      getContext: () => context,
      setAttribute(name, value) { this[name] = value; },
      toDataURL() {
        encodedPhotos.push({ source, points });
        return source ? 'data:image/jpeg;base64,/9j/' : 'data:image/jpeg;base64,R1VJREU=';
      },
    };
  };
  const canvas = makeCanvas();
  const elements = new Map([['preview', canvas]]);
  const makeElement = () => Object.assign(new EventTarget(), {
    children: [],
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this[name] = value; },
    toggleAttribute() {},
    removeAttribute(name) { delete this[name]; },
    focus() {},
    scrollIntoView() { this.scrolled = true; },
  });
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  };
  element('mode').value = 'pose';
  element('intent').value = '4:3';
  element('pose-instruction').value = '';
  element('example').value = 'refine-cafe';
  const browserGlobals = {
    document: { getElementById: element, createElement: tag => tag === 'canvas' ? makeCanvas() : makeElement() },
    window: new EventTarget(),
    location: { origin: 'http://127.0.0.1:4174' },
    Image: class {
      naturalWidth = imageSize[0];
      naturalHeight = imageSize[1];
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
    requests.push({ path, body: body && JSON.parse(body), signal, resolve });
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
  const click = (id) => element(id).dispatchEvent(new Event('click'));
  const change = (id, value) => {
    element(id).value = value;
    element(id).dispatchEvent(new Event('change'));
  };
  const finish = async (request, result, status = 200) => {
    request.resolve(Response.json(result, { status }));
    await setImmediate();
  };
  const landmarks = Array.from({ length: 17 }, (_, index) => ({
    x: index % 2 === 0 ? 0.6 : 0.4,
    y: 0.2 + index * 0.025,
    visibility: index === 4 ? 0 : 1,
  }));
  const secondPose = { landmarks, suggestion: '把手抬到胸前。' };
  const generated = { url: 'https://cdn.example.test/demo.png' };

  await choose('first.jpg');
  assert.equal(requests.length, 1);
  assert.equal(requests[0].path, '/api/shuttermuse');
  assert.deepEqual(requests[0].body, { task: 'pose', poseChoices: true, mime: 'image/jpeg', image: '/9j/' });

  await choose('second.jpg');
  assert.equal(requests.length, 2);
  assert.equal(requests[0].signal.aborted, true);
  await finish(requests[1], secondPose);
  await finish(requests[0], { ...secondPose, suggestion: '向左移动一步。' });
  assert.equal(element('filename').textContent, 'second.jpg');
  assert.equal(element('preview')['aria-label'], '叠加建议姿势骨架的照片');
  assert.equal(element('suggestion').textContent, secondPose.suggestion);
  assert.equal(element('pose-choices').hidden, true);
  assert.equal(element('target-legend').hidden, false);
  assert.equal(element('generated-preview').hidden, false);
  const renderedPoints = [...canvas.points];
  assert.equal(renderedPoints.length, 16);
  assert.deepEqual(renderedPoints[0], [60, 20]);

  click('original');
  assert.equal(element('preview')['aria-label'], '原片');
  assert.equal(element('target-legend').hidden, true);
  assert.equal(element('suggestion').textContent, secondPose.suggestion);
  assert.equal(element('pose').hidden, false);
  assert.equal(canvas.points.length, 0);
  click('pose');
  assert.deepEqual(canvas.points, renderedPoints);
  assert.equal(requests.length, 2);

  const encodedCount = encodedPhotos.length;
  click('generate-preview');
  const canceled = requests.at(-1);
  assert.equal(canceled.path, '/api/shuttermuse/preview');
  assert.deepEqual(canceled.body, {
    suggestion: secondPose.suggestion, landmarks, mime: 'image/jpeg', image: '/9j/', targetPoseGuide: 'R1VJREU=',
  });
  assert.equal(encodedPhotos.length, encodedCount + 2);
  assert.deepEqual(encodedPhotos.at(-2).points, renderedPoints);
  assert.equal(encodedPhotos.at(-2).source, undefined);
  assert.deepEqual(encodedPhotos.at(-1).points, []);
  assert.ok(encodedPhotos.at(-1).source instanceof Image);
  click('cancel-preview');
  assert.equal(canceled.signal.aborted, true);
  assert.equal(element('generate-preview').disabled, false);
  click('generate-preview');
  const outdated = requests.at(-1);
  await choose('third.jpg');
  assert.equal(outdated.signal.aborted, true);
  assert.equal(element('generated-preview').hidden, true);
  await finish(canceled, generated);
  await finish(outdated, generated);
  await finish(requests.at(-1), { landmarks: landmarks.map(point => ({ ...point, visibility: -1 })) });
  assert.equal(element('generated-photo').src, undefined);
  assert.equal(element('suggestion').hidden, true);
  assert.equal(element('retry-advice').hidden, false);
  assert.equal(element('alternate-pose').hidden, true);
  assert.match(element('status').textContent, /没有可用的身体姿势/);
  assert.equal(element('filename').textContent, 'third.jpg');

  click('retry-advice');
  await finish(requests.at(-1), { landmarks });
  const fallback = element('suggestion').textContent;
  assert.equal(fallback, '可以参考图中的身体姿势。');
  assert.equal(element('target-legend').hidden, false);
  assert.equal(element('retry-advice').hidden, true);

  click('generate-preview');
  await finish(requests.at(-1), { error: '生成失败' }, 503);
  assert.equal(element('generate-preview').textContent, '重试生成 AI 示意图');
  assert.equal(element('preview-result').hidden, true);
  click('generate-preview');
  await finish(requests.at(-1), { url: 'javascript:alert(1)' });
  assert.equal(element('preview-result').hidden, true);
  click('generate-preview');
  const oldGeneration = requests.at(-1);
  assert.equal(oldGeneration.body.suggestion, fallback);
  click('alternate-pose');
  assert.equal(oldGeneration.signal.aborted, true);
  const alternative = requests.at(-1);
  assert.deepEqual(alternative.body, { ...requests[0].body, previousSuggestion: fallback });
  await finish(alternative, { landmarks, suggestion: '这个姿势没有变化。' });
  assert.equal(element('suggestion').textContent, fallback);
  assert.deepEqual(canvas.points, renderedPoints);
  assert.match(element('status').textContent, /与上一种很接近/);

  click('alternate-pose');
  const nextLandmarks = landmarks.map((point, index) => index === 9 ? { ...point, x: 0.75 } : point);
  const newSuggestion = '抬起左手。';
  await finish(requests.at(-1), { landmarks: nextLandmarks, suggestion: newSuggestion });
  await finish(oldGeneration, generated);
  assert.equal(element('suggestion').textContent, newSuggestion);
  assert.equal(element('preview-result').hidden, true);
  assert.notDeepEqual(canvas.points, renderedPoints);
  assert.equal(requests.at(-1).path, '/api/shuttermuse');
  click('generate-preview');
  assert.equal(requests.at(-1).body.suggestion, newSuggestion);
  assert.deepEqual(requests.at(-1).body.landmarks, nextLandmarks);
  assert.deepEqual(encodedPhotos.at(-1).points, []);
  await finish(requests.at(-1), generated);
  assert.equal(element('generated-photo').src, generated.url);
  assert.equal(element('preview-result').hidden, false);
  assert.equal(element('photo-kind').textContent, '原片');
  assert.equal(element('original')['aria-pressed'], 'true');
  assert.equal(element('photo-pair').scrolled, true);
  assert.equal(canvas.points.length, 0);
  click('pose');
  assert.equal(element('preview-result').hidden, false);
  assert.equal(element('generated-photo').src, generated.url);
  assert.equal(canvas.points.length, renderedPoints.length);
  assert.equal(element('filename').textContent, 'third.jpg');

  const sample = async (key) => {
    element('example').value = key;
    click('sample');
    const source = requests.at(-1);
    assert.equal(source.path, `/pose-examples/${key}.png`);
    source.resolve(new Response(new Blob([], { type: 'image/png' })));
    await setImmediate();
    return requests.at(-1);
  };
  const composition = await sample('composition-wide');
  assert.equal(composition.body.task, 'composition');
  assert.equal(element('preview-result').hidden, true);
  await finish(composition, { decision: 'refine', box: [0.25, 0.25, 0.75, 0.75], suggestion: '收紧左侧边缘。' });
  assert.deepEqual(rects.at(-1), [25, 25, 50, 50]);
  assert.equal(element('suggestion').textContent, '收紧左侧边缘。');
  assert.equal(element('filename').textContent, '论文示例原片 · 宽景');
  assert.deepEqual(draws.at(-2).coordinates, [25, 25, 50, 50, 0, 0, 50, 50]);
  click('following');
  assert.equal(element('following')['aria-pressed'], 'true');
  assert.equal(element('preview')['aria-label'], '按建议裁切的照片');
  assert.deepEqual([canvas.width, canvas.height], [50, 50]);
  click('original');
  assert.deepEqual([canvas.width, canvas.height], [100, 100]);

  const tilt = await sample('composition-tilt');
  await finish(tilt, { decision: 'reject', suggestion: '放平相机再拍。' });
  assert.equal(element('following').hidden, true);
  assert.equal(element('alternate-pose').hidden, true);
  assert.match(element('status').textContent, /重新取景/);
  const keep = await sample('composition-keep');
  await finish(keep, { decision: 'keep' });
  assert.match(element('status').textContent, /保留原构图/);

  imageSize = [240, 360];
  const ratio = await sample('crop-ratio');
  assert.equal(ratio.body.task, 'crop');
  assert.equal(ratio.body.intent, '4:3');
  await finish(ratio, { decision: 'refine', box: [0.25, 0.25, 0.75, 0.75] });
  assert.deepEqual(rects.at(-1), [60, 135, 120, 90]);
  click('following');
  assert.deepEqual([canvas.width, canvas.height], [120, 90]);

  change('intent', 'half-body');
  assert.equal(element('following').hidden, true);
  click('retry-advice');
  const half = requests.at(-1);
  assert.equal(half.body.intent, 'half-body');
  const halfSample = await sample('crop-half');
  assert.equal(half.signal.aborted, true);
  await finish(half, { decision: 'keep' });
  await finish(halfSample, { decision: 'refine', box: [0.25, 0.25, 0.75, 0.75] });
  click('following');
  assert.deepEqual([canvas.width, canvas.height], [120, 180]);

  const scene = await sample('scene-stairs');
  assert.equal(scene.body.task, 'scene-pose');
  await finish(scene, secondPose);
  assert.equal(element('following').hidden, true);
  assert.equal(element('generated-preview').hidden, false);
  assert.equal(element('alternate-pose').hidden, false);
  click('alternate-pose');
  const sceneAlternative = requests.at(-1);
  assert.deepEqual(sceneAlternative.body, { ...scene.body, previousSuggestion: secondPose.suggestion });
  await finish(sceneAlternative, { landmarks: nextLandmarks, suggestion: newSuggestion });
  click('generate-preview');
  const sceneRender = requests.at(-1);
  assert.equal(sceneRender.body.previewMode, 'scene-pose');
  assert.deepEqual(sceneRender.body.landmarks, nextLandmarks);
  assert.equal(Object.hasOwn(sceneRender.body, 'poseGuide'), false);

  const joint = await sample('joint-orchard');
  assert.equal(sceneRender.signal.aborted, true);
  assert.equal(joint.body.task, 'joint');
  const jointBox = [0.25, 0.125, 0.75, 0.875];
  await finish(joint, { ...secondPose, box: jointBox });
  assert.equal(element('preview')['aria-label'], '叠加建议裁切框和姿势骨架的照片');
  assert.deepEqual(rects.at(-1), [60, 45, 120, 270]);
  assert.deepEqual(canvas.points[0], [144, 72]);
  assert.equal(element('generated-preview').hidden, false);
  assert.equal(element('alternate-pose').hidden, false);
  click('alternate-pose');
  const jointAlternative = requests.at(-1);
  assert.deepEqual(jointAlternative.body, { ...joint.body, previousSuggestion: secondPose.suggestion });
  await finish(jointAlternative, { ...secondPose, box: [0, 0.1, 0.9, 0.9] });
  assert.match(element('status').textContent, /与上一种很接近/);
  assert.deepEqual(rects.at(-1), [60, 45, 120, 270]);
  assert.deepEqual(canvas.points[0], [144, 72]);
  click('original');
  assert.equal(element('preview')['aria-label'], '原片');
  assert.deepEqual([canvas.width, canvas.height], [240, 360]);
  click('pose');
  click('generate-preview');
  const jointRender = requests.at(-1);
  assert.equal(jointRender.body.previewMode, 'joint');
  assert.equal(Object.hasOwn(jointRender.body, 'poseGuide'), false);
  assert.deepEqual(encodedPhotos.at(-1).points, []);
  assert.ok(Math.abs(jointRender.body.landmarks[0].x - 0.7) < 1e-8);
  assert.ok(Math.abs(jointRender.body.landmarks[0].y - 0.1) < 1e-8);
  assert.deepEqual(draws.findLast(draw => draw.coordinates.length === 8).coordinates, [60, 45, 120, 270, 0, 0, 120, 270]);
  await finish(sceneRender, generated);
  assert.equal(element('preview-result').hidden, true);
  await finish(jointRender, generated);
  assert.equal(element('preview-result').hidden, false);

  change('mode', 'pose');
  const instruction = '让人物侧身，手自然垂下。';
  element('pose-instruction').value = '  ' + instruction + '  ';
  await choose('with-direction.jpg');
  const plan = requests.at(-1);
  assert.deepEqual(plan.body, { ...requests[0].body, instruction });
  const thirdSuggestion = '把右手轻搭在膝盖上。';
  const thirdLandmarks = landmarks.map((point, index) => index === 9 ? { ...point, x: 0.5 } : point);
  const choices = [secondPose.suggestion, newSuggestion, thirdSuggestion];
  await finish(plan, { ...secondPose, suggestions: [...choices, '往右跨一步。'] });
  const chooser = element('pose-choices');
  const [first, second, third] = chooser.children;
  const thumbnail = (button) => button.children[0].children[0];
  const feedback = (button) => button.children[0].children[1].textContent;
  const selectChoice = (button) => button.dispatchEvent(new Event('click'));
  const secondGrounding = requests.at(-1);
  assert.equal(chooser.hidden, false);
  assert.deepEqual(chooser.children.map(button => button.children[1].textContent), choices);
  assert.equal(first['aria-pressed'], 'true');
  assert.equal(thumbnail(first).points.length, 16);
  assert.equal(thumbnail(second).hidden, true);
  assert.match(feedback(second), /正在生成/);
  assert.match(feedback(third), /等待/);
  assert.deepEqual(secondGrounding.body, {
    task: 'pose', instruction, selectedSuggestion: newSuggestion, mime: 'image/jpeg', image: '/9j/',
  });
  element('pose-instruction').value = '还没应用的要求';
  selectChoice(third);
  selectChoice(third);
  assert.equal(requests.at(-1), secondGrounding);
  assert.equal(first['aria-pressed'], 'true');
  await finish(secondGrounding, { landmarks, suggestion: newSuggestion });
  const thirdGrounding = requests.at(-1);
  assert.deepEqual(thirdGrounding.body, { ...secondGrounding.body, selectedSuggestion: thirdSuggestion });
  assert.match(feedback(second), /点击重试/);
  assert.equal(thumbnail(second).hidden, true);
  selectChoice(first);
  await finish(thirdGrounding, { landmarks: thirdLandmarks, suggestion: thirdSuggestion });
  assert.equal(thumbnail(third).hidden, false);
  assert.equal(first['aria-pressed'], 'true');
  selectChoice(third);
  assert.equal(third['aria-pressed'], 'true');
  assert.equal(requests.at(-1), thirdGrounding);
  assert.equal(element('suggestion').textContent, thirdSuggestion);
  click('generate-preview');
  const thirdPreview = requests.at(-1);
  assert.equal(thirdPreview.body.suggestion, thirdSuggestion);
  assert.equal(thirdPreview.body.instruction, instruction);
  assert.equal(thirdPreview.body.targetPoseGuide, 'R1VJREU=');
  assert.deepEqual(thirdPreview.body.landmarks, thirdLandmarks);

  selectChoice(second);
  const failedRetry = requests.at(-1);
  assert.equal(thirdPreview.signal.aborted, true);
  assert.deepEqual(failedRetry.body, { ...secondGrounding.body, retryGrounding: true });
  await finish(failedRetry, { landmarks: nextLandmarks, suggestion: thirdSuggestion });
  assert.equal(third['aria-pressed'], 'true');
  assert.equal(element('suggestion').textContent, thirdSuggestion);
  assert.match(element('status').textContent, /未返回所选姿势/);
  assert.match(feedback(second), /点击重试/);
  selectChoice(second);
  await finish(requests.at(-1), { landmarks: nextLandmarks, suggestion: newSuggestion });
  assert.equal(second['aria-pressed'], 'true');
  assert.equal(element('suggestion').textContent, newSuggestion);
  assert.equal(thumbnail(second).hidden, false);
  assert.notDeepEqual(thumbnail(second).points, thumbnail(first).points);

  await choose('single-plan.jpg');
  const singlePlan = requests.at(-1);
  const draft = '还没应用的要求';
  await finish(singlePlan, { ...secondPose, suggestions: [secondPose.suggestion] });
  const [singleFirst, missingSecond, missingThird] = chooser.children;
  const fillSecond = requests.at(-1);
  assert.equal(chooser.hidden, false);
  assert.equal(singleFirst['aria-pressed'], 'true');
  assert.equal(element('generated-preview').hidden, false);
  assert.equal(missingSecond.children[1].textContent, '姿势 2');
  assert.equal(thumbnail(missingSecond).hidden, true);
  assert.deepEqual(fillSecond.body, {
    task: 'pose', instruction: draft, previousSuggestion: secondPose.suggestion,
    mime: 'image/jpeg', image: '/9j/',
  });
  await finish(fillSecond, { landmarks: nextLandmarks, suggestion: newSuggestion });
  assert.equal(missingSecond.children[1].textContent, newSuggestion);
  assert.equal(thumbnail(missingSecond).hidden, false);
  const fillThird = requests.at(-1);
  assert.deepEqual(fillThird.body, { ...fillSecond.body, previousSuggestion: newSuggestion });
  selectChoice(missingThird);
  assert.equal(requests.at(-1), fillThird);

  window.dispatchEvent(new Event('pagehide'));
  assert.equal(fillThird.signal.aborted, true);
  assert.equal(missingSecond.disabled, true);
  window.dispatchEvent(Object.assign(new Event('pageshow'), { persisted: true }));
  const resumedThird = requests.at(-1);
  assert.notEqual(resumedThird, fillThird);
  assert.deepEqual(resumedThird.body, fillThird.body);
  assert.equal(missingSecond.disabled, false);
  await finish(fillThird, { landmarks: thirdLandmarks, suggestion: thirdSuggestion });
  assert.equal(thumbnail(missingThird).hidden, true);
  await finish(resumedThird, { landmarks: thirdLandmarks, suggestion: '把手抬到胸前！' });
  assert.equal(thumbnail(missingThird).hidden, true);
  assert.match(feedback(missingThird), /点击重试/);
  assert.equal(singleFirst['aria-pressed'], 'true');
  selectChoice(missingThird);
  const fillRetry = requests.at(-1);
  assert.deepEqual(fillRetry.body, fillThird.body);
  await finish(fillRetry, { landmarks: thirdLandmarks, suggestion: thirdSuggestion });
  assert.equal(missingThird['aria-pressed'], 'true');
  assert.equal(thumbnail(missingThird).hidden, false);
  assert.equal(missingThird.children[1].textContent, thirdSuggestion);

  element('pose-direction').dispatchEvent(new Event('submit', { cancelable: true }));
  const revisedPlan = requests.at(-1);
  assert.equal(revisedPlan.body.instruction, '还没应用的要求');
  assert.equal(chooser.hidden, true);
  await finish(revisedPlan, { ...secondPose, suggestions: choices });
  const outdatedChoice = requests.at(-1);
  change('mode', 'composition');
  assert.equal(element('pose-direction').hidden, true);
  assert.equal(outdatedChoice.signal.aborted, true);
  await finish(outdatedChoice, { landmarks: nextLandmarks, suggestion: newSuggestion });
  await finish(thirdPreview, generated);
  assert.equal(chooser.hidden, true);
  assert.equal(element('preview-result').hidden, true);
  assert.equal(requests.at(-1), outdatedChoice);
});
