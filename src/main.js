import './style.css';
import { isLocalStudio, requestPhotoAdvice } from './photo-advice.js';
import { poses, guidance, validatePose, placeReference } from './poses.js';
import { drawSkeleton, drawPoseThumbnail } from './draw.js';
import { analyzePixels, rankPoses } from './scene-analysis.js';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as mobilenet from '@tensorflow-models/mobilenet';
import '@tensorflow/tfjs';

const app=document.querySelector('#app');
app.innerHTML=`<main class="shell"><header><div><span class="eyebrow">POSE STUDIO · 姿势灵感</span><h1>记录自然的你<span>✳</span></h1><p>让眼前的场景，给你一点拍照灵感。</p></div><button id="switch" class="icon-button" aria-label="切换相机">⇄</button></header><section class="camera"><video id="video" autoplay playsinline muted></video><img id="uploaded" alt="上传的照片预览" hidden><canvas id="overlay"></canvas><div class="camera-empty" id="empty"><div class="focus-icon">⌗</div><strong>准备好拍一张了吗？</strong><span>开启相机，寻找适合此刻的姿势</span><button id="start" class="primary">开启相机</button><button id="upload-start" class="secondary">上传照片</button></div><div class="camera-top"><span class="live" id="source-label">● 实时取景</span><span id="model-status">等待相机</span><button id="upload" class="camera-upload">上传照片</button></div><div id="hint" class="hint">选择一个姿势，开始引导</div></section><section class="controls"><div class="section-heading"><div><span class="eyebrow">FIND YOUR POSE</span><h2>选择拍摄场景</h2></div><span id="scene-status" class="status">手动选择</span></div><div class="chips" id="chips"><button data-scene="park">🌿 公园</button><button data-scene="street">◧ 街道</button><button data-scene="indoor">⌂ 室内</button><button data-scene="mountain">⛰ 山景</button></div><div class="section-heading poses-heading"><div><span class="eyebrow">POSE IDEAS</span><h2>为你推荐的姿势</h2></div><div><button id="more" class="text-button">换一组</button><button id="import" class="text-button">＋ 导入素材</button></div></div><div class="pose-list" id="pose-list"></div><p class="privacy">✳ 画面分析在本机进行 · 构图位置仅供参考 · <a href="./shuttermuse.html" style="color:inherit">ShutterMuse 拍照建议</a></p></section><div class="bottom"><button id="capture" class="shutter" aria-label="拍照"><span></span></button><span>${isLocalStudio ? '拍照后在本机获取建议' : '轻触拍摄'}</span></div></main><dialog id="preview"><div class="dialog-head"><strong>拍摄预览</strong><button id="close" class="icon-button" aria-label="关闭">×</button></div><img id="photo" alt="拍摄照片预览"><p id="photo-advice" role="status" hidden></p><button type="button" id="retry-advice" class="text-button" hidden>重试建议</button><a id="save" class="primary" download="pose-photo.png">保存照片</a></dialog><input type="file" id="scene-file" accept="image/*" hidden><input type="file" id="json-file" accept="application/json" hidden><input type="file" id="image-file" accept="image/*" hidden>`;
const $=s=>document.querySelector(s), video=$('#video'), canvas=$('#overlay'), ctx=canvas.getContext('2d');
let library=[...poses], scene=null, selected=poses[0], stream=null, facing='environment', landmarker=null, classifier=null, frame=0, lastScene=0, sceneLocked=false, imageUrl=null, uploadedUrl=null, stillLandmarker=null, stillPoints=null, inputMode='camera', analysis=null, lastLayout=0, poseLocked=false, page=0;
const labels={park:'公园',street:'街道',indoor:'室内',mountain:'山景'};
/** 场景不确定时展示通用姿势；已选场景则选择该场景的首个姿势。 */
function defaultPose(){return rankPoses(library,scene,analysis)[0] || library[0];}
/** 缩小画面抽样，避免逐帧进行高成本图像计算。 */
function analyzeFrame(source){const sample=document.createElement('canvas');sample.width=96;sample.height=96;const c=sample.getContext('2d',{willReadFrequently:true});c.drawImage(source,0,0,96,96);return analyzePixels(c.getImageData(0,0,96,96).data,96,96);}
/** 网络不可用时及时降级，避免模型加载无限等待。 */
function withTimeout(promise,ms=12000){let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('模型加载超时')),ms);})]).finally(()=>clearTimeout(timer));}
/** 刷新场景按钮与推荐卡片。 */
function render() {
 document.querySelectorAll('[data-scene]').forEach(b=>b.classList.toggle('active',b.dataset.scene===scene));
 const ranked=rankPoses(library,scene);if(page>=ranked.length)page=0;const items=ranked.slice(page,page+3);$('#more').hidden=ranked.length<=3;
 $('#pose-list').replaceChildren(...items.map(p=>{const b=document.createElement('button');b.className='pose-card'+(selected?.id===p.id?' selected':'');const thumb=document.createElement('canvas');thumb.width=150;thumb.height=180;const c=thumb.getContext('2d');c.fillStyle='#e9eee8';c.fillRect(0,0,150,180);drawPoseThumbnail(c,p.points,150,180,p.imageAspect ?? 1);const name=document.createElement('strong');name.textContent=p.name;const sub=document.createElement('small');sub.textContent=p.description; b.append(thumb,name,sub);b.onclick=()=>{poseLocked=true;selected=p;$('#hint').textContent=p.description;render();if(inputMode==='photo')drawStill();};return b;}));
}
/** 启动设备相机并处理浏览器拒绝与设备错误。 */
async function startCamera() {
 try{if(!navigator.mediaDevices?.getUserMedia)throw new Error('相机需要 HTTPS 或 localhost 安全环境');const next=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing}},audio:false});stream?.getTracks().forEach(t=>t.stop());stream=next;inputMode='camera', analysis=null, lastLayout=0, poseLocked=false, page=0;$('#source-label').textContent='● 实时取景';$('#uploaded').hidden=true;video.hidden=false;video.srcObject=stream;await video.play();$('#empty').hidden=true;$('#model-status').textContent='正在加载识别模型';if(!frame)frame=requestAnimationFrame(loop);initModels();}catch(e){$('#empty').hidden=false;$('#empty strong').textContent='无法使用相机';$('#empty span').textContent=`${e.message || '请检查相机权限和设备'}，允许访问后重试`;$('#start').textContent='重试';}
}
/** 加载姿态及图像分类模型；模型失败时允许手动选择场景。 */
async function initModels(){
 const poseTask=(async()=>{const vision=await withTimeout(FilesetResolver.forVisionTasks('/wasm'));return withTimeout(PoseLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/pose_landmarker_lite.task'},runningMode:'VIDEO',numPoses:1}));})();
 poseTask.then(model=>{landmarker=model;$('#model-status').textContent='骨架引导已就绪';}).catch(()=>{$('#model-status').textContent='骨架模型不可用';});
 withTimeout(mobilenet.load()).then(model=>{classifier=model;}).catch(()=>{$('#scene-status').textContent='场景不确定，可手动选择';});
}
/** 将通用分类器的部分标签转换为保守的场景建议。 */
function classifyScene(predictions){const matches=[['mountain',/alp|mountain|cliff|valley|volcano|peak/i],['park',/park|garden|forest|tree|lawn/i],['street',/street|road|traffic|crosswalk|car|bus/i],['indoor',/room|studio|office|restaurant|sofa|desk|bookcase/i]];for(const pred of predictions){if(pred.probability<.25)continue;for(const [scene,pattern] of matches)if(pattern.test(pred.className))return scene;}return null;}
/** 按视频原始比例绘制骨架并周期性抽帧分类。 */
async function loop(now){frame=requestAnimationFrame(loop);if(inputMode!=='camera'||!video.videoWidth)return;canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.clearRect(0,0,canvas.width,canvas.height);const mirror=facing==='user';if(now-lastLayout>2000){lastLayout=now;try{analysis=analyzeFrame(video);if(mirror&&analysis){analysis.side=analysis.side==='left'?'right':'left';analysis.x=1-analysis.x;}if(!poseLocked){selected=defaultPose();page=0;}render();}catch{analysis=null;}}try{if(landmarker){const result=landmarker.detectForVideo(video,now);const points=result.landmarks?.[0];if(selected){drawSkeleton(ctx,placeReference(selected.points,mirror&&points?points.map(p=>({...p,x:1-p.x})):points,analysis,selected.imageAspect ?? canvas.width/canvas.height,canvas.width/canvas.height),canvas.width,canvas.height,'rgba(195,255,125,.9)',false);if(points)drawSkeleton(ctx,points,canvas.width,canvas.height,'#d8ff8c',mirror);$('#hint').textContent=guidance(mirror && points ? points.map(p=>({...p,x:1-p.x})) : points,selected.points,mirror);}}}catch(e){$('#model-status').textContent='骨架识别暂不可用';}
 if(classifier&&!sceneLocked&&now-lastScene>3000){lastScene=now;try{const found=classifyScene(await classifier.classify(video,5));if(found){scene=found;if(!poseLocked)selected=defaultPose();$('#scene-status').textContent=`识别建议：${labels[found]}`;render();}else{$('#scene-status').textContent='识别不确定，请选择场景';}}catch{$('#scene-status').textContent='场景不确定，可手动选择';}}
}
/** 把参考姿势叠加在建议位置，并在检测到人物时绘制当前骨架。 */
function drawStill(){
 const img=$('#uploaded');if(!img.naturalWidth)return;
 canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.clearRect(0,0,canvas.width,canvas.height);
 if(!selected)return;
 drawSkeleton(ctx,placeReference(selected.points,stillPoints,analysis,selected.imageAspect ?? canvas.width/canvas.height,canvas.width/canvas.height),canvas.width,canvas.height,'rgba(197,255,125,.95)');
 if(stillPoints){drawSkeleton(ctx,stillPoints,canvas.width,canvas.height,'#ffffff');const aligned=[11,12,23,24].every(i=>(stillPoints[i]?.visibility??0)>=.45);const text=guidance(stillPoints,selected.points);$('#hint').textContent=aligned?(text==='请后退一点，让全身进入画面'?'照片未显示完整身体，可参考上半身姿势':text):'人物关键点不足，亮色骨架为构图参考';}
 else $('#hint').textContent=analysis?`${analysis.edgeDifference<2?'构图参考':analysis.side==='left'?'左侧纹理较少':'右侧纹理较少'} · ${selected.name}`:`推荐姿势：${selected.name}`;
}
/** 在本机加载图片，使用轻量模型识别场景及人体关键点。 */
async function loadScenePhoto(file){
 if(!file?.type.startsWith('image/')){ $('#hint').textContent='请选择图片文件';return; }
 const img=$('#uploaded'),url=URL.createObjectURL(file);
 try{await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('图片无法读取'));img.src=url;});
  if(uploadedUrl)URL.revokeObjectURL(uploadedUrl);uploadedUrl=url;stream?.getTracks().forEach(t=>t.stop());stream=null;inputMode='photo';$('#source-label').textContent='照片分析';stillPoints=null;sceneLocked=false;video.hidden=true;img.hidden=false;$('#empty').hidden=true;$('#model-status').textContent='分析照片中';$('#scene-status').textContent='分析场景中';scene=null;poseLocked=false;page=0;analysis=analyzeFrame(img);selected=defaultPose();render();drawStill();
  if (isLocalStudio) showPhotoPreview(img);
  const photoUrl=url;
  const poseTask=(async()=>{
    if(!stillLandmarker){const vision=await withTimeout(FilesetResolver.forVisionTasks('/wasm'));stillLandmarker=await withTimeout(PoseLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/pose_landmarker_lite.task'},runningMode:'IMAGE',numPoses:1}));}
    return stillLandmarker.detect(img).landmarks?.[0]||null;
  })();
  poseTask.then(points=>{if(uploadedUrl!==photoUrl||inputMode!=='photo')return;stillPoints=points;$('#model-status').textContent=points?'已检测人物，正在对齐':'未检测到人物';drawStill();}).catch(()=>{if(uploadedUrl!==photoUrl||inputMode!=='photo')return;$('#model-status').textContent='骨架识别不可用';drawStill();});
  const sceneTask=classifier?Promise.resolve(classifier):withTimeout(mobilenet.load());
  sceneTask.then(async model=>{classifier=model;const found=classifyScene(await model.classify(img,5));if(uploadedUrl!==photoUrl||inputMode!=='photo'||sceneLocked)return;scene=found;if(!poseLocked)selected=defaultPose();$('#scene-status').textContent=found?`识别建议：${labels[found]}`:'通用推荐 · 可手动选场景';render();drawStill();}).catch(()=>{if(uploadedUrl!==photoUrl||inputMode!=='photo'||sceneLocked)return;$('#scene-status').textContent='场景识别不可用，可手动选择';});

 }catch(e){if(uploadedUrl!==url)URL.revokeObjectURL(url);$('#model-status').textContent='识别暂不可用';$('#scene-status').textContent='通用推荐 · 可手动选场景';drawStill();}
}
$('#more').onclick=()=>{const ranked=rankPoses(library,scene);poseLocked=true;page=page+3>=ranked.length?0:page+3;render();};
$('#upload-start').onclick=$('#upload').onclick=()=>$('#scene-file').click();
$('#scene-file').onchange=e=>{const file=e.target.files?.[0];if(file)loadScenePhoto(file);e.target.value='';};
$('#start').onclick=startCamera;$('#switch').onclick=()=>{facing=facing==='user'?'environment':'user';video.classList.toggle('mirror',facing==='user');if(stream || inputMode==='photo')startCamera();};
$('#chips').onclick=e=>{const b=e.target.closest('[data-scene]');if(!b)return;scene=b.dataset.scene;sceneLocked=true;poseLocked=false;page=0;selected=defaultPose();$('#scene-status').textContent='已手动选择';$('#hint').textContent=`推荐姿势：${selected.name}`;render();if(inputMode==='photo')drawStill();};
const preview = $('#preview');
const adviceStatus = $('#photo-advice');
const retryAdvice = $('#retry-advice');
let adviceRequest;
let capturedPhoto;

async function analyzeCapture() {
  adviceRequest?.abort();
  const request = new AbortController();
  adviceRequest = request;
  retryAdvice.hidden = true;
  adviceStatus.hidden = false;
  adviceStatus.textContent = '正在本机看这张照片…';
  try {
    const suggestion = await requestPhotoAdvice(capturedPhoto, request.signal);
    if (!request.signal.aborted) adviceStatus.textContent = suggestion;
  } catch {
    if (request.signal.aborted) return;
    adviceStatus.textContent = '暂时无法获取拍摄建议，照片仍可保存。';
    retryAdvice.hidden = false;
  }
}

function showPhotoPreview(source, mirror = false) {
  const out = document.createElement('canvas');
  out.width = source.videoWidth || source.naturalWidth;
  out.height = source.videoHeight || source.naturalHeight;
  if (!out.width || !out.height) return;
  const context = out.getContext('2d');
  if (mirror) {
    context.translate(out.width, 0);
    context.scale(-1, 1);
  }
  context.drawImage(source, 0, 0);
  const url = out.toDataURL('image/png');
  $('#photo').src = url;
  $('#save').href = url;
  if (!preview.open) preview.showModal();
  if (isLocalStudio) {
    capturedPhoto = out;
    void analyzeCapture();
  }
}

$('#capture').onclick = () => {
  if (inputMode === 'photo') return showPhotoPreview($('#uploaded'));
  if (video.videoWidth) return showPhotoPreview(video, facing === 'user');
  $('#hint').textContent = '请先开启相机';
};
retryAdvice.onclick = analyzeCapture;
$('#close').onclick = () => preview.close();
preview.addEventListener('close', () => {
  if (preview.open) return;
  adviceRequest?.abort();
  capturedPhoto = null;
  retryAdvice.hidden = adviceStatus.hidden = true;
});
window.addEventListener('pagehide', () => adviceRequest?.abort());
let pending=null;$('#import').onclick=()=>$('#json-file').click();$('#json-file').onchange=async e=>{try{pending=JSON.parse(await e.target.files[0].text());if(!validatePose(pending))throw Error('JSON 缺少名称、场景或 33 个有效关键点');$('#image-file').click();}catch(err){alert(`导入失败：${err.message}`);}e.target.value='';};
/** 导入图片仅用于读取宽高比；推荐卡片只绘制关键点。 */
$('#image-file').onchange=e=>{const file=e.target.files[0];if(!file||!pending)return;const imported={...pending,id:`import-${Date.now()}`,source:'用户导入 · 授权待自行确认'};const url=URL.createObjectURL(file);const image=new Image();image.onload=()=>{imported.imageAspect=image.naturalWidth/image.naturalHeight;URL.revokeObjectURL(url);render();if(selected?.id===imported.id&&inputMode==='photo')drawStill();};image.onerror=()=>{URL.revokeObjectURL(url);$('#scene-status').textContent='导入图片无法读取，请重试';};image.src=url;library.unshift(imported);scene=imported.scene;sceneLocked=true;page=0;selected=imported;$('#scene-status').textContent='已导入素材';pending=null;render();e.target.value='';};
window.addEventListener('pagehide',()=>stream?.getTracks().forEach(t=>t.stop()));render();


