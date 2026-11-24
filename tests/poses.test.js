import test from 'node:test';
import assert from 'node:assert/strict';
import { poses, guidance, validatePose, placeReference } from '../src/poses.js';
import { posesDiffer } from '../src/photo-advice.js';

test('pose choices distinguish a distant wrist movement from reframing the same pose', () => {
  const small = Array.from({ length: 17 }, (_, i) => ({
    x: 0.4 + (i % 2) * 0.015, y: 0.4 + i * 0.008, visibility: 1,
  }));
  const reframed = small.map(p => ({ ...p, x: 1.2 * p.x + 0.04, y: 1.2 * p.y - 0.02 }));
  assert.equal(posesDiffer(small, reframed, 1.5), false);
  const raisedWrist = small.map((p, i) => i === 9 ? { ...p, y: p.y - 0.03 } : p);
  assert.equal(posesDiffer(small, raisedWrist, 1.5), true);
});
test('演示姿势覆盖三种场景且数据有效',()=>{for(const scene of ['park','street','indoor'])assert.ok(poses.filter(p=>p.scene===scene).length>=5);assert.ok(poses.every(validatePose));});
test('缺少关键点时要求重新入镜',()=>assert.match(guidance([],poses[0].points),/全身/));
test('与参考一致时保持姿势',()=>assert.equal(guidance(poses[0].points,poses[0].points),'很好，保持这个姿势'));
test('拒绝不完整的素材',()=>assert.equal(validatePose({name:'x',scene:'park',points:[]}),false));

test('无人像时姿势放在默认画面位置',()=>{const result=placeReference(poses[0].points);assert.ok(result[11].x < .5);assert.ok(result[27].y < 1);});
test('检测到人物时推荐骨架对齐人物中心',()=>{const shifted=poses[0].points.map(p=>({...p,x:p.x+.15}));const result=placeReference(poses[0].points,shifted);assert.ok(Math.abs((result[11].x+result[12].x)/2-.65)<.001);});

test('人物靠近画面边缘时肩部锚点不被强制拉回中心',()=>{const person=poses[0].points.map(p=>({...p,x:p.x+.3}));const result=placeReference(poses[0].points,person);assert.ok(Math.abs((result[11].x+result[12].x)/2-.8)<.001);});
test('脚部遮挡时仍按照可见躯干对齐',()=>{const person=poses[0].points.map(p=>({...p,x:p.x+.1,y:p.y+.08}));person[27].visibility=person[28].visibility=0;const result=placeReference(poses[0].points,person);assert.ok(Math.abs((result[11].x+result[12].x)/2-.6)<.001);assert.ok(Math.abs((result[11].y+result[12].y)/2-.36)<.001);});
test('人物缩小时肩部和躯干按检测结果缩放',()=>{const person=poses[0].points.map(p=>({...p,x:.7+(p.x-.5)*.4,y:.25+(p.y-.28)*.4}));const result=placeReference(poses[0].points,person);assert.ok(Math.abs(result[11].x-person[11].x)<.001);assert.ok(Math.abs(result[23].y-person[23].y)<.001);});

test('山景参考素材含图片与模型关键点',()=>{const mountain=poses.find(p=>p.id==='mountain-sunrise');assert.equal(mountain.scene,'mountain');assert.ok(mountain.image);assert.equal(mountain.points.length,33);assert.ok(validatePose(mountain));});

test('横图参考叠加竖图时保留像素宽高比例并对齐肩部',()=>{const source=poses[0].points,person=source.map(p=>({...p,x:.5+(p.x-.5)*.4,y:.3+(p.y-.28)*.4}));const sourceAspect=1.5,targetAspect=.6,result=placeReference(source,person,null,sourceAspect,targetAspect);const sourceDx=(source[12].x-source[11].x)*sourceAspect,sourceDy=source[15].y-source[11].y;const targetDx=(result[12].x-result[11].x)*targetAspect,targetDy=result[15].y-result[11].y;assert.ok(Math.abs(targetDx/targetDy-sourceDx/sourceDy)<1e-9);assert.ok(Math.abs((result[11].x+result[12].x)/2-(person[11].x+person[12].x)/2)<1e-9);});
test('没有人物时横图参考也按像素比例放置',()=>{const p=poses[0].points,r=placeReference(p,null,{x:.5},1.5,.6);assert.ok(Math.abs(((r[12].x-r[11].x)*.6)/(r[15].y-r[11].y)-((p[12].x-p[11].x)*1.5)/(p[15].y-p[11].y))<1e-9);});
