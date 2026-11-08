// Synthetic alerts verify rendering decisions, not pose accuracy.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import ts from 'typescript';
const root=fileURLToPath(new URL('../',import.meta.url));
const output=path.resolve(root,'../data/overlay-check');
await mkdir(output,{recursive:true});
for(const name of ['motion','overlay']){
  let code=ts.transpileModule(await readFile(path.join(root,`src/${name}.ts`),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
  code=code.replaceAll("'./motion'","'./motion.mjs'");
  await writeFile(path.join(output,`${name}.mjs`),code);
}
const {annotations,highlights,segmentKey}=await import(pathToFileURL(path.join(output,'overlay.mjs')).href);
const names=['left_hip','left_knee','left_ankle'];
const alert={kind:'reference_difference',metric:'left_knee_flexion_deg',delta_deg:-22,joints:names,segments:[['left_hip','left_knee'],['left_knee','left_ankle']]};
const frame={landmarks:names.map((_,i)=>({x:.4,y:.2+i*.25,visibility:1,presence:1})),comparison:{status:'differences',alerts:[alert]}};
assert.equal(highlights(frame,names).segments.has(segmentKey('left_knee','left_hip')),true);
assert.deepEqual(annotations(frame,names),[{x:.4,y:.45,text:'Rodilla izq.: 22° de diferencia'}]);
frame.landmarks[1].visibility=.2;
assert.equal(highlights(frame,names).joints.size,0);
assert.deepEqual(annotations(frame,names),[]);
frame.landmarks[1].visibility=1;frame.landmarks[1].x=NaN;
assert.deepEqual(annotations(frame,names),[]);
frame.landmarks[1].x=.4;frame.comparison.status='not_evaluable';
assert.deepEqual(annotations(frame,names),[]);
frame.comparison.status='differences';frame.comparison.alerts=[{...alert,joints:['missing_joint']}];
assert.deepEqual(annotations(frame,names),[]);
assert.deepEqual(annotations(undefined,names),[]);
console.log('Overlay: red segments, joint-anchored labels, low-confidence/invalid/missing-data suppression passed.');
