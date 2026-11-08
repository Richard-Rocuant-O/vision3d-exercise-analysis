// Server-render the actual component to verify labels and missing-data states.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const frontend=fileURLToPath(new URL('../',import.meta.url));
const output=path.resolve(frontend,'../data/feedback-check');
await mkdir(output,{recursive:true});
let code=ts.transpileModule(await readFile(path.join(frontend,'src/components/ExerciseFeedback.tsx'),'utf8'),{
  compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX},
}).outputText;
code=code.replaceAll('"react/jsx-runtime"',JSON.stringify(import.meta.resolve('react/jsx-runtime')));
const compiled=path.join(output,'ExerciseFeedback.mjs');
await writeFile(compiled,code);
const {default:Feedback}=await import(pathToFileURL(compiled).href);
const motion=JSON.parse(await readFile(process.argv[2]||path.join(output,'video-pose.json'),'utf8'));
const frame=motion.frames.find(frame=>frame.landmarks?.length);
assert.ok(frame,'Needs a real detected frame');
const render=(current,outside=false)=>renderToStaticMarkup(React.createElement(Feedback,{frame:current,motion:{...motion,exercise_summary:undefined},time:(current?.timestamp_ms??0)/1000,outside}));
let html=render({...frame,exercise:{status:'identified',label:'Squat',name:'Sentadilla'},technique:{status:'pending_review',reason:'reference_not_reviewed'}});
assert.ok(html.includes('Sentadilla')&&html.includes('DIFERENCIAS CON EL EJEMPLO'));
assert.ok(!html.includes('Técnica correcta')&&!html.includes('Técnica incorrecta'));
html=render({...frame,comparison:{status:'differences',alerts:[{metric:'trunk_inclination_deg',message:'Tronco: 22° de diferencia con el ejemplo durante bajada.'}]}});
assert.ok(html.includes('1 zona con diferencia')&&html.includes('Tronco: 22°'));
assert.ok(html.includes('feedback-card differences'));
html=render({...frame,exercise:{status:'uncertain',label:null,name:null,candidate_name:'Zancadas'},comparison:{status:'not_evaluable',reason:'exercise_uncertain',alerts:[]}});
assert.ok(html.includes('Ejercicio no detectado'));
assert.ok(!html.includes('Posible')&&!html.includes('Zancadas'));
assert.ok(html.includes('No se compara la técnica porque no se ha identificado el ejercicio.'));
assert.ok(!html.includes('squat_01.mp4'),'An unrelated reference must not look like it was used');
html=render({...frame,measurements:{left_knee_flexion_deg:null,right_knee_flexion_deg:null,trunk_inclination_deg:null}});
assert.ok(!html.includes('NaN')&&!html.includes('undefined'));
assert.equal((html.match(/No evaluable/g)||[]).length>=3,true);
html=render(undefined,true);assert.ok(html.includes('Fuera del tramo analizado'));
html=render({...frame,measurements:{left_knee_flexion_deg:{value:0,visibility:1},right_knee_flexion_deg:null,trunk_inclination_deg:null}});
assert.ok(html.includes('0°'),'A real zero angle must not be treated as missing');
html=renderToStaticMarkup(React.createElement(Feedback,{frame:{...frame,exercise:{status:'identified',label:'Lunges',name:'Zancadas'}},motion:{...motion,exercise_summary:{status:'identified',label:'Squat',name:'Sentadilla'}},time:3,outside:false}));
assert.ok(html.includes('Sentadilla')&&!html.includes('>Zancadas<'),'Whole-segment identity must not flicker with a common posture');
await writeFile(path.join(output,'panel.html'),render(frame));
console.log(JSON.stringify({checks:'recognized exercise, unknown exercise without candidate suggestions, difference alerts, missing angles, zero angle, outside analyzed segment',render:'React server render; no browser screenshot'}));
