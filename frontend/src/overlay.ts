import { reliable, type PoseFrame } from './motion';

export const COLORS = { visible:'#a4f27d', uncertain:'#9a9f9d', difference:'#ff645d' };
export const segmentKey = (a:string,b:string) => [a,b].sort().join(':');

/** Positions stay in image coordinates so labels follow the same fit as the video. */
export function annotations(frame:PoseFrame|undefined,names:string[]) {
  const marked=highlights(frame,names);
  return (frame?.comparison?.alerts??[]).flatMap(alert=>{
    if(alert.kind!=='reference_difference'||!Number.isFinite(alert.delta_deg)||!alert.joints.length||!alert.joints.every(name=>marked.joints.has(name)))return [];
    const points=alert.joints.map(name=>frame!.landmarks[names.indexOf(name)]!);
    const knee=alert.joints.find(name=>name.endsWith('_knee'));
    const anchor=knee?frame!.landmarks[names.indexOf(knee)]!: {
      x:points.reduce((sum,p)=>sum+p.x,0)/points.length,
      y:points.reduce((sum,p)=>sum+p.y,0)/points.length,
    };
    const label=alert.metric==='left_knee_flexion_deg'?'Rodilla izq.':alert.metric==='right_knee_flexion_deg'?'Rodilla der.':'Tronco';
    return [{x:anchor.x,y:anchor.y,text:`${label}: ${Math.abs(alert.delta_deg).toFixed(0)}° de diferencia`}];
  });
}

/** A low-confidence joint invalidates its entire alert, including connected bones. */
export function highlights(frame:PoseFrame|undefined,names:string[]) {
  const joints=new Set<string>(),segments=new Set<string>();
  const ids=new Map(names.map((name,i)=>[name,i]));
  if(frame?.comparison?.status==='differences'){
    for(const alert of frame.comparison.alerts){
      if(alert.kind!=='reference_difference'||!alert.joints.length||!alert.joints.every(name=>reliable(frame.landmarks[ids.get(name)??-1])))continue;
      alert.joints.forEach(name=>joints.add(name));
      alert.segments.forEach(([a,b])=>segments.add(segmentKey(a,b)));
    }
  }
  return {joints,segments};
}
