import { useEffect, useRef } from 'react';
import { CONNECTIONS, reliable, type Motion, type PoseFrame } from '../motion';
import { COLORS, annotations, highlights, segmentKey } from '../overlay';

export default function PoseOverlay({ frame, motion, visible }: { frame?: PoseFrame; motion: Motion | null; visible: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const draw = () => {
      const { width, height } = el.getBoundingClientRect();
      const ratio = Math.min(devicePixelRatio, 2);
      el.width = Math.round(width * ratio); el.height = Math.round(height * ratio);
      const ctx = el.getContext('2d');
      if (!ctx || !motion || !frame || !visible) return;
      ctx.scale(ratio, ratio);
      // Match object-fit: contain, including black margins for portrait images.
      const scale = Math.min(width / motion.width, height / motion.height);
      const w = motion.width * scale, h = motion.height * scale;
      const ox = (width - w) / 2, oy = (height - h) / 2;
      const names = new Map(motion.landmark_names.map((n, i) => [n, i]));
      const point = (n: string) => frame.landmarks[names.get(n) ?? -1];
      const marked=highlights(frame,motion.landmark_names);
      for (const [a, b] of CONNECTIONS) {
        const p = point(a), q = point(b);
        if (!p || !q || ![p.x,p.y,q.x,q.y].every(Number.isFinite)) continue;
        const red=reliable(p)&&reliable(q)&&marked.segments.has(segmentKey(a,b));
        ctx.strokeStyle = red ? COLORS.difference : reliable(p) && reliable(q) ? COLORS.visible : COLORS.uncertain;
        ctx.lineWidth=red?4.5:2.5;
        ctx.shadowBlur=red?8:0;ctx.shadowColor=COLORS.difference;
        ctx.setLineDash(reliable(p) && reliable(q) ? [] : [4, 5]);
        ctx.beginPath(); ctx.moveTo(ox + p.x * w, oy + p.y * h); ctx.lineTo(ox + q.x * w, oy + q.y * h); ctx.stroke();
      }
      ctx.setLineDash([]);
      frame.landmarks.forEach((p,i)=>{
        if (!p || ![p.x,p.y].every(Number.isFinite)) return;
        const red=reliable(p)&&marked.joints.has(motion.landmark_names[i]);
        ctx.shadowBlur=red?8:0;ctx.fillStyle = red?COLORS.difference:reliable(p)?'#eaffdd':COLORS.uncertain;
        ctx.beginPath(); ctx.arc(ox + p.x * w, oy + p.y * h, red?5:3.5, 0, Math.PI * 2); ctx.fill();
      });
      // Label the measured difference at its joint, without suggesting a target position.
      ctx.shadowBlur=0;ctx.font='600 12px system-ui';ctx.textBaseline='middle';
      const boxes:{x:number;y:number;w:number;h:number}[]=[];
      for(const label of annotations(frame,motion.landmark_names)){
        const px=ox+label.x*w,py=oy+label.y*h;
        if(px<ox||px>ox+w||py<oy||py>oy+h)continue;
        const boxWidth=Math.min(ctx.measureText(label.text).width+16,Math.max(0,width-8));
        const boxX=Math.max(4,Math.min(px+18,width-boxWidth-4));
        let boxY=Math.max(4,Math.min(py-32,height-30));
        for(let attempt=0;attempt<boxes.length+1;attempt++){
          if(!boxes.some(b=>boxX<b.x+b.w&&boxX+boxWidth>b.x&&boxY<b.y+b.h&&boxY+26>b.y))break;
          boxY=boxY+30<=height-30?boxY+30:Math.max(4,boxY-30);
        }
        ctx.strokeStyle=COLORS.difference;ctx.lineWidth=1.5;
        ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(boxX,boxY+13);ctx.stroke();
        ctx.fillStyle='#251714';ctx.fillRect(boxX,boxY,boxWidth,26);ctx.strokeRect(boxX,boxY,boxWidth,26);
        ctx.fillStyle='#ffb1aa';ctx.fillText(label.text,boxX+8,boxY+13,Math.max(1,boxWidth-16));
        boxes.push({x:boxX,y:boxY,w:boxWidth,h:26});
      }
    };
    draw();
    const observer = new ResizeObserver(draw); observer.observe(el);
    return () => observer.disconnect();
  }, [frame, motion, visible]);
  return <canvas className="overlay" ref={canvas} aria-hidden="true" />;
}
