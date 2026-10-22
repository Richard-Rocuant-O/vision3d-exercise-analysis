import { useEffect, useRef } from 'react';
import { CONNECTIONS, reliable, type Motion, type PoseFrame } from '../motion';

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
      ctx.lineWidth = 2.5;
      for (const [a, b] of CONNECTIONS) {
        const p = point(a), q = point(b);
        if (!p || !q) continue;
        ctx.strokeStyle = reliable(p) && reliable(q) ? '#a4f27d' : '#9a9f9d';
        ctx.setLineDash(reliable(p) && reliable(q) ? [] : [4, 5]);
        ctx.beginPath(); ctx.moveTo(ox + p.x * w, oy + p.y * h); ctx.lineTo(ox + q.x * w, oy + q.y * h); ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const p of frame.landmarks) {
        if (!p) continue;
        ctx.fillStyle = reliable(p) ? '#eaffdd' : '#939a96';
        ctx.beginPath(); ctx.arc(ox + p.x * w, oy + p.y * h, 3.5, 0, Math.PI * 2); ctx.fill();
      }
    };
    draw();
    const observer = new ResizeObserver(draw); observer.observe(el);
    return () => observer.disconnect();
  }, [frame, motion, visible]);
  return <canvas className="overlay" ref={canvas} aria-hidden="true" />;
}
