import { useEffect, useRef, useState } from 'react';
import AvatarView from './components/AvatarView';
import PoseOverlay from './components/PoseOverlay';
import { frameAt, reliable, type Motion } from './motion';

const ACCEPT = '.jpg,.jpeg,.png,.webp,.mp4,.m4v,.mov,image/jpeg,image/png,image/webp,video/mp4';
function mediaKind(file: File): 'image' | 'video' {
  return file.type.startsWith('image/') || /\.(jpe?g|png|webp)$/i.test(file.name) ? 'image' : 'video';
}

export default function App() {
  const video = useRef<HTMLVideoElement>(null);
  const pending = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [motion, setMotion] = useState<Motion | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [time, setTime] = useState(0);
  const [showPoints, setShowPoints] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const kind = file ? mediaKind(file) : 'image';
  const frame = frameAt(motion, time * 1000);
  const visible = frame?.landmarks.filter(reliable).length ?? 0;

  useEffect(() => {
    if (!file) { setUrl(''); return; }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  useEffect(() => () => pending.current?.abort(), []);
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const tick = () => { if (video.current) setTime(video.current.currentTime); id = requestAnimationFrame(tick); };
    tick();
    return () => cancelAnimationFrame(id);
  }, [playing]);
  useEffect(() => { if(video.current) video.current.playbackRate = speed; }, [speed, url]);

  const selectFile = (next: File | null) => {
    pending.current?.abort(); pending.current = null;
    setFile(next); setMotion(null); setError(''); setTime(0); setBusy(false); setPlaying(false);
  };
  const analyze = async () => {
    if (!file) return;
    pending.current?.abort();
    const controller = new AbortController(); pending.current = controller;
    video.current?.pause();
    setBusy(true); setError(''); setMotion(null);
    try {
      const body = new FormData(); body.append(kind === 'image' ? 'image' : 'video', file);
      const response = await fetch(kind === 'image' ? '/api/analyze-image' : '/api/analyze', { method:'POST', body, signal:controller.signal });
      if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('No se pudo conectar con el servicio Python. Comprueba que esté abierto en el puerto 8765.');
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'No se pudo analizar este archivo.');
      if (pending.current !== controller) return;
      if (!Array.isArray(data.frames) || !data.frames.length) throw new Error('El archivo no produjo fotogramas utilizables.');
      setMotion(data);
      const t = video.current?.currentTime ?? 0; setTime(t);
    } catch(e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'No se pudo conectar con el servicio Python.');
    } finally {
      if (pending.current === controller) { pending.current = null; setBusy(false); }
    }
  };
  const download = () => {
    if(!motion)return;
    const data = URL.createObjectURL(new Blob([JSON.stringify(motion)], {type:'application/json'}));
    const a = document.createElement('a'); a.href=data; a.download='vision3d-pose.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(data),1000);
  };
  const fileInput = (label: string, className: string) => <label className={className}>{label}<input aria-label={label} type="file" accept={ACCEPT} onChange={e=>{selectFile(e.target.files?.[0]??null);e.target.value='';}} /></label>;

  return <main>
    <header>
      <div className="brand"><span className="mark">V</span><div><b>VISION3D <em>AI</em></b><small>ESTIMACIÓN DE POSTURA</small></div></div>
      <span className="step">PROTOTIPO LOCAL</span>
    </header>
    <section className="intro">
      <p className="eyebrow">UNA PERSONA · UNA IMAGEN O VIDEO</p>
      <h1>Tu postura,<br/><i>punto por punto.</i></h1>
      <p className="sub">Carga una foto y observa su postura estimada en un muñeco 3D. Con un video, el muñeco sigue el movimiento.</p>
    </section>
    <section className="workspace">
      <div className="video-card">
        <div className="card-head"><span><b>01</b> {file ? kind==='image'?'IMAGEN ORIGINAL':'VIDEO ORIGINAL' : 'TU ARCHIVO'}</span>{fileInput(file?'Cambiar archivo':'Elegir archivo','upload')}</div>
        <div className="stage">
          {url ? <>
            {kind==='image' ? <img src={url} alt="Imagen seleccionada para estimar la postura" onError={()=>setError('El navegador no pudo abrir esta imagen. Prueba JPG o PNG.')} /> :
              <video ref={video} src={url} controls onTimeUpdate={e=>setTime(e.currentTarget.currentTime)} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)} onError={()=>setError('No se pudo reproducir este video. Prueba MP4 con códec H.264.')} />}
            <PoseOverlay frame={frame} motion={motion} visible={showPoints} />
          </> : <div className="empty"><div className="upload-icon">↑</div><b>Empieza con una foto o un video</b><span>Cuerpo completo · una sola persona</span>{fileInput('Seleccionar archivo','pick')}<small>JPG, PNG, WebP · MP4, M4V, MOV</small></div>}
          {busy && <div className="processing" role="status"><span className="spinner" />Detectando puntos corporales…</div>}
        </div>
        <div className="video-footer"><span title={file?.name}>{file?.name??'Imagen: hasta 20 MB · video: 100 MB / 60 s'}</span><button className="analyze" disabled={!file||busy} onClick={analyze}>{busy?'Procesando…':'Estimar postura →'}</button></div>
      </div>
      <div className="rig-card">
        <div className="card-head"><span><b>02</b> MUÑECO 3D ESTIMADO</span><span className="live"><i/>{motion?frame?.landmarks.length?'POSE DETECTADA':'SIN DETECCIÓN':'EN ESPERA'}</span></div>
        <AvatarView frame={frame} motion={motion} />
        <div className="rig-footer"><span><i className="legend-dot green"/> Mayor visibilidad</span><span><i className="legend-dot gray"/> Baja confianza</span></div>
      </div>
    </section>
    {error && <p className="error" role="alert">{error}</p>}
    {motion && <section className="results">
      <div className="result-row"><label className="toggle"><input type="checkbox" checked={showPoints} onChange={e=>setShowPoints(e.target.checked)}/> Puntos sobre el original</label><span>{visible} / 33 puntos con visibilidad suficiente</span><button className="text-button" onClick={download}>Descargar JSON ↓</button></div>
      {!frame?.landmarks.length && <p className="warning">No se detectó una persona en este instante. Prueba una toma de cuerpo completo, con buena luz y sin obstrucciones.</p>}
      {!!frame?.landmarks.length && visible<33 && <p className="quality-note">Los puntos grises son estimaciones poco confiables: sus articulaciones quedan como <b>no evaluables</b>. El color verde indica visibilidad, no técnica correcta.</p>}
      {kind==='video' && <div className="playback">
        <button className="play-button" onClick={()=>{if(!video.current)return;if(video.current.paused)video.current.play().catch(()=>setError('Pulsa reproducir en el video.'));else video.current.pause();}}>{playing?'Pausar':'Reproducir'}</button>
        <input aria-label="Línea de tiempo" type="range" min="0" max={motion.duration_ms/1000} step=".01" value={time} onChange={e=>{const t=Number(e.target.value);if(video.current)video.current.currentTime=t;setTime(t);}}/>
        <span>{time.toFixed(1)} / {(motion.duration_ms/1000).toFixed(1)} s</span>
        <select aria-label="Velocidad de reproducción" value={speed} onChange={e=>setSpeed(Number(e.target.value))}>{[.25,.5,1,1.5,2].map(s=><option key={s} value={s}>{s}×</option>)}</select>
      </div>}
    </section>}
    <section className="notice"><span>ⓘ</span><p><b>Postura estimada, no evaluación médica.</b> Una imagen ofrece una pose estática. La cámara, ropa, iluminación y partes ocultas afectan el resultado. Girar el muñeco permite explorar la estimación; no revela información que la cámara no captó.</p></section>
    <footer>VISION3D AI <span>·</span> Procesamiento local · tus archivos se eliminan del servicio al terminar</footer>
  </main>;
}
