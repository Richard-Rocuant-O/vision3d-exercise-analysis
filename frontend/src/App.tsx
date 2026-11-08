import { useEffect, useRef, useState } from 'react';
import PoseOverlay from './components/PoseOverlay';
import DatasetBrowser, { type DatasetExample } from './components/DatasetBrowser';
import ExerciseFeedback from './components/ExerciseFeedback';
import { frameAt, reliable, type Motion } from './motion';

const ACCEPT = '.jpg,.jpeg,.png,.webp,.mp4,.m4v,.mov,image/jpeg,image/png,image/webp,video/mp4';
const VIDEO_ACCEPT = '.mp4,.m4v,.mov,video/mp4';
function mediaKind(file: File): 'image' | 'video' {
  return file.type.startsWith('image/') || /\.(jpe?g|png|webp)$/i.test(file.name) ? 'image' : 'video';
}

export default function App() {
  const video = useRef<HTMLVideoElement>(null);
  const pending = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [datasetSource, setDatasetSource] = useState<DatasetExample | null>(null);
  const [url, setUrl] = useState('');
  const [motion, setMotion] = useState<Motion | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [time, setTime] = useState(0);
  const [showPoints, setShowPoints] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [compareExample, setCompareExample] = useState(true);
  const [tolerance, setTolerance] = useState('15');
  const [sourceDuration, setSourceDuration] = useState<number | null>(null);
  const [segmentStart, setSegmentStart] = useState('0');
  const [segmentDuration, setSegmentDuration] = useState('15');
  const [refFile, setRefFile] = useState<File | null>(null);
  const [refUrl, setRefUrl] = useState('');
  const [refError, setRefError] = useState('');
  const kind = file ? mediaKind(file) : 'image';
  const frame = frameAt(motion, time * 1000);
  const visible = frame?.landmarks.filter(reliable).length ?? 0;
  const playbackStart = (motion?.analysis_window?.start_ms ?? 0) / 1000;
  const playbackEnd = (motion?.analysis_window?.end_ms ?? motion?.duration_ms ?? 0) / 1000;
  const outsideWindow = !!motion?.analysis_window && (time < playbackStart || time > playbackEnd);
  const startValue = Number(segmentStart), durationValue = Number(segmentDuration);
  const segmentValid = segmentStart.trim() !== '' && segmentDuration.trim() !== '' && Number.isFinite(startValue) && startValue >= 0 && Number.isFinite(durationValue) && durationValue > 0 && durationValue <= 60 && (sourceDuration === null || startValue < sourceDuration);
  const toleranceValid = !compareExample || (tolerance.trim()!==''&&Number.isFinite(Number(tolerance))&&Number(tolerance)>=5&&Number(tolerance)<=45);

  useEffect(() => {
    if (!file) { setUrl(''); return; }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  useEffect(() => {
    if (!refFile) { setRefUrl(''); return; }
    const u = URL.createObjectURL(refFile);
    setRefUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [refFile]);
  useEffect(() => () => pending.current?.abort(), []);
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const tick = () => {
      if (video.current) {
        if (motion && video.current.currentTime >= playbackEnd) { video.current.pause(); video.current.currentTime = playbackEnd; }
        setTime(video.current.currentTime);
      }
      id = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(id);
  }, [playing, motion, playbackEnd]);
  useEffect(() => { if(video.current) video.current.playbackRate = speed; }, [speed, url]);

  const selectFile = (next: File | null) => {
    pending.current?.abort(); pending.current = null;
    setFile(next); setDatasetSource(null); setMotion(null); setError(''); setTime(0); setBusy(false); setPlaying(false);
    setSourceDuration(null); setSegmentStart('0'); setSegmentDuration('15');
  };
  const selectReference = (next: File | null) => {
    if (next && mediaKind(next) !== 'video') { setRefError('La referencia debe ser un video MP4, M4V o MOV.'); return; }
    setRefFile(next); setRefError('');
  };
  const analyze = async () => {
    if (!file) return;
    if (kind === 'video' && !segmentValid) { setError('Elige un inicio dentro del video y una duración de más de 0 y hasta 60 segundos.'); return; }
    if(kind==='video'&&!toleranceValid){setError('La tolerancia visual debe estar entre 5 y 45 grados.');return;}
    pending.current?.abort();
    const controller = new AbortController(); pending.current = controller;
    video.current?.pause();
    setBusy(true); setError(''); setMotion(null);
    try {
      const body = new FormData(); body.append(kind === 'image' ? 'image' : 'video', file);
      if (kind === 'video') {
        body.append('start_seconds',segmentStart); body.append('duration_seconds',segmentDuration);
        body.append('compare_to_example',String(compareExample));body.append('tolerance_degrees',compareExample?tolerance:'15');
      }
      const response = await fetch(kind === 'image' ? '/api/analyze-image' : '/api/analyze', { method:'POST', body, signal:controller.signal });
      if (!response.headers.get('content-type')?.includes('application/json')) {
        throw new Error(response.status === 500
          ? 'El servicio Python falló al procesar el archivo. Revisa el registro backend-error.log en data/runtime.'
          : response.status === 502 || response.status === 503 || response.status === 504
            ? 'El servicio de análisis no está disponible. Comprueba que Python esté abierto en el puerto 8765.'
            : `El servicio devolvió una respuesta inesperada (HTTP ${response.status}).`);
      }
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'No se pudo analizar este archivo.');
      if (pending.current !== controller) return;
      if (!Array.isArray(data.frames) || !data.frames.length) throw new Error('El archivo no produjo fotogramas utilizables.');
      setMotion(datasetSource ? { ...data, dataset_source: {
        label: datasetSource.label, source_row: datasetSource.source_row,
        image_hash: datasetSource.image_hash, validation_fold: datasetSource.validation_fold,
        version: datasetSource.version,
      } } : data);
      const t = (data.analysis_window?.start_ms ?? 0) / 1000;
      if (video.current) video.current.currentTime = t;
      setTime(t);
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

  const referenceInput = (label: string, className: string) => <label className={className}>{label}<input aria-label={label} type="file" accept={VIDEO_ACCEPT} onChange={e=>{selectReference(e.target.files?.[0]??null);e.target.value='';}} /></label>;

  return <main>
    <header>
      <div className="brand"><span className="mark">V</span><div><b>VISION3D <em>AI</em></b><small>ESTIMACIÓN DE POSTURA</small></div></div>
      <span className="step">PROTOTIPO LOCAL</span>
    </header>
    <section className="intro">
      <p className="eyebrow">UNA PERSONA · UNA IMAGEN O VIDEO</p>
      <h1>Tu postura,<br/><i>punto por punto.</i></h1>
      <p className="sub">Analiza un video y observa las articulaciones sobre tu movimiento. Las diferencias con el ejemplo se resaltan en rojo.</p>
    </section>
    {file&&kind==='video'&&<section className="analysis-options">
      <div className="segment-controls">
        <label>Inicio (segundos)<input type="number" min="0" max={sourceDuration??undefined} step="0.1" value={segmentStart} disabled={busy} onChange={e=>setSegmentStart(e.target.value)}/></label>
        <label>Duración (segundos)<input type="number" min="0.1" max="60" step="0.1" value={segmentDuration} disabled={busy} onChange={e=>setSegmentDuration(e.target.value)}/></label>
        <span>{sourceDuration!==null?`Video: ${sourceDuration.toFixed(1)} s. `:''}Se analiza solo el tramo elegido, hasta 60 s. El archivo original no se modifica.</span>
      </div>
      <div className="comparison-options"><label className="toggle"><input type="checkbox" checked={compareExample} disabled={busy} onChange={e=>setCompareExample(e.target.checked)}/> Comparar con el ejemplo de sentadilla · experimental</label>{compareExample&&<label>Tolerancia visual <input aria-label="Tolerancia visual en grados" type="number" min="5" max="45" value={tolerance} disabled={busy} onChange={e=>setTolerance(e.target.value)}/> °</label>}</div>
      {compareExample&&<p>Rojo indica diferencia con el ejemplo, no un error confirmado. La tolerancia controla el resaltado; no define una técnica correcta. Se requiere vista lateral y una repetición completa.</p>}
      {!segmentValid&&<p role="status">Elige un inicio dentro del video y una duración de más de 0 y hasta 60 segundos.</p>}
    </section>}
    {datasetSource && <div className="dataset-selection" role="status"><span>Imagen del dataset · <b>{datasetSource.label}</b> · fila de origen {datasetSource.source_row}</span><span>Pulsa <b>Analizar</b> para detectar sus articulaciones.</span></div>}
    <section className="workspace">
      <div className="video-card">
        <div className="card-head"><span><b>01</b> {file ? kind==='image'?'IMAGEN ORIGINAL':'VIDEO ORIGINAL' : 'TU ARCHIVO'}</span>{fileInput(file?'Cambiar archivo':'Elegir archivo','upload')}</div>
        <div className="stage">
          {url ? <>
            {kind==='image' ? <img src={url} alt="Imagen seleccionada para estimar la postura" onError={()=>setError('El navegador no pudo abrir esta imagen. Prueba JPG o PNG.')} /> :
              <video ref={video} src={url} controls onLoadedMetadata={e=>{const d=e.currentTarget.duration;setSourceDuration(Number.isFinite(d)&&d>0?d:null);}} onTimeUpdate={e=>setTime(e.currentTarget.currentTime)} onPlay={e=>{if(motion&&(e.currentTarget.currentTime<playbackStart||e.currentTarget.currentTime>=playbackEnd))e.currentTarget.currentTime=playbackStart;setPlaying(true);}} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)} onError={()=>setError('No se pudo reproducir este video. Prueba MP4 con códec H.264.')} />}
            <PoseOverlay frame={frame} motion={motion} visible={showPoints} />
          </> : <div className="empty"><div className="upload-icon">↑</div><b>Empieza con una foto o un video</b><span>Cuerpo completo · una sola persona</span>{fileInput('Seleccionar archivo','pick')}<small>JPG, PNG, WebP · MP4, M4V, MOV</small></div>}
          {busy && <div className="processing" role="status"><span className="spinner" />{kind==='video'?'Estimando el movimiento…':'Detectando puntos corporales…'}</div>}
        </div>
        <div className="video-footer"><span title={file?.name}>{file?.name??'Imagen: 20 MB · video: 100 MB / tramos de 60 s'}</span><button className="analyze" disabled={!file||busy||(kind==='video'&&(!segmentValid||!toleranceValid))} onClick={analyze}>{busy?'Procesando…':'Analizar →'}</button></div>
      </div>
      <div className="video-card reference-card">
        <div className="card-head"><span><b>02</b> VIDEO DE REFERENCIA</span>{referenceInput(refFile?'Cambiar video de referencia':'Elegir video de referencia','upload')}</div>
        <div className="stage">
          {refUrl ? <video src={refUrl} controls onError={()=>setRefError('No se pudo reproducir este video. Prueba MP4 con códec H.264.')} /> :
            <div className="empty"><div className="upload-icon">↑</div><b>Sube el video de referencia</b><span>Cuerpo completo · una sola persona</span>{referenceInput('Seleccionar video','pick')}<small>MP4, M4V, MOV</small></div>}
        </div>
        <div className="video-footer"><span title={refFile?.name}>{refFile?.name??'Video: hasta 100 MB'}</span>{refFile&&<button className="text-button" onClick={()=>selectReference(null)}>Quitar</button>}</div>
        {refError && <p className="error" role="alert">{refError}</p>}
      </div>
    </section>
    <div className="overlay-legend"><span><i className="legend-dot green"/> Articulación visible</span><span><i className="legend-dot gray"/> No evaluable</span><span><i className="legend-dot red"/> Diferencia con el ejemplo · experimental</span></div>
    {motion&&<ExerciseFeedback frame={frame} motion={motion} time={time} outside={outsideWindow}/>}
    {error && <p className="error" role="alert">{error}</p>}
    {motion && <section className="results">
      <div className="result-row"><label className="toggle"><input type="checkbox" checked={showPoints} onChange={e=>setShowPoints(e.target.checked)}/> Articulaciones y resaltado</label><span>{visible} / 33 puntos visibles</span><button className="text-button" onClick={download}>Descargar JSON ↓</button></div>
      {outsideWindow&&<p className="warning">Fuera del tramo analizado ({playbackStart.toFixed(1)}–{playbackEnd.toFixed(1)} s). Usa la línea de tiempo inferior o vuelve a reproducir ese tramo.</p>}
      {!outsideWindow&&!frame?.landmarks.length && <p className="warning">No se detectó una persona en este instante. Prueba una toma de cuerpo completo, con buena luz y sin obstrucciones.</p>}
      {!!frame?.landmarks.length && visible<33 && <p className="quality-note">Los puntos grises son estimaciones poco confiables: sus articulaciones quedan como <b>no evaluables</b>. El color verde indica visibilidad, no técnica correcta.</p>}
      {kind==='video' && <div className="playback">
        <button className="play-button" onClick={()=>{if(!video.current)return;if(video.current.paused)video.current.play().catch(()=>setError('Pulsa reproducir en el video.'));else video.current.pause();}}>{playing?'Pausar':'Reproducir'}</button>
        <input aria-label="Línea de tiempo del tramo analizado" type="range" min={playbackStart} max={playbackEnd} step=".01" value={Math.min(playbackEnd,Math.max(playbackStart,time))} onChange={e=>{const t=Number(e.target.value);if(video.current)video.current.currentTime=t;setTime(t);}}/>
        <span>{time.toFixed(1)} s · tramo {playbackStart.toFixed(1)}–{playbackEnd.toFixed(1)} s</span>
        <select aria-label="Velocidad de reproducción" value={speed} onChange={e=>setSpeed(Number(e.target.value))}>{[.25,.5,1,1.5,2].map(s=><option key={s} value={s}>{s}×</option>)}</select>
      </div>}
    </section>}
    <DatasetBrowser onSelect={(image, example)=>{selectFile(image);setDatasetSource(example);window.scrollTo({top:0,behavior:'smooth'});}} />
    <section className="notice"><span>ⓘ</span><p><b>Comparación experimental, no evaluación médica.</b> El rojo no certifica un error ni la ausencia de rojo una técnica perfecta. La cámara, ropa, iluminación y partes ocultas afectan la estimación.</p></section>
    <footer>VISION3D AI <span>·</span> Análisis temporal · biblioteca CSV guardada solo en este equipo</footer>
  </main>;
}
