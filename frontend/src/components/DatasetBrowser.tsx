import { useEffect, useRef, useState } from 'react';

export type DatasetExample = {
  id: number; name: string; label: string; source_row: string;
  validation_fold: string | null; image_hash: string | null; version: string; image_url: string;
};
type Summary = { loaded: boolean; filename?: string; total: number; labels: { label: string; count: number }[]; version?: string };
type Examples = { items: DatasetExample[]; total: number; page: number; pages: number; version: string };
const LABELS: Record<string,string> = {
  'Squat':'Sentadilla', 'Bent-over row':'Remo inclinado', 'Chest dips':'Fondos de pecho',
  'Glute bridge':'Puente de glúteos', 'Lunges':'Zancadas', 'Plank':'Plancha',
  'Pull-up':'Dominadas', 'Push-up':'Flexiones', 'Shoulder press':'Press de hombros',
};
const format = (n: number) => n.toLocaleString('es-CL');
async function readJson(response: Response) {
  if(!response.headers.get('content-type')?.includes('application/json')) throw new Error('No se pudo conectar con la biblioteca. Comprueba que el servicio Python esté abierto.');
  const data=await response.json();
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:'No se pudo abrir el dataset.');
  return data;
}

export default function DatasetBrowser({ onSelect }: { onSelect: (file: File, example: DatasetExample) => void }) {
  const [info,setInfo]=useState<Summary|null>(null);
  const [examples,setExamples]=useState<Examples|null>(null);
  const [label,setLabel]=useState('Squat');
  const [page,setPage]=useState(1);
  const [loading,setLoading]=useState(false);
  const [importing,setImporting]=useState(false);
  const [choosing,setChoosing]=useState<number|null>(null);
  const [error,setError]=useState('');
  const importRequest=useRef<AbortController|null>(null);
  const choiceRequest=useRef<AbortController|null>(null);

  useEffect(()=>{
    const controller=new AbortController();
    fetch('/api/dataset',{signal:controller.signal}).then(readJson).then((data:Summary)=>{
      setInfo(data); setLabel(data.labels.some(l=>l.label==='Squat')?'Squat':'');
    }).catch(e=>{if(!controller.signal.aborted)setError(e.message);});
    return ()=>controller.abort();
  },[]);
  useEffect(()=>()=>{importRequest.current?.abort();choiceRequest.current?.abort();},[]);
  useEffect(()=>{
    if(!info?.loaded)return;
    const controller=new AbortController();
    setLoading(true);setExamples(null);
    const params=new URLSearchParams({page:String(page),page_size:'8'});
    if(label)params.set('label',label);
    fetch('/api/dataset/examples?'+params,{signal:controller.signal}).then(readJson).then(setExamples)
      .catch(e=>{if(!controller.signal.aborted)setError(e.message);})
      .finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return ()=>{controller.abort();choiceRequest.current?.abort();};
  },[info?.version,info?.loaded,label,page]);

  const importFile=async(file:File)=>{
    importRequest.current?.abort();
    const controller=new AbortController();importRequest.current=controller;
    choiceRequest.current?.abort();
    setImporting(true);setError('');
    try{
      const body=new FormData();body.append('dataset',file);
      const data:Summary=await readJson(await fetch('/api/dataset/import',{method:'POST',body,signal:controller.signal}));
      if(controller.signal.aborted)return;
      setInfo(data);setPage(1);setLabel(data.labels.some(l=>l.label==='Squat')?'Squat':'');
    }catch(e){if(!controller.signal.aborted)setError(e instanceof Error?e.message:'No se pudo importar el CSV.');}
    finally{if(importRequest.current===controller){setImporting(false);importRequest.current=null;}}
  };
  const choose=async(example:DatasetExample)=>{
    choiceRequest.current?.abort();
    const controller=new AbortController();choiceRequest.current=controller;
    setChoosing(example.id);setError('');
    try{
      const response=await fetch(example.image_url,{signal:controller.signal});
      if(!response.ok){await readJson(response);return;}
      const blob=await response.blob();
      if(controller.signal.aborted)return;
      onSelect(new File([blob],example.name.replace(/\.[^.]+$/,'')+'.png',{type:'image/png'}),example);
    }catch(e){if(!controller.signal.aborted)setError(e instanceof Error?e.message:'No se pudo abrir la imagen.');}
    finally{if(choiceRequest.current===controller){setChoosing(null);choiceRequest.current=null;}}
  };

  return <section className="dataset-panel" aria-label="Biblioteca local de ejercicios">
    <div className="dataset-heading">
      <div><p className="eyebrow">EJEMPLOS DEL DATASET</p><h2>Explora tus ejercicios</h2><p className="dataset-description">Elige una imagen para abrirla en el visor y estimar su postura.</p></div>
      <label className="upload">{importing?'Importando CSV…':info?.loaded?'Cambiar CSV':'Cargar CSV preparado'}<input type="file" accept=".csv,text/csv" disabled={importing} aria-label="Cargar dataset CSV" onChange={e=>{const file=e.target.files?.[0];if(file)importFile(file);e.target.value='';}}/></label>
    </div>
    <p className="dataset-disclaimer">Las etiquetas identifican el ejercicio. No indican si su técnica es correcta o incorrecta. El muñeco se estima de nuevo desde cada imagen.</p>
    {error&&<p className="error" role="alert">{error}</p>}
    {!info&&!error&&<p className="dataset-state" role="status">Abriendo biblioteca local…</p>}
    {info&&!info.loaded&&<p className="dataset-state">Carga dataset_preparado.csv para explorar sus imágenes. Se guardará solo en este equipo.</p>}
    {info?.loaded&&<>
      <div className="dataset-toolbar">
        <span className="dataset-count">{format(info.total)} imágenes · {info.labels.length} ejercicios</span>
        <label>Ejercicio <select value={label} disabled={importing} onChange={e=>{setLabel(e.target.value);setPage(1);setError('');}}>
          <option value="">Todos los ejercicios</option>
          {info.labels.map(item=><option key={item.label} value={item.label}>{LABELS[item.label]??item.label} ({format(item.count)})</option>)}
        </select></label>
      </div>
      {loading&&<p className="dataset-state" role="status">Cargando imágenes…</p>}
      {examples&&!loading&&<>
        <div className="dataset-grid">{examples.items.map(example=><button className="example-card" key={example.id} disabled={importing||choosing!==null} onClick={()=>choose(example)}>
          <div className="example-image"><img src={example.image_url+'&thumbnail=true'} loading="lazy" alt={LABELS[example.label]??example.label}/></div>
          <div className="example-info"><b>{LABELS[example.label]??example.label}</b><span title={example.name}>{example.name}</span><small>Fila de origen {example.source_row}{example.validation_fold?' · fold '+example.validation_fold:''}</small><strong>{choosing===example.id?'Abriendo…':'Abrir en visor ↗'}</strong></div>
        </button>)}</div>
        {!examples.total&&<p className="dataset-state">No hay imágenes para este filtro.</p>}
        <div className="dataset-pages"><span>{format(examples.total)} ejemplos en este filtro</span><div><button className="text-button" disabled={page===1||importing||choosing!==null} onClick={()=>setPage(p=>p-1)}>← Anterior</button><span>Página {page} de {examples.pages}</span><button className="text-button" disabled={page>=examples.pages||importing||choosing!==null} onClick={()=>setPage(p=>p+1)}>Siguiente →</button></div></div>
      </>}
    </>}
  </section>;
}
