# VISION3D AI

Prototipo inicial para visualizar articulaciones estimadas en un video de sentadilla. Esta etapa procesa **un video** y dibuja los puntos detectados sobre sus fotogramas. No evalúa si la técnica es correcta ni ofrece consejo médico.

## Alcance actual

- Servicio Python con FastAPI, OpenCV y MediaPipe Pose Landmarker.
- Interfaz web con React y Three.js preparada para el visor 3D.
- Los videos se procesan localmente y no se guardan en Git.

## Preparación

Requisitos: Python 3.12 y Node.js 22 o compatible.

### Servicio de análisis

```powershell
cd backend
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
python scripts\download_pose_model.py
python -m uvicorn app.main:app --reload --port 8000
```

### Interfaz web

En otra terminal:

```powershell
cd frontend
npm install
npm run dev
```

Abre la dirección que indique Vite, selecciona un video y pulsa **Analizar video**. El análisis extrae puntos de pose con sus tiempos y visibilidad; la interfaz los muestra como esqueleto 2D sobre el video. El video de prueba local `Ejercicicos/squat/Video 1.mp4` alterna vistas frontal y lateral; para empezar, conviene seleccionar el tramo lateral.

## Limitaciones

La estimación depende del ángulo, la ropa, la iluminación y las articulaciones ocultas. El resultado es una estimación visual, no una medición clínica. Los puntos con baja visibilidad se indican como inciertos. El siguiente hito será retargetear esta secuencia a un muñeco 3D y reproducirla.

## Vercel y Modal (futuro)

El frontend Vite puede configurarse en Vercel con `frontend` como raíz del proyecto y `npm run build` como comando de compilación. La integración del servicio Python con Modal queda para una etapa futura, cuando el procesamiento local esté consolidado. No se despliega nada desde este repositorio.

## Atribución

MediaPipe Pose Landmarker es una dependencia de terceros; consulta sus términos y licencia antes de distribuir el producto. No se incluyen videos ni modelos descargados en el repositorio.

