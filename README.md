# VISION3D AI

Prototipo local: carga una imagen o video de una persona para ver los puntos detectados y un muñeco articulado simple en Three.js. Una imagen produce una pose estática; un video produce una secuencia. La comparación con una referencia será una etapa posterior.

## Preparación

Requisitos: Python 3.12, Node.js 22 y uv.

Desde la raíz del proyecto:

```powershell
cd backend
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/download_pose_model.py
cd ../frontend
npm ci
cd ..
.\start-local.ps1
```

Abre **http://127.0.0.1:5173**. El lanzador ejecuta ambos servicios en segundo plano y guarda registros en `data/runtime/`. También reconoce `backend/.deps`, usado en esta máquina de desarrollo. Si los puertos 8765 o 5173 ya están ocupados, comprueba que correspondan a VISION3D antes de reutilizarlos.

Alternativa en dos terminales, que puedes detener con Ctrl+C:

```powershell
# Terminal 1, desde backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# Terminal 2, desde frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

## Uso

1. Selecciona una foto JPG, PNG o WebP (20 MB, 30 megapíxeles máximo), o un video MP4, M4V o MOV (100 MB, 60 segundos máximo).
2. Pulsa **Estimar postura**. Para video en el navegador, usa códec H.264.
3. Revisa los puntos sobre el original y el muñeco al lado.
4. Arrastra sobre el muñeco para girarlo; usa la rueda para acercarlo.
5. Para videos, usa reproducir, pausar, velocidad y línea de tiempo. Una foto muestra una pose fija.
6. Usa **Descargar JSON** para conservar las coordenadas y los tiempos.

Los puntos verdes indican visibilidad/presencia suficientes bajo un umbral experimental de 0,5. No indican técnica correcta ni precisión conocida. Los puntos grises quedan como **no evaluables**. Los instantes sin detección muestran un muñeco vacío.

El servicio elimina sus archivos temporales al terminar. Guarda tus imágenes y videos propios dentro de `data/`, que está excluida de Git.

## Arquitectura

- `backend/app/pose.py`: OpenCV y MediaPipe Pose Landmarker (modos IMAGE y VIDEO).
- `backend/app/main.py`: API local; inferencia fuera del hilo de solicitudes.
- `frontend/src/components/PoseOverlay.tsx`: puntos 2D respetando proporciones y márgenes.
- `frontend/src/viewer/SimpleAvatar.ts`: adaptación de pose a geometrías simples; preparado para reemplazar por un avatar GLB/glTF.
- `frontend/src/components/AvatarView.tsx`: escena Three.js y controles de cámara.

API: `GET /api/health`, `POST /api/analyze-image` (campo multipart `image`), `POST /api/analyze` (campo `video`).

JSON 2.0: metadatos del estimador y fotogramas con `timestamp_ms`, `frame_index`, `landmarks` de imagen y `world_landmarks` en metros estimados, centrados en cadera. Los puntos no finitos son null y los fotogramas sin detección tienen listas vacías. Se guardan por separado ambos sistemas de coordenadas.

El video se muestrea aproximadamente cada 100 ms. Se usan tiempos del decodificador OpenCV; si no son válidos o crecientes, se usa una aproximación con FPS y se registra en `timestamp_source`.

## Límites

La profundidad es una estimación desde una cámara. Girar el muñeco no descubre partes que la cámara no observó. El avatar se presenta sobre un suelo visual usando los pies observables, sin reconstruir el desplazamiento absoluto de la persona por una habitación.

Cámara, ropa, iluminación, perspectiva y articulaciones ocultas afectan la estimación. Una foto no permite evaluar fases, velocidad ni repeticiones. El sistema no reconoce automáticamente el tipo de ejercicio ni comprueba que haya exactamente una persona. No calcula ángulos, puntuaciones ni diagnósticos en esta versión.

## Comprobación reproducible

Desde backend, con el entorno preparado:

```powershell
.venv/Scripts/python.exe scripts/check_local.py "ruta/al/video.mp4"
```

Comprueba imagen y video, coordenadas 3D, tiempos crecientes, una imagen sintética negra sin detección y rechazo de contenido inválido. Guarda una imagen derivada y resultados en `data/local-check/`: son una prueba funcional, no un dataset ni una medición de precisión.

## Vercel y Modal, a futuro

Vercel: raíz `frontend`, comando `npm run build`, salida `dist`. El proxy Vite solo funciona durante desarrollo; una web alojada necesitará configurar la URL de API y los orígenes CORS. Modal alojaría el mismo núcleo Python y el modelo descargado. La integración de nube queda pendiente. Nada se publica o despliega automáticamente.

La visión futura está en `docs/project-vision.md`; el alcance ejecutable actual está en este README.

Fuentes: [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) y [Three.js OrbitControls](https://threejs.org/docs/pages/OrbitControls.html).
