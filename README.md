# VISION3D AI

Prototipo local: carga una imagen o video de una persona para ver los puntos detectados sobre el original. El muñeco 3D se retiró. En videos de sentadilla lateral se comparan rodillas y tronco por fase, resaltando diferencias con un ejemplo en rojo y añadiendo etiquetas junto a las articulaciones. El ejemplo sigue sin revisión de técnica: el rojo no confirma un error y su ausencia no certifica una ejecución correcta.

**Uso actual y verificación del resaltado:** [docs/video-highlights.md](docs/video-highlights.md). Las secciones históricas de avatar y MotionBERT de este documento describen experimentos anteriores; esos visores y métodos ya no forman parte de la interfaz actual.

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

1. Selecciona una foto JPG, PNG o WebP (20 MB, 30 megapíxeles máximo), o un video MP4, M4V o MOV (archivo de hasta 100 MB).
2. En video, elige **Inicio** y **Duración** del tramo (15 segundos inicialmente, hasta 60 por análisis). El archivo puede durar más: se procesa solo ese intervalo. Pulsa **Estimar postura**. Para reproducirlo en el navegador, usa códec H.264.
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
- `frontend/src/viewer/HumanAvatar.ts`: avatar humano genérico con segmentos de volumen, manos, cabeza y ropa deportiva; conserva la interfaz para un futuro rig GLB/glTF.
- `frontend/src/components/AvatarView.tsx`: escena Three.js y controles de cámara.

API: `GET /api/health`, `GET /api/estimators`, `POST /api/analyze-image` (campo multipart `image`), `POST /api/analyze` (campos `video`, `estimator=mediapipe|motionbert`, `start_seconds` y `duration_seconds`; estos tres últimos opcionales). Por API se usa inicio 0 y duración 60 si se omiten.

JSON 2.0: metadatos del estimador y fotogramas con `timestamp_ms`, `frame_index`, `landmarks` de imagen y `world_landmarks` en metros estimados, centrados en cadera. Los puntos no finitos son null y los fotogramas sin detección tienen listas vacías. Se guardan por separado ambos sistemas de coordenadas.

El tramo elegido se procesa en todos sus fotogramas decodificables, conservando la frecuencia original, hasta 60 segundos y 3600 fotogramas por análisis. No es necesario recortar el archivo original. `analysis_window` documenta el intervalo real y el solicitado; los tiempos e índices corresponden al video original. El visor se vacía al buscar fuera del tramo. `sample_interval_ms=0` significa que no se saltean fotogramas del intervalo; `source_fps` y `sampling` documentan la extracción. Se usan tiempos del decodificador OpenCV; si no son válidos o crecientes, se usa una aproximación con FPS y se registra en `timestamp_source`.

## Límites

La profundidad es una estimación desde una cámara. Girar el muñeco no descubre partes que la cámara no observó. El avatar se presenta sobre un suelo visual usando los pies observables, sin reconstruir el desplazamiento absoluto de la persona por una habitación.

Cámara, ropa, iluminación, perspectiva y articulaciones ocultas afectan la estimación. Una foto no permite evaluar fases, velocidad ni repeticiones. El reconocimiento del ejercicio es experimental y puede abstenerse. Se muestran ángulos 2D aproximados de rodillas y tronco, que requieren vista lateral. No se comprueba automáticamente que haya exactamente una persona ni se emiten diagnósticos o una calificación de técnica correcta/incorrecta.

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

## Biblioteca del dataset CSV

La sección **Explora tus ejercicios** permite filtrar por etiqueta, navegar sus imágenes y pulsar **Abrir en visor**. Después pulsa **Estimar postura** para obtener los puntos y el muñeco. En esta máquina ya se copió el dataset preparado académico a `data/dataset/dataset_preparado.csv`. El original permanece intacto.

En otro equipo, usa **Cargar CSV preparado** o copia el archivo a esa ruta local. Debe estar en UTF-8 con las columnas `imageName`, `base64encoded` y `label`. Se conservan `source_row`, `image_hash` y `validation_fold` cuando están disponibles. El importador admite hasta 100 MB y 10.000 imágenes; un archivo inválido no reemplaza la biblioteca anterior.

El CSV persiste solo en el equipo local. No se incluye en Git y cargarlo no inicia un entrenamiento. Hay scripts explícitos de entrenamiento experimental, descritos abajo. Las etiquetas describen ejercicios y no validan su técnica. Los `poseLandmarks` antiguos del CSV no se interpretan como coordenadas 3D: se vuelve a ejecutar MediaPipe sobre la imagen elegida. El JSON descargado conserva la etiqueta, fila de origen, fold y hash de versión del CSV en `dataset_source`.

API de biblioteca: `GET /api/dataset`, `GET /api/dataset/examples?label=Squat&page=1`, `GET /api/dataset/examples/{id}/image?version={sha256}` y `POST /api/dataset/import` (campo multipart `dataset`). La imagen original se sirve como PNG y las miniaturas como JPEG; el listado no envía todo el base64 al navegador.

Con la app iniciada, comprueba el flujo de biblioteca y análisis con:

```powershell
# Desde backend
.venv/Scripts/python.exe scripts/check_dataset.py
```

## Avatar humano

El visor usa una silueta humana genérica con cabeza, cuello, torso, pelvis, extremidades de volumen variable, manos, ropa deportiva y calzado. Su iluminación incluye sombras y un suelo visual. El botón **Articulaciones** muestra u oculta los puntos sobre el cuerpo. Las zonas poco confiables se mantienen grises.

La forma del cuerpo, dedos, cara y vestimenta son detalles visuales genéricos: no reconstruyen la identidad ni la anatomía exacta de la persona, y no añaden precisión a MediaPipe. La postura sigue las coordenadas estimadas del archivo.

Comprobación de geometría sin GPU, desde frontend, usando un JSON de pose de video:

```powershell
node scripts/check-avatar.mjs "ruta/al/video-pose.json"
```

Verifica transformaciones finitas, poses vacías/degeneradas, zonas de baja confianza y el control de articulaciones. Genera datos de diagnóstico en `data/avatar-check/`, excluido de Git.

## MotionBERT opcional para videos

Al analizar un video, la interfaz usa MotionBERT automáticamente cuando está disponible; si no está instalado, solicita MediaPipe. Después, **Ver movimiento de** permite alternar ambos estimadores cuando el resultado contiene los dos. Las fotos y la biblioteca CSV siguen usando MediaPipe. En esta máquina ya están instalados PyTorch para CPU y los pesos oficiales verificados.

Preparación en otro equipo, desde `backend`:

```powershell
# Opción CPU, suficiente para probar sin configurar CUDA:
uv pip install --python .venv/Scripts/python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe scripts/download_motionbert_model.py
```

Reinicia el servicio Python después de instalar y recarga la web. `requirements-motionbert.txt` declara la dependencia opcional si preparas otro entorno PyTorch. El modelo está excluido de Git y no se descarga durante una solicitud. Sin PyTorch o sin pesos, la web utiliza MediaPipe; el resultado identifica el estimador que realmente se ejecutó. Si MotionBERT falla durante su carga o inferencia, se muestra el error sin cambiar de método silenciosamente.

MotionBERT añade `motionbert_landmarks` (17 articulaciones) y metadatos `motionbert` en JSON 2.1. Conserva intactos los 33 `landmarks` y `world_landmarks` originales. Sus coordenadas son relativas a la cadera, **en unidades del modelo, no metros**; una escala constante se aplica exclusivamente al avatar. Manos y orientación de pies son detalles genéricos grises porque faltan esos puntos en H36M17. La confianza procede de la visibilidad 2D, no de una precisión 3D conocida. No se rellenan intervalos sin persona.

```powershell
# Desde backend, con la app abierta:
.venv/Scripts/python.exe scripts/test_motionbert.py
.venv/Scripts/python.exe scripts/test_video_segment.py
.venv/Scripts/python.exe scripts/check_motionbert.py "ruta/al/video.mp4"
# Para comprobar un tramo conservando sus tiempos originales:
.venv/Scripts/python.exe scripts/check_motionbert.py "ruta/al/video.mp4" --start 15 --duration 10
```

Detalles del adaptador, licencia, pruebas y uso futuro en Modal: [docs/motionbert.md](docs/motionbert.md). Un movimiento más suave no garantiza mayor precisión; hace falta revisar nuestros videos antes de elegir un estimador por defecto.

## Ejercicio y mediciones durante la reproducción

Después de analizar, el panel **Análisis del ejercicio** sigue la misma marca de tiempo que el video y el muñeco, incluso al pausar, cambiar velocidad o buscar otro instante. El nombre del ejercicio se estima usando el contexto del tramo completo, suponiendo un único ejercicio: puede mostrar un candidato **Posible…** o **sin identificar** cuando la evidencia no basta. Las mediciones son del instante actual. No toma la etiqueta del nombre del archivo subido ni de la selección de la galería.

Las mediciones muestran flexión de rodilla izquierda/derecha e inclinación del tronco en la imagen. Una articulación poco confiable o ausente devuelve **No evaluable**, nunca un ángulo de cero inventado. Se vacían fuera del tramo seleccionado. El procesamiento ocurre antes de reproducir; esta versión no es una cámara en vivo.

La referencia de prueba elegida es `Ejercicios/squat/squat_01.mp4`, registrada como **pendiente de revisión** en `backend/config/squat-reference.json`. Todavía no se compara con ella ni se afirma que un movimiento está bien o mal: faltan criterios revisados por fase y validación de ese motor. La identificación del ejercicio y su técnica son tareas separadas.

La aplicación prioriza el clasificador adaptado con ejemplos de video cuando existe; los ensayos anteriores con fotos/puntos del CSV quedan como alternativas experimentales. Los modelos y reportes se guardan solo en `data/models/`, fuera de Git. En esta máquina ya se prepararon las herramientas de entrenamiento; en otro equipo:

```powershell
# Desde backend, entorno base ya preparado:
uv pip install --python .venv/Scripts/python.exe -r requirements-training.txt
.venv/Scripts/python.exe scripts/train_exercise_classifier.py ../data/dataset/dataset_preparado.csv
# Requiere los videos locales indicados en el manifiesto:
.venv/Scripts/python.exe scripts/train_video_classifier.py
.venv/Scripts/python.exe scripts/test_feedback.py
# Con la app abierta, usa un video propio:
.venv/Scripts/python.exe scripts/check_feedback.py ../Ejercicios/squat/squat_02.mp4 --start 0 --duration 5
```

No se afirma una precisión general: el dataset de fotos produjo baselines débiles y los pocos videos de prueba no representan todos los ejercicios, personas o cámaras. Procedencia, reportes, ensayos visuales y límites: [docs/exercise-feedback.md](docs/exercise-feedback.md).
