# MotionBERT en VISION3D

Integración experimental de video, independiente del visor y de futuras reglas
de comparación. No determina si una sentadilla es correcta ni emite diagnósticos.

## Recorrido de los datos

1. OpenCV decodifica todos los fotogramas del tramo elegido (hasta 60 s / 3600
   fotogramas por análisis). El archivo original puede durar más. `start_seconds`
   y `duration_seconds` seleccionan el intervalo; los tiempos e índices se
   conservan respecto al original y `analysis_window` documenta la selección.
   MediaPipe ejecuta VIDEO con marcas de tiempo crecientes. El FPS original se
   conserva; no se exige 30 FPS ni se duplican muestras de 10 FPS.
2. `backend/app/motionbert.py` convierte los 33 puntos de imagen a H36M17.
   Pelvis = centro de caderas; cuello = centro de hombros; columna = centro
   pelvis/cuello; cabeza = centro de orejas como **aproximación explícita**.
   Esta última no equivale a un punto observado de coronilla. El detector 2D
   oficial usa AlphaPose/Halpe, por lo que MediaPipe es una adaptación que
   requiere validación con nuestros clips.
3. La normalización sigue el modo `--pixel` de `infer_wild.py`: convertir a
   píxeles, restar el centro de imagen y dividir por la mitad de la dimensión
   menor. Se conserva la proporción de la imagen. El tercer canal es el mínimo
   de `visibility` y `presence`; los centros conservan la menor confianza de sus
   fuentes. Un punto ausente/no finito entra con confianza cero.
4. El checkpoint oficial Lite de estimación 3D procesa ventanas de hasta 243
   fotogramas, con ensamblado de la predicción original y reflejada. Ventanas
   consecutivas se solapan al menos 32 fotogramas y se promedian con pesos
   mayores en el centro para reducir discontinuidades. Las articulaciones
   se reflejan intercambiando izquierda y derecha, además de invertir X.
5. Los intervalos sin persona o con saltos de tiempo se separan antes de la
   inferencia. Se exige pelvis/cuello presentes y al menos seis entradas con
   visibilidad suficiente. No se conecta una detección anterior con otra
   posterior atravesando un intervalo descartado. Los puntos ausentes siguen
   siendo `null`; los poco confiables conservan su condición aunque el modelo
   produzca posiciones suaves.
6. Cada predicción se centra en su pelvis. Se guarda en unidades del modelo,
   separada de los metros estimados de MediaPipe. Para dibujar, una sola escala
   por clip coloca la longitud mediana de torso observable en 0,5 unidades de
   avatar. **No es una calibración de tamaño corporal.**

## API y formato

`GET /api/estimators` indica qué dependencias/pesos están disponibles. La primera
inferencia comprueba también SHA-256, compatibilidad del checkpoint y dispositivo.
La web consulta esta disponibilidad al pulsar **Estimar postura** y solicita
MotionBERT cuando está disponible, o MediaPipe si falta la instalación opcional.
El estimador ejecutado se identifica en el resultado. No hay un selector previo
de método; se puede alternar la visualización después si existen ambos resultados.

`POST /api/analyze`: multipart `video`, `start_seconds` (0 por defecto),
`duration_seconds` (60 por defecto, máximo 60) y `estimator` (`mediapipe`, predeterminado,
o `motionbert`). Un intervalo inválido o estimador desconocido devuelve 422. Si se elige MotionBERT y
faltan dependencias/pesos, devuelve 503 con instrucciones; nunca sustituye el
estimador silenciosamente. Los temporales se eliminan también ante errores.

JSON 2.1 con MotionBERT:

- `landmark_names`, `frames[].landmarks`, `frames[].world_landmarks`:
  los 33 puntos originales de MediaPipe, sin reemplazar ni mezclar sistemas.
- `motionbert.landmark_names`, `frames[].motionbert_landmarks`:
  los 17 puntos H36M en unidades del modelo. Lista vacía si no hay entrada
  utilizable; puntos individuales ausentes como `null`.
- `motionbert`: checkpoint, SHA-256, versión de PyTorch, dispositivo,
  normalización, procedencia de confianza, ventana, solapamiento y escala visual.
- `selected_estimator`: método solicitado; la selección del visor no modifica
  el archivo de resultados. Ambos movimientos se conservan para comparar.

El umbral 0,5 solo controla visibilidad experimental. La confianza 2D no mide
incertidumbre en profundidad. Una articulación oculta puede estar mal estimada
aunque el detector le dé confianza alta.

## Visor y retargeting

`frontend/src/viewer/poseAdapter.ts` elige la secuencia y aplica la escala solo al
dibujar. El visor sigue independiente de PyTorch. `HumanAvatar` acepta nombres
de articulación de los dos esquemas; la cabeza usa el centro aproximado y las
manos/pies sin seguimiento detallado se dibujan con orientación genérica y en
gris. Esos detalles no se añaden a los puntos exportados como observaciones.

El avatar actual orienta segmentos mediante sus extremos; un GLB con esqueleto
necesitará otro adaptador para convertir posiciones a rotaciones y respetar su
pose de reposo. MotionBERT no entrega esas rotaciones.

## Ejecución y pruebas

La preparación CPU está en el README. En esta máquina el lanzador usa
`backend/.deps`; para reproducir allí una instalación opcional con esa estructura:

```powershell
uv pip install --python backend/.venv/Scripts/python.exe --target backend/.deps torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python backend/scripts/download_motionbert_model.py
```

`VISION3D_MOTIONBERT_DEVICE=cpu` es el valor predeterminado. Se puede seleccionar
`cuda` únicamente con una instalación PyTorch/CUDA compatible; no se instala
CUDA automáticamente. El modelo se carga una vez y las inferencias se
serializan para limitar memoria. CPU utiliza hasta cuatro hilos PyTorch.

`test_motionbert.py` comprueba mapeo izquierda/derecha, proporciones, confianza
de puntos derivados, ausencias, cortes temporales, cobertura de ventanas y
errores de disponibilidad. No necesita pesos ni ejecutar PyTorch.

`test_video_segment.py` usa un decodificador y detector simulados para verificar
archivos de más de un minuto, búsqueda posterior al segundo 60, tiempos/índices
del original, fin de archivo y validación de intervalos. No estima calidad de pose.

`check_motionbert.py` envía un video real a través del proxy web y verifica
tiempos, secuencia de fotogramas, 33/17 articulaciones, resultados finitos,
centro de pelvis y rechazo de métodos desconocidos. Escribe JSON y tiempos
observados en `data/motionbert-check/`; los tiempos no son un benchmark general.
Admite `--start 15 --duration 10` para comprobar un intervalo distinto del inicio.

La calidad requiere revisar varios clips (cámara lateral, ocultación, luz y
ropa distintas), observar saltos y fases de movimiento y, para afirmar precisión
3D, disponer de referencia 3D medida. Suavidad y exactitud son cosas distintas.

## Procedencia y licencia

Arquitectura oficial vendorizada con revisión fija y Apache 2.0, incluyendo
licencia y avisos: `backend/app/vendor/motionbert/NOTICE.md`. El único cambio
del código del modelo es una importación relativa. Se carga un checkpoint
oficial descargado por separado y verificado mediante SHA-256; nunca pesos
arbitrarios subidos por el usuario. PyTorch usa `weights_only=True` y la carga
de estado es estricta. No se cargan modelos con pesos aleatorios como resultado.

La licencia del código no implica que todos los datasets/pesos compartan las
mismas condiciones; revisar la tarjeta y Human3.6M antes de redistribuir un
producto entrenado. No se redistribuyen videos ni los pesos en este repositorio.

Fuentes oficiales:

- [MotionBERT y publicación](https://github.com/Walter0807/MotionBERT)
- [Inferencia de video](https://github.com/Walter0807/MotionBERT/blob/main/infer_wild.py)
- [Adaptador de entrada oficial](https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_wild.py)
- [Checkpoint del autor](https://huggingface.co/walterzhu/MotionBERT)
- [Condiciones de Human3.6M](http://vision.imar.ro/human3.6m/description.php)

## Modal y Vercel

Para Modal, incluir los requisitos base, PyTorch compatible con el dispositivo,
el código vendorizado y ambos modelos en una imagen/volumen; reutilizar el módulo
Python y mantener carga en caché. Medir primero latencia y memoria para decidir
CPU/GPU y concurrencia. Faltan implementar autenticación de API, CORS del dominio
real y gestión de trabajos largos antes de ofrecerlo públicamente.

Vercel solo serviría el frontend Three.js; ni PyTorch ni la inferencia se añaden
al navegador. El proxy actual es de desarrollo. No se ha desplegado nada.
