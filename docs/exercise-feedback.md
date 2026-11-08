# Reconocimiento y feedback sincronizado

El usuario confirmó que «en tiempo real» significa **durante la reproducción del
video cargado**. El backend prepara los resultados del tramo antes de reproducir;
React muestra las mediciones del mismo fotograma que mueve el muñeco. Las
estimaciones por postura usan un filtro causal; el nombre del ejercicio del
tramo utiliza el contexto completo, disponible porque el video ya se procesó.
No se simula inferencia de cámara en directo.

## Datos y tareas diferentes

El CSV local contiene 3321 imágenes, nueve etiquetas de ejercicio y cinco folds.
No tiene anotaciones de técnica correcta/incorrecta ni criterios por articulación.
La galería no equivale a un clasificador entrenado. Cargar otro CSV no entrena
automáticamente ni cambia los modelos activos.

Se prepararon ensayos reproducibles:

- Pose normalizada: puntos 2D de hombros, caderas, codos, muñecas, rodillas,
  tobillos y nariz, conservando proporciones. Se restan caderas y se normaliza por
  longitud de torso. Los puntos no observables se codifican como ausentes.
  Se conservan 2191 filas y se descartan 1130 sin pose suficiente.
- Baseline de vecinos y Random Forest sobre pose. Random Forest obtuvo
  56,1 % de acierto antes de abstención en las 442 imágenes utilizables del fold 5.
  No alcanza para una identificación general fiable.
- Ensayo visual: codificador MobileNetV3-Small ImageNet1K congelado y clasificador
  lineal supervisado sobre sus representaciones. Se conserva el cuerpo completo
  con letterbox 224×224. Obtuvo 51,5 % en 664 imágenes del fold 5, también
  insuficiente como solución general. No se presenta la precisión sobre casos
  aceptados como si fuera el acierto de todas las entradas.
- Adaptación con clips etiquetados según las carpetas del usuario: un video por
  ejercicio para entrenamiento y **tres videos completos de sentadilla separados**
  para comprobar la transferencia. El manifiesto y las particiones están en
  `backend/config/exercise-training-videos.json`. Solo se procesan los primeros
  15 segundos de cada clip y se toman ejemplos de entrenamiento cada 100 ms.

Las etiquetas de carpeta identifican ejercicios, no su ejecución correcta.
Los clips de evaluación no entran en el ajuste. Se comprueban hashes para impedir
que el mismo archivo aparezca en ambos grupos. No hay suficientes sujetos o clips
por clase para afirmar generalización a otras personas ni a los nueve ejercicios.

Los scripts reservan fold 5 antes de ajustar umbrales de rechazo con folds 1–4.
Los modelos finales se reajustan con los datos de entrenamiento disponibles.
No hay IDs fiables de persona/sesión en el CSV; un split de imágenes puede
compartir personas o imágenes muy similares. Los resultados de imágenes pertenecen
al baseline antes de adaptarlo a video y no se atribuyen al modelo adaptado.

## Modelo activo y resultados

`app/exercises.py` carga primero `data/models/exercise_video_classifier.npz` si
existe. En su ausencia se puede usar el ensayo visual preparado y, sin él, el
baseline de pose. Todos son experimentales y pueden abstenerse. El JSON declara
el método que se ejecutó y sus datos de evaluación.

Random Forest se exporta a arrays NumPy; en inferencia no se necesita scikit-learn
ni cargar pickle. El exportador comprueba sus probabilidades contra el modelo de
scikit-learn. El entrenamiento se instala en un entorno separado de la app cuando
se utiliza `.training-deps`.

La etiqueta se filtra de forma causal (EMA de 500 ms y estabilidad de 240 ms).
Una pérdida de detección o salto temporal reinicia el estado, sin mantener una
etiqueta sobre un cuerpo ausente. La puntuación interna no es una probabilidad
calibrada y no se muestra como porcentaje de certeza. Un candidato insuficiente
puede aparecer como «Posible…», con el reconocimiento sin confirmar.

En videos, `exercise_summary` agrupa votos por duración para estimar el ejercicio
del tramo completo, suponiendo una sola actividad. Se exige al menos un segundo
de datos, dos tercios de predominio, margen de un tercio sobre la siguiente clase
y medio segundo de estimaciones aceptadas para confirmar la etiqueta. Son filtros
experimentales, no garantías de acierto. Esta agrupación **usa el tramo completo**;
el filtro causal sigue describiendo únicamente las predicciones por fotograma.
Un tramo con varios ejercicios requiere segmentación adicional.

En la comprobación funcional, el fragmento 3–6 s de `squat_01` se identificó como
sentadilla, pero ese video forma parte del entrenamiento y no valida precisión.
Los tres clips separados quedaron sin confirmar; las propuestas fueron zancadas
para `squat_02` y sentadilla para `squat_03`/`squat_4`. Esta confusión impide
presentar el reconocedor como fiable. Los reportes de muestras por video están en
`data/models/exercise_video_classifier.report.json`; la agrupación de tramo se
comprueba por separado y no convierte los aciertos por fotograma en una métrica
de reconocimiento de videos.

Cada fotograma exporta:

- `exercise`: estado, etiqueta/nombre cuando se acepta, candidato si corresponde,
  puntuación y su significado. No se usa el nombre del archivo para inferir.
- `measurements`: flexión de ambas rodillas e inclinación del tronco, o `null`.
- `technique`: `not_evaluable`, `unsupported` o `pending_review`; no contiene una
  valoración de técnica correcta/incorrecta.

Las mediciones siguen cambiando al reproducir o buscar en el video; el nombre
del ejercicio corresponde al resumen del tramo. Ambos se vacían visualmente al
salir del intervalo analizado.

`exercise_classifier` registra el modelo, versión de características, hash del
dataset, evaluación y filtro temporal. `feedback` documenta las definiciones y
el estado de la referencia. Se conservan los campos originales de pose.

## Qué significan las mediciones

Son ángulos **proyectados en la imagen**, adecuados solo como estimaciones bajo
la suposición de vista lateral. Se convierten X/Y a píxeles antes de medir, para
que una imagen rectangular no deforme los ángulos.

- Rodilla: 180° menos el ángulo interior cadera–rodilla–tobillo. Una pierna
  aproximadamente extendida corresponde a 0°, sin un umbral de técnica correcta.
- Tronco: ángulo entre hombro–cadera y la vertical de la imagen. Se usa el lado
  observable con mayor visibilidad; no se determina si la inclinación va hacia
  delante o atrás.

Se exige visibilidad/presencia experimental mínima de 0,5 y coordenadas finitas.
Un vector degenerado o una articulación oculta devuelve `null`. No se interpreta
la confianza de MediaPipe como precisión biomecánica. Los puntos 3D de MotionBERT
no se mezclan con los puntos 2D para calcular estas mediciones.

## Valoración de técnica pendiente

El usuario permitió elegir cualquiera de sus videos. Se registró
`Ejercicios/squat/squat_01.mp4` como **ejemplo no revisado**. Esto no convierte su
movimiento en una ejecución correcta ni en una referencia clínica.

`backend/config/squat-reference.json` conserva `review.status=pending` y una
lista de criterios vacía. No se calcula una puntuación de técnica ni se activa
«bien/mal», aunque una postura coincida con ese ejemplo o el reconocedor indique
sentadilla. El motor de comparación por fases y sus criterios revisados todavía
no están implementados. Cambiar una bandera del JSON no valida un movimiento.

Para habilitar esa valoración faltan una referencia revisada para la variante,
criterios observables por fase, alineación de repeticiones y pruebas de falsas
alertas con casos adecuados. No se fijan grados universales a partir de un clip.

## Preparación y comprobación

Las instrucciones del baseline y la adaptación están en el README. Para repetir
el ensayo visual opcional:

```powershell
# Desde backend, entorno Python base con requisitos de entrenamiento:
uv pip install --python .venv/Scripts/python.exe -r requirements-recognition.txt --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe scripts/train_image_classifier.py ../data/dataset/dataset_preparado.csv --download-encoder
```

Los pesos oficiales del codificador se verifican y descargan únicamente durante
la preparación, nunca al recibir un video. El CSV, las imágenes, videos, cachés
y modelos derivados permanecen en este equipo y están excluidos de Git. La
licencia del código no sustituye las condiciones de los pesos/datasets originales;
revisarlas antes de distribuir un producto.

`test_feedback.py` verifica geometría conocida, proporciones de imagen, ausencia
de ángulos inventados, no valoración con referencia pendiente, causalidad y
reinicio tras pérdidas de detección. `check_feedback.py` prueba el endpoint con
un clip real y conserva el resultado en `data/feedback-check/`.

Desde frontend, `node scripts/check-feedback-ui.mjs` renderiza el componente real
con React y comprueba estados confirmados/tentativos, referencia pendiente,
ángulos nulos/cero y búsqueda fuera de tramo. Es una comprobación de contenido,
no una captura de un navegador.

Fuentes técnicas primarias: [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[MobileNetV3-Small](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html),
[combinación PyTorch/torchvision](https://pytorch.org/get-started/previous-versions/).
