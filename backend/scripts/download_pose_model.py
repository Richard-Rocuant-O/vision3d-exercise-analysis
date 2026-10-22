from pathlib import Path
from urllib.request import urlretrieve

MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if MODEL.exists():
    print(f"Modelo ya disponible: {MODEL}")
else:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    print("Descargando el modelo oficial de MediaPipe…")
    urlretrieve(URL, MODEL)
    print(f"Modelo guardado en: {MODEL}")
