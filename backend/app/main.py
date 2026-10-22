from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
SAMPLE_MS = 100
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

app = FastAPI(title="VISION3D AI - Pose Prototype", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ready" if MODEL_PATH.is_file() else "missing"}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)) -> dict:
    if video.content_type not in {"video/mp4", "video/quicktime", "video/x-m4v", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Formato no admitido. Sube un video MP4 o M4V.")
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Falta el modelo. Ejecuta backend/scripts/download_pose_model.py.")

    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".m4v", ".mov"}:
        suffix = ".mp4"

    temp_path: str | None = None
    try:
        data = await video.read(MAX_UPLOAD_BYTES + 1)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="El video debe pesar menos de 100 MB.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(data)
            temp_path = temp.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=422, detail="No se pudo abrir el video.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            capture.release()
            raise HTTPException(status_code=422, detail="El video no informa una duración o frecuencia válida.")

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False,
        )
        poses: list[dict] = []
        next_sample = 0
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = round(frame_index * 1000 / fps)
                if timestamp_ms >= next_sample:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                    )
                    landmarks = []
                    if result.pose_landmarks:
                        landmarks = [
                            {
                                "x": point.x,
                                "y": point.y,
                                "z": point.z,
                                "visibility": point.visibility,
                                "presence": point.presence,
                            }
                            for point in result.pose_landmarks[0]
                        ]
                    poses.append({"timestamp_ms": timestamp_ms, "landmarks": landmarks})
                    next_sample = timestamp_ms + SAMPLE_MS
                frame_index += 1
        capture.release()
        return {
            "filename": video.filename,
            "duration_ms": round(frame_count * 1000 / fps),
            "width": width,
            "height": height,
            "sample_interval_ms": SAMPLE_MS,
            "landmark_schema": "MediaPipe Pose Landmarker; normalized image coordinates",
            "landmark_names": list(LANDMARK_NAMES),
            "frames": poses,
        }
    finally:
        await video.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
