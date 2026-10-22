"""MediaPipe adapter. Image coordinates and world estimates stay separate."""
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
SAMPLE_MS = 100
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def create_landmarker(mode):
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mode, num_poses=1, output_segmentation_masks=False,
    ))


def serialize(result, timestamp_ms, frame_index):
    def points(values):
        return [dict(x=p.x, y=p.y, z=p.z, visibility=p.visibility or 0, presence=p.presence or 0)
                if all(math.isfinite(v) for v in (p.x, p.y, p.z)) else None for p in values]
    return {
        "timestamp_ms": timestamp_ms, "frame_index": frame_index,
        "landmarks": points(result.pose_landmarks[0]) if result.pose_landmarks else [],
        "world_landmarks": points(result.pose_world_landmarks[0]) if result.pose_world_landmarks else [],
    }


def response(kind, width, height, frames, duration_ms=0, timestamp_source="static"):
    return {
        "schema_version": "2.0", "input_type": kind,
        "width": width, "height": height, "duration_ms": duration_ms,
        "sample_interval_ms": SAMPLE_MS if kind == "video" else 0,
        "timestamp_source": timestamp_source,
        "coordinate_systems": {
            "landmarks": "normalized image: x right, y down; z relative estimate (not used as meters)",
            "world_landmarks": "MediaPipe estimated meters: hip-centered; x right, y down, z away from camera",
        },
        "estimator": {"name": "MediaPipe Pose Landmarker", "version": mp.__version__, "model": "lite/float16/1"},
        "landmark_names": list(LANDMARK_NAMES), "frames": frames,
        "detected_frames": sum(bool(f["landmarks"]) for f in frames),
    }


def rgb_image(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def extract_image(path):
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo leer la imagen. Prueba un archivo JPG, PNG o WebP.")
    height, width = frame.shape[:2]
    if width * height > 30_000_000:
        raise ValueError("La imagen supera los 30 megapíxeles. Reduce su tamaño antes de analizarla.")
    if max(width, height) > 1920:
        scale = 1920 / max(width, height)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    with create_landmarker(vision.RunningMode.IMAGE) as detector:
        result = detector.detect(rgb_image(frame))
    return response("image", width, height, [serialize(result, 0, 0)])


def extract_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("No se pudo abrir el video. Prueba un MP4 con códec H.264.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError("El video no informa una duración válida.")
        if count / fps > 60:
            raise ValueError("Para esta prueba, selecciona un video de hasta 60 segundos.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, index, last_decoded, next_sample = [], 0, -1, 0
        fallback = False
        with create_landmarker(vision.RunningMode.VIDEO) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = round(pts) if math.isfinite(pts) and pts >= 0 else -1
                if timestamp <= last_decoded:
                    timestamp = max(last_decoded + 1, round(index * 1000 / fps))
                    fallback = True
                last_decoded = timestamp
                if timestamp > 60_000:
                    raise ValueError("El video supera los 60 segundos.")
                if timestamp >= next_sample:
                    if max(frame.shape[:2]) > 1920:
                        scale = 1920 / max(frame.shape[:2])
                        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                    result = detector.detect_for_video(rgb_image(frame), timestamp)
                    frames.append(serialize(result, timestamp, index))
                    next_sample = timestamp + SAMPLE_MS
                index += 1
        if not frames:
            raise ValueError("El archivo no contiene fotogramas que se puedan leer.")
        return response("video", width, height, frames, max(round(count * 1000 / fps), last_decoded),
                        "opencv_pts_with_fps_fallback" if fallback else "opencv_pts")
    finally:
        capture.release()
