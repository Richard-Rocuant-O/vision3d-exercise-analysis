"""MediaPipe adapter. Image coordinates and world estimates stay separate."""
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
# Decode and estimate every frame. Temporal estimators need continuous input.
SAMPLE_MS = 0
MAX_VIDEO_FRAMES = 3600
MAX_SEGMENT_SECONDS = 60
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


def extract_video(path, start_seconds=0.0, duration_seconds=MAX_SEGMENT_SECONDS):
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("El inicio del tramo debe ser un número igual o mayor que cero.")
    if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= MAX_SEGMENT_SECONDS:
        raise ValueError("Elige un tramo de más de 0 y hasta 60 segundos.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("No se pudo abrir el video. Prueba un MP4 con códec H.264.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError("El video no informa una duración válida.")
        source_duration_ms = round(count * 1000 / fps)
        start_ms = round(start_seconds * 1000)
        end_ms = start_ms + round(duration_seconds * 1000)
        if start_ms >= source_duration_ms:
            raise ValueError("El inicio del tramo queda fuera del video. Elige un segundo anterior al final.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, index, last_decoded = [], 0, -1
        fallback = False
        if start_ms:
            if not capture.set(cv2.CAP_PROP_POS_MSEC, start_ms):
                raise ValueError("No se pudo buscar ese tramo. Prueba desde el segundo 0 o usa un MP4 H.264.")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if math.isfinite(position) and position >= 0:
                index = round(position)
        with create_landmarker(vision.RunningMode.VIDEO) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = round(pts) if math.isfinite(pts) and pts >= 0 else -1
                if timestamp <= last_decoded or (not frames and start_ms and index > 0 and timestamp == 0):
                    timestamp = max(last_decoded + 1, round(index * 1000 / fps))
                    fallback = True
                last_decoded = timestamp
                if timestamp >= end_ms:
                    break
                if timestamp < start_ms:
                    index += 1
                    continue
                if len(frames) >= MAX_VIDEO_FRAMES:
                    raise ValueError("El tramo supera los 3600 fotogramas. Elige una duración menor o reduce sus FPS.")
                if max(frame.shape[:2]) > 1920:
                    scale = 1920 / max(frame.shape[:2])
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                result = detector.detect_for_video(rgb_image(frame), timestamp)
                frames.append(serialize(result, timestamp, index))
                index += 1
        if not frames:
            raise ValueError("El archivo no contiene fotogramas que se puedan leer.")
        actual_end_ms = min(end_ms, frames[-1]["timestamp_ms"] + max(1, round(1000 / fps)))
        motion = response("video", width, height, frames, max(source_duration_ms, actual_end_ms),
                          "opencv_pts_with_fps_fallback" if fallback else "opencv_pts")
        motion["source_fps"] = fps
        motion["sampling"] = "every_decoded_frame"
        motion["analysis_window"] = {
            "start_ms": frames[0]["timestamp_ms"], "end_ms": actual_end_ms,
            "requested_start_ms": start_ms, "requested_duration_ms": round(duration_seconds * 1000),
            "source_duration_ms": source_duration_ms,
        }
        return motion
    finally:
        capture.release()
