"""Optional temporal lifting. Never overwrite MediaPipe's image/world estimates.

The adapter follows upstream infer_wild.py (--pixel normalization, flip ensemble).
Output is hip-relative, in model units, NOT measured meters. Input visibility is
retained as an observability gate, not advertised as 3D confidence.
"""
from functools import lru_cache, partial
import hashlib
import importlib.util
import math
import os
import pickle
from pathlib import Path
from threading import RLock

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "motionbert-lite.bin"
MODEL_SHA256 = "9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f"
CHECKPOINT = "FT_MB_lite_MB_ft_h36m_global_lite"
WINDOW = 243
OVERLAP = 32
QUALITY_THRESHOLD = 0.5
NAMES = ["pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
         "spine", "neck", "nose", "head", "left_shoulder", "left_elbow", "left_wrist",
         "right_shoulder", "right_elbow", "right_wrist"]
FLIP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
_lock = RLock()


class MotionBertUnavailable(RuntimeError):
    pass


def status():
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "Falta PyTorch. Sigue la preparación opcional de MotionBERT en el README."}
    if not MODEL_PATH.is_file():
        return {"available": False, "reason": "Faltan los pesos. Ejecuta scripts/download_motionbert_model.py desde backend."}
    return {"available": True, "reason": None}


@lru_cache(maxsize=1)
def _load_model():
    state = status()
    if not state["available"]:
        raise MotionBertUnavailable(state["reason"])
    try:
        import torch
        from .vendor.motionbert.DSTformer import DSTformer
    except (ImportError, OSError) as exc:
        raise MotionBertUnavailable("No se pudo cargar PyTorch. Revisa la instalación opcional del README.") from exc
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise MotionBertUnavailable("Los pesos no coinciden con el modelo oficial. Vuelve a ejecutar su descarga.")
    device_name = os.getenv("VISION3D_MOTIONBERT_DEVICE", "cpu")
    if device_name not in ("cpu", "cuda") or (device_name == "cuda" and not torch.cuda.is_available()):
        raise MotionBertUnavailable("El dispositivo MotionBERT no está disponible. Usa cpu o instala PyTorch con CUDA compatible.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=4, maxlen=WINDOW, num_joints=17,
                      norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        weights = checkpoint["model_pos"]
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        model.load_state_dict(weights, strict=True)
        model.to(device_name).eval()
    except (RuntimeError, KeyError, ValueError, pickle.UnpicklingError) as exc:
        raise MotionBertUnavailable("No se pudo cargar el checkpoint de pose MotionBERT. Revisa sus pesos y versión de PyTorch.") from exc
    return model, torch, device_name


def _point(p):
    if not p or not all(math.isfinite(p.get(k, float("nan"))) for k in ("x", "y")):
        return None
    confidence = min(p.get("visibility", 0), p.get("presence", 0))
    confidence = float(np.clip(confidence, 0, 1)) if math.isfinite(confidence) else 0.0
    return np.array([p["x"], p["y"], confidence], dtype=np.float32)


def _midpoint(a, b):
    if a is None or b is None:
        return None
    result = (a + b) / 2
    result[2] = min(a[2], b[2])
    return result


def map_frame(landmarks, width, height):
    """MediaPipe33 -> H36M17; synthesized centers keep the weakest confidence.

    MediaPipe has no head-top joint. Ear midpoint is an explicit head-center
    proxy; it is never presented as an observed crown position.
    """
    source = [_point(landmarks[i]) if i < len(landmarks) else None for i in range(33)]
    pelvis = _midpoint(source[23], source[24])
    neck = _midpoint(source[11], source[12])
    mapped = [pelvis, source[24], source[26], source[28], source[23], source[25], source[27],
              _midpoint(pelvis, neck), neck, source[0], _midpoint(source[7], source[8]),
              source[11], source[13], source[15], source[12], source[14], source[16]]
    result = np.zeros((17, 3), dtype=np.float32)
    for i, p in enumerate(mapped):
        if p is not None:
            result[i, :2] = (p[:2] * [width, height] - np.array([width, height]) / 2) / (min(width, height) / 2)
            result[i, 2] = p[2]
    return result


def window_starts(length):
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, WINDOW - OVERLAP))
    if starts[-1] != length - WINDOW:
        starts.append(length - WINDOW)
    return starts


def segments(inputs, times):
    """Split missing-person intervals and timestamp gaps; never fill them in."""
    differences = np.diff(times)
    expected = float(np.median(differences)) if len(differences) else 33.0
    gap_limit = max(100.0, expected * 2.5)
    start = None
    for i, pose in enumerate(inputs):
        usable = pose[0, 2] > 0 and pose[8, 2] > 0 and np.count_nonzero(pose[:, 2] >= QUALITY_THRESHOLD) >= 6
        gap = i > 0 and times[i] - times[i - 1] > gap_limit
        if start is not None and (not usable or gap):
            yield start, i
            start = None
        if usable and start is None:
            start = i
    if start is not None:
        yield start, len(inputs)


def _predict(inputs, model, torch, device):
    accumulated = np.zeros(inputs.shape, dtype=np.float32)
    weights = np.zeros((len(inputs), 1, 1), dtype=np.float32)
    with torch.inference_mode():
        for start in window_starts(len(inputs)):
            batch = inputs[start:start + WINDOW]
            tensor = torch.from_numpy(batch[None]).to(device)
            flipped = tensor[:, :, FLIP].clone()
            flipped[..., 0] *= -1
            first = model(tensor)
            second = model(flipped)[:, :, FLIP]
            second[..., 0] *= -1
            prediction = ((first + second) / 2)[0].cpu().numpy()
            # Cosmetic root centering; raw checkpoint output has no metric guarantee.
            prediction -= prediction[:, :1, :]
            ramp = np.minimum(np.arange(len(batch)) + 1, np.arange(len(batch), 0, -1))
            ramp = np.minimum(ramp, OVERLAP).astype(np.float32)[:, None, None]
            accumulated[start:start + len(batch)] += prediction * ramp
            weights[start:start + len(batch)] += ramp
    return accumulated / weights


def lift(motion):
    if motion["input_type"] != "video":
        raise ValueError("MotionBERT está habilitado para videos; las fotos usan MediaPipe.")
    frames = motion["frames"]
    inputs = np.stack([map_frame(frame["landmarks"], motion["width"], motion["height"]) for frame in frames])
    times = [frame["timestamp_ms"] for frame in frames]
    for frame in frames:
        frame["motionbert_landmarks"] = []
    with _lock:
        model, torch, device = _load_model()
        for start, end in segments(inputs, times):
            prediction = _predict(inputs[start:end], model, torch, device)
            if not np.isfinite(prediction).all():
                raise ValueError("MotionBERT produjo coordenadas no finitas.")
            for i, points in enumerate(prediction, start):
                root_confidence = inputs[i, 0, 2]
                frames[i]["motionbert_landmarks"] = [
                    dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                         visibility=float(min(inputs[i, j, 2], root_confidence)),
                         presence=float(min(inputs[i, j, 2], root_confidence)))
                    if inputs[i, j, 2] > 0 else None for j, p in enumerate(points)
                ]
    # One constant display scale for the complete clip, never a per-frame zoom.
    torso_lengths = [float(np.linalg.norm(np.array([p[8][k] - p[0][k] for k in ("x", "y", "z")])))
                     for frame in frames if (p := frame["motionbert_landmarks"]) and p[0] and p[8]
                     and min(p[0]["visibility"], p[8]["visibility"]) >= QUALITY_THRESHOLD]
    median = float(np.median(torso_lengths)) if torso_lengths else 0
    motion["schema_version"] = "2.1"
    motion["selected_estimator"] = "motionbert"
    motion["coordinate_systems"]["motionbert_landmarks"] = "hip-centered model units (NOT meters); x right, y down, z camera depth"
    motion["motionbert"] = {
        "name": "MotionBERT-Lite", "checkpoint": CHECKPOINT, "sha256": MODEL_SHA256,
        "landmark_names": NAMES, "device": device, "torch_version": torch.__version__,
        "window_frames": WINDOW, "overlap_frames": OVERLAP,
        "normalization": "image center / half of shortest image dimension (upstream --pixel)",
        "confidence_source": "minimum MediaPipe 2D visibility/presence; not calibrated 3D confidence",
        "head_mapping": "ear midpoint proxy, not observed head top",
        "render_scale": 0.5 / median if median > 1e-6 else 1.0,
        "render_scale_units": "generic avatar units; torso median displayed as 0.5, not measured meters",
        "detected_frames": sum(bool(frame["motionbert_landmarks"]) for frame in frames),
    }
    return motion
