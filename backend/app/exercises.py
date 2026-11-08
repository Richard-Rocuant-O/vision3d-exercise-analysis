"""Experimental posture classifier. Exercise identity is separate from technique."""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "data/models/exercise_classifier.npz"
VIDEO_MODEL_PATH = MODEL_PATH.with_name('exercise_video_classifier.npz')
FEATURE_VERSION = "pose2d-torso-v1"
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
LABELS_ES = {
    "Squat": "Sentadilla", "Bent-over row": "Remo inclinado", "Chest dips": "Fondos de pecho",
    "Glute bridge": "Puente de glúteos", "Lunges": "Zancadas", "Plank": "Plancha",
    "Pull-up": "Dominadas", "Push-up": "Flexiones", "Shoulder press": "Press de hombros",
}


def features(landmarks, width, height):
    if len(landmarks) != 33 or width <= 0 or height <= 0:
        return None
    coordinates, visible = [], []
    for i in JOINTS:
        p = landmarks[i]
        valid = bool(p and all(math.isfinite(p.get(k, float('nan'))) for k in ('x', 'y'))
                     and min(p.get('visibility', 0), p.get('presence', p.get('visibility', 0))) >= .5)
        visible.append(valid)
        coordinates.append([p['x'] * width, p['y'] * height] if valid else [0, 0])
    coordinates, visible = np.array(coordinates, dtype=np.float32), np.array(visible)
    # At least one observable shoulder/hip, and most of the body. Never fill hidden joints.
    if visible.sum() < 9 or not visible[[1, 2]].any() or not visible[[7, 8]].any():
        return None
    shoulders = coordinates[[i for i in (1, 2) if visible[i]]].mean(axis=0)
    hips = coordinates[[i for i in (7, 8) if visible[i]]].mean(axis=0)
    scale = float(np.linalg.norm(shoulders - hips))
    if scale < 5:
        return None
    normalized = np.clip((coordinates - hips) / scale, -5, 5)
    normalized[~visible] = 0
    # Missingness is part of the input; zero coordinate alone is ambiguous.
    return np.column_stack((normalized, visible.astype(np.float32) * .5)).flatten()


def mirrored(vectors):
    result = np.asarray(vectors).reshape(-1, len(JOINTS), 3)[:, MIRROR].copy()
    result[:, :, 0] *= -1
    return result.reshape(-1, len(JOINTS) * 3)


def neighbors(query, training, training_labels, classes, k=7):
    """Distance-weighted kNN; a mirrored copy cannot vote twice for one image."""
    query, training = np.asarray(query, dtype=np.float32), np.asarray(training, dtype=np.float32)
    alternate = mirrored(training)
    scores, distances = [], []
    for start in range(0, len(query), 128):
        block = query[start:start + 128]
        squared = (block * block).sum(1)[:, None] + (training * training).sum(1)[None, :] - 2 * block @ training.T
        flipped = (block * block).sum(1)[:, None] + (alternate * alternate).sum(1)[None, :] - 2 * block @ alternate.T
        distance = np.sqrt(np.maximum(0, np.minimum(squared, flipped)) / training.shape[1])
        indices = np.argsort(distance, axis=1)[:, :min(k, len(training))]
        local = np.take_along_axis(distance, indices, axis=1)
        weights = 1 / (local + .05)
        votes = np.zeros((len(block), classes), dtype=np.float32)
        for j in range(indices.shape[1]):
            votes[np.arange(len(block)), training_labels[indices[:, j]]] += weights[:, j]
        votes /= weights.sum(1)[:, None]
        class_distances = np.column_stack([distance[:, training_labels == label].min(1) for label in range(classes)])
        scores.append(votes)
        distances.append(class_distances)
    return np.concatenate(scores), np.concatenate(distances)


def forest_scores(vectors, model):
    """Portable tree inference. No pickle or scikit-learn needed in the API."""
    vectors = np.asarray(vectors, dtype=np.float32)
    result = np.zeros((len(vectors), model['tree_values'].shape[1]), dtype=np.float32)
    for root in model['tree_roots']:
        nodes = np.full(len(vectors), root, dtype=np.int64)
        for _ in range(64):
            active = model['tree_left'][nodes] >= 0
            if not active.any():
                break
            rows = np.flatnonzero(active)
            branch = nodes[rows]
            left = vectors[rows, model['tree_features'][branch]] <= model['tree_thresholds'][branch]
            nodes[rows] = np.where(left, model['tree_left'][branch], model['tree_right'][branch])
        else:
            raise ValueError('El modelo de reconocimiento contiene árboles inválidos.')
        result += model['tree_values'][nodes]
    return result / len(model['tree_roots'])


@lru_cache(maxsize=1)
def _load(path, signature):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['feature_version'] != FEATURE_VERSION:
            raise ValueError('El clasificador necesita volver a entrenarse con el formato actual.')
        keys = ('features', 'labels', 'thresholds', 'radii', 'tree_roots', 'tree_left', 'tree_right',
                'tree_features', 'tree_thresholds', 'tree_values')
        return {key: archive[key] for key in keys}, metadata


def load_model():
    path = VIDEO_MODEL_PATH if VIDEO_MODEL_PATH.is_file() else MODEL_PATH
    if not path.is_file():
        return None
    stat = path.stat()
    return _load(str(path), (stat.st_mtime_ns, stat.st_size))


def attach(motion):
    """Precompute causal labels; the frontend displays the label at the playhead."""
    loaded = load_model()
    if loaded is None:
        for frame in motion['frames']:
            frame['exercise'] = {'status': 'model_missing', 'label': None, 'name': None}
        motion['exercise_classifier'] = {'available': False}
        return motion
    model, metadata = loaded
    usable, indices = [], []
    for i, frame in enumerate(motion['frames']):
        vector = features(frame['landmarks'], motion['width'], motion['height'])
        frame['exercise'] = {'status': 'not_evaluable', 'label': None, 'name': None}
        if vector is not None:
            usable.append(vector)
            indices.append(i)
    if usable:
        _, distances = neighbors(usable, model['features'], model['labels'], len(metadata['classes']))
        scores = forest_scores(usable, model)
        smooth, previous, pending_label, since = None, None, None, 0
        for row, i in enumerate(indices):
            frame = motion['frames'][i]
            timestamp = frame['timestamp_ms']
            if previous is None or i != previous[0] + 1 or timestamp - previous[1] > 200:
                smooth, pending_label, since = scores[row].copy(), None, timestamp
            else:
                alpha = 1 - math.exp(-(timestamp - previous[1]) / 500)
                smooth = smooth * (1 - alpha) + scores[row] * alpha
            label = int(np.argmax(smooth))
            score = float(smooth[label])
            margin = float(score - np.partition(smooth, -2)[-2])
            supported = (score >= model['thresholds'][label] and margin >= .15
                         and distances[row, label] <= model['radii'][label]
                         and int(np.argmax(scores[row])) == label)
            if not supported or pending_label != label:
                pending_label, since = label if supported else None, timestamp
            accepted = supported and (motion['input_type'] == 'image' or timestamp - since >= 240)
            name = metadata['classes'][label]
            frame['exercise'] = {
                'status': 'identified' if accepted else 'uncertain', 'label': name if accepted else None,
                'name': LABELS_ES.get(name, name) if accepted else None,
                'candidate_name': LABELS_ES.get(name, name) if score >= .45 else None,
                'candidate_label': name,
                'support': score, 'support_kind': 'forest vote, NOT calibrated probability',
            }
            previous = (i, timestamp)
    motion['exercise_classifier'] = {
        'available': True, 'name': metadata.get('algorithm', 'pose Random Forest') + ' (experimental)', 'feature_version': FEATURE_VERSION,
        'dataset_sha256': metadata['dataset_sha256'], 'classes': metadata['classes'],
        'validation': {'kind': 'heldout_squat_videos', 'videos': metadata['video_holdout']} if 'video_holdout' in metadata else metadata['holdout'],
        'scope': metadata.get('validation_note', 'image-level holdout; not validated on unseen people or video sessions'),
        'temporal_filter': 'causal 500ms EMA, 240ms stability; no future frames',
    }
    return motion
