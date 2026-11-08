"""Optional image-based exercise recognition, sampled and displayed causally."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from .exercises import LABELS_ES

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = ROOT / 'backend/models/mobilenet_v3_small-047dcff4.pth'
ENCODER_URL = 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
HEAD_PATH = ROOT / 'data/models/exercise_image_classifier.npz'
VERSION = 'mobilenet-v3-small-imagenet1k-v1-letterbox224-l2'
SAMPLE_MS = 200
_lock = RLock()


def download_encoder():
    import torch
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.load_state_dict_from_url(ENCODER_URL, model_dir=str(ENCODER_PATH.parent),
                                       check_hash=True, weights_only=True, progress=True)


@lru_cache(maxsize=1)
def encoder():
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError('Faltan los pesos del reconocedor visual; sigue docs/exercise-feedback.md.')
    import torch
    from torchvision.models import mobilenet_v3_small
    if not hashlib.sha256(ENCODER_PATH.read_bytes()).hexdigest().startswith('047dcff4'):
        raise ValueError('La huella del codificador visual no coincide con el modelo oficial.')
    torch.set_num_threads(4)
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu', weights_only=True), strict=True)
    model.classifier = torch.nn.Identity()
    return model.eval(), torch


def letterbox(image):
    height, width = image.shape[:2]
    ratio = 224 / max(width, height)
    w, h = max(1, round(width * ratio)), max(1, round(height * ratio))
    result = np.full((224, 224, 3), 127, dtype=np.uint8)
    x, y = (224 - w) // 2, (224 - h) // 2
    result[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return result


def embeddings(images):
    model, torch = encoder()
    result = []
    with _lock, torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images[start:start + 32]])
            tensor = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).float() / 255
            tensor = (tensor - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
            values = model(tensor).cpu().numpy()
            result.append(values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8))
    return np.concatenate(result)


def scores(vectors, coefficients, intercept):
    logits = vectors @ coefficients.T + intercept
    logits -= logits.max(1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(1, keepdims=True)


def distances(vectors, training, labels, classes):
    result = []
    for start in range(0, len(vectors), 64):
        block = vectors[start:start + 64]
        squared = ((block * block).sum(1)[:, None] + (training * training).sum(1)[None, :] - 2 * block @ training.T)
        distance = np.sqrt(np.maximum(0, squared))
        result.append(np.column_stack([distance[:, labels == label].min(1) for label in range(classes)]))
    return np.concatenate(result)


@lru_cache(maxsize=1)
def _head(signature):
    with np.load(HEAD_PATH, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['feature_version'] != VERSION:
            raise ValueError('El reconocedor visual requiere volver a entrenarse.')
        return {key: archive[key] for key in ('features', 'labels', 'coefficients', 'intercept', 'thresholds', 'radii')}, metadata


def sample_images(motion, path):
    if motion['input_type'] == 'image':
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('No se pudo abrir la imagen para reconocer el ejercicio.')
        return [0], [letterbox(image)]
    selected, next_sample = [], -1
    for i, frame in enumerate(motion['frames']):
        if frame['timestamp_ms'] >= next_sample:
            selected.append(i)
            next_sample = frame['timestamp_ms'] + SAMPLE_MS
    targets = {motion['frames'][i]['frame_index']: i for i in selected}
    capture = cv2.VideoCapture(str(path))
    try:
        first, last = min(targets), max(targets)
        if not capture.isOpened() or (first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first)):
            raise ValueError('No se pudo leer el tramo para reconocer el ejercicio.')
        index = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if index != first:
            raise ValueError('No se pudo sincronizar el reconocimiento con el video original.')
        indices, images = [], []
        while index <= last:
            ok, image = capture.read()
            if not ok:
                break
            if index in targets:
                indices.append(targets[index])
                images.append(letterbox(image))
            index += 1
        if not images:
            raise ValueError('El tramo no contiene imágenes utilizables para reconocimiento.')
        return indices, images
    finally:
        capture.release()


def attach(motion, path):
    if not HEAD_PATH.is_file():
        return False
    stat = HEAD_PATH.stat()
    model, metadata = _head((stat.st_mtime_ns, stat.st_size))
    if not ENCODER_PATH.is_file() or hashlib.sha256(ENCODER_PATH.read_bytes()).hexdigest() != metadata['encoder_sha256']:
        raise ValueError('Falta el codificador visual con el que se entrenó el clasificador.')
    indices, images = sample_images(motion, path)
    vectors = embeddings(images)
    probabilities = scores(vectors, model['coefficients'], model['intercept'])
    proximity = distances(vectors, model['features'], model['labels'], len(metadata['classes']))
    assign_predictions(motion, indices, probabilities, proximity, model, metadata['classes'])
    motion['exercise_classifier'] = {
        'available': True, 'name': 'MobileNetV3-Small + supervised linear classifier (experimental)',
        'feature_version': VERSION, 'dataset_sha256': metadata['dataset_sha256'], 'classes': metadata['classes'],
        'validation': metadata['holdout'], 'scope': 'image-level holdout; not validated on unseen people or video sessions',
        'sample_interval_ms': SAMPLE_MS, 'temporal_filter': 'causal 500ms EMA and 240ms stability; no future samples',
    }
    return True


def assign_predictions(motion, indices, probabilities, proximity, model, names):
    """Hold only preceding samples; missing detections reset the temporal state."""
    selected = {index: row for row, index in enumerate(indices)}
    smooth, last_sample, pending, since, latest = None, None, None, 0, None
    for i, frame in enumerate(motion['frames']):
        timestamp = frame['timestamp_ms']
        visible = sum(bool(p and all(math.isfinite(p.get(k, float('nan'))) for k in ('x', 'y', 'visibility', 'presence'))
                           and min(p['visibility'], p['presence']) >= .5) for p in frame['landmarks'])
        if visible < 6:
            frame['exercise'] = {'status': 'not_evaluable', 'label': None, 'name': None}
            smooth, last_sample, pending, latest = None, None, None, None
            continue
        if i in selected:
            row = selected[i]
            if smooth is None or timestamp - last_sample > 600:
                smooth, pending, since = probabilities[row].copy(), None, timestamp
            else:
                alpha = 1 - math.exp(-(timestamp - last_sample) / 500)
                smooth = smooth * (1 - alpha) + probabilities[row] * alpha
            label = int(np.argmax(smooth))
            support = float(smooth[label])
            margin = support - float(np.partition(smooth, -2)[-2])
            supported = (support >= model['thresholds'][label] and margin >= .15
                         and proximity[row, label] <= model['radii'][label]
                         and int(np.argmax(probabilities[row])) == label)
            if not supported or pending != label:
                pending, since = label if supported else None, timestamp
            accepted = supported and (motion['input_type'] == 'image' or timestamp - since >= 240)
            name = names[label]
            latest = {'status': 'identified' if accepted else 'uncertain', 'label': name if accepted else None,
                      'name': LABELS_ES.get(name, name) if accepted else None,
                      'candidate_name': LABELS_ES.get(name, name) if support >= .45 else None,
                      'candidate_label': name,
                      'support': support, 'support_kind': 'classifier score, NOT calibrated probability',
                      'sample_timestamp_ms': timestamp}
            last_sample = timestamp
        frame['exercise'] = dict(latest) if latest and timestamp - last_sample <= 600 else {'status': 'uncertain', 'label': None, 'name': None}
