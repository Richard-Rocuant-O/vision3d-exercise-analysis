"""Adapt the pose baseline with labeled clips; hold out whole squat videos."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pose import extract_video
from app.exercises import MODEL_PATH, features, neighbors, forest_scores
from train_exercise_classifier import fit, export_forest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'backend/config/exercise-training-videos.json'
TARGET = ROOT / 'data/models/exercise_video_classifier.npz'


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with np.load(MODEL_PATH, allow_pickle=False) as source:
        baseline = {key: source[key] for key in ('features', 'labels', 'thresholds', 'radii')}
        metadata = json.loads(str(source['metadata']))
    cache_dir = ROOT / 'data/models/video-training'
    cache_dir.mkdir(parents=True, exist_ok=True)
    training_hashes, heldout_hashes, loaded = set(), set(), []
    for partition in ('training', 'holdout'):
        for item in manifest[partition]:
            path = ROOT / item['path']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (training_hashes if partition == 'training' else heldout_hashes).add(digest)
            cached = cache_dir / (digest + '.json')
            if cached.is_file():
                motion = json.loads(cached.read_text(encoding='utf-8'))
            else:
                print(f"Extrayendo {partition}: {path.name}", flush=True)
                motion = extract_video(path, 0, 15)
                cached.write_text(json.dumps(motion, allow_nan=False), encoding='utf-8')
            indices, vectors = [], []
            next_sample = -1
            for i, frame in enumerate(motion['frames']):
                if frame['timestamp_ms'] < next_sample:
                    continue
                next_sample = frame['timestamp_ms'] + 100
                vector = features(frame['landmarks'], motion['width'], motion['height'])
                if vector is not None:
                    indices.append(i)
                    vectors.append(vector)
            loaded.append((partition, item, digest, motion, np.array(vectors), indices))
    if training_hashes & heldout_hashes:
        raise ValueError('Un video aparece en entrenamiento y evaluación.')
    names = metadata['classes']
    train_x = [baseline['features']]
    train_y = [baseline['labels']]
    for partition, item, digest, motion, x, indices in loaded:
        if partition == 'training' and len(x):
            train_x.append(x)
            train_y.append(np.full(len(x), names.index(item['label']), dtype=np.int64))
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    estimator = fit(x, y)
    forest = export_forest(estimator)
    report = []
    for partition, item, digest, motion, sample, indices in loaded:
        if partition != 'holdout':
            continue
        if not len(sample):
            report.append({'video': item['path'], 'sha256': digest, 'expected': item['label'],
                           'usable_samples': 0, 'accepted_samples': 0, 'correct_accepted': 0,
                           'status': 'not_evaluable'})
            continue
        predicted = forest_scores(sample, forest)
        _, proximity = neighbors(sample, x, y, len(names))
        best = predicted.argmax(1)
        support = predicted.max(1)
        margin = support - np.sort(predicted, axis=1)[:, -2]
        accepted = ((support >= baseline['thresholds'][best]) & (margin >= .15)
                    & (proximity[np.arange(len(best)), best] <= baseline['radii'][best]))
        expected = names.index(item['label'])
        report.append({'video': item['path'], 'sha256': digest, 'expected': item['label'],
                       'usable_samples': len(sample), 'top1_accuracy': float((best == expected).mean()),
                       'accepted_samples': int(accepted.sum()),
                       'correct_accepted': int((accepted & (best == expected)).sum()),
                       'predicted_counts': {names[i]: int((best == i).sum()) for i in set(best.tolist())}})
    metadata = dict(metadata, algorithm='RandomForest + labeled video examples',
                    training_video_manifest=manifest['training'], training_video_hashes=sorted(training_hashes),
                    video_holdout=report, training_examples_after_refit=len(y),
                    validation_note='Static image holdout belongs to baseline BEFORE video adaptation; video test includes only three squat clips, first 15 seconds. Not a general video accuracy metric.')
    np.savez_compressed(TARGET, features=x, labels=y, thresholds=baseline['thresholds'], radii=baseline['radii'],
                        **forest, metadata=np.array(json.dumps(metadata)))
    TARGET.with_suffix('.report.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(TARGET), 'training_examples': len(y), 'heldout_videos': report}))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
