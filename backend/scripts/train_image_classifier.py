"""Frozen visual encoder + supervised exercise head, with an untouched holdout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dataset import decode_base64
from app.exercise_images import (ENCODER_PATH, HEAD_PATH, VERSION, download_encoder, letterbox, embeddings, scores, distances)
from train_exercise_classifier import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('--download-encoder', action='store_true')
    args = parser.parse_args()
    if args.download_encoder:
        download_encoder()
    dataset_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    encoder_hash = hashlib.sha256(ENCODER_PATH.read_bytes()).hexdigest()
    csv.field_size_limit(100 * 1024 * 1024)
    with args.csv.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len({row['image_hash'] for row in rows}) != len(rows):
        raise ValueError('Agrupa imágenes duplicadas antes de separar entrenamiento y evaluación.')
    names = sorted({row['label'] for row in rows})
    y = np.array([names.index(row['label']) for row in rows])
    folds = np.array([int(row['validation_fold']) for row in rows])
    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = HEAD_PATH.with_name('exercise_image_features.npz')
    x = None
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            if str(archive['dataset_hash']) == dataset_hash and str(archive['version']) == VERSION and str(archive['encoder_hash']) == encoder_hash:
                x = archive['features']
    if x is None:
        batches = []
        for start in range(0, len(rows), 64):
            images = []
            for row in rows[start:start + 64]:
                raw = np.frombuffer(decode_base64(row['base64encoded']), dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Una imagen del CSV no se pudo decodificar.')
                images.append(letterbox(image))
            batches.append(embeddings(images))
            if start % 512 == 0:
                print(f'Codificadas {min(start + 64, len(rows))}/{len(rows)} imágenes', flush=True)
        x = np.concatenate(batches)
        np.savez_compressed(cache, features=x, dataset_hash=np.array(dataset_hash), version=np.array(VERSION), encoder_hash=np.array(encoder_hash))
    development, holdout = folds != 5, folds == 5
    fit = lambda mask: LogisticRegression(C=10, class_weight='balanced', max_iter=2000, random_state=42).fit(x[mask], y[mask])
    for mask in (development, holdout):
        if set(y[mask]) != set(range(len(names))):
            raise ValueError('Faltan clases en entrenamiento o evaluación.')
    out_scores, out_distances = np.zeros((len(y), len(names))), np.zeros((len(y), len(names)))
    for fold in (1, 2, 3, 4):
        training, validation = development & (folds != fold), folds == fold
        model = fit(training)
        out_scores[validation] = model.predict_proba(x[validation])
        out_distances[validation] = distances(x[validation], x[training], y[training], len(names))
    thresholds, radii = np.full(len(names), 1.1), np.zeros(len(names))
    predictions = out_scores.argmax(1)
    for label in range(len(names)):
        chosen = development & (predictions == label)
        correct = chosen & (y == label)
        if correct.sum() < 15:
            continue
        radius = float(np.quantile(out_distances[correct, label], .95))
        for threshold in (.4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95):
            accepted = chosen & (out_scores[:, label] >= threshold) & (out_distances[:, label] <= radius)
            if accepted.sum() >= 15 and (y[accepted] == label).mean() >= .85:
                thresholds[label], radii[label] = threshold, radius
                break
    model = fit(development)
    heldout_scores = model.predict_proba(x[holdout])
    np.testing.assert_allclose(scores(x[holdout], model.coef_, model.intercept_), heldout_scores, atol=1e-7)
    report = metrics(heldout_scores, distances(x[holdout], x[development], y[development], len(names)), y[holdout], thresholds, radii, names)
    metadata = {'feature_version': VERSION, 'classes': names, 'dataset_sha256': dataset_hash, 'encoder_sha256': encoder_hash,
                'training_examples_after_refit': len(y), 'holdout': report,
                'threshold_selection': 'folds 1-4 only; development target .85 precision, >=15 cases',
                'holdout_split': 'existing fold 5; no person/session identifiers',
                'limitations': 'No correct/incorrect technique labels; no video/negative-class benchmark; image holdout can share near-duplicates and people.'}
    model = fit(np.ones(len(y), dtype=bool))
    np.savez_compressed(HEAD_PATH, features=x, labels=y, coefficients=model.coef_, intercept=model.intercept_,
                        thresholds=thresholds, radii=radii, metadata=np.array(json.dumps(metadata)))
    HEAD_PATH.with_suffix('.report.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(HEAD_PATH), 'training_examples': len(y), 'thresholds': thresholds.tolist(), 'holdout': report}))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
