"""Train a small reproducible baseline, with threshold selection separate from holdout."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.exercises import FEATURE_VERSION, MODEL_PATH, features, neighbors, mirrored, forest_scores


def parse_pose(value):
    if not value or value.strip() in ('[]', 'nan', 'None'):
        return []
    values = ast.literal_eval(value)
    if not isinstance(values, list) or len(values) != 33:
        return []
    result = []
    for p in values:
        if isinstance(p, str):
            p = {key: float(number) for key, number in re.findall(r'(x|y|z|visibility):\s*([-+0-9.eE]+)', p)}
        result.append(p if isinstance(p, dict) else None)
    return result


def fit(x, y):
    return RandomForestClassifier(n_estimators=240, min_samples_leaf=2, max_depth=24,
                                  class_weight='balanced', random_state=42, n_jobs=4).fit(
                                      np.concatenate((x, mirrored(x))), np.tile(y, 2))


def export_forest(model):
    roots, left, right, indices, thresholds, values = [], [], [], [], [], []
    offset = 0
    for estimator in model.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        left.append(np.where(tree.children_left >= 0, tree.children_left + offset, -1))
        right.append(np.where(tree.children_right >= 0, tree.children_right + offset, -1))
        indices.append(tree.feature)
        thresholds.append(tree.threshold)
        probabilities = tree.value[:, 0, :]
        values.append(probabilities / probabilities.sum(1, keepdims=True))
        offset += tree.node_count
    return {'tree_roots': np.array(roots, dtype=np.int64), 'tree_left': np.concatenate(left),
            'tree_right': np.concatenate(right), 'tree_features': np.concatenate(indices),
            'tree_thresholds': np.concatenate(thresholds), 'tree_values': np.concatenate(values).astype(np.float32)}


def metrics(scores, distances, truth, thresholds, radii, names):
    prediction = scores.argmax(1)
    confidence = scores.max(1)
    sorted_scores = np.sort(scores, axis=1)
    accepted = ((confidence >= thresholds[prediction]) & (distances[np.arange(len(truth)), prediction] <= radii[prediction])
                & (sorted_scores[:, -1] - sorted_scores[:, -2] >= .15))
    confusion = np.zeros((len(names), len(names)), dtype=int)
    for expected, actual in zip(truth, prediction):
        confusion[expected, actual] += 1
    per_class = {}
    for label, name in enumerate(names):
        chosen = accepted & (prediction == label)
        per_class[name] = {'examples': int((truth == label).sum()), 'accepted_predictions': int(chosen.sum()),
                           'precision_among_accepted': float((prediction[chosen] == truth[chosen]).mean()) if chosen.any() else None}
    return {'examples': len(truth), 'accuracy_before_abstention': float((prediction == truth).mean()),
            'coverage': float(accepted.mean()),
            'accuracy_among_accepted': float((prediction[accepted] == truth[accepted]).mean()) if accepted.any() else None,
            'confusion_matrix': confusion.tolist(), 'per_class': per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    vectors, labels, folds, hashes, discarded = [], [], [], [], Counter()
    csv.field_size_limit(100 * 1024 * 1024)
    with args.csv.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                vector = features(parse_pose(row['poseLandmarks']), float(row['imageWidth']), float(row['imageHeight']))
                if vector is None:
                    discarded['missing_or_unreliable_pose'] += 1
                    continue
                fold = int(row['validation_fold'])
                if fold not in (1, 2, 3, 4, 5):
                    raise ValueError('Invalid fold')
                vectors.append(vector)
                labels.append(row['label'])
                folds.append(fold)
                hashes.append(row['image_hash'])
            except (ValueError, KeyError, TypeError, SyntaxError):
                discarded['invalid_row'] += 1
    if len(set(hashes)) != len(hashes):
        raise ValueError('Hay imágenes duplicadas. Agrúpalas antes de separar entrenamiento y evaluación.')
    names = sorted(set(labels))
    x = np.array(vectors, dtype=np.float32)
    y = np.array([names.index(label) for label in labels], dtype=np.int64)
    fold_ids = np.array(folds)
    development, holdout = fold_ids != 5, fold_ids == 5
    for mask in (development, holdout):
        if set(y[mask]) != set(range(len(names))):
            raise ValueError('Faltan clases en entrenamiento o evaluación.')
    # Fold 5 is never used to choose rejection thresholds or distance radii.
    out_scores = np.zeros((len(y), len(names)), dtype=np.float32)
    out_distances = np.zeros_like(out_scores)
    for fold in (1, 2, 3, 4):
        training, validation = development & (fold_ids != fold), fold_ids == fold
        estimator = fit(x[training], y[training])
        out_scores[validation] = estimator.predict_proba(x[validation])
        _, out_distances[validation] = neighbors(x[validation], x[training], y[training], len(names))
    thresholds = np.full(len(names), 1.1, dtype=np.float32)  # disabled until development evidence suffices
    radii = np.zeros(len(names), dtype=np.float32)
    predictions = out_scores.argmax(1)
    for label in range(len(names)):
        chosen = development & (predictions == label)
        correct = chosen & (y == label)
        if correct.sum() < 15:
            continue
        radius = float(np.quantile(out_distances[correct, label], .95))
        for threshold in (.35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95):
            accepted = chosen & (out_scores[:, label] >= threshold) & (out_distances[:, label] <= radius)
            if accepted.sum() >= 15 and (y[accepted] == label).mean() >= .85:
                thresholds[label], radii[label] = threshold, radius
                break
    estimator = fit(x[development], y[development])
    test_scores = estimator.predict_proba(x[holdout])
    _, test_distances = neighbors(x[holdout], x[development], y[development], len(names))
    # Export must preserve the predictions actually evaluated before we refit all rows.
    exported = export_forest(estimator)
    np.testing.assert_allclose(forest_scores(x[holdout], exported), test_scores, atol=2e-6)
    report = metrics(test_scores, test_distances, y[holdout], thresholds, radii, names)
    metadata = {'feature_version': FEATURE_VERSION, 'algorithm': 'RandomForest', 'classes': names,
                'dataset_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                'training_examples_after_refit': len(y), 'discarded': dict(discarded),
                'class_counts': dict(Counter(labels)), 'holdout': report,
                'threshold_selection': 'folds 1-4 only; development target .85 precision with >=15 accepted cases',
                'holdout_split': 'existing fold 5; image-level only; person/session IDs unavailable',
                'training_pose_source': 'poseLandmarks supplied by dataset; not re-extracted by current MediaPipe',
                'limitations': 'No technique labels, negative exercises, unseen-person validation or video benchmark.'}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, features=x, labels=y, thresholds=thresholds, radii=radii, **export_forest(fit(x, y)),
                        metadata=np.array(json.dumps(metadata)))
    report_path = MODEL_PATH.with_suffix('.report.json')
    report_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(MODEL_PATH), 'usable_examples': len(y), 'discarded': dict(discarded),
                      'classes': names, 'thresholds': thresholds.tolist(), 'holdout': report}, ensure_ascii=True))


if __name__ == '__main__':
    main()
