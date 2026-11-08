"""Experimental phase-matched differences, never a claim of correct technique."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GAP_MS = 250
MIN_RANGE_DEG = 15  # Alignment observability, NOT an acceptable knee-flexion threshold.
PERSIST_MS = 240
PHASE_NAMES = {'top': 'posición alta', 'bottom': 'posición baja', 'descending': 'bajada', 'ascending': 'subida'}
KNEES = ('left_knee_flexion_deg', 'right_knee_flexion_deg')


def knee(frame):
    values = [frame.get('measurements', {}).get(key) for key in KNEES]
    values = [p for p in values if p is not None and p['visibility'] >= .5]
    return max(values, key=lambda p: p['visibility'])['value'] if values else None


def phases(frames, anchor_top=None, tolerance=15):
    """Require a top-bottom-top cycle. Never bridge missing detections or time gaps."""
    result = [None] * len(frames)
    groups, group = [], []
    for i, frame in enumerate(frames):
        value = knee(frame)
        if value is None or (group and frame['timestamp_ms'] - frames[group[-1]]['timestamp_ms'] > GAP_MS):
            if group:
                groups.append(group)
                group = []
        if value is not None:
            group.append(i)
    if group:
        groups.append(group)
    for indices in groups:
        if len(indices) < 7:
            continue
        values = np.array([knee(frames[i]) for i in indices])
        smooth = np.array([np.median(values[max(0, i - 2):min(len(values), i + 3)]) for i in range(len(values))])
        low, high = np.quantile(smooth, [.1, .9])
        if high - low < MIN_RANGE_DEG or (anchor_top is not None and abs(low - anchor_top) > tolerance):
            continue
        progress = np.clip((smooth - low) / (high - low), 0, 1)
        tops = np.flatnonzero(progress <= .15)
        if len(tops) < 2:
            continue
        valid = np.zeros(len(indices), dtype=bool)
        # Merge each run of high positions, then retain only complete intervening cycles.
        runs = np.split(tops, np.flatnonzero(np.diff(tops) > 1) + 1)
        for before, after in zip(runs, runs[1:]):
            start, end = int(before[-1]), int(after[0])
            if np.max(progress[start:end + 1]) >= .85:
                valid[int(before[0]):int(after[-1]) + 1] = True
        for j, i in enumerate(indices):
            if not valid[j]:
                continue
            if progress[j] <= .1:
                phase = 'top'
            elif progress[j] >= .9:
                phase = 'bottom'
            else:
                delta = smooth[min(j + 2, len(indices) - 1)] - smooth[max(0, j - 2)]
                if abs(delta) < .02 * (high - low):
                    continue
                phase = 'descending' if delta > 0 else 'ascending'
            result[i] = {'phase': phase, 'progress': float(progress[j]), 'top': float(low)}
    return result


def side_view(frame):
    """Reject clearly frontal projections. Passing this heuristic does not verify camera pose."""
    points = frame.get('landmarks', [])
    if len(points) != 33:
        return False
    ps = [points[i] for i in (11, 12, 23, 24)]
    if not all(p and min(p['visibility'], p['presence']) >= .5 for p in ps):
        return False
    # Coordinates are normalized to image size; use the same pixel aspect as measurements.
    width, height = frame['_image_size']
    xy = np.array([[p['x'] * width, p['y'] * height] for p in ps])
    torso = np.linalg.norm(xy[:2].mean(0) - xy[2:].mean(0))
    if torso < 5:
        return False
    return bool(np.linalg.norm(xy[0] - xy[1]) / torso < .65 and np.linalg.norm(xy[2] - xy[3]) / torso < .6)


@lru_cache(maxsize=1)
def _read(path, signature):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(profile, measure):
    path = (ROOT / profile.get('motion_file', 'data/references/squat_01.json')).resolve()
    if not path.is_relative_to(ROOT / 'data') or not path.is_file():
        return None
    stat = path.stat()
    raw = _read(str(path), (stat.st_mtime_ns, stat.st_size))
    frames = []
    for frame in raw['frames']:
        frames.append({**frame, 'measurements': measure(frame, raw['width'], raw['height']), '_image_size': (raw['width'], raw['height'])})
    return frames


def body_parts(metric, frame):
    if metric in KNEES:
        side = 'left' if metric.startswith('left') else 'right'
        return [f'{side}_hip', f'{side}_knee', f'{side}_ankle'], [[f'{side}_hip', f'{side}_knee'], [f'{side}_knee', f'{side}_ankle']]
    points = frame['landmarks']
    choices = []
    for side, a, b in (('left', 11, 23), ('right', 12, 24)):
        if points[a] and points[b]:
            quality = min(points[a]['visibility'], points[a]['presence'], points[b]['visibility'], points[b]['presence'])
            choices.append((quality, side))
    side = max(choices)[1] if choices else 'left'
    return [f'{side}_shoulder', f'{side}_hip'], [[f'{side}_shoulder', f'{side}_hip']]


def compare(motion, profile, measure, enabled=True, tolerance=15.0):
    def unavailable(reason):
        for frame in motion['frames']:
            frame['comparison'] = {'status': 'not_evaluable', 'reason': reason, 'alerts': []}

    metadata = {'kind': 'experimental_reference_difference', 'reference_video': profile['reference_video'],
                'reference_reviewed': profile['review']['status'] == 'reviewed',
                'visual_tolerance_deg': tolerance, 'tolerance_is_not_a_technique_limit': True,
                'alignment': 'complete top-bottom-top cycles; phase and relative progress; offline',
                'red_means': 'difference from the selected example, NOT incorrect technique'}
    motion['reference_comparison'] = metadata
    if not enabled:
        unavailable('disabled')
        return
    if motion['input_type'] != 'video':
        unavailable('video_required')
        return
    if motion.get('exercise_summary', {}).get('label') != 'Squat':
        unavailable('exercise_uncertain')
        return
    refs = reference(profile, measure)
    if refs is None:
        unavailable('reference_missing')
        return
    reference_phases = phases(refs)
    usable_ref = [(i, p) for i, p in enumerate(reference_phases) if p and side_view(refs[i])]
    if not usable_ref:
        unavailable('reference_phase_unavailable')
        return
    top = float(np.median([p['top'] for _, p in usable_ref]))
    current = [{**frame, '_image_size': (motion['width'], motion['height'])} for frame in motion['frames']]
    current_phases = phases(current, anchor_top=top, tolerance=tolerance)
    pending = {}
    for i, (frame, phase) in enumerate(zip(current, current_phases)):
        original = motion['frames'][i]
        if not phase or not side_view(frame):
            original['comparison'] = {'status': 'not_evaluable', 'reason': 'phase_or_view_unavailable', 'alerts': []}
            pending.clear()
            continue
        matches = [(j, p) for j, p in usable_ref if p['phase'] == phase['phase'] and abs(p['progress'] - phase['progress']) <= .12]
        if not matches:
            original['comparison'] = {'status': 'not_evaluable', 'reason': 'phase_match_unavailable', 'alerts': []}
            pending.clear()
            continue
        alerts, differences, present = [], {}, set()
        for metric, value in frame['measurements'].items():
            if value is None:
                pending.pop(metric, None)
                continue
            reference_values = []
            for j, _ in matches:
                m = refs[j]['measurements']
                target = m.get(metric)
                if target is None and metric in KNEES:
                    target = max((m[k] for k in KNEES if m[k] is not None), key=lambda v: v['visibility'], default=None)
                if target is not None:
                    reference_values.append(target['value'])
            if not reference_values:
                pending.pop(metric, None)
                continue
            target = float(np.median(reference_values))
            delta = value['value'] - target
            differences[metric] = {'current_deg': value['value'], 'example_deg': target, 'delta_deg': delta}
            if abs(delta) <= tolerance:
                pending.pop(metric, None)
                continue
            timestamp = frame['timestamp_ms']
            previous = pending.get(metric)
            start = previous[0] if previous and timestamp - previous[1] <= GAP_MS else timestamp
            pending[metric] = (start, timestamp)
            present.add(metric)
            if timestamp - start < PERSIST_MS:
                continue
            joints, segments = body_parts(metric, frame)
            label = 'rodilla izquierda' if metric.startswith('left') else 'rodilla derecha' if metric.startswith('right') else 'tronco'
            alerts.append({'metric': metric, 'kind': 'reference_difference', 'joints': joints, 'segments': segments,
                           'delta_deg': delta, 'message': f'{label.capitalize()}: {abs(delta):.0f}° de diferencia con el ejemplo durante {PHASE_NAMES[phase["phase"]]}.'})
        pending = {key: value for key, value in pending.items() if key in present}
        original['comparison'] = {'status': 'differences' if alerts else 'within_display_tolerance',
                                  'phase': phase['phase'], 'alerts': alerts, 'differences': differences,
                                  'reference_timestamp_ms': refs[min(matches, key=lambda p: abs(p[1]['progress'] - phase['progress']))[0]]['timestamp_ms']}
