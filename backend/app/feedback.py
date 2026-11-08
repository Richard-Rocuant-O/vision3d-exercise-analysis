"""Observable geometry and reviewed criteria, separate from exercise recognition."""
import json
import math
from pathlib import Path

from .exercises import attach as recognize, LABELS_ES
from .reference_compare import compare

PROFILE_PATH = Path(__file__).resolve().parents[1] / 'config/squat-reference.json'


def point(frame, index, width, height):
    values = frame['landmarks']
    p = values[index] if index < len(values) else None
    if not p or not all(math.isfinite(p.get(key, float('nan'))) for key in ('x', 'y', 'visibility', 'presence')):
        return None
    if min(p['visibility'], p['presence']) < .5:
        return None
    return p['x'] * width, p['y'] * height, min(p['visibility'], p['presence'])


def flexion(a, b, c):
    if a is None or b is None or c is None:
        return None
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    lengths = math.hypot(*u) * math.hypot(*v)
    if lengths < 1:
        return None
    cosine = max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / lengths))
    return {'value': 180 - math.degrees(math.acos(cosine)), 'visibility': min(a[2], b[2], c[2])}


def measurements(frame, width, height):
    get = lambda i: point(frame, i, width, height)
    result = {'left_knee_flexion_deg': flexion(get(23), get(25), get(27)),
              'right_knee_flexion_deg': flexion(get(24), get(26), get(28))}
    trunks = []
    for shoulder_id, hip_id in ((11, 23), (12, 24)):
        shoulder, hip = get(shoulder_id), get(hip_id)
        if shoulder and hip and math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]) >= 1:
            trunks.append({'value': math.degrees(math.atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1]))),
                           'visibility': min(shoulder[2], hip[2])})
    result['trunk_inclination_deg'] = max(trunks, key=lambda p: p['visibility']) if trunks else None
    return result


def evaluate(exercise, values, profile):
    if exercise.get('status') != 'identified':
        return {'status': 'not_evaluable', 'reason': 'exercise_uncertain', 'alerts': []}
    if exercise['label'] != 'Squat':
        return {'status': 'unsupported', 'reason': 'no_reviewed_criteria_for_exercise', 'alerts': []}
    if not values['trunk_inclination_deg'] or not (values['left_knee_flexion_deg'] or values['right_knee_flexion_deg']):
        return {'status': 'not_evaluable', 'reason': 'unreliable_joints', 'alerts': []}
    # A user choosing a demo file does not make it a reviewed technical reference.
    return {'status': 'pending_review', 'reason': 'reference_and_phase_criteria_not_reviewed', 'alerts': [],
            'reference': profile['reference_video'], 'reference_version': profile['version']}


def segment_identity(motion):
    """A loaded clip may use whole-segment context; this is not live-camera inference."""
    if motion['input_type'] != 'video':
        return None
    votes, supported, total = {}, {}, 0
    frames = motion['frames']
    for i, frame in enumerate(frames):
        exercise = frame['exercise']
        label = exercise.get('label') or exercise.get('candidate_label')
        if label is None:
            continue
        interval = frames[i + 1]['timestamp_ms'] - frame['timestamp_ms'] if i + 1 < len(frames) else 40
        weight = min(200, max(1, interval))
        votes[label] = votes.get(label, 0) + weight
        total += weight
        if exercise['status'] == 'identified':
            supported[label] = supported.get(label, 0) + weight
    if not votes:
        return {'status': 'uncertain', 'label': None, 'name': None, 'candidate_name': None,
                'scope': 'whole analyzed segment, assuming one exercise'}
    ordered = sorted(votes, key=votes.get, reverse=True)
    best = ordered[0]
    dominance = votes[best] / total
    margin = (votes[best] - (votes[ordered[1]] if len(ordered) > 1 else 0)) / total
    accepted = total >= 1000 and dominance >= 2 / 3 and margin >= 1 / 3 and supported.get(best, 0) >= 500
    candidate = dominance >= .4 and margin >= .05
    return {'status': 'identified' if accepted else 'uncertain', 'label': best if accepted else None,
            'name': LABELS_ES.get(best, best) if accepted else None,
            'candidate_name': LABELS_ES.get(best, best) if candidate else None,
            'dominance': dominance, 'support_kind': 'duration-weighted posture votes; NOT probability of correctness',
            'scope': 'whole analyzed segment, assuming one exercise; uses all processed frames'}


def analyze_feedback(motion, compare_to_example=True, tolerance_degrees=15.0):
    try:
        recognize(motion)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        motion['exercise_classifier'] = {'available': False, 'error': str(exc)}
        for frame in motion['frames']:
            frame['exercise'] = {'status': 'model_unavailable', 'label': None, 'name': None}
    profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    summary = segment_identity(motion)
    if summary is not None:
        motion['exercise_summary'] = summary
    for frame in motion['frames']:
        values = measurements(frame, motion['width'], motion['height'])
        frame['measurements'] = values
        frame['technique'] = evaluate(summary or frame['exercise'], values, profile)
    compare(motion, profile, measurements, enabled=compare_to_example, tolerance=tolerance_degrees)
    motion['feedback'] = {
        'mode': 'precomputed, synchronized to original video timestamps during playback',
        'measurements': '2D projected angles; lateral view required; not measured anatomical angles',
        'knee_definition': '180 minus hip-knee-ankle interior angle; 0 approximately extended',
        'trunk_definition': 'angle to image vertical; no forward/backward direction assessment',
        'reference': profile, 'technique_evaluation_available': False,
    }
    return motion
