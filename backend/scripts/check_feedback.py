"""Exercise/measurement API smoke check on a user-supplied video."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from api_check_utils import upload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    parser.add_argument('--start', type=float, default=3)
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--expected-label', help='Optional functional check for a known control clip, not a benchmark.')
    args = parser.parse_args()
    result = upload(args.url, args.video, args.start, args.duration)
    assert result['exercise_classifier']['available'], result['exercise_classifier']
    assert not result['feedback']['technique_evaluation_available']
    if args.expected_label:
        assert result['exercise_summary']['label'] == args.expected_label, result['exercise_summary']
    times = [f['timestamp_ms'] for f in result['frames']]
    assert times[0] >= args.start * 1000
    assert all(a < b for a, b in zip(times, times[1:]))
    measured = 0
    for frame in result['frames']:
        assert frame['exercise']['status'] in ('identified', 'uncertain', 'not_evaluable')
        assert frame['technique']['status'] in ('not_evaluable', 'pending_review', 'unsupported')
        assert frame['comparison']['status'] in ('not_evaluable', 'within_display_tolerance', 'differences')
        assert frame['comparison'].get('reason') != 'disabled'
        for alert in frame['comparison']['alerts']:
            assert alert['kind'] == 'reference_difference'
            assert alert['joints'] and alert['segments'] and math.isfinite(alert['delta_deg'])
        for name, angle in frame['measurements'].items():
            if angle is not None:
                assert math.isfinite(angle['value']) and 0 <= angle['value'] <= 180
                assert angle['visibility'] >= .5
                measured += 1
        if not frame['landmarks']:
            assert frame['exercise']['status'] == 'not_evaluable'
            assert not any(frame['measurements'].values())
    assert measured > 0
    output = Path(__file__).resolve().parents[2] / 'data/feedback-check'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'video-pose.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(json.dumps({'frames': len(result['frames']), 'model': result['exercise_classifier']['name'],
                      'exercise_summary': result['exercise_summary'],
                      'exercises': dict(Counter(f['exercise'].get('label') or f['exercise']['status'] for f in result['frames'])),
                      'technique_states': dict(Counter(f['technique']['status'] for f in result['frames'])),
                      'comparison_states': dict(Counter(f['comparison']['status'] for f in result['frames'])),
                      'finite_measurements': measured, 'window': result['analysis_window']}))


if __name__ == '__main__':
    main()
