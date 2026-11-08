"""Extract the chosen demonstration. This does not certify its technique."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pose import extract_video

ROOT = Path(__file__).resolve().parents[2]


def main():
    profile = json.loads((ROOT / 'backend/config/squat-reference.json').read_text(encoding='utf-8'))
    source = ROOT / profile['reference_video']
    target = (ROOT / profile['motion_file']).resolve()
    if not target.is_relative_to(ROOT / 'data'):
        raise ValueError('La referencia procesada debe guardarse dentro de data/.')
    result = extract_video(source)
    result['reference_source'] = {'filename': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                'review_status': profile['review']['status'], 'is_technique_ground_truth': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(json.dumps({'reference': str(target), 'frames': len(result['frames']), 'review_status': profile['review']['status']}))


if __name__ == '__main__':
    main()
