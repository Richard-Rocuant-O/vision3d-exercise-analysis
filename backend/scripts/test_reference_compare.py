"""Synthetic movement fixtures test alignment and alerts, not accuracy or technique."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import reference_compare as module


def cycle(step=100, trunk=20):
    frames = []
    for timestamp in range(0, 4001, step):
        progress = min(timestamp / 2000, (4000 - timestamp) / 2000)
        landmarks = [dict(x=.5, y=.5, presence=1., visibility=1.) for _ in range(33)]
        for index, x, y in [(11, .4, .2), (12, .42, .2), (23, .4, .6), (24, .42, .6)]:
            landmarks[index].update(x=x, y=y)
        frames.append({'timestamp_ms': timestamp, 'landmarks': landmarks, '_image_size': (640, 480),
                       'measurements': {**{key: {'value': 90 * progress, 'visibility': 1.} for key in module.KNEES},
                                        'trunk_inclination_deg': {'value': trunk, 'visibility': 1.}}})
    return frames


class ComparisonTests(unittest.TestCase):
    def compare(self, frames):
        motion = {'input_type': 'video', 'width': 640, 'height': 480, 'frames': frames,
                  'exercise_summary': {'label': 'Squat'}}
        profile = {'reference_video': 'fixture.mp4', 'review': {'status': 'pending'}}
        with patch.object(module, 'reference', return_value=cycle(step=80)):
            module.compare(motion, profile, None)
        return motion

    def test_same_movement_different_sampling_no_red(self):
        frames = cycle()
        # A slower video must align by phase, not raw time.
        for frame in frames:
            frame['timestamp_ms'] *= 2
        self.compare(frames)
        self.assertTrue(any(f['comparison']['status'] == 'within_display_tolerance' for f in frames))
        self.assertFalse(any(f['comparison']['alerts'] for f in frames))

    def test_persistent_trunk_difference_targets_torso(self):
        motion = self.compare(cycle(trunk=50))
        alerts = [a for f in motion['frames'] for a in f['comparison']['alerts']]
        self.assertTrue(alerts)
        self.assertEqual({a['metric'] for a in alerts}, {'trunk_inclination_deg'})
        self.assertTrue(all(a['kind'] == 'reference_difference' for a in alerts))
        self.assertFalse(motion['reference_comparison']['reference_reviewed'])

    def test_missing_knees_do_not_bridge_a_cycle(self):
        frames = cycle()
        frames[20]['measurements'] = {key: None for key in frames[20]['measurements']}
        self.assertFalse(any(module.phases(frames)))

    def test_partial_cycle_is_not_evaluable(self):
        frames = cycle()[:21]
        self.compare(frames)
        self.assertTrue(all(f['comparison']['status'] == 'not_evaluable' for f in frames))

    def test_front_view_and_low_confidence_have_no_red(self):
        for low_confidence in (False, True):
            frames = cycle(trunk=50)
            for frame in frames:
                if low_confidence:
                    frame['landmarks'][11]['visibility'] = .1
                else:
                    frame['landmarks'][11]['x'] = .1
                    frame['landmarks'][12]['x'] = .9
            self.compare(frames)
            self.assertFalse(any(f['comparison']['alerts'] for f in frames))

    def test_short_spike_does_not_alert(self):
        frames = cycle()
        frames[20]['measurements']['trunk_inclination_deg']['value'] = 55
        self.compare(frames)
        self.assertFalse(any(f['comparison']['alerts'] for f in frames))


if __name__ == '__main__':
    unittest.main()
