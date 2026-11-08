"""Geometry and synchronization contracts, not validation of exercise technique."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.feedback import flexion, measurements, evaluate, segment_identity
from app.exercise_images import assign_predictions, letterbox
from app import exercises


def landmark(x=.5, y=.5, quality=1):
    return dict(x=x, y=y, z=0, visibility=quality, presence=quality)


class FeedbackTests(unittest.TestCase):
    def test_knee_definition(self):
        self.assertAlmostEqual(flexion((0, 0, 1), (0, 1, 1), (0, 2, 1))['value'], 0)
        self.assertAlmostEqual(flexion((0, 0, 1), (0, 1, 1), (1, 1, 1))['value'], 90)
        self.assertIsNone(flexion(None, (0, 1, 1), (1, 1, 1)))
        self.assertIsNone(flexion((0, 0, 1), (0, 0, 1), (1, 1, 1)))

    def test_angles_use_image_aspect_and_confidence(self):
        points = [landmark(quality=0) for _ in range(33)]
        points[11], points[23] = landmark(.4, .2), landmark(.2, .6)
        # In a 200x100 image, the physical offsets are 40px horizontally/vertically.
        values = measurements({'landmarks': points}, 200, 100)
        self.assertAlmostEqual(values['trunk_inclination_deg']['value'], 45)
        self.assertIsNone(values['left_knee_flexion_deg'])
        points[11]['visibility'] = float('nan')
        self.assertIsNone(measurements({'landmarks': points}, 200, 100)['trunk_inclination_deg'])

    def test_unknown_or_unreviewed_reference_never_means_correct(self):
        valid = {key: {'value': 45, 'visibility': 1} for key in ('left_knee_flexion_deg', 'right_knee_flexion_deg', 'trunk_inclination_deg')}
        profile = {'reference_video': 'squat_01.mp4', 'version': 'demo', 'review': {'status': 'pending'}, 'criteria': []}
        self.assertEqual(evaluate({'status': 'uncertain'}, valid, profile)['status'], 'not_evaluable')
        self.assertEqual(evaluate({'status': 'identified', 'label': 'Squat'}, valid, profile)['status'], 'pending_review')
        self.assertEqual(evaluate({'status': 'identified', 'label': 'Push-up'}, valid, profile)['status'], 'unsupported')

    def motion(self, count):
        return {'input_type': 'video', 'frames': [{'timestamp_ms': i * 200, 'landmarks': [landmark() for _ in range(33)]} for i in range(count)]}

    def assign(self, motion, probabilities):
        count = len(motion['frames'])
        assign_predictions(motion, list(range(count)), np.array(probabilities), np.zeros((count, 2)),
                           {'thresholds': np.array([.6, .6]), 'radii': np.array([1., 1.])}, ['Squat', 'Push-up'])

    def test_future_samples_do_not_change_previous_feedback(self):
        short, longer = self.motion(4), self.motion(6)
        self.assign(short, [[.95, .05]] * 4)
        self.assign(longer, [[.95, .05]] * 4 + [[.05, .95]] * 2)
        self.assertEqual([f['exercise'] for f in short['frames']], [f['exercise'] for f in longer['frames'][:4]])
        self.assertEqual(short['frames'][0]['exercise']['status'], 'uncertain')
        self.assertEqual(short['frames'][2]['exercise']['label'], 'Squat')

    def test_missing_person_resets_confirmed_identity(self):
        motion = self.motion(6)
        motion['frames'][3]['landmarks'] = []
        self.assign(motion, [[.95, .05]] * 6)
        self.assertEqual(motion['frames'][2]['exercise']['label'], 'Squat')
        self.assertEqual(motion['frames'][3]['exercise']['status'], 'not_evaluable')
        self.assertEqual(motion['frames'][4]['exercise']['status'], 'uncertain')

    def test_no_labels_from_future_or_stale_samples(self):
        motion = self.motion(7)
        assign_predictions(motion, [2], np.array([[.99, .01]]), np.zeros((1, 2)),
                           {'thresholds': np.array([.6, .6]), 'radii': np.array([1., 1.])}, ['Squat', 'Push-up'])
        self.assertIsNone(motion['frames'][0]['exercise']['label'])
        self.assertIsNone(motion['frames'][-1]['exercise']['label'])

    def test_letterbox_preserves_whole_body_aspect(self):
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        boxed = letterbox(image)
        self.assertEqual(boxed.shape, (224, 224, 3))
        self.assertTrue((boxed[:, :56] == 127).all())
        self.assertTrue((boxed[:, 56:168] == 0).all())

    def test_pose_classifier_is_also_causal(self):
        def classify(count, probabilities):
            motion = self.motion(count)
            motion.update(width=200, height=100)
            for frame in motion['frames']:
                for i in (11, 12):
                    frame['landmarks'][i] = landmark(.5, .2)
                for i in (23, 24):
                    frame['landmarks'][i] = landmark(.5, .6)
            model = {'thresholds': np.array([.6, .6]), 'radii': np.array([1., 1.]), 'features': [], 'labels': []}
            metadata = {'classes': ['Squat', 'Push-up'], 'dataset_sha256': 'test', 'holdout': {}}
            with patch.object(exercises, 'load_model', return_value=(model, metadata)), patch.object(exercises, 'neighbors', return_value=(np.zeros((count, 2)), np.zeros((count, 2)))), patch.object(exercises, 'forest_scores', return_value=np.array(probabilities)):
                exercises.attach(motion)
            return motion
        short = classify(4, [[.95, .05]] * 4)
        long = classify(6, [[.95, .05]] * 4 + [[.05, .95]] * 2)
        self.assertEqual([f['exercise'] for f in short['frames']], [f['exercise'] for f in long['frames'][:4]])
        self.assertEqual(short['frames'][2]['exercise']['label'], 'Squat')

    def test_segment_identity_is_not_a_technique_judgment(self):
        motion = self.motion(8)
        for frame in motion['frames']:
            frame['exercise'] = {'status': 'identified', 'label': 'Squat', 'name': 'Sentadilla'}
        summary = segment_identity(motion)
        self.assertEqual(summary['label'], 'Squat')
        self.assertIn('all processed frames', summary['scope'])
        self.assertNotIn('correct', summary)
        motion['frames'][-1]['exercise'] = {'status': 'uncertain', 'candidate_label': 'Lunges'}
        self.assertEqual(segment_identity(motion)['label'], 'Squat')


if __name__ == '__main__':
    unittest.main()
