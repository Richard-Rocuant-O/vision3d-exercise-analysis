"""Time-window extraction contracts; detection is mocked, not a pose benchmark."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import pose


class Capture:
    def __init__(self, seconds=80, fps=10):
        self.fps, self.count = fps, seconds * fps
        self.index, self.last = 0, -1
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_POS_FRAMES: self.index,
                cv2.CAP_PROP_POS_MSEC: self.last * 1000 / self.fps}.get(prop, 0)

    def set(self, prop, value):
        self.index = math.floor(value * self.fps / 1000)
        return True

    def read(self):
        if self.index >= self.count:
            return False, None
        self.last = self.index
        self.index += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class Detector:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def detect_for_video(self, image, timestamp):
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])


class SegmentTests(unittest.TestCase):
    def extract(self, capture, start=0, duration=60):
        with patch.object(pose.cv2, 'VideoCapture', return_value=capture), patch.object(pose, 'create_landmarker', return_value=Detector()):
            return pose.extract_video('fixture.mp4', start, duration)

    def test_long_file_accepts_default_first_minute(self):
        capture = Capture(seconds=80)
        result = self.extract(capture)
        self.assertEqual(len(result['frames']), 600)
        self.assertEqual(result['duration_ms'], 80000)
        self.assertEqual(result['analysis_window']['end_ms'], 60000)
        self.assertTrue(capture.released)

    def test_segment_after_first_minute_keeps_original_timestamps_and_indices(self):
        result = self.extract(Capture(seconds=80), start=65, duration=3)
        frames = result['frames']
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0]['timestamp_ms'], 65000)
        self.assertEqual(frames[0]['frame_index'], 650)
        self.assertEqual(frames[-1]['timestamp_ms'], 67900)
        self.assertEqual(result['analysis_window']['end_ms'], 68000)

    def test_duration_is_clipped_at_actual_end_of_source(self):
        result = self.extract(Capture(seconds=80), start=78, duration=15)
        self.assertEqual(len(result['frames']), 20)
        self.assertEqual(result['analysis_window']['end_ms'], 80000)

    def test_start_outside_source_closes_capture(self):
        capture = Capture(seconds=5)
        with self.assertRaisesRegex(ValueError, 'fuera del video'):
            self.extract(capture, start=5)
        self.assertTrue(capture.released)

    def test_invalid_ranges_are_rejected(self):
        for start, duration in ((-1, 2), (0, 0), (0, 61), (float('nan'), 2), (0, float('inf'))):
            with self.subTest(start=start, duration=duration), self.assertRaises(ValueError):
                pose.extract_video('fixture.mp4', start, duration)


if __name__ == '__main__':
    unittest.main()
