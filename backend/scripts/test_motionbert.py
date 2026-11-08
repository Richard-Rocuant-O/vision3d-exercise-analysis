"""Adapter contracts without a download, PyTorch, video, or network."""
import asyncio
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from starlette.datastructures import UploadFile
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import motionbert as mb
from app.main import analyze


def point(x=0.5, y=0.5, confidence=1):
    return dict(x=x, y=y, z=0, visibility=confidence, presence=confidence)


class AdapterTests(unittest.TestCase):
    def test_joint_identity_and_head_proxy(self):
        source = [point(i / 100, i / 100) for i in range(33)]
        mapped = mb.map_frame(source, 100, 100)
        # In a square frame, 0..1 maps to -1..1; these are identities, not a swap.
        self.assertAlmostEqual(mapped[2, 0], 2 * .26 - 1)  # right knee
        self.assertAlmostEqual(mapped[5, 0], 2 * .25 - 1)  # left knee
        self.assertAlmostEqual(mapped[0, 0], 2 * .235 - 1)  # hip center
        self.assertAlmostEqual(mapped[10, 0], 2 * .075 - 1)  # ear midpoint proxy

    def test_aspect_ratio_preserved(self):
        source = [point(.75, .75) for _ in range(33)]
        mapped = mb.map_frame(source, 1920, 1080)
        self.assertAlmostEqual(mapped[9, 0], 480 / 540)
        self.assertAlmostEqual(mapped[9, 1], .5)

    def test_centers_do_not_inflate_visibility(self):
        source = [point() for _ in range(33)]
        source[23] = point(confidence=.1)
        source[12] = point(confidence=.3)
        mapped = mb.map_frame(source, 100, 100)
        self.assertAlmostEqual(mapped[0, 2], .1)
        self.assertAlmostEqual(mapped[8, 2], .3)
        self.assertAlmostEqual(mapped[7, 2], .1)

    def test_missing_and_nonfinite_points_have_zero_confidence(self):
        source = [point() for _ in range(33)]
        source[25] = None
        source[26] = point(x=float('nan'))
        mapped = mb.map_frame(source, 100, 100)
        self.assertEqual(mapped[2, 2], 0)
        self.assertEqual(mapped[5, 2], 0)
        self.assertTrue(np.isfinite(mapped).all())
        self.assertFalse(mb.map_frame([], 100, 100).any())

    def test_no_temporal_bridge_over_missing_person_or_time_gap(self):
        inputs = np.ones((8, 17, 3), dtype=np.float32)
        inputs[2:4] = 0
        self.assertEqual(list(mb.segments(inputs, [0, 33, 66, 99, 132, 165, 1000, 1033])),
                         [(0, 2), (4, 6), (6, 8)])

    def test_clip_windows_cover_tail_without_exceeding_model_limit(self):
        for length in (1, 242, 243, 244, 454, 486, 3600):
            coverage = np.zeros(length, dtype=int)
            for start in mb.window_starts(length):
                coverage[start:min(start + mb.WINDOW, length)] += 1
            self.assertTrue((coverage > 0).all())
            self.assertEqual(mb.window_starts(length)[-1], max(0, length - mb.WINDOW))

    def test_unavailable_model_returns_503_and_closes_upload(self):
        upload = UploadFile(file=io.BytesIO(b"clip"), filename="example.mp4")
        with patch.object(mb, 'status', return_value={"available": False, "reason": "missing checkpoint"}):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(analyze(upload, "motionbert"))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertTrue(upload.file.closed)


if __name__ == "__main__":
    unittest.main()
