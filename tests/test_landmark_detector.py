"""Tests for src/landmark_detector.py. Requires the real MediaPipe Tasks
model files under models/; skipped gracefully if they are missing so the
suite still passes in environments where they haven't been downloaded."""

import os
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config


def _models_available():
    return os.path.exists(config.HAND_LANDMARKER_MODEL_PATH) and os.path.exists(
        config.POSE_LANDMARKER_MODEL_PATH
    )


@unittest.skipUnless(_models_available(), "MediaPipe Tasks model files not found under models/")
class TestLandmarkDetector(unittest.TestCase):
    def setUp(self):
        # A fresh detector per test avoids VIDEO-mode's strictly-increasing
        # timestamp requirement leaking state across independent test cases.
        from landmark_detector import LandmarkDetector
        self.detector = LandmarkDetector()

    def tearDown(self):
        self.detector.close()

    def test_initializes_without_error(self):
        self.assertIsNotNone(self.detector)

    def test_detect_on_blank_frame_returns_empty_results(self):
        blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        result = self.detector.detect(blank, 0)
        self.assertIn("hand_result", result)
        self.assertIn("pose_result", result)
        # A blank frame should yield no detections.
        self.assertEqual(result["hand_result"].hand_landmarks, [])
        self.assertEqual(result["pose_result"].pose_landmarks, [])

    def test_detect_accepts_increasing_timestamps(self):
        blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        # VIDEO running mode requires strictly increasing timestamps; this
        # should not raise.
        self.detector.detect(blank, 1000)
        self.detector.detect(blank, 1001)
        self.detector.detect(blank, 1002)


if __name__ == "__main__":
    unittest.main()
