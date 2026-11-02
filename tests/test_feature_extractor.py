"""Tests for src/feature_extractor.py: fixed-size output, missing-hand
handling, and normalization behavior."""

import unittest

import numpy as np
import test_helpers as th

import config
from feature_extractor import extract_features, _normalize_hand, _normalize_pose


class TestFeatureVectorShape(unittest.TestCase):
    def test_fixed_length_with_both_hands_and_pose(self):
        fv, info = extract_features(th.make_two_hand_result(), th.make_pose_result())
        self.assertEqual(fv.shape, (config.FEATURE_VECTOR_LENGTH,))
        self.assertEqual(fv.dtype, np.float32)
        self.assertTrue(info["left_hand"])
        self.assertTrue(info["right_hand"])
        self.assertTrue(info["pose"])

    def test_fixed_length_with_nothing_detected(self):
        fv, info = extract_features(th.make_no_hand_result(), th.make_no_pose_result())
        self.assertEqual(fv.shape, (config.FEATURE_VECTOR_LENGTH,))
        self.assertFalse(info["left_hand"])
        self.assertFalse(info["right_hand"])
        self.assertFalse(info["pose"])


class TestMissingHandZeroFill(unittest.TestCase):
    def test_missing_right_hand_is_zero_filled(self):
        fv, info = extract_features(th.make_left_only_hand_result(), th.make_no_pose_result())
        self.assertTrue(info["left_hand"])
        self.assertFalse(info["right_hand"])

        left_slice = fv[0:config.HAND_FEATURE_LEN]
        right_slice = fv[config.HAND_FEATURE_LEN:config.HAND_FEATURE_LEN * 2]

        self.assertFalse(np.allclose(left_slice, 0.0), "detected hand should not be all zeros")
        self.assertTrue(np.allclose(right_slice, 0.0), "missing hand must be zero-filled")

    def test_missing_pose_is_zero_filled(self):
        fv, info = extract_features(th.make_no_hand_result(), th.make_no_pose_result())
        pose_start = config.HAND_FEATURE_LEN * 2
        pose_slice = fv[pose_start:pose_start + config.POSE_FEATURE_LEN]
        self.assertTrue(np.allclose(pose_slice, 0.0))

    def test_detection_flags_are_last_three_values(self):
        fv, info = extract_features(th.make_left_only_hand_result(), th.make_pose_result())
        flags = fv[-config.NUM_DETECTION_FLAGS:]
        np.testing.assert_array_equal(flags, np.array([1.0, 0.0, 1.0], dtype=np.float32))


class TestNormalization(unittest.TestCase):
    def test_hand_wrist_moved_to_origin(self):
        points = np.random.rand(21, 3).astype(np.float32)
        normalized = _normalize_hand(points)
        np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_hand_normalization_handles_degenerate_scale(self):
        """All landmarks identical -> scale reference distance is 0; must
        not raise or produce NaN/Inf."""
        points = np.zeros((21, 3), dtype=np.float32)
        normalized = _normalize_hand(points)
        self.assertFalse(np.any(np.isnan(normalized)))
        self.assertFalse(np.any(np.isinf(normalized)))

    def test_pose_shoulder_midpoint_moved_to_origin(self):
        points = np.array(
            [[0.3, 0.4, 0.0], [0.7, 0.4, 0.0], [0.2, 0.6, 0.0],
             [0.8, 0.6, 0.0], [0.15, 0.8, 0.0], [0.85, 0.8, 0.0]],
            dtype=np.float32,
        )
        normalized = _normalize_pose(points)
        midpoint = (normalized[0] + normalized[1]) / 2.0
        np.testing.assert_allclose(midpoint, [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
