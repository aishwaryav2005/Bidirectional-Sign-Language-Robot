"""Tests for src/temporal_sampling.py: uniform frame-index sampling for
variable-length videos."""

import unittest

import numpy as np
import test_helpers  # noqa: F401

from temporal_sampling import sample_frame_indices


class TestTemporalSampling(unittest.TestCase):
    def test_downsampling_long_video(self):
        indices = sample_frame_indices(source_length=180, target_length=60)
        self.assertEqual(indices.shape, (60,))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 179)
        self.assertTrue(np.all(np.diff(indices) >= 0), "indices must be non-decreasing")

    def test_upsampling_short_video_repeats_not_invents_frames(self):
        indices = sample_frame_indices(source_length=20, target_length=60)
        self.assertEqual(indices.shape, (60,))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 19)
        self.assertTrue(set(indices.tolist()).issubset(set(range(20))))

    def test_exact_length_is_identity(self):
        indices = sample_frame_indices(source_length=60, target_length=60)
        np.testing.assert_array_equal(indices, np.arange(60))

    def test_single_frame_video(self):
        indices = sample_frame_indices(source_length=1, target_length=60)
        self.assertEqual(indices.shape, (60,))
        self.assertTrue(np.all(indices == 0))

    def test_spans_full_video_never_crops(self):
        """Regression guard: must not just take the first target_length
        frames of a long video -- must span first to last frame."""
        indices = sample_frame_indices(source_length=1000, target_length=60)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 999)
        self.assertGreater(indices[30], 400, "should span the whole range, not cluster near the start")

    def test_rejects_invalid_lengths(self):
        with self.assertRaises(ValueError):
            sample_frame_indices(source_length=0, target_length=60)
        with self.assertRaises(ValueError):
            sample_frame_indices(source_length=60, target_length=0)

    def test_indices_always_in_bounds(self):
        for source_length in [1, 2, 5, 30, 59, 60, 61, 200, 10000]:
            indices = sample_frame_indices(source_length, target_length=60)
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < source_length))
            self.assertEqual(indices.shape, (60,))


if __name__ == "__main__":
    unittest.main()
