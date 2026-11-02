"""Tests for src/validate_dataset.py: sample counts, per-label counts,
missing-hand stats, and corrupted-file detection."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
from validate_dataset import validate_dataset


class TestValidateDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_sample(self, label, filename, seq_len=5, feat_dim=None, left_detected_frames=5):
        feat_dim = feat_dim or config.FEATURE_VECTOR_LENGTH
        label_dir = os.path.join(self.tmp_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        arr = np.random.rand(seq_len, feat_dim).astype(np.float32)
        if feat_dim >= config.NUM_DETECTION_FLAGS:
            arr[:, -3] = 0.0  # left_hand_detected = False for all frames
            arr[:left_detected_frames, -3] = 1.0
            arr[:, -2] = 1.0  # right_hand_detected
            arr[:, -1] = 1.0  # pose_detected
        np.save(os.path.join(label_dir, filename), arr)

    def test_counts_samples_per_label(self):
        self._write_sample("hello", "sample_001.npy")
        self._write_sample("hello", "sample_002.npy")
        self._write_sample("thank_you", "sample_001.npy")

        report = validate_dataset(sequences_dir=self.tmp_dir)
        self.assertEqual(report["labels"]["hello"], 2)
        self.assertEqual(report["labels"]["thank_you"], 1)
        self.assertEqual(report["total_samples"], 3)

    def test_detects_consistent_sequence_length_and_feature_dim(self):
        self._write_sample("hello", "sample_001.npy", seq_len=30, feat_dim=147)
        self._write_sample("hello", "sample_002.npy", seq_len=30, feat_dim=147)

        report = validate_dataset(sequences_dir=self.tmp_dir)
        self.assertEqual(report["sequence_lengths"], {30})
        self.assertEqual(report["feature_dimensions"], {147})

    def test_flags_inconsistent_dimensions(self):
        self._write_sample("hello", "sample_001.npy", seq_len=30, feat_dim=147)
        self._write_sample("hello", "sample_002.npy", seq_len=25, feat_dim=147)  # wrong length

        report = validate_dataset(sequences_dir=self.tmp_dir)
        self.assertEqual(report["sequence_lengths"], {30, 25})

    def test_missing_hand_statistics(self):
        self._write_sample("hello", "sample_001.npy", seq_len=10, left_detected_frames=4)

        report = validate_dataset(sequences_dir=self.tmp_dir)
        stats = report["missing_hand_stats"]["hello"]
        self.assertEqual(stats["total_frames"], 10)
        self.assertEqual(stats["left_missing_frames"], 6)  # 10 - 4 detected

    def test_corrupted_file_detected(self):
        label_dir = os.path.join(self.tmp_dir, "hello")
        os.makedirs(label_dir, exist_ok=True)
        # Write garbage bytes instead of a valid .npy file.
        with open(os.path.join(label_dir, "sample_001.npy"), "wb") as f:
            f.write(b"not a real npy file")

        report = validate_dataset(sequences_dir=self.tmp_dir)
        self.assertEqual(len(report["corrupted_files"]), 1)
        self.assertEqual(report["labels"]["hello"], 0)

    def test_empty_dataset(self):
        report = validate_dataset(sequences_dir=self.tmp_dir)
        self.assertEqual(report["total_samples"], 0)
        self.assertEqual(report["labels"], {})

    def test_missing_directory_does_not_raise(self):
        nonexistent = os.path.join(self.tmp_dir, "does_not_exist")
        report = validate_dataset(sequences_dir=nonexistent)
        self.assertEqual(report["total_samples"], 0)


if __name__ == "__main__":
    unittest.main()
