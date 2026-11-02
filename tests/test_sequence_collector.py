"""Tests for src/sequence_collector.py: sequence length, save/load, metadata."""

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
from sequence_collector import SequenceCollector


class TestSequenceCollector(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.tmp_dir, "metadata.csv")
        self.collector = SequenceCollector(
            sequence_length=5,
            feature_dim=config.FEATURE_VECTOR_LENGTH,
            sequences_dir=self.tmp_dir,
            metadata_path=self.metadata_path,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_frame(self):
        return (
            np.random.rand(config.FEATURE_VECTOR_LENGTH).astype(np.float32),
            {"left_hand": True, "right_hand": False, "pose": True},
        )

    def test_completes_after_exactly_sequence_length_frames(self):
        self.collector.start_recording()
        completed_flags = []
        for _ in range(5):
            fv, info = self._fake_frame()
            completed_flags.append(self.collector.add_frame(fv, info))
        self.assertEqual(completed_flags, [False, False, False, False, True])
        self.assertFalse(self.collector.is_recording)

    def test_add_frame_rejects_wrong_length_vector(self):
        self.collector.start_recording()
        with self.assertRaises(ValueError):
            self.collector.add_frame(np.zeros(10, dtype=np.float32), {"left_hand": False, "right_hand": False, "pose": False})

    def test_save_and_reload_round_trip(self):
        self.collector.start_recording()
        for _ in range(5):
            fv, info = self._fake_frame()
            self.collector.add_frame(fv, info)

        path = self.collector.save_sequence("hello", camera_resolution=(640, 480), signer_id="signer_1")
        self.assertTrue(os.path.exists(path))

        loaded = np.load(path)
        self.assertEqual(loaded.shape, (5, config.FEATURE_VECTOR_LENGTH))
        self.assertEqual(loaded.dtype, np.float32)

    def test_metadata_row_written_correctly(self):
        self.collector.start_recording()
        for _ in range(5):
            fv, info = self._fake_frame()
            self.collector.add_frame(fv, info)
        self.collector.save_sequence("hello", camera_resolution=(640, 480), signer_id="signer_1")

        with open(self.metadata_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["label"], "hello")
        self.assertEqual(int(row["sequence_length"]), 5)
        self.assertEqual(int(row["feature_dimension"]), config.FEATURE_VECTOR_LENGTH)
        self.assertEqual(row["signer_id"], "signer_1")
        self.assertEqual(int(row["frames_left_hand_detected"]), 5)
        self.assertEqual(int(row["frames_right_hand_detected"]), 0)

    def test_multiple_samples_same_label_increment_filename(self):
        for _ in range(2):
            self.collector.start_recording()
            for _ in range(5):
                fv, info = self._fake_frame()
                self.collector.add_frame(fv, info)
            self.collector.save_sequence("hello")

        label_dir = os.path.join(self.tmp_dir, "hello")
        files = sorted(os.listdir(label_dir))
        self.assertEqual(files, ["sample_001.npy", "sample_002.npy"])

    def test_save_before_complete_raises(self):
        self.collector.start_recording()
        fv, info = self._fake_frame()
        self.collector.add_frame(fv, info)  # only 1 of 5 frames
        with self.assertRaises(ValueError):
            self.collector.save_sequence("hello")

    def test_cancel_recording_clears_buffer(self):
        self.collector.start_recording()
        fv, info = self._fake_frame()
        self.collector.add_frame(fv, info)
        self.collector.cancel_recording()
        self.assertFalse(self.collector.is_recording)
        self.assertEqual(self.collector.current_frame_count, 0)


if __name__ == "__main__":
    unittest.main()
