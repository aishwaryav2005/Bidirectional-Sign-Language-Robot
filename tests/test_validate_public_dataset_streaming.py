"""Tests for src/validate_public_dataset_streaming.py using small synthetic
fixtures (not real dataset content). Confirms it's read-only and correctly
flags every failure mode it claims to check."""

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
from validate_public_dataset_streaming import validate, _all_checks_passed


def _write_npy(path, shape=(60, 147), fill=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.full(shape, fill, dtype=np.float32))


def _write_metadata(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["video_id", "processing_status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_dataset_csv(path, n_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "gloss", "duration", "category"])
        for i in range(n_rows):
            w.writerow([f"vid{i}", f"gloss{i}", "5", "misc"])


class TestStreamingValidator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.sequences_dir = os.path.join(self.tmp_dir, "sequences")
        self.metadata_path = os.path.join(self.tmp_dir, "metadata", "processing_metadata.csv")
        self.dataset_csv = os.path.join(self.tmp_dir, "dataset.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, progress_every=0):
        return validate(
            sequences_dir=self.sequences_dir,
            metadata_path=self.metadata_path,
            dataset_csv_path=self.dataset_csv,
            progress_every=progress_every,
        )

    def test_all_valid_passes(self):
        for i in range(3):
            _write_npy(os.path.join(self.sequences_dir, "label", f"vid{i}.npy"))
        _write_metadata(self.metadata_path, [
            {"video_id": f"vid{i}", "processing_status": "success"} for i in range(3)
        ])
        _write_dataset_csv(self.dataset_csv, 3)

        report = self._run()
        self.assertEqual(report["total_files"], 3)
        self.assertEqual(report["correct_shape"], 3)
        self.assertEqual(report["wrong_shape"], 0)
        self.assertEqual(report["nan_count"], 0)
        self.assertEqual(report["inf_count"], 0)
        self.assertEqual(report["missing_metadata"], 0)
        self.assertEqual(report["orphan_metadata"], 0)
        self.assertTrue(_all_checks_passed(report))

    def test_detects_wrong_shape(self):
        _write_npy(os.path.join(self.sequences_dir, "label", "vid0.npy"), shape=(30, 147))
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["wrong_shape"], 1)
        self.assertEqual(report["correct_shape"], 0)
        self.assertFalse(_all_checks_passed(report))

    def test_detects_nan(self):
        path = os.path.join(self.sequences_dir, "label", "vid0.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32)
        arr[0, 0] = np.nan
        np.save(path, arr)
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["nan_count"], 1)
        self.assertFalse(_all_checks_passed(report))

    def test_detects_inf(self):
        path = os.path.join(self.sequences_dir, "label", "vid0.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32)
        arr[0, 0] = np.inf
        np.save(path, arr)
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["inf_count"], 1)
        self.assertFalse(_all_checks_passed(report))

    def test_detects_missing_metadata(self):
        """An .npy file exists on disk with no corresponding metadata row."""
        _write_npy(os.path.join(self.sequences_dir, "label", "vid0.npy"))
        _write_metadata(self.metadata_path, [])  # no rows at all
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["missing_metadata"], 1)
        self.assertFalse(_all_checks_passed(report))

    def test_detects_orphan_metadata(self):
        """A metadata row claims success but the .npy file doesn't exist."""
        _write_metadata(self.metadata_path, [{"video_id": "ghost", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["orphan_metadata"], 1)
        self.assertFalse(_all_checks_passed(report))

    def test_detects_duplicate_on_disk(self):
        _write_npy(os.path.join(self.sequences_dir, "label_a", "vid0.npy"))
        _write_npy(os.path.join(self.sequences_dir, "label_b", "vid0.npy"))
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        report = self._run()
        self.assertEqual(report["duplicate_video_ids_on_disk"], ["vid0"])
        self.assertFalse(_all_checks_passed(report))

    def test_detects_count_mismatch_against_dataset_csv(self):
        """Fewer .npy files than dataset.csv rows must fail overall, even
        if every individual file that DOES exist is perfectly valid."""
        _write_npy(os.path.join(self.sequences_dir, "label", "vid0.npy"))
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 5)  # dataset.csv claims 5, only 1 processed

        report = self._run()
        self.assertEqual(report["expected_total"], 5)
        self.assertEqual(report["total_files"], 1)
        self.assertFalse(_all_checks_passed(report))

    def test_read_only_does_not_modify_anything(self):
        path = os.path.join(self.sequences_dir, "label", "vid0.npy")
        _write_npy(path)
        _write_metadata(self.metadata_path, [{"video_id": "vid0", "processing_status": "success"}])
        _write_dataset_csv(self.dataset_csv, 1)

        before_npy = os.path.getmtime(path)
        before_meta = os.path.getmtime(self.metadata_path)
        self._run()
        self.assertEqual(os.path.getmtime(path), before_npy)
        self.assertEqual(os.path.getmtime(self.metadata_path), before_meta)


if __name__ == "__main__":
    unittest.main()
