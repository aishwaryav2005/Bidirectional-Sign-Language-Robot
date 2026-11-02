"""Tests for src/generate_cislr_split_215class.py: duplicate-content
detection and safe assignment (synthetic fixture, includes a deliberate
duplicate pair), plus exact real-dataset counts. Never touches
data/splits/cislr_82class/, data/public_processed/, data/public_raw/, or
data/sequences/."""

import csv
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import test_helpers  # noqa: F401

import config
import generate_cislr_split_215class as gcs215

_META_FIELDS = [
    "video_id", "original_filename", "label", "class_id",
    "original_video_length_frames", "sampled_frame_indices",
    "processing_status", "failure_reason", "sequence_length",
    "feature_dimension", "left_hand_detection_rate",
    "right_hand_detection_rate", "pose_detection_rate",
    "sequence_path", "source_metadata", "processing_time_seconds",
]


def _write_video(path, content_seed, n_frames=10, width=64, height=48, fps=10):
    """Writes a small synthetic .mp4. Same content_seed -> byte-identical
    file (used to simulate exact-duplicate source videos)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (content_seed * 7 + i * 3) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestDuplicateAwareSplit(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.tmp_dir, "videos")
        self.sequences_dir = os.path.join(self.tmp_dir, "sequences")
        self.metadata_path = os.path.join(self.tmp_dir, "processing_metadata.csv")
        self.output_dir = os.path.join(self.tmp_dir, "splits", "cislr_215class")
        os.makedirs(self.videos_dir)

        # alpha: 4 samples, no duplicates.
        # beta: 4 samples, 2 of which (beta_0, beta_2) are byte-identical
        #       raw videos (content_seed=99 for both) -- the duplicate pair.
        # gamma: 5 samples, no duplicates.
        # toolow: 2 samples -- below the min_samples=4 threshold, excluded.
        self.plan = {
            "alpha": [("alpha_0", 1), ("alpha_1", 2), ("alpha_2", 3), ("alpha_3", 4)],
            "beta": [("beta_0", 99), ("beta_1", 5), ("beta_2", 99), ("beta_3", 6)],
            "gamma": [("gamma_0", 10), ("gamma_1", 11), ("gamma_2", 12), ("gamma_3", 13), ("gamma_4", 14)],
            "toolow": [("toolow_0", 20), ("toolow_1", 21)],
        }
        self._write_fixture()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_fixture(self):
        rows = []
        for gloss, members in self.plan.items():
            for video_id, content_seed in members:
                filename = f"{video_id}.mp4"
                _write_video(os.path.join(self.videos_dir, filename), content_seed)

                label_dir = os.path.join(self.sequences_dir, gloss)
                os.makedirs(label_dir, exist_ok=True)
                seq_path_abs = os.path.join(label_dir, f"{video_id}.npy")
                np.save(seq_path_abs, np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32))
                seq_path_rel = os.path.relpath(seq_path_abs, self.tmp_dir)

                rows.append({
                    "video_id": video_id, "original_filename": filename, "label": gloss,
                    "class_id": "0", "original_video_length_frames": "10", "sampled_frame_indices": "[]",
                    "processing_status": "success", "failure_reason": "",
                    "sequence_length": "60", "feature_dimension": "147",
                    "left_hand_detection_rate": "0.5", "right_hand_detection_rate": "0.5", "pose_detection_rate": "1.0",
                    "sequence_path": seq_path_rel,
                    "source_metadata": json.dumps({"category": "test", "duration": "1"}),
                    "processing_time_seconds": "0.1",
                })
        with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_META_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def test_finds_the_known_duplicate_pair(self):
        _, eligible = None, None
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        self.assertEqual(len(duplicate_groups), 1)
        self.assertEqual(duplicate_groups[0], ["beta_0", "beta_2"])

    def test_toolow_gloss_excluded(self):
        split, eligible, _, _ = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        self.assertEqual(set(eligible.keys()), {"alpha", "beta", "gamma"})
        all_glosses = {r["gloss"] for rows in split.values() for r in rows}
        self.assertNotIn("toolow", all_glosses)

    def test_duplicate_pair_never_split_across_val_test(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        beta_val = [r["video_id"] for r in split["val"] if r["gloss"] == "beta"]
        beta_test = [r["video_id"] for r in split["test"] if r["gloss"] == "beta"]
        beta_train = [r["video_id"] for r in split["train"] if r["gloss"] == "beta"]

        dup_pair = {"beta_0", "beta_2"}
        # Neither duplicate member may be in val AND the other in test (or train).
        self.assertFalse(dup_pair.issubset(set(beta_val)))
        self.assertFalse(dup_pair.issubset(set(beta_test)))
        # The pair must be entirely within ONE split.
        locations = set()
        for vid in dup_pair:
            if vid in beta_val:
                locations.add("val")
            elif vid in beta_test:
                locations.add("test")
            elif vid in beta_train:
                locations.add("train")
        self.assertEqual(len(locations), 1, f"duplicate pair split across: {locations}")

    def test_no_conflicts_when_enough_safe_samples(self):
        # beta has 2 safe (non-duplicated) samples (beta_1, beta_3) -- enough
        # to fill val+test without needing the fallback.
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        self.assertEqual(conflicts, [])

    def test_every_eligible_class_has_at_least_two_train_samples(self):
        split, eligible, _, _ = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        for gloss in eligible:
            train_count = sum(1 for r in split["train"] if r["gloss"] == gloss)
            self.assertGreaterEqual(train_count, 2, f"{gloss} has < 2 train samples")

    def test_every_class_exactly_one_val_one_test(self):
        split, eligible, _, _ = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        for gloss in eligible:
            val_count = sum(1 for r in split["val"] if r["gloss"] == gloss)
            test_count = sum(1 for r in split["test"] if r["gloss"] == gloss)
            self.assertEqual(val_count, 1)
            self.assertEqual(test_count, 1)

    def test_deterministic_with_fixed_seed(self):
        split_a, *_ = gcs215.generate_duplicate_aware_split(self.metadata_path, self.videos_dir, seed=42, min_samples=4)
        split_b, *_ = gcs215.generate_duplicate_aware_split(self.metadata_path, self.videos_dir, seed=42, min_samples=4)
        for name in ("train", "val", "test"):
            ids_a = [r["video_id"] for r in split_a[name]]
            ids_b = [r["video_id"] for r in split_b[name]]
            self.assertEqual(ids_a, ids_b)

    def test_no_video_id_overlap_across_splits(self):
        split, *_ = gcs215.generate_duplicate_aware_split(self.metadata_path, self.videos_dir, seed=42, min_samples=4)
        train_ids = {r["video_id"] for r in split["train"]}
        val_ids = {r["video_id"] for r in split["val"]}
        test_ids = {r["video_id"] for r in split["test"]}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(val_ids & test_ids, set())

    def test_write_and_validate_end_to_end_no_cross_split_duplicate_content(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        gcs215.write_manifests(split, self.output_dir)
        gcs215.write_split_metadata_215(
            self.output_dir, split, eligible, duplicate_groups, conflicts,
            seed=42, min_samples=4, metadata_path=self.metadata_path, videos_root=self.videos_dir,
        )
        report, loaded = gcs215.validate_manifests_with_duplicates(self.output_dir, self.videos_dir, self.metadata_path, project_root=self.tmp_dir)
        self.assertEqual(report["cross_split_duplicate_content"], [])
        self.assertTrue(report["passed"])

    def test_split_metadata_documents_duplicates(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            self.metadata_path, self.videos_dir, seed=42, min_samples=4
        )
        gcs215.write_manifests(split, self.output_dir)
        _, info = gcs215.write_split_metadata_215(
            self.output_dir, split, eligible, duplicate_groups, conflicts,
            seed=42, min_samples=4, metadata_path=self.metadata_path, videos_root=self.videos_dir,
        )
        self.assertEqual(info["duplicate_handling"]["num_duplicate_groups"], 1)
        self.assertIn(["beta_0", "beta_2"], info["duplicate_handling"]["duplicate_groups_found"])
        self.assertFalse(info["signer_independent_splitting"])


class TestRealDataset215Class(unittest.TestCase):
    """Verifies the exact counts against the REAL processed CISLR
    metadata and videos. Skipped automatically if that data isn't
    present. Read-only: never writes into data/splits/cislr_215class/
    (uses a temp output dir)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(config.PUBLIC_METADATA_PATH):
            raise unittest.SkipTest("Real processed CISLR metadata not found")
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_exact_215_classes_1024_samples(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            config.PUBLIC_METADATA_PATH, gcs215.DEFAULT_VIDEOS_ROOT, seed=42, min_samples=4
        )
        self.assertEqual(len(eligible), 215)
        total = sum(len(rows) for rows in split.values())
        self.assertEqual(total, 1024)

    def test_exact_594_215_215_split(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            config.PUBLIC_METADATA_PATH, gcs215.DEFAULT_VIDEOS_ROOT, seed=42, min_samples=4
        )
        self.assertEqual(len(split["train"]), 594)
        self.assertEqual(len(split["val"]), 215)
        self.assertEqual(len(split["test"]), 215)

    def test_finds_exactly_the_known_green_duplicate(self):
        split, eligible, duplicate_groups, conflicts = gcs215.generate_duplicate_aware_split(
            config.PUBLIC_METADATA_PATH, gcs215.DEFAULT_VIDEOS_ROOT, seed=42, min_samples=4
        )
        self.assertEqual(len(duplicate_groups), 1)
        self.assertEqual(duplicate_groups[0], ["D9ihUabRKIQ_2", "wwQV7dGvw0s_2"])
        self.assertEqual(conflicts, [])  # green had enough safe samples, no fallback needed

    def test_82class_split_untouched(self):
        path_82 = os.path.join(config.DATA_DIR, "splits", "cislr_82class", "train.csv")
        with open(path_82, newline="", encoding="utf-8") as f:
            rows_before = list(csv.DictReader(f))
        gcs215.generate_duplicate_aware_split(
            config.PUBLIC_METADATA_PATH, gcs215.DEFAULT_VIDEOS_ROOT, seed=42, min_samples=4
        )
        with open(path_82, newline="", encoding="utf-8") as f:
            rows_after = list(csv.DictReader(f))
        self.assertEqual(rows_before, rows_after)


if __name__ == "__main__":
    unittest.main()
