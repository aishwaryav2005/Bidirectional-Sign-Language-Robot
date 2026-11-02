"""Tests for src/generate_cislr_split.py using a small synthetic metadata
fixture (not the real 7,050-row dataset), so the suite stays fast and
self-contained. Never touches data/public_processed/, data/public_raw/,
or data/sequences/."""

import csv
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
import generate_cislr_split as gcs

_META_FIELDS = [
    "video_id", "original_filename", "label", "class_id",
    "original_video_length_frames", "sampled_frame_indices",
    "processing_status", "failure_reason", "sequence_length",
    "feature_dimension", "left_hand_detection_rate",
    "right_hand_detection_rate", "pose_detection_rate",
    "sequence_path", "source_metadata", "processing_time_seconds",
]


class TestGenerateCislrSplit(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.sequences_dir = os.path.join(self.tmp_dir, "sequences")
        self.metadata_path = os.path.join(self.tmp_dir, "processing_metadata.csv")
        self.output_dir = os.path.join(self.tmp_dir, "splits", "cislr_82class")

        # Synthetic vocabulary: 5 eligible glosses (>=5 samples) with
        # varying counts, plus 2 ineligible glosses (1 and 4 samples)
        # that must be excluded entirely.
        self.plan = {
            "alpha": 5,
            "beta": 6,
            "gamma": 7,
            "delta": 5,
            "epsilon": 8,
            "toolow_one": 1,
            "toolow_four": 4,
        }
        self._write_fixture()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_fixture(self):
        rows = []
        for gloss, n in self.plan.items():
            for i in range(n):
                video_id = f"{gloss}_{i:02d}"
                label_dir = os.path.join(self.sequences_dir, gloss)
                os.makedirs(label_dir, exist_ok=True)
                seq_path_abs = os.path.join(label_dir, f"{video_id}.npy")
                np.save(seq_path_abs, np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32))
                seq_path_rel = os.path.relpath(seq_path_abs, self.tmp_dir)
                rows.append({
                    "video_id": video_id,
                    "original_filename": f"{video_id}.mp4",
                    "label": gloss,
                    "class_id": "0",
                    "original_video_length_frames": "90",
                    "sampled_frame_indices": "[]",
                    "processing_status": "success",
                    "failure_reason": "",
                    "sequence_length": "60",
                    "feature_dimension": "147",
                    "left_hand_detection_rate": "0.5",
                    "right_hand_detection_rate": "0.5",
                    "pose_detection_rate": "1.0",
                    "sequence_path": seq_path_rel,
                    "source_metadata": json.dumps({"category": "test_cat", "duration": "5"}),
                    "processing_time_seconds": "1.0",
                })
        with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_META_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _eligible_counts(self):
        return {g: n for g, n in self.plan.items() if n >= 5}

    def test_only_eligible_glosses_included(self):
        split, eligible = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        self.assertEqual(set(eligible.keys()), set(self._eligible_counts().keys()))
        all_glosses_in_output = {r["gloss"] for rows in split.values() for r in rows}
        self.assertNotIn("toolow_one", all_glosses_in_output)
        self.assertNotIn("toolow_four", all_glosses_in_output)

    def test_exact_class_and_sample_counts(self):
        split, eligible = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        self.assertEqual(len(eligible), 5)  # alpha, beta, gamma, delta, epsilon
        total_expected = sum(self._eligible_counts().values())  # 5+6+7+5+8 = 31
        total_actual = sum(len(rows) for rows in split.values())
        self.assertEqual(total_actual, total_expected)
        self.assertEqual(len(split["val"]), 5)   # exactly 1 per eligible gloss
        self.assertEqual(len(split["test"]), 5)  # exactly 1 per eligible gloss
        self.assertEqual(len(split["train"]), total_expected - 10)

    def test_every_class_in_all_three_splits_with_min_one_train(self):
        split, eligible = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        for gloss in eligible:
            for name in ("train", "val", "test"):
                glosses_here = {r["gloss"] for r in split[name]}
                self.assertIn(gloss, glosses_here, f"{gloss} missing from {name}")
            train_count_for_gloss = sum(1 for r in split["train"] if r["gloss"] == gloss)
            self.assertGreaterEqual(train_count_for_gloss, 1)
            val_count_for_gloss = sum(1 for r in split["val"] if r["gloss"] == gloss)
            test_count_for_gloss = sum(1 for r in split["test"] if r["gloss"] == gloss)
            self.assertEqual(val_count_for_gloss, 1)
            self.assertEqual(test_count_for_gloss, 1)

    def test_no_video_id_overlap_across_splits(self):
        split, _ = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        train_ids = {r["video_id"] for r in split["train"]}
        val_ids = {r["video_id"] for r in split["val"]}
        test_ids = {r["video_id"] for r in split["test"]}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(val_ids & test_ids, set())
        # And no duplicates within a single split.
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            raw_list = [r["video_id"] for r in split[name]]
            self.assertEqual(len(raw_list), len(set(raw_list)), f"duplicate video_id within {name}")

    def test_deterministic_with_fixed_seed(self):
        split_a, _ = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        split_b, _ = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        for name in ("train", "val", "test"):
            ids_a = [r["video_id"] for r in split_a[name]]
            ids_b = [r["video_id"] for r in split_b[name]]
            self.assertEqual(ids_a, ids_b, f"{name} split differs across identical-seed runs")

    def test_different_seed_can_change_val_test_assignment(self):
        split_42, _ = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        split_7, _ = gcs.generate_split(self.metadata_path, seed=7, min_samples=5)
        val_42 = [r["video_id"] for r in split_42["val"]]
        val_7 = [r["video_id"] for r in split_7["val"]]
        # Not a strict requirement that they must differ, but with 5
        # eligible glosses and small pools it's overwhelmingly likely;
        # this guards against the seed being silently ignored.
        self.assertNotEqual(val_42, val_7)

    def test_original_metadata_file_not_modified(self):
        with open(self.metadata_path, "rb") as f:
            before = f.read()
        before_mtime = os.path.getmtime(self.metadata_path)

        gcs.generate_split(self.metadata_path, seed=42, min_samples=5)

        with open(self.metadata_path, "rb") as f:
            after = f.read()
        self.assertEqual(before, after)
        self.assertEqual(os.path.getmtime(self.metadata_path), before_mtime)

    def test_write_and_validate_manifests_end_to_end(self):
        split, eligible = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        manifest_paths = gcs.write_manifests(split, self.output_dir)
        for name in ("train", "val", "test"):
            self.assertTrue(os.path.exists(manifest_paths[name]))

        meta_json_path, info = gcs.write_split_metadata(
            self.output_dir, split, eligible, seed=42, min_samples=5, metadata_path=self.metadata_path
        )
        self.assertTrue(os.path.exists(meta_json_path))
        with open(meta_json_path) as f:
            saved_info = json.load(f)
        self.assertEqual(saved_info["num_classes"], 5)
        self.assertEqual(saved_info["random_seed"], 42)
        self.assertFalse(saved_info["signer_independent_splitting"])
        self.assertIn("vocabulary_rule", saved_info)
        self.assertIn("timestamp_utc", saved_info)

        report, loaded = gcs.validate_manifests(self.output_dir, project_root=self.tmp_dir)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["duplicate_ids_within_split"], {})
        self.assertEqual(report["cross_split_overlap"], [])
        self.assertEqual(report["missing_sequence_files"], [])
        self.assertEqual(report["classes_missing_from_a_split"], [])

    def test_validate_manifests_detects_missing_sequence_file(self):
        split, eligible = gcs.generate_split(self.metadata_path, seed=42, min_samples=5)
        gcs.write_manifests(split, self.output_dir)
        gcs.write_split_metadata(self.output_dir, split, eligible, seed=42, min_samples=5, metadata_path=self.metadata_path)

        # Delete one referenced sequence file to prove the check catches it.
        victim = split["train"][0]
        victim_path = os.path.join(self.tmp_dir, victim["sequence_path"])
        os.remove(victim_path)

        report, _ = gcs.validate_manifests(self.output_dir, project_root=self.tmp_dir)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing_sequence_files"]), 1)
        self.assertEqual(report["missing_sequence_files"][0]["video_id"], victim["video_id"])


if __name__ == "__main__":
    unittest.main()
