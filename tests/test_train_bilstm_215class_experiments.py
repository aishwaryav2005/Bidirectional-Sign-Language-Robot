"""Tests for src/train_bilstm_215class_experiments.py. Uses small
synthetic fixtures for fast checks, plus real-data checks (skipped
automatically if the 215-class split/experiment artifacts don't exist
yet) for the exact 215-class output dimension and selected-model
verification. Never touches data/splits/cislr_82class/,
models/cislr_bilstm_baseline/, or models/cislr_bilstm_improved/."""

import csv
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
import train_bilstm_215class_experiments as exp215
from train_bilstm_baseline import build_model as build_baseline_model
from train_bilstm_improved import build_regularized_bilstm_model, build_gru_model, build_augmented_training_set

_MANIFEST_FIELDS = ["video_id", "gloss", "category", "sequence_path", "split"]


def _write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TestModelDimensionsAt215(unittest.TestCase):
    """The 215-class experiment reuses generic model-builders from
    train_bilstm_baseline.py / train_bilstm_improved.py -- verify they
    behave correctly at num_classes=215 specifically (not just the
    smaller numbers already exercised by their own test suites)."""

    def test_baseline_model_215_output(self):
        model = build_baseline_model(num_classes=215)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 215))

    def test_regularized_model_215_output(self):
        model = build_regularized_bilstm_model(num_classes=215)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 215))

    def test_gru_model_215_output(self):
        model = build_gru_model(num_classes=215)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 215))


class TestManifestLoadingAndClassMapping(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_manifest_reads_train_val_test(self):
        for name, gloss in [("train", "alpha"), ("val", "alpha"), ("test", "alpha")]:
            path = os.path.join(self.tmp_dir, f"{name}.csv")
            _write_manifest(path, [{"video_id": f"{name}_1", "gloss": gloss, "category": "", "sequence_path": "", "split": name}])
        train_rows = exp215.load_manifest(os.path.join(self.tmp_dir, "train.csv"))
        val_rows = exp215.load_manifest(os.path.join(self.tmp_dir, "val.csv"))
        test_rows = exp215.load_manifest(os.path.join(self.tmp_dir, "test.csv"))
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(len(val_rows), 1)
        self.assertEqual(len(test_rows), 1)

    def test_class_mapping_reproducible(self):
        manifest = [
            {"video_id": "1", "gloss": "zeta", "category": "", "sequence_path": "", "split": "train"},
            {"video_id": "2", "gloss": "alpha", "category": "", "sequence_path": "", "split": "train"},
        ]
        m1, _ = exp215.build_class_mapping(manifest)
        m2, _ = exp215.build_class_mapping(manifest)
        self.assertEqual(m1, m2)
        self.assertEqual(m1, {"alpha": 0, "zeta": 1})


class TestAugmentationIsolation(unittest.TestCase):
    """Confirms, in this script's specific calling pattern, that
    augmentation only ever touches training data."""

    def _make_seq(self, rng):
        seq = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32)
        seq[:, :config.HAND_FEATURE_LEN] = rng.normal(0, 0.3, size=(config.SEQUENCE_LENGTH, config.HAND_FEATURE_LEN))
        flags_start = config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN
        seq[:, flags_start] = 1.0
        return seq

    def test_augmentation_only_affects_training_set(self):
        rng = np.random.default_rng(0)
        X_train = np.stack([self._make_seq(rng) for _ in range(4)])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.stack([self._make_seq(rng) for _ in range(2)])
        X_test = np.stack([self._make_seq(rng) for _ in range(2)])
        X_val_before, X_test_before = X_val.copy(), X_test.copy()

        X_train_aug, y_train_aug = build_augmented_training_set(X_train, y_train, seed=42, augment_factor=3)

        self.assertEqual(X_train_aug.shape[0], 12)  # 4 original + 4 + 4
        np.testing.assert_array_equal(X_val, X_val_before)
        np.testing.assert_array_equal(X_test, X_test_before)

    def test_no_nan_or_inf_after_augmentation(self):
        rng = np.random.default_rng(0)
        X_train = np.stack([self._make_seq(rng) for _ in range(5)])
        y_train = np.array([0, 1, 2, 0, 1])
        X_train_aug, _ = build_augmented_training_set(X_train, y_train, seed=42, augment_factor=3)
        self.assertFalse(np.any(np.isnan(X_train_aug)))
        self.assertFalse(np.any(np.isinf(X_train_aug)))

    def test_deterministic_with_seed_42(self):
        rng = np.random.default_rng(0)
        X_train = np.stack([self._make_seq(rng) for _ in range(4)])
        y_train = np.array([0, 1, 0, 1])
        X_aug_1, _ = build_augmented_training_set(X_train, y_train, seed=42, augment_factor=2)
        X_aug_2, _ = build_augmented_training_set(X_train, y_train, seed=42, augment_factor=2)
        np.testing.assert_array_equal(X_aug_1, X_aug_2)


class TestExperimentResultGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_experiment_and_json_round_trip_tiny_synthetic(self):
        from train_bilstm_improved import run_experiment
        exp215.set_global_seed(42)
        rng = np.random.default_rng(0)

        def make(n):
            return np.stack([np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32) + rng.normal(0, 0.1) for _ in range(n)])

        num_classes = 3
        X_train, y_train = make(9), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        X_val, y_val = make(3), np.array([0, 1, 2])
        X_test, y_test = make(3), np.array([0, 1, 2])
        class_names = ["a", "b", "c"]

        model = build_gru_model(num_classes=num_classes, gru1_units=4, gru2_units=4, normalization_adapt_data=X_train)
        result = run_experiment(
            "tiny", model, X_train, y_train, X_val, y_val, X_test, y_test,
            class_names, self.tmp_dir, max_epochs=2, patience=2, batch_size=3,
        )
        self.assertIn("val_accuracy", result)
        self.assertIn("test_accuracy", result)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "tiny", "model.keras")))

        serializable = {k: v for k, v in result.items() if not k.startswith("_")}
        path = os.path.join(self.tmp_dir, "results.json")
        exp215.save_json(path, {"experiments": {"tiny": serializable}})
        with open(path) as f:
            loaded = json.load(f)
        self.assertNotIn("_test_confusion_matrix", loaded["experiments"]["tiny"])


@unittest.skipUnless(
    os.path.exists(os.path.join(config.DATA_DIR, "splits", "cislr_215class", "train.csv")),
    "215-class split not found",
)
class TestRealSplitLoading(unittest.TestCase):
    """Verifies the actual 215-class manifests load correctly. Read-only."""

    def test_real_manifest_counts(self):
        splits_dir = os.path.join(config.DATA_DIR, "splits", "cislr_215class")
        train_rows = exp215.load_manifest(os.path.join(splits_dir, "train.csv"))
        val_rows = exp215.load_manifest(os.path.join(splits_dir, "val.csv"))
        test_rows = exp215.load_manifest(os.path.join(splits_dir, "test.csv"))
        self.assertEqual(len(train_rows), 594)
        self.assertEqual(len(val_rows), 215)
        self.assertEqual(len(test_rows), 215)

    def test_real_class_mapping_has_215_classes(self):
        splits_dir = os.path.join(config.DATA_DIR, "splits", "cislr_215class")
        train_rows = exp215.load_manifest(os.path.join(splits_dir, "train.csv"))
        val_rows = exp215.load_manifest(os.path.join(splits_dir, "val.csv"))
        test_rows = exp215.load_manifest(os.path.join(splits_dir, "test.csv"))
        gloss_to_id, _ = exp215.build_class_mapping(train_rows, val_rows, test_rows)
        self.assertEqual(len(gloss_to_id), 215)


@unittest.skipUnless(
    os.path.exists(os.path.join(config.PROJECT_ROOT, "models", "cislr_215class_experiments", "experiment_results.json")),
    "215-class experiment has not been run yet",
)
class TestRealExperimentArtifacts(unittest.TestCase):
    """Verifies the actual completed experiment's artifacts. Read-only;
    runs only after the real experiment script has been executed."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(config.PROJECT_ROOT, "models", "cislr_215class_experiments")
        with open(os.path.join(cls.output_dir, "experiment_results.json"), encoding="utf-8") as f:
            cls.results = json.load(f)

    def test_selected_model_artifact_exists(self):
        model_path = os.path.join(self.output_dir, "selected_model", "model.keras")
        self.assertTrue(os.path.exists(model_path))

    def test_selected_model_uses_215_class_output(self):
        import keras
        model_path = os.path.join(self.output_dir, "selected_model", "model.keras")
        model = keras.models.load_model(model_path)
        self.assertEqual(model.output_shape, (None, 215))
        self.assertEqual(model.input_shape, (None, 60, 147))

    def test_experiment_results_json_has_all_four_candidates(self):
        self.assertEqual(self.results["num_classes"], 215)
        for name in ("baseline_bilstm", "regularized_bilstm", "augmented_bilstm", "gru"):
            self.assertIn(name, self.results["experiments"])

    def test_selection_used_validation_not_test(self):
        self.assertIn("validation accuracy only", self.results["selection_criterion"])

    def test_82class_reference_present_and_labeled_not_comparable(self):
        self.assertIn("reference_82class_experiment", self.results)
        self.assertEqual(self.results["reference_82class_experiment"]["num_classes"], 82)
        self.assertIn("NOT directly comparable", self.results["comparison_note"])


if __name__ == "__main__":
    unittest.main()
