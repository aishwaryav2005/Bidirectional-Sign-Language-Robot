"""Tests for src/train_bilstm_baseline.py using a small synthetic
manifest/sequence fixture (not the real 492-sample CISLR subset), so the
suite stays fast. Never touches data/public_processed/, data/public_raw/,
data/sequences/, or data/splits/cislr_82class/."""

import csv
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
import train_bilstm_baseline as tbb

_MANIFEST_FIELDS = ["video_id", "gloss", "category", "sequence_path", "split"]


def _write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TestManifestAndClassMapping(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_manifest_reads_all_rows_in_order(self):
        path = os.path.join(self.tmp_dir, "train.csv")
        rows = [
            {"video_id": "a1", "gloss": "hello", "category": "greeting", "sequence_path": "x/a1.npy", "split": "train"},
            {"video_id": "a2", "gloss": "thanks", "category": "greeting", "sequence_path": "x/a2.npy", "split": "train"},
        ]
        _write_manifest(path, rows)
        loaded = tbb.load_manifest(path)
        self.assertEqual([r["video_id"] for r in loaded], ["a1", "a2"])
        self.assertEqual([r["gloss"] for r in loaded], ["hello", "thanks"])

    def test_class_mapping_is_deterministic_alphabetical(self):
        manifest = [
            {"video_id": "1", "gloss": "zebra", "category": "", "sequence_path": "", "split": "train"},
            {"video_id": "2", "gloss": "apple", "category": "", "sequence_path": "", "split": "train"},
            {"video_id": "3", "gloss": "mango", "category": "", "sequence_path": "", "split": "train"},
        ]
        gloss_to_id, id_to_gloss = tbb.build_class_mapping(manifest)
        self.assertEqual(gloss_to_id, {"apple": 0, "mango": 1, "zebra": 2})
        self.assertEqual(id_to_gloss, {0: "apple", 1: "mango", 2: "zebra"})

    def test_class_mapping_stable_across_repeated_calls(self):
        manifest = [{"video_id": str(i), "gloss": g, "category": "", "sequence_path": "", "split": "train"}
                    for i, g in enumerate(["dog", "cat", "bird", "ant"])]
        m1, _ = tbb.build_class_mapping(manifest)
        m2, _ = tbb.build_class_mapping(manifest)
        self.assertEqual(m1, m2)

    def test_class_mapping_union_across_multiple_manifests(self):
        train = [{"video_id": "1", "gloss": "cat", "category": "", "sequence_path": "", "split": "train"}]
        val = [{"video_id": "2", "gloss": "dog", "category": "", "sequence_path": "", "split": "val"}]
        gloss_to_id, _ = tbb.build_class_mapping(train, val)
        self.assertEqual(gloss_to_id, {"cat": 0, "dog": 1})

    def test_class_mapping_not_dependent_on_row_order(self):
        forward = [{"video_id": "1", "gloss": "b", "category": "", "sequence_path": "", "split": "t"},
                   {"video_id": "2", "gloss": "a", "category": "", "sequence_path": "", "split": "t"}]
        reversed_rows = list(reversed(forward))
        m1, _ = tbb.build_class_mapping(forward)
        m2, _ = tbb.build_class_mapping(reversed_rows)
        self.assertEqual(m1, m2)


class TestSequenceLoading(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_sequence_accepts_correct_shape(self):
        path = os.path.join(self.tmp_dir, "ok.npy")
        np.save(path, np.zeros((60, 147), dtype=np.float32))
        arr = tbb.load_sequence("ok.npy", self.tmp_dir)
        self.assertEqual(arr.shape, (60, 147))
        self.assertEqual(arr.dtype, np.float32)

    def test_load_sequence_rejects_wrong_shape(self):
        path = os.path.join(self.tmp_dir, "bad.npy")
        np.save(path, np.zeros((30, 147), dtype=np.float32))
        with self.assertRaises(ValueError):
            tbb.load_sequence("bad.npy", self.tmp_dir)

    def test_load_split_arrays_shapes_and_labels(self):
        gloss_to_id = {"alpha": 0, "beta": 1}
        rows = []
        for gloss in ("alpha", "beta"):
            for i in range(2):
                vid = f"{gloss}_{i}"
                path = os.path.join(self.tmp_dir, f"{vid}.npy")
                np.save(path, np.full((60, 147), gloss_to_id[gloss], dtype=np.float32))
                rows.append({"video_id": vid, "gloss": gloss, "category": "", "sequence_path": f"{vid}.npy", "split": "train"})

        X, y, video_ids = tbb.load_split_arrays(rows, gloss_to_id, self.tmp_dir)
        self.assertEqual(X.shape, (4, 60, 147))
        self.assertEqual(y.shape, (4,))
        self.assertEqual(video_ids, [r["video_id"] for r in rows])
        # values were filled with their class id -- confirms correct file->label pairing
        np.testing.assert_array_equal(X[0], np.full((60, 147), 0, dtype=np.float32))
        np.testing.assert_array_equal(X[2], np.full((60, 147), 1, dtype=np.float32))
        np.testing.assert_array_equal(y, [0, 0, 1, 1])

    def test_load_split_arrays_only_touches_given_rows(self):
        """Regression guard: must not scan/load anything beyond the rows
        it's given (i.e. never the full processed dataset)."""
        gloss_to_id = {"only": 0}
        path = os.path.join(self.tmp_dir, "only.npy")
        np.save(path, np.zeros((60, 147), dtype=np.float32))
        # An unrelated, unreferenced .npy file sitting in the same dir.
        np.save(os.path.join(self.tmp_dir, "unrelated.npy"), np.ones((60, 147), dtype=np.float32))

        rows = [{"video_id": "only", "gloss": "only", "category": "", "sequence_path": "only.npy", "split": "train"}]
        X, y, video_ids = tbb.load_split_arrays(rows, gloss_to_id, self.tmp_dir)
        self.assertEqual(X.shape[0], 1)
        self.assertEqual(video_ids, ["only"])


class TestModel(unittest.TestCase):
    def test_build_model_input_output_shapes(self):
        model = tbb.build_model(num_classes=5, lstm1_units=8, lstm2_units=4)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 5))

    def test_build_model_param_count_is_reasonable(self):
        model = tbb.build_model(num_classes=82, lstm1_units=64, lstm2_units=32)
        n_params = model.count_params()
        # Should be a compact model: comfortably under 1M params for this size.
        self.assertGreater(n_params, 0)
        self.assertLess(n_params, 1_000_000)

    def test_model_fits_and_predicts_on_tiny_synthetic_data(self):
        """Fast smoke test: end-to-end fit + predict on a handful of
        synthetic samples, not the real dataset -- proves the training
        loop and output shapes are wired correctly."""
        tbb.set_global_seed(42)
        rng = np.random.default_rng(0)
        num_classes = 3
        X_train = rng.normal(size=(9, 60, 147)).astype(np.float32)
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        X_val = rng.normal(size=(3, 60, 147)).astype(np.float32)
        y_val = np.array([0, 1, 2], dtype=np.int64)

        model = tbb.build_model(num_classes=num_classes, lstm1_units=4, lstm2_units=4, normalization_adapt_data=X_train)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=3, verbose=0)

        self.assertIn("loss", history.history)
        self.assertIn("val_accuracy", history.history)

        preds = model.predict(X_val, verbose=0)
        self.assertEqual(preds.shape, (3, num_classes))
        # softmax outputs should sum to ~1 per row
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(3), atol=1e-4)


class TestEvaluationAndArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_evaluate_predictions_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        results = tbb.evaluate_predictions(y_true, y_pred, class_names=["a", "b", "c"])
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["macro_f1"], 1.0)
        self.assertEqual(results["weighted_f1"], 1.0)
        self.assertEqual(results["confusion_matrix"].shape, (3, 3))
        self.assertEqual(results["confusion_matrix"].sum(), 6)

    def test_evaluate_predictions_detects_errors(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 2, 2])  # one wrong
        results = tbb.evaluate_predictions(y_true, y_pred, class_names=["a", "b", "c"])
        self.assertLess(results["accuracy"], 1.0)
        self.assertIn("precision", results["classification_report"])

    def test_save_json_creates_file_and_parent_dirs(self):
        path = os.path.join(self.tmp_dir, "nested", "dir", "out.json")
        tbb.save_json(path, {"a": 1, "b": [1, 2, 3]})
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded, {"a": 1, "b": [1, 2, 3]})

    def test_save_confusion_matrix_plot_creates_png(self):
        cm = np.array([[3, 1], [0, 4]])
        out_path = os.path.join(self.tmp_dir, "cm.png")
        tbb.save_confusion_matrix_plot(cm, ["classA", "classB"], out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)


class TestSeeding(unittest.TestCase):
    def test_set_global_seed_is_reproducible(self):
        tbb.set_global_seed(42)
        a1 = np.random.rand(5)
        tbb.set_global_seed(42)
        a2 = np.random.rand(5)
        np.testing.assert_array_equal(a1, a2)


if __name__ == "__main__":
    unittest.main()
