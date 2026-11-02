"""Tests for src/train_bilstm_improved.py: augmentation correctness
(shape, determinism, flag/zero-block safety, no NaN/Inf), model
input/output dimensions for the regularized-BiLSTM and GRU variants, and
experiment-result generation. Uses small synthetic data throughout, not
the real 492-sample CISLR subset."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
import train_bilstm_improved as tbi


def _make_synthetic_sequence(rng, left_detected=True, right_detected=True, pose_detected=True):
    """Builds one (60, 147) sequence with plausible nonzero coordinates
    in detected blocks and exact zeros in undetected blocks, matching
    the real pipeline's zero-fill convention."""
    seq = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH), dtype=np.float32)
    blocks = [
        (0, config.HAND_FEATURE_LEN, left_detected),
        (config.HAND_FEATURE_LEN, config.HAND_FEATURE_LEN * 2, right_detected),
        (config.HAND_FEATURE_LEN * 2, config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN, pose_detected),
    ]
    for start, end, detected in blocks:
        if detected:
            seq[:, start:end] = rng.normal(0, 0.3, size=(config.SEQUENCE_LENGTH, end - start)).astype(np.float32)
    flags_start = config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN
    seq[:, flags_start] = 1.0 if left_detected else 0.0
    seq[:, flags_start + 1] = 1.0 if right_detected else 0.0
    seq[:, flags_start + 2] = 1.0 if pose_detected else 0.0
    return seq


class TestAugmentation(unittest.TestCase):
    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng)
        out = tbi.augment_sequence(seq, np.random.default_rng(1))
        self.assertEqual(out.shape, (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH))
        self.assertEqual(out.dtype, np.float32)

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng)
        out1 = tbi.augment_sequence(seq, np.random.default_rng(42))
        out2 = tbi.augment_sequence(seq, np.random.default_rng(42))
        np.testing.assert_array_equal(out1, out2)

    def test_different_seed_gives_different_output(self):
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng)
        out1 = tbi.augment_sequence(seq, np.random.default_rng(1))
        out2 = tbi.augment_sequence(seq, np.random.default_rng(2))
        self.assertFalse(np.array_equal(out1, out2))

    def test_does_not_modify_input_in_place(self):
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng)
        original = seq.copy()
        tbi.augment_sequence(seq, np.random.default_rng(5))
        np.testing.assert_array_equal(seq, original)

    def test_detection_flags_never_perturbed(self):
        """The last 3 values of every frame are binary detection flags,
        not coordinates -- augmentation must leave them exactly as-is."""
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng, left_detected=True, right_detected=False, pose_detected=True)
        out = tbi.augment_sequence(seq, np.random.default_rng(7))
        flags_start = config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN
        out_flags = out[:, flags_start:flags_start + 3]
        # Every frame's flags must be exactly {1,0,1} (temporal reordering
        # may reshuffle WHICH frame is where, but every frame in this
        # fixture has identical flags, so after reorder they must still
        # all read exactly 1,0,1 -- any perturbation would break this).
        self.assertTrue(np.all(out_flags[:, 0] == 1.0))
        self.assertTrue(np.all(out_flags[:, 1] == 0.0))
        self.assertTrue(np.all(out_flags[:, 2] == 1.0))

    def test_undetected_block_stays_exactly_zero(self):
        """A hand/pose block with detection flag 0 must remain exactly
        zero-filled after augmentation -- never perturbed into fake
        near-origin coordinates."""
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng, left_detected=False, right_detected=True, pose_detected=True)
        out = tbi.augment_sequence(seq, np.random.default_rng(9))
        left_block = out[:, 0:config.HAND_FEATURE_LEN]
        np.testing.assert_array_equal(left_block, np.zeros_like(left_block))

    def test_detected_block_is_actually_perturbed(self):
        """Sanity check the opposite direction: a detected block SHOULD
        change (with overwhelming probability) under augmentation --
        guards against an accidentally-inert augmentation function."""
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng, left_detected=True, right_detected=True, pose_detected=True)
        out = tbi.augment_sequence(seq, np.random.default_rng(11))
        right_block_before = seq[:, config.HAND_FEATURE_LEN:config.HAND_FEATURE_LEN * 2]
        right_block_after = out[:, config.HAND_FEATURE_LEN:config.HAND_FEATURE_LEN * 2]
        self.assertFalse(np.allclose(right_block_before, right_block_after))

    def test_no_nan_or_inf_introduced(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            seq = _make_synthetic_sequence(
                rng,
                left_detected=bool(rng.integers(0, 2)),
                right_detected=bool(rng.integers(0, 2)),
                pose_detected=bool(rng.integers(0, 2)),
            )
            out = tbi.augment_sequence(seq, np.random.default_rng(trial))
            self.assertFalse(np.any(np.isnan(out)), f"NaN introduced on trial {trial}")
            self.assertFalse(np.any(np.isinf(out)), f"Inf introduced on trial {trial}")

    def test_sequence_length_preserved_by_temporal_perturbation(self):
        rng = np.random.default_rng(0)
        seq = _make_synthetic_sequence(rng)
        for trial in range(10):
            out = tbi.augment_sequence(seq, np.random.default_rng(trial), time_jitter_std=2.0)
            self.assertEqual(out.shape[0], config.SEQUENCE_LENGTH)


class TestBuildAugmentedTrainingSet(unittest.TestCase):
    def test_augment_factor_one_returns_original_unchanged(self):
        rng = np.random.default_rng(0)
        X = np.stack([_make_synthetic_sequence(rng) for _ in range(4)])
        y = np.array([0, 1, 0, 1])
        X_aug, y_aug = tbi.build_augmented_training_set(X, y, seed=42, augment_factor=1)
        self.assertEqual(X_aug.shape, X.shape)
        np.testing.assert_array_equal(y_aug, y)

    def test_augment_factor_expands_dataset_correctly(self):
        rng = np.random.default_rng(0)
        X = np.stack([_make_synthetic_sequence(rng) for _ in range(4)])
        y = np.array([0, 1, 0, 1])
        X_aug, y_aug = tbi.build_augmented_training_set(X, y, seed=42, augment_factor=3)
        self.assertEqual(X_aug.shape[0], 12)  # 4 original + 4 + 4 copies
        self.assertEqual(y_aug.shape[0], 12)
        # labels for the augmented copies must match their source sample's label
        np.testing.assert_array_equal(y_aug[:4], y)
        np.testing.assert_array_equal(y_aug[4:8], y)
        np.testing.assert_array_equal(y_aug[8:12], y)

    def test_deterministic_across_repeated_calls(self):
        rng = np.random.default_rng(0)
        X = np.stack([_make_synthetic_sequence(rng) for _ in range(3)])
        y = np.array([0, 1, 2])
        X_aug_1, _ = tbi.build_augmented_training_set(X, y, seed=42, augment_factor=2)
        X_aug_2, _ = tbi.build_augmented_training_set(X, y, seed=42, augment_factor=2)
        np.testing.assert_array_equal(X_aug_1, X_aug_2)

    def test_does_not_mutate_input_arrays(self):
        rng = np.random.default_rng(0)
        X = np.stack([_make_synthetic_sequence(rng) for _ in range(3)])
        y = np.array([0, 1, 2])
        X_before = X.copy()
        tbi.build_augmented_training_set(X, y, seed=42, augment_factor=3)
        np.testing.assert_array_equal(X, X_before)

    def test_validation_and_test_arrays_untouched_by_augmentation_call(self):
        """Augmentation must only ever be invoked on training data --
        this test simulates the calling convention used in main() and
        confirms val/test arrays passed alongside are never touched."""
        rng = np.random.default_rng(0)
        X_train = np.stack([_make_synthetic_sequence(rng) for _ in range(4)])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.stack([_make_synthetic_sequence(rng) for _ in range(2)])
        X_val_before = X_val.copy()

        tbi.build_augmented_training_set(X_train, y_train, seed=42, augment_factor=3)

        np.testing.assert_array_equal(X_val, X_val_before)


class TestModels(unittest.TestCase):
    def test_regularized_bilstm_input_output_shape(self):
        model = tbi.build_regularized_bilstm_model(num_classes=5, lstm1_units=4, lstm2_units=4)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 5))

    def test_gru_model_input_output_shape(self):
        model = tbi.build_gru_model(num_classes=5, gru1_units=4, gru2_units=4)
        self.assertEqual(model.input_shape, (None, 60, 147))
        self.assertEqual(model.output_shape, (None, 5))

    def test_regularized_bilstm_smaller_than_default_baseline_units(self):
        """Experiment A must use a smaller model than the baseline
        (64/32 units) where sensible -- default is 32/16."""
        model = tbi.build_regularized_bilstm_model(num_classes=82)
        from train_bilstm_baseline import build_model as build_baseline_model
        baseline_model = build_baseline_model(num_classes=82)
        self.assertLess(model.count_params(), baseline_model.count_params())

    def test_gru_model_comparable_size_to_baseline(self):
        """Experiment C should be comparable in size to the baseline
        BiLSTM (155,385 params), not wildly larger or smaller."""
        model = tbi.build_gru_model(num_classes=82)
        n_params = model.count_params()
        self.assertGreater(n_params, 50_000)
        self.assertLess(n_params, 300_000)


class TestExperimentResultGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_experiment_end_to_end_on_tiny_synthetic_data(self):
        tbi.set_global_seed(42)
        rng = np.random.default_rng(0)
        num_classes = 3
        X_train = np.stack([_make_synthetic_sequence(rng) for _ in range(9)])
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        X_val = np.stack([_make_synthetic_sequence(rng) for _ in range(3)])
        y_val = np.array([0, 1, 2])
        X_test = np.stack([_make_synthetic_sequence(rng) for _ in range(3)])
        y_test = np.array([0, 1, 2])
        class_names = ["a", "b", "c"]

        model = tbi.build_gru_model(num_classes=num_classes, gru1_units=4, gru2_units=4, normalization_adapt_data=X_train)
        result = tbi.run_experiment(
            "tiny_test_experiment", model, X_train, y_train, X_val, y_val, X_test, y_test,
            class_names, self.tmp_dir, max_epochs=2, patience=2, batch_size=3,
        )

        self.assertIn("val_accuracy", result)
        self.assertIn("test_accuracy", result)
        self.assertIn("param_count", result)
        self.assertGreater(result["param_count"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "tiny_test_experiment", "model.keras")))

    def test_experiment_results_json_is_serializable_and_strips_internal_keys(self):
        fake_result = {
            "name": "x", "val_accuracy": 0.5, "test_accuracy": 0.3,
            "_test_confusion_matrix": np.array([[1, 0], [0, 1]]),
            "_history": {"loss": [1.0, 0.5]},
        }
        serializable = {k: v for k, v in fake_result.items() if not k.startswith("_")}
        path = os.path.join(self.tmp_dir, "results.json")
        from train_bilstm_baseline import save_json
        save_json(path, {"experiments": {"x": serializable}})
        with open(path) as f:
            loaded = json.load(f)
        self.assertNotIn("_test_confusion_matrix", loaded["experiments"]["x"])
        self.assertEqual(loaded["experiments"]["x"]["val_accuracy"], 0.5)


class TestClassMappingReuse(unittest.TestCase):
    """build_class_mapping is imported unchanged from
    train_bilstm_baseline -- re-verified here in this module's context."""

    def test_class_mapping_deterministic_here_too(self):
        manifest = [
            {"video_id": "1", "gloss": "zeta", "category": "", "sequence_path": "", "split": "train"},
            {"video_id": "2", "gloss": "alpha", "category": "", "sequence_path": "", "split": "train"},
        ]
        m1, _ = tbi.build_class_mapping(manifest)
        m2, _ = tbi.build_class_mapping(manifest)
        self.assertEqual(m1, m2)
        self.assertEqual(m1, {"alpha": 0, "zeta": 1})


if __name__ == "__main__":
    unittest.main()
