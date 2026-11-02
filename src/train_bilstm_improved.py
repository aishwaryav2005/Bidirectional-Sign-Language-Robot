"""
Controlled model-improvement experiment, run AFTER the baseline
(src/train_bilstm_baseline.py, models/cislr_bilstm_baseline/ -- not
modified or retrained here). Goal: determine whether the baseline's
overfitting (train ~99% / val ~20-27%) can be reduced by regularization
and/or landmark-level data augmentation, and whether a GRU cell changes
anything, using the SAME 82-class split, SAME test set, and SAME
evaluation code as the baseline.

Reuses (imports, never modifies) src/train_bilstm_baseline.py's
load_manifest, build_class_mapping, load_split_arrays,
evaluate_predictions, save_json, save_confusion_matrix_plot, and
set_global_seed, so every experiment here is guaranteed to use the
identical data-loading and metric code as the baseline -- a fair,
apples-to-apples comparison.

Three experiments, each isolating one variable against the baseline:
    A. Regularized BiLSTM   -- smaller units + L2 + recurrent dropout,
                               SAME (unaugmented) training data.
    B. Augmented BiLSTM     -- SAME architecture as the baseline,
                               trained on augmented training data only.
    C. GRU baseline         -- same style/size as the baseline BiLSTM,
                               GRU cells instead of LSTM, SAME data.

Model selection is by VALIDATION accuracy only. The test set is loaded
once per experiment (for reporting) but is evaluated and reported ONLY
for the single model selected by validation performance -- never used
to choose between experiments.

Usage:
    python src/train_bilstm_improved.py --seed 42
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_bilstm_baseline import (
    load_manifest, build_class_mapping, load_split_arrays,
    evaluate_predictions, save_json, save_confusion_matrix_plot,
    set_global_seed, EXPECTED_SHAPE,
)

DEFAULT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")
DEFAULT_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_improved")
DEFAULT_REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "cislr_model_improvement_experiment.md")

# Feature-vector layout (src/feature_extractor.py, config.py) -- confirmed
# by inspection, not assumed: [left_hand(0:63) | right_hand(63:126) |
# pose(126:144) | detection_flags(144:147)]. Only the first 144 values are
# (x, y, z) coordinates; the last 3 are binary detection flags and must
# NEVER be perturbed. A zero-filled block (flag == 0, hand/pose not
# detected) must also never be perturbed -- adding noise to it would turn
# a legitimate "not detected" signal into fake near-origin coordinates.
_COORD_BLOCKS = [
    (0, config.HAND_FEATURE_LEN, 0),                                              # left hand, flag index 0
    (config.HAND_FEATURE_LEN, config.HAND_FEATURE_LEN * 2, 1),                    # right hand, flag index 1
    (config.HAND_FEATURE_LEN * 2, config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN, 2),  # pose, flag index 2
]
_FLAGS_START = config.HAND_FEATURE_LEN * 2 + config.POSE_FEATURE_LEN  # 144


# ---------------------------------------------------------------------------
# Experiment B: landmark-level augmentation (training data only)
# ---------------------------------------------------------------------------

def augment_sequence(seq, rng, noise_std=0.01, translation_std=0.02,
                      scale_range=(0.95, 1.05), time_jitter_std=1.0):
    """Returns a NEW augmented (60, 147) array; never modifies `seq` in
    place. `rng` must be a seeded numpy Generator for determinism.

    - Detection flags (last 3 values/frame) are copied through verbatim,
      never perturbed.
    - Zero-filled blocks (hand/pose not detected in that frame) are left
      as exact zeros -- never perturbed.
    - Each detected block (left hand / right hand / pose) gets ONE
      translation + ONE scale for the whole sequence (physically
      coherent -- "the same clip filmed slightly differently"), plus
      small per-frame Gaussian noise (simulates per-frame detection
      jitter).
    - Temporal perturbation resequences which original frames appear
      (small monotonic index jitter, then nearest-neighbor rounding) --
      it never blends/interpolates between frames, so it can't fabricate
      values that cross a detected/undetected boundary. Sequence length
      is always exactly preserved.
    """
    seq = np.asarray(seq, dtype=np.float32).copy()
    T = seq.shape[0]

    base_idx = np.arange(T, dtype=np.float64)
    jitter = rng.normal(0.0, time_jitter_std, size=T)
    warped = np.sort(base_idx + jitter)  # keep monotonic non-decreasing -> preserves temporal order
    idx = np.clip(np.round(warped), 0, T - 1).astype(int)
    seq = seq[idx]

    flags = seq[:, _FLAGS_START:_FLAGS_START + 3]

    for start, end, flag_idx in _COORD_BLOCKS:
        mask = flags[:, flag_idx] > 0.5
        if not np.any(mask):
            continue
        n_landmarks = (end - start) // 3
        block_xyz = seq[:, start:end].reshape(T, n_landmarks, 3)

        translation = rng.normal(0.0, translation_std, size=(1, 1, 3))
        scale = rng.uniform(scale_range[0], scale_range[1])
        noise = rng.normal(0.0, noise_std, size=block_xyz.shape)

        perturbed = block_xyz * scale + translation + noise
        block_xyz = np.where(mask[:, None, None], perturbed, block_xyz)
        seq[:, start:end] = block_xyz.reshape(T, end - start)

    return seq.astype(np.float32)


def build_augmented_training_set(X_train, y_train, seed, augment_factor=3):
    """Returns (X_aug, y_aug): the original training set PLUS
    (augment_factor - 1) deterministically-augmented copies of each
    sample, using independent, reproducible per-sample RNGs spawned from
    one seeded parent generator. augment_factor=1 returns the original
    data unchanged (no augmentation). Never touches validation/test data
    -- this function is only ever called with the training split."""
    if augment_factor <= 1:
        return X_train.copy(), y_train.copy()

    n = X_train.shape[0]
    parent_rng = np.random.default_rng(seed)
    # One independent child generator per (sample, copy) -- deterministic
    # and reproducible for a fixed seed, regardless of iteration order.
    n_copies = augment_factor - 1
    child_rngs = parent_rng.spawn(n * n_copies)

    X_parts = [X_train]
    y_parts = [y_train]
    rng_i = 0
    for _copy in range(n_copies):
        X_copy = np.zeros_like(X_train)
        for i in range(n):
            X_copy[i] = augment_sequence(X_train[i], child_rngs[rng_i])
            rng_i += 1
        X_parts.append(X_copy)
        y_parts.append(y_train.copy())

    X_aug = np.concatenate(X_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_regularized_bilstm_model(num_classes, seq_len=config.SEQUENCE_LENGTH,
                                    feat_dim=config.FEATURE_VECTOR_LENGTH,
                                    lstm1_units=32, lstm2_units=16, dropout_rate=0.4,
                                    recurrent_dropout=0.2, l2_reg=1e-4, learning_rate=1e-3,
                                    normalization_adapt_data=None):
    """Experiment A: smaller than the baseline (32/16 vs 64/32 units),
    with recurrent dropout and L2 weight regularization added -- same
    (unaugmented) data as the baseline, isolating "does a more
    regularized/smaller model generalize better" as the only variable."""
    import keras
    from keras import regularizers

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm1_units, return_sequences=True,
            dropout=dropout_rate, recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
        ), name="bilstm_1",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm2_units,
            dropout=dropout_rate, recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
        ), name="bilstm_2",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="gloss_softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)

    model = keras.Model(inputs, outputs, name="cislr_bilstm_regularized")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru_model(num_classes, seq_len=config.SEQUENCE_LENGTH,
                     feat_dim=config.FEATURE_VECTOR_LENGTH,
                     gru1_units=64, gru2_units=32, dropout_rate=0.3, learning_rate=1e-3,
                     normalization_adapt_data=None):
    """Experiment C: same style/size as the baseline BiLSTM (64/32 units,
    dropout 0.3, no extra regularization), with GRU cells in place of
    LSTM -- isolating "does the recurrent cell type matter" as the only
    variable, same (unaugmented) data as the baseline."""
    import keras

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(gru1_units, return_sequences=True), name="bigru_1"
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(gru2_units), name="bigru_2"
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="gloss_softmax")(x)

    model = keras.Model(inputs, outputs, name="cislr_gru_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Generic experiment runner
# ---------------------------------------------------------------------------

def run_experiment(name, model, X_train, y_train, X_val, y_val, X_test, y_test,
                    class_names, output_dir, max_epochs=200, patience=20, batch_size=16):
    """Trains `model`, evaluates it on val (for selection) and test (for
    reporting only -- selection must NOT use this), saves the model and
    returns a results dict. Test-set numbers are computed and stored so
    they're available for the report, but main() must not use them to
    pick the best experiment."""
    import keras

    exp_dir = os.path.join(output_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    model_path = os.path.join(exp_dir, "model.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=max_epochs, batch_size=batch_size, shuffle=True,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.perf_counter() - start
    epochs_trained = len(history.history["loss"])

    best_model = keras.models.load_model(model_path)

    val_pred = np.argmax(best_model.predict(X_val, verbose=0), axis=1)
    val_results = evaluate_predictions(y_val, val_pred, class_names)

    test_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    test_results = evaluate_predictions(y_test, test_pred, class_names)

    train_pred = np.argmax(best_model.predict(X_train, verbose=0), axis=1)
    train_accuracy = float(np.mean(train_pred == y_train))

    return {
        "name": name,
        "model_path": os.path.relpath(model_path, config.PROJECT_ROOT),
        "param_count": int(model.count_params()),
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(elapsed, 3),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_results["accuracy"],
        "val_macro_precision": val_results["macro_precision"],
        "val_macro_recall": val_results["macro_recall"],
        "val_macro_f1": val_results["macro_f1"],
        "val_weighted_f1": val_results["weighted_f1"],
        "test_accuracy": test_results["accuracy"],
        "test_macro_precision": test_results["macro_precision"],
        "test_macro_recall": test_results["macro_recall"],
        "test_macro_f1": test_results["macro_f1"],
        "test_weighted_f1": test_results["weighted_f1"],
        "_test_confusion_matrix": test_results["confusion_matrix"],
        "_test_classification_report": test_results["classification_report"],
        "_history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled model-improvement experiment (regularization / augmentation / GRU) vs the CISLR 82-class BiLSTM baseline.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--augment-factor", type=int, default=3, help="Training set size multiplier for Experiment B (1 = no augmentation)")
    parser.add_argument("--baseline-val-accuracy", type=float, default=0.26829269528388977, help="Baseline's already-recorded best validation accuracy, for comparison (baseline is not retrained here)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    train_rows = load_manifest(os.path.join(args.splits_dir, "train.csv"))
    val_rows = load_manifest(os.path.join(args.splits_dir, "val.csv"))
    test_rows = load_manifest(os.path.join(args.splits_dir, "test.csv"))

    gloss_to_id, id_to_gloss = build_class_mapping(train_rows, val_rows, test_rows)
    num_classes = len(gloss_to_id)
    class_names = [id_to_gloss[i] for i in range(num_classes)]

    print("=" * 60)
    print("CISLR Model Improvement Experiment")
    print("=" * 60)
    print(f"Classes: {num_classes}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    X_train, y_train, train_ids = load_split_arrays(train_rows, gloss_to_id, config.PROJECT_ROOT)
    X_val, y_val, val_ids = load_split_arrays(val_rows, gloss_to_id, config.PROJECT_ROOT)
    X_test, y_test, test_ids = load_split_arrays(test_rows, gloss_to_id, config.PROJECT_ROOT)

    assert not (set(train_ids) & set(val_ids)), "train/val video_id overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test video_id overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test video_id overlap"

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ---- Experiment A: regularized BiLSTM, unaugmented data ----
    print("-" * 60)
    print("Experiment A: Regularized BiLSTM")
    set_global_seed(args.seed)
    model_a = build_regularized_bilstm_model(num_classes, normalization_adapt_data=X_train)
    results["experiment_a_regularized_bilstm"] = run_experiment(
        "experiment_a_regularized_bilstm", model_a, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )

    # ---- Experiment B: baseline-style BiLSTM, augmented training data ----
    print("-" * 60)
    print(f"Experiment B: Augmented BiLSTM (augment_factor={args.augment_factor})")
    from train_bilstm_baseline import build_model as build_baseline_model
    X_train_aug, y_train_aug = build_augmented_training_set(X_train, y_train, args.seed, args.augment_factor)
    print(f"Augmented training set size: {X_train_aug.shape[0]} (from {X_train.shape[0]})")
    assert not np.any(np.isnan(X_train_aug)), "augmentation introduced NaN"
    assert not np.any(np.isinf(X_train_aug)), "augmentation introduced Inf"
    set_global_seed(args.seed)
    model_b = build_baseline_model(num_classes, normalization_adapt_data=X_train_aug)
    results["experiment_b_augmented_bilstm"] = run_experiment(
        "experiment_b_augmented_bilstm", model_b, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )
    results["experiment_b_augmented_bilstm"]["augment_factor"] = args.augment_factor
    results["experiment_b_augmented_bilstm"]["augmented_train_samples"] = int(X_train_aug.shape[0])

    # ---- Experiment C: GRU, unaugmented data ----
    print("-" * 60)
    print("Experiment C: GRU baseline")
    set_global_seed(args.seed)
    model_c = build_gru_model(num_classes, normalization_adapt_data=X_train)
    results["experiment_c_gru"] = run_experiment(
        "experiment_c_gru", model_c, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )

    # ---- Model selection: VALIDATION accuracy only, baseline included for reference ----
    print("-" * 60)
    print("Selecting best model by VALIDATION accuracy (test set not used for this decision)...")
    candidates = {"baseline_bilstm": args.baseline_val_accuracy}
    for key, r in results.items():
        candidates[key] = r["val_accuracy"]
    for name, val_acc in candidates.items():
        print(f"  {name:35s} val_accuracy={val_acc:.4f}")

    selected_name = max(candidates, key=candidates.get)
    print(f"Selected: {selected_name} (val_accuracy={candidates[selected_name]:.4f})")

    if selected_name == "baseline_bilstm":
        final_test_results = None  # baseline's test result is already reported in its own report; not re-evaluated here
        print("Baseline remains the best model; see reports/cislr_bilstm_baseline_report.md for its (already final) test result. No new test evaluation performed here.")
    else:
        selected = results[selected_name]
        print(f"Final untouched TEST evaluation of {selected_name} (single evaluation):")
        print(f"  test_accuracy={selected['test_accuracy']:.4f}  test_macro_f1={selected['test_macro_f1']:.4f}")

        selected_dir = os.path.join(args.output_dir, "selected_model")
        os.makedirs(selected_dir, exist_ok=True)
        import shutil
        shutil.copyfile(
            os.path.join(config.PROJECT_ROOT, selected["model_path"]),
            os.path.join(selected_dir, "model.keras"),
        )
        save_json(os.path.join(selected_dir, "class_mapping.json"), {
            "gloss_to_id": gloss_to_id,
            "id_to_gloss": {str(k): v for k, v in id_to_gloss.items()},
            "num_classes": num_classes,
        })
        with open(os.path.join(selected_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(results[selected_name]["_test_classification_report"])
        np.savetxt(
            os.path.join(selected_dir, "confusion_matrix.csv"),
            results[selected_name]["_test_confusion_matrix"], fmt="%d", delimiter=",",
        )
        save_confusion_matrix_plot(
            results[selected_name]["_test_confusion_matrix"], class_names,
            os.path.join(selected_dir, "confusion_matrix.png"),
        )

    # ---- machine-readable results (strip large/non-serializable internals) ----
    serializable_results = {}
    for key, r in results.items():
        r2 = {k: v for k, v in r.items() if not k.startswith("_")}
        serializable_results[key] = r2

    experiment_summary = {
        "seed": args.seed,
        "num_classes": num_classes,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "baseline_val_accuracy": args.baseline_val_accuracy,
        "candidates_val_accuracy": candidates,
        "selected_model": selected_name,
        "selection_criterion": "validation accuracy only; test set was not used to choose between experiments",
        "experiments": serializable_results,
    }
    save_json(os.path.join(args.output_dir, "experiment_results.json"), experiment_summary)

    print("=" * 60)
    print(f"experiment_results.json written to: {os.path.join(args.output_dir, 'experiment_results.json')}")
    print("=" * 60)

    return experiment_summary


if __name__ == "__main__":
    main()
