"""
Trains a compact BiLSTM baseline classifier on the 82-gloss, >=5-sample
CISLR subset defined by data/splits/cislr_82class/{train,val,test}.csv
(see reports/cislr_split_strategy_report.md and
reports/cislr_class_distribution_report.md for how that vocabulary and
split were chosen).

Pipeline: manifest CSVs -> load only the 492 referenced .npy sequences
(never the full 7,050-sequence dataset) -> BiLSTM classifier -> train
with early stopping on validation -> evaluate ONCE on test -> save all
artifacts under models/cislr_bilstm_baseline/.

Does not touch data/public_processed/, data/public_raw/, data/sequences/,
existing preprocessing scripts, feature extraction, or the split
manifests themselves (read-only).

This is an initial 82-gloss CISLR landmark-based baseline, not a claim
about general Indian Sign Language recognition, and NOT a
signer-independent evaluation (see class_mapping/training_config notes
and reports/cislr_split_strategy_report.md) -- CISLR provides no
signer/session identifier.

Usage:
    python src/train_bilstm_baseline.py --seed 42
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DEFAULT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")
DEFAULT_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_baseline")
DEFAULT_REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "cislr_bilstm_baseline_report.md")

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)  # (60, 147)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed):
    """Seeds Python, NumPy, and (if importable) Keras/TensorFlow's global
    random state, and requests deterministic ops where supported. Safe to
    call even if TensorFlow isn't needed by the caller yet."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Manifest / data loading (pure, testable, no TensorFlow dependency)
# ---------------------------------------------------------------------------

def load_manifest(csv_path):
    """Reads one split manifest CSV. Returns a list of row dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_class_mapping(*manifests):
    """Deterministic gloss -> integer class ID mapping: the sorted
    (alphabetical) list of unique glosses across all given manifests,
    indexed in order. NOT dependent on filesystem or CSV row order.
    Returns (gloss_to_id, id_to_gloss)."""
    glosses = set()
    for manifest in manifests:
        glosses.update(row["gloss"] for row in manifest)
    sorted_glosses = sorted(glosses)
    gloss_to_id = {gloss: i for i, gloss in enumerate(sorted_glosses)}
    id_to_gloss = {i: gloss for gloss, i in gloss_to_id.items()}
    return gloss_to_id, id_to_gloss


def load_sequence(sequence_path, project_root, expected_shape=EXPECTED_SHAPE):
    """Loads and shape-validates exactly one .npy sequence file."""
    full_path = os.path.join(project_root, sequence_path)
    arr = np.load(full_path)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{full_path}: expected shape {expected_shape}, got {arr.shape}"
        )
    return arr.astype(np.float32)


def load_split_arrays(manifest_rows, gloss_to_id, project_root):
    """Loads sequences ONE AT A TIME (never the full processed dataset --
    only exactly the rows in the given manifest) into a preallocated
    array. Returns (X: float32[N, 60, 147], y: int64[N], video_ids: [N])."""
    n = len(manifest_rows)
    X = np.zeros((n,) + EXPECTED_SHAPE, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    video_ids = []
    for i, row in enumerate(manifest_rows):
        X[i] = load_sequence(row["sequence_path"], project_root)
        y[i] = gloss_to_id[row["gloss"]]
        video_ids.append(row["video_id"])
    return X, y, video_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes, seq_len=config.SEQUENCE_LENGTH, feat_dim=config.FEATURE_VECTOR_LENGTH,
                 lstm1_units=64, lstm2_units=32, dropout_rate=0.3, learning_rate=1e-3,
                 normalization_adapt_data=None):
    """Compact BiLSTM classifier:
    Input(60,147) -> Normalization -> BiLSTM(64) -> Dropout
                   -> BiLSTM(32) -> Dropout -> Dense(num_classes, softmax)
    `normalization_adapt_data` should be the TRAINING set only (never
    val/test) to avoid leaking their statistics into the model."""
    import keras

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm1_units, return_sequences=True), name="bilstm_1"
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm2_units), name="bilstm_2"
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="gloss_softmax")(x)

    model = keras.Model(inputs, outputs, name="cislr_bilstm_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation / artifacts
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, class_names):
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, f1_score,
        classification_report, confusion_matrix,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report_str = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": weighted_f1,
        "classification_report": report_str,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_plot(cm, class_names, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.18), max(12, n * 0.18)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicted gloss")
    ax.set_ylabel("True gloss")
    ax.set_title(f"Confusion Matrix -- {n}-class CISLR BiLSTM baseline (test set)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a compact BiLSTM baseline on the 82-class CISLR subset.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm1-units", type=int, default=64)
    parser.add_argument("--lstm2-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    import keras  # imported after seeding so keras' own init is seeded too

    train_rows = load_manifest(os.path.join(args.splits_dir, "train.csv"))
    val_rows = load_manifest(os.path.join(args.splits_dir, "val.csv"))
    test_rows = load_manifest(os.path.join(args.splits_dir, "test.csv"))

    gloss_to_id, id_to_gloss = build_class_mapping(train_rows, val_rows, test_rows)
    num_classes = len(gloss_to_id)
    class_names = [id_to_gloss[i] for i in range(num_classes)]

    print("=" * 60)
    print("CISLR BiLSTM Baseline Training")
    print("=" * 60)
    print(f"Classes: {num_classes}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")
    print(f"Seed: {args.seed}")

    X_train, y_train, train_ids = load_split_arrays(train_rows, gloss_to_id, config.PROJECT_ROOT)
    X_val, y_val, val_ids = load_split_arrays(val_rows, gloss_to_id, config.PROJECT_ROOT)
    X_test, y_test, test_ids = load_split_arrays(test_rows, gloss_to_id, config.PROJECT_ROOT)
    print(f"Loaded arrays: X_train{X_train.shape} X_val{X_val.shape} X_test{X_test.shape}")

    # Leakage guard: video_id sets must be disjoint (already guaranteed by
    # generate_cislr_split.py, re-checked here defensively).
    assert not (set(train_ids) & set(val_ids)), "train/val video_id overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test video_id overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test video_id overlap"

    model = build_model(
        num_classes,
        lstm1_units=args.lstm1_units,
        lstm2_units=args.lstm2_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        normalization_adapt_data=X_train,
    )
    model.summary()
    param_count = model.count_params()

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.keras")
    final_model_path = os.path.join(args.output_dir, "final_model.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            best_model_path, monitor="val_accuracy", mode="max", save_best_only=True,
        ),
    ]

    print("-" * 60)
    print("Training...")
    start_time = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    train_elapsed = time.perf_counter() - start_time
    epochs_trained = len(history.history["loss"])
    best_val_accuracy = float(max(history.history["val_accuracy"]))

    # model's in-memory weights are already the best-val-accuracy epoch's
    # weights (EarlyStopping restore_best_weights=True); saving now yields
    # the SAME weights as best_model.keras (documented explicitly below,
    # not a distinct "last epoch" checkpoint).
    model.save(final_model_path)

    print(f"Training finished in {train_elapsed:.2f}s ({epochs_trained} epochs)")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # ---- artifacts: class mapping, config, history ----
    save_json(os.path.join(args.output_dir, "class_mapping.json"), {
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": {str(k): v for k, v in id_to_gloss.items()},
        "num_classes": num_classes,
        "mapping_rule": "sorted alphabetically over the union of train/val/test manifest glosses -- not dependent on filesystem order",
    })

    import tensorflow as tf
    import sklearn
    training_config = {
        "seed": args.seed,
        "splits_dir": os.path.relpath(args.splits_dir, config.PROJECT_ROOT),
        "num_classes": num_classes,
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dimension": config.FEATURE_VECTOR_LENGTH,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "architecture": {
            "input_normalization": "keras.layers.Normalization(axis=-1), adapted on train set only",
            "lstm1_units_bidirectional": args.lstm1_units,
            "lstm2_units_bidirectional": args.lstm2_units,
            "dropout_rate": args.dropout,
            "output": f"Dense({num_classes}, activation=softmax)",
        },
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "loss": "sparse_categorical_crossentropy",
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_monitor": "val_accuracy",
        "model_selection": "best_model.keras and final_model.keras hold IDENTICAL weights: EarlyStopping(restore_best_weights=True) restores the best-val-accuracy epoch's weights into the model before final_model.keras is saved, and ModelCheckpoint independently persisted the same best epoch to best_model.keras during training.",
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(train_elapsed, 3),
        "best_validation_accuracy": best_val_accuracy,
        "model_parameter_count": int(param_count),
        "framework_versions": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "test_set_usage": "Test set was loaded but NOT used for training, validation, early stopping, or model selection -- evaluated exactly once, after training completed, for the final reported result.",
        "signer_independent_evaluation": False,
        "signer_independent_note": "CISLR provides no signer/session identifier; this split is a seeded stratified random split by gloss, not signer-independent. See reports/cislr_split_strategy_report.md.",
    }
    save_json(os.path.join(args.output_dir, "training_config.json"), training_config)

    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(os.path.join(args.output_dir, "training_history.json"), history_serializable)

    # ---- final, single test evaluation ----
    print("-" * 60)
    print("Evaluating on TEST set (once, final result)...")
    best_model = keras.models.load_model(best_model_path)
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    results = evaluate_predictions(y_test, y_pred, class_names)

    save_json(os.path.join(args.output_dir, "evaluation_results.json"), {
        "test_accuracy": results["accuracy"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "test_samples": len(test_rows),
        "num_classes": num_classes,
    })

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(results["classification_report"])

    np.savetxt(
        os.path.join(args.output_dir, "confusion_matrix.csv"),
        results["confusion_matrix"], fmt="%d", delimiter=",",
    )
    save_confusion_matrix_plot(
        results["confusion_matrix"], class_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test accuracy:     {results['accuracy']:.4f}")
    print(f"Macro precision:   {results['macro_precision']:.4f}")
    print(f"Macro recall:      {results['macro_recall']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Weighted F1:       {results['weighted_f1']:.4f}")
    print(f"Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Epochs trained:    {epochs_trained}")
    print(f"Model parameters:  {param_count:,}")
    print(f"Artifacts saved to: {args.output_dir}")
    print("=" * 60)

    return {
        "results": results,
        "training_config": training_config,
        "class_names": class_names,
        "history": history_serializable,
    }


if __name__ == "__main__":
    main()
