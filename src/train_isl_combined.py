"""
Phase 6/7/8: trains BiLSTM and GRU classifiers on the harmonized
CISLR + INCLUDE combined dataset (data/splits/isl_combined/, built by
src/build_combined_dataset.py -- not regenerated here).

Reuses, unmodified, from src/train_bilstm_baseline.py: set_global_seed,
evaluate_predictions, save_json, save_confusion_matrix_plot.
Reuses, unmodified, from src/train_bilstm_improved.py: augment_sequence,
build_augmented_training_set (landmark-level augmentation -- detection
flags and zero-filled undetected blocks are never perturbed).

Does NOT touch models/cislr_bilstm_baseline/, models/cislr_bilstm_improved/,
models/cislr_215class_experiments/, or any existing split/report.
All new artifacts go under models/isl_combined/.

Usage:
    python src/train_isl_combined.py --seed 42
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_bilstm_baseline import (
    set_global_seed, evaluate_predictions, save_json, save_confusion_matrix_plot,
)
from train_bilstm_improved import augment_sequence, build_augmented_training_set

SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined")
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "isl_combined")
REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "combined_dataset_training_report.md")
EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)

# Reference numbers from prior, completed CISLR-only experiments (read-only,
# for comparison in the report -- not recomputed here).
CISLR_82CLASS_BASELINE_TEST_ACCURACY = 0.2561
CISLR_82CLASS_BASELINE_MACRO_F1 = 0.1898
CISLR_215CLASS_TEST_ACCURACY = 0.2140
CISLR_215CLASS_MACRO_F1 = 0.1646


def load_split(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_arrays(rows, project_root):
    n = len(rows)
    X = np.zeros((n,) + EXPECTED_SHAPE, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    video_ids = []
    for i, row in enumerate(rows):
        path = os.path.join(project_root, row["sequence_path"])
        arr = np.load(path)
        if arr.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {arr.shape}")
        X[i] = arr.astype(np.float32)
        y[i] = int(row["class_id"])
        video_ids.append(row["video_id"])
    return X, y, video_ids


def build_model(cell_type, num_classes, seq_len=config.SEQUENCE_LENGTH, feat_dim=config.FEATURE_VECTOR_LENGTH,
                 units1=64, units2=32, dropout_rate=0.3, learning_rate=1e-3,
                 normalization_adapt_data=None):
    import keras

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)

    if cell_type == "bilstm":
        x = keras.layers.Bidirectional(keras.layers.LSTM(units1, return_sequences=True), name="bilstm_1")(x)
        x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(units2), name="bilstm_2")(x)
    elif cell_type == "gru":
        x = keras.layers.Bidirectional(keras.layers.GRU(units1, return_sequences=True), name="bigru_1")(x)
        x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(units2), name="bigru_2")(x)
    else:
        raise ValueError(cell_type)

    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_softmax")(x)

    model = keras.Model(inputs, outputs, name=f"isl_combined_{cell_type}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train, num_classes):
    """sklearn-style balanced class weights: n_samples / (n_classes * count_c).
    Only used when imbalance is severe (max/min count ratio > 5), per the
    task's 'if class imbalance is severe' instruction."""
    counts = np.bincount(y_train, minlength=num_classes)
    counts_nonzero = counts[counts > 0]
    ratio = counts_nonzero.max() / counts_nonzero.min()
    if ratio <= 5:
        return None, ratio
    n = len(y_train)
    weights = {}
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = float(n / (num_classes * counts[c]))
    return weights, ratio


def train_one(cell_type, X_train, y_train, X_val, y_val, num_classes, args, output_subdir):
    import keras

    model = build_model(cell_type, num_classes, dropout_rate=args.dropout,
                         learning_rate=args.learning_rate, normalization_adapt_data=X_train)
    param_count = model.count_params()

    class_weight, imbalance_ratio = compute_class_weights(y_train, num_classes)

    os.makedirs(output_subdir, exist_ok=True)
    best_model_path = os.path.join(output_subdir, "best_model.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(3, args.patience // 3), min_lr=1e-6),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.perf_counter() - start
    epochs_trained = len(history.history["loss"])
    best_val_acc = float(max(history.history["val_accuracy"]))

    save_json(os.path.join(output_subdir, "training_history.json"),
              {k: [float(v) for v in vals] for k, vals in history.history.items()})

    return {
        "model_path": best_model_path,
        "param_count": int(param_count),
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(elapsed, 3),
        "best_val_accuracy": best_val_acc,
        "class_weight_used": class_weight is not None,
        "imbalance_ratio": round(float(imbalance_ratio), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--augment-factor", type=int, default=3)
    args = parser.parse_args()

    set_global_seed(args.seed)
    import keras

    train_rows = load_split(os.path.join(SPLITS_DIR, "train.csv"))
    val_rows = load_split(os.path.join(SPLITS_DIR, "val.csv"))
    test_rows = load_split(os.path.join(SPLITS_DIR, "test.csv"))

    num_classes = max(int(r["class_id"]) for r in train_rows + val_rows + test_rows) + 1
    class_names = [None] * num_classes
    for r in train_rows + val_rows + test_rows:
        class_names[int(r["class_id"])] = r["normalized_label"]

    print("=" * 60)
    print("ISL Combined (CISLR + INCLUDE) Model Training")
    print("=" * 60)
    print(f"Classes: {num_classes}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    X_train, y_train, train_ids = load_arrays(train_rows, config.PROJECT_ROOT)
    X_val, y_val, val_ids = load_arrays(val_rows, config.PROJECT_ROOT)
    X_test, y_test, test_ids = load_arrays(test_rows, config.PROJECT_ROOT)

    assert not (set(train_ids) & set(val_ids)), "train/val video_id overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test video_id overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test video_id overlap"
    print("Leakage guard passed.")

    print(f"Augmenting training set (factor={args.augment_factor})...")
    X_train_aug, y_train_aug = build_augmented_training_set(X_train, y_train, args.seed, args.augment_factor)
    print(f"Training set: {X_train.shape[0]} -> {X_train_aug.shape[0]} (augmented)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(os.path.join(OUTPUT_DIR, "class_mapping.json"), {
        "class_id_to_label": {str(i): class_names[i] for i in range(num_classes)},
        "num_classes": num_classes,
        "note": "class_id assignment comes from reports/isl_label_mapping.csv (built by build_combined_dataset.py) -- CISLR's 82 existing glosses keep their original class IDs; INCLUDE labels are appended.",
    })
    save_json(os.path.join(OUTPUT_DIR, "preprocessing_config.json"), {
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dimension": config.FEATURE_VECTOR_LENGTH,
        "feature_layout": "[left_hand(63) | right_hand(63) | pose(18) | detection_flags(3)]",
        "normalization": "keras.layers.Normalization(axis=-1), adapted per-model on that model's augmented training set only",
        "augmentation": {
            "factor": args.augment_factor,
            "method": "src/train_bilstm_improved.py:augment_sequence -- per-sequence translation+scale per detected block, per-frame Gaussian noise, small monotonic temporal index jitter; detection flags and zero-filled undetected blocks never perturbed",
            "seed": args.seed,
        },
    })

    results = {}
    for cell_type in ("bilstm", "gru"):
        print("-" * 60)
        print(f"Training {cell_type.upper()}...")
        subdir = os.path.join(OUTPUT_DIR, cell_type)
        results[cell_type] = train_one(cell_type, X_train_aug, y_train_aug, X_val, y_val, num_classes, args, subdir)
        print(f"{cell_type}: best_val_acc={results[cell_type]['best_val_accuracy']:.4f} "
              f"epochs={results[cell_type]['epochs_trained']} time={results[cell_type]['training_time_seconds']:.1f}s "
              f"class_weighted={results[cell_type]['class_weight_used']} (imbalance ratio={results[cell_type]['imbalance_ratio']})")

    selected_type = max(results, key=lambda k: results[k]["best_val_accuracy"])
    print("-" * 60)
    print(f"Selected model (by validation accuracy): {selected_type}")

    best_model = keras.models.load_model(results[selected_type]["model_path"])
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    eval_results = evaluate_predictions(y_test, y_pred, class_names)

    selected_dir = os.path.join(OUTPUT_DIR, "selected_model")
    os.makedirs(selected_dir, exist_ok=True)
    best_model.save(os.path.join(selected_dir, "model.keras"))
    save_json(os.path.join(selected_dir, "selection_info.json"), {
        "selected_architecture": selected_type,
        "selection_criterion": "highest validation accuracy",
        "bilstm_val_accuracy": results["bilstm"]["best_val_accuracy"],
        "gru_val_accuracy": results["gru"]["best_val_accuracy"],
    })
    save_json(os.path.join(selected_dir, "evaluation_results.json"), {
        "test_accuracy": eval_results["accuracy"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "macro_f1": eval_results["macro_f1"],
        "weighted_f1": eval_results["weighted_f1"],
        "test_samples": len(test_rows),
        "num_classes": num_classes,
    })
    with open(os.path.join(selected_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(eval_results["classification_report"])
    np.savetxt(os.path.join(selected_dir, "confusion_matrix.csv"), eval_results["confusion_matrix"], fmt="%d", delimiter=",")
    save_confusion_matrix_plot(eval_results["confusion_matrix"], class_names, os.path.join(selected_dir, "confusion_matrix.png"))

    with open(os.path.join(selected_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "source_dataset", "true_label", "predicted_label", "correct"])
        for i, row in enumerate(test_rows):
            writer.writerow([row["video_id"], row["source_dataset"], class_names[y_test[i]], class_names[y_pred[i]], int(y_test[i] == y_pred[i])])

    print("=" * 60)
    print("FINAL TEST RESULTS (selected model:", selected_type, ")")
    print(f"Test accuracy:  {eval_results['accuracy']:.4f}")
    print(f"Macro F1:       {eval_results['macro_f1']:.4f}")
    print(f"Weighted F1:    {eval_results['weighted_f1']:.4f}")
    print("=" * 60)

    improved_vs_82 = eval_results["accuracy"] > CISLR_82CLASS_BASELINE_TEST_ACCURACY

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"""# Combined Dataset (CISLR + INCLUDE) Training Report

## Dataset

- Total combined classes: {num_classes}
- Train / val / test samples (post-augmentation train shown separately): {len(train_rows)} / {len(val_rows)} / {len(test_rows)}
- Augmented train samples actually used for fitting: {X_train_aug.shape[0]}

## Models Trained

| Model | Params | Best val acc | Epochs | Train time (s) | Class-weighted |
|---|---:|---:|---:|---:|:---:|
| BiLSTM | {results['bilstm']['param_count']:,} | {results['bilstm']['best_val_accuracy']:.4f} | {results['bilstm']['epochs_trained']} | {results['bilstm']['training_time_seconds']:.1f} | {results['bilstm']['class_weight_used']} |
| GRU | {results['gru']['param_count']:,} | {results['gru']['best_val_accuracy']:.4f} | {results['gru']['epochs_trained']} | {results['gru']['training_time_seconds']:.1f} | {results['gru']['class_weight_used']} |

**Selected (highest validation accuracy): {selected_type.upper()}**

## Final Test Performance (selected model, evaluated once)

| Metric | Value |
|---|---:|
| Test accuracy | {eval_results['accuracy']:.4f} |
| Macro precision | {eval_results['macro_precision']:.4f} |
| Macro recall | {eval_results['macro_recall']:.4f} |
| Macro F1 | {eval_results['macro_f1']:.4f} |
| Weighted F1 | {eval_results['weighted_f1']:.4f} |

## Comparison with Existing CISLR-Only Baselines

| | CISLR 82-class baseline | CISLR 215-class | Combined (this run) |
|---|---:|---:|---:|
| Classes | 82 | 215 | {num_classes} |
| Test accuracy | {CISLR_82CLASS_BASELINE_TEST_ACCURACY:.4f} | {CISLR_215CLASS_TEST_ACCURACY:.4f} | {eval_results['accuracy']:.4f} |
| Macro F1 | {CISLR_82CLASS_BASELINE_MACRO_F1:.4f} | {CISLR_215CLASS_TEST_ACCURACY:.4f} | {eval_results['macro_f1']:.4f} |

**Raw accuracy is not directly comparable across different class counts and
different, non-overlapping (mostly) vocabularies** -- more classes means a
lower random-chance baseline. Test accuracy vs. the 82-class baseline
{'IMPROVED' if improved_vs_82 else 'did NOT improve'} in raw terms; see
random-chance-normalized comparison before drawing conclusions.

## Limitations

- Not signer-independent for either dataset portion (see
  reports/isl_combined_split_report.md).
- The INCLUDE portion is a small subset of the full INCLUDE corpus (see
  reports/include_dataset_analysis.md), processed under a strict time/
  storage budget -- not the full 5,250-sample dataset.
- Single seed, single run per architecture -- results are directional, not
  a robust estimate of variance.
- Existing CISLR-only models/splits were not modified, retrained, or
  reused as weights (fully independent training run).
""")
    print(f"Wrote report: {REPORT_PATH}")

    return {"selected_type": selected_type, "eval_results": eval_results, "results": results, "num_classes": num_classes}


if __name__ == "__main__":
    main()
