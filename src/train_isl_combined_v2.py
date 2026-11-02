"""
Phase 6/7 (isl_expansion_v2): trains BiLSTM and GRU on the EXPANDED
combined dataset (data/splits/isl_combined_v2/, built by
src/build_combined_dataset_v2.py -- Seasons + Electronics + People).

DIAGNOSTIC FIX vs. v1 (src/train_isl_combined.py): the v1 controlled
CISLR-test evaluation (reports/controlled_cislr_evaluation.md) showed
that, despite a large apparent gain on the COMBINED test set (46.55% vs
the CISLR baseline's 25.61%), accuracy on the ORIGINAL frozen CISLR
82-class test set was completely FLAT (25.61% -> 25.61%) and macro F1
slightly WORSE (0.1898 -> 0.1864). Diagnosis: v1's class-weighting only
activated above a 5x imbalance ratio, and the actual ratio (3.67) fell
under that threshold -- so CISLR's sparse classes (~4 samples/class)
received no compensating weight against INCLUDE's much denser classes
(~14-20 samples/class), and the shared model capacity was pulled toward
the easier, better-represented INCLUDE classes. **This version always
applies balanced class weighting** (removing the >5 threshold), directly
targeting that diagnosed cause, so this run is a genuine test of whether
correcting the imbalance recovers CISLR-side performance -- not a blind
re-run.

Otherwise identical methodology to v1: reuses set_global_seed,
evaluate_predictions, save_json, save_confusion_matrix_plot (from
train_bilstm_baseline.py) and augment_sequence, build_augmented_training_set
(from train_bilstm_improved.py) unmodified. Detection flags and zero-filled
undetected blocks are never perturbed by augmentation. Does NOT touch
models/isl_combined/ (v1) or any CISLR-only frozen artifact.

Usage:
    python src/train_isl_combined_v2.py --seed 42
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
from train_bilstm_improved import build_augmented_training_set
from train_isl_combined import build_model, load_split, load_arrays

SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined_v2")
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "isl_combined_v2")
REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "isl_expansion_v2_training_report.md")
EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)

CISLR_82CLASS_BASELINE_TEST_ACCURACY = 0.2561
CISLR_82CLASS_BASELINE_MACRO_F1 = 0.1898
V1_COMBINED_TEST_ACCURACY = 0.4655
V1_COMBINED_MACRO_F1 = 0.3121
V1_CONTROLLED_CISLR_TEST_ACCURACY = 0.2561
V1_CONTROLLED_CISLR_MACRO_F1 = 0.1864


def compute_balanced_class_weights_always(y_train, num_classes):
    """Always-balanced class weights (sklearn 'balanced' formula:
    n_samples / (n_classes * count_c)) -- unlike v1's
    compute_class_weights, this is NEVER skipped, regardless of the
    imbalance ratio. Directly targets the diagnosed cause of v1's flat
    controlled-CISLR result (see module docstring)."""
    counts = np.bincount(y_train, minlength=num_classes)
    counts_nonzero = counts[counts > 0]
    ratio = float(counts_nonzero.max() / counts_nonzero.min())
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

    class_weight, imbalance_ratio = compute_balanced_class_weights_always(y_train, num_classes)

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
    final_train_acc = float(history.history["accuracy"][-1])

    save_json(os.path.join(output_subdir, "training_history.json"),
              {k: [float(v) for v in vals] for k, vals in history.history.items()})

    return {
        "model_path": best_model_path,
        "param_count": int(param_count),
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(elapsed, 3),
        "best_val_accuracy": best_val_acc,
        "final_train_accuracy": final_train_acc,
        "class_weight_used": True,
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
    print("ISL Combined v2 (CISLR + INCLUDE expanded) Model Training")
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
        "note": "class_id assignment comes from reports/isl_label_mapping_v2.csv -- CISLR's 82 existing glosses keep their original class IDs 0-81; INCLUDE labels are appended.",
    })
    save_json(os.path.join(OUTPUT_DIR, "preprocessing_config.json"), {
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dimension": config.FEATURE_VECTOR_LENGTH,
        "feature_layout": "[left_hand(63) | right_hand(63) | pose(18) | detection_flags(3)]",
        "normalization": "keras.layers.Normalization(axis=-1), adapted per-model on that model's augmented training set only",
        "class_weighting": "ALWAYS balanced (sklearn-style n/(n_classes*count_c)), unlike v1 which only applied it above a 5x imbalance ratio -- fix motivated by v1's flat controlled-CISLR result",
        "augmentation": {
            "factor": args.augment_factor,
            "method": "src/train_bilstm_improved.py:augment_sequence, unmodified",
            "seed": args.seed,
        },
    })

    results = {}
    for cell_type in ("bilstm", "gru"):
        print("-" * 60)
        print(f"Training {cell_type.upper()}...")
        subdir = os.path.join(OUTPUT_DIR, cell_type)
        results[cell_type] = train_one(cell_type, X_train_aug, y_train_aug, X_val, y_val, num_classes, args, subdir)
        print(f"{cell_type}: train_acc={results[cell_type]['final_train_accuracy']:.4f} "
              f"best_val_acc={results[cell_type]['best_val_accuracy']:.4f} "
              f"epochs={results[cell_type]['epochs_trained']} time={results[cell_type]['training_time_seconds']:.1f}s "
              f"(imbalance ratio={results[cell_type]['imbalance_ratio']}, always class-weighted)")

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

    save_json(REPORT_PATH.replace(".md", ".json"), {
        "num_classes": num_classes,
        "train_samples": len(train_rows), "val_samples": len(val_rows), "test_samples": len(test_rows),
        "augmented_train_samples": int(X_train_aug.shape[0]),
        "bilstm": results["bilstm"], "gru": results["gru"],
        "selected_architecture": selected_type,
        "test_evaluation": {
            "accuracy": eval_results["accuracy"], "macro_precision": eval_results["macro_precision"],
            "macro_recall": eval_results["macro_recall"], "macro_f1": eval_results["macro_f1"],
            "weighted_f1": eval_results["weighted_f1"],
        },
    })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"""# ISL Expansion v2 Training Report (Combined Test-Set Performance)

## Dataset

- Total combined classes: {num_classes}
- Train / val / test (pre-augmentation): {len(train_rows)} / {len(val_rows)} / {len(test_rows)}
- Augmented train samples used for fitting: {X_train_aug.shape[0]}
- Class weighting: **always balanced** (fix vs. v1 -- see module docstring in src/train_isl_combined_v2.py)

## Models Trained

| Model | Params | Train acc | Best val acc | Epochs | Train time (s) |
|---|---:|---:|---:|---:|---:|
| BiLSTM | {results['bilstm']['param_count']:,} | {results['bilstm']['final_train_accuracy']:.4f} | {results['bilstm']['best_val_accuracy']:.4f} | {results['bilstm']['epochs_trained']} | {results['bilstm']['training_time_seconds']:.1f} |
| GRU | {results['gru']['param_count']:,} | {results['gru']['final_train_accuracy']:.4f} | {results['gru']['best_val_accuracy']:.4f} | {results['gru']['epochs_trained']} | {results['gru']['training_time_seconds']:.1f} |

**Selected (highest validation accuracy): {selected_type.upper()}**

## Final Test Performance on the COMBINED v2 Test Set (evaluated once)

| Metric | v1 (Seasons+Electronics only) | v2 (this run) |
|---|---:|---:|
| Test accuracy | {V1_COMBINED_TEST_ACCURACY:.4f} | {eval_results['accuracy']:.4f} |
| Macro F1 | {V1_COMBINED_MACRO_F1:.4f} | {eval_results['macro_f1']:.4f} |
| Macro precision | -- | {eval_results['macro_precision']:.4f} |
| Macro recall | -- | {eval_results['macro_recall']:.4f} |
| Weighted F1 | -- | {eval_results['weighted_f1']:.4f} |

**This table alone does NOT tell us whether CISLR recognition improved** --
see `reports/controlled_cislr_evaluation_v2.md` for that measurement,
which is the one that actually matters for the Phase 8 decision.

## Limitations

- Not signer-independent for either dataset portion.
- Still a partial INCLUDE subset (3 of 15 categories: Seasons, Electronics, People).
- Single seed, single run per architecture.
""")
    print(f"Wrote report: {REPORT_PATH}")

    return {"selected_type": selected_type, "eval_results": eval_results, "results": results, "num_classes": num_classes}


if __name__ == "__main__":
    main()
