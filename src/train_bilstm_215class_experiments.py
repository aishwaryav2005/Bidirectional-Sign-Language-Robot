"""
Controlled model-comparison experiment on the NEW 215-class CISLR split
(data/splits/cislr_215class/, 594/215/215 train/val/test -- see
reports/cislr_215class_split_report.md). Goal: determine whether more
real training data (594 vs the 82-class experiment's 328) improves
generalization, using the same four model families already validated at
82 classes.

This is a NEW, isolated script. It does not modify, retrain, or read
for training src/train_bilstm_baseline.py, src/train_bilstm_improved.py,
models/cislr_bilstm_baseline/, or models/cislr_bilstm_improved/ -- it
only IMPORTS (unmodified) their reusable, already-parameterized
functions: build_model, build_regularized_bilstm_model, build_gru_model,
augment_sequence, build_augmented_training_set, run_experiment,
evaluate_predictions, load_manifest, build_class_mapping,
load_split_arrays, set_global_seed, save_json, save_confusion_matrix_plot.
All of these were already generic over num_classes and the manifest
directory, so no changes to that code were needed for 215 classes.

Four experiments, all trained fresh here (unlike the 82-class
improvement experiment, where "baseline" was an already-trained
reference; there is no pre-existing 215-class baseline model, so all
four are new runs on the SAME 215-class data):
    A. baseline_bilstm     -- same architecture as the original 82-class
                               baseline (64/32 units, dropout 0.3).
    B. regularized_bilstm   -- same regularized architecture as the
                               82-class improvement experiment (32/16
                               units, dropout 0.4, recurrent dropout 0.2,
                               L2 1e-4), adapted to 215 classes.
    C. augmented_bilstm     -- same architecture as baseline_bilstm,
                               trained on the augmented (factor=3)
                               training set, using the SAME
                               augment_sequence function (already
                               flag-safe / zero-block-safe / seeded).
    D. gru                  -- same GRU architecture as the 82-class
                               experiment (64/32 units, dropout 0.3),
                               adapted to 215 classes.

Model selection is by VALIDATION accuracy only -- the 215-class test set
is loaded for reporting but never used to choose between experiments.

Usage:
    python src/train_bilstm_215class_experiments.py --seed 42
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_bilstm_baseline import (
    load_manifest, build_class_mapping, load_split_arrays,
    save_json, set_global_seed, build_model as build_baseline_model,
)
from train_bilstm_improved import (
    build_regularized_bilstm_model, build_gru_model,
    build_augmented_training_set, run_experiment,
)

DEFAULT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_215class")
DEFAULT_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "cislr_215class_experiments")
DEFAULT_REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "cislr_215class_model_experiment.md")

# Reference-only: the already-trained 82-class baseline's recorded
# results, read (never retrained/modified) for the comparison section.
BASELINE_82CLASS_CONFIG_PATH = os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_baseline", "training_config.json")
BASELINE_82CLASS_EVAL_PATH = os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_baseline", "evaluation_results.json")


def load_82class_reference():
    """Read-only: loads the already-recorded 82-class baseline results
    for comparison context. Never trains or modifies anything."""
    import json
    with open(BASELINE_82CLASS_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(BASELINE_82CLASS_EVAL_PATH, encoding="utf-8") as f:
        eval_results = json.load(f)
    return {
        "num_classes": cfg["num_classes"],
        "train_samples": cfg["train_samples"],
        "val_samples": cfg["val_samples"],
        "test_samples": cfg["test_samples"],
        "best_validation_accuracy": cfg["best_validation_accuracy"],
        "test_accuracy": eval_results["test_accuracy"],
        "test_macro_f1": eval_results["macro_f1"],
        "test_weighted_f1": eval_results["weighted_f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled model-comparison experiment on the 215-class CISLR split.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--augment-factor", type=int, default=3)
    args = parser.parse_args()

    set_global_seed(args.seed)

    train_rows = load_manifest(os.path.join(args.splits_dir, "train.csv"))
    val_rows = load_manifest(os.path.join(args.splits_dir, "val.csv"))
    test_rows = load_manifest(os.path.join(args.splits_dir, "test.csv"))

    gloss_to_id, id_to_gloss = build_class_mapping(train_rows, val_rows, test_rows)
    num_classes = len(gloss_to_id)
    class_names = [id_to_gloss[i] for i in range(num_classes)]

    print("=" * 60)
    print("CISLR 215-Class Model Comparison Experiment")
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

    # ---- A: baseline_bilstm ----
    print("-" * 60)
    print("Experiment A: Baseline BiLSTM (215 classes)")
    set_global_seed(args.seed)
    model_a = build_baseline_model(num_classes, normalization_adapt_data=X_train)
    results["baseline_bilstm"] = run_experiment(
        "baseline_bilstm", model_a, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )

    # ---- B: regularized_bilstm ----
    print("-" * 60)
    print("Experiment B: Regularized BiLSTM (215 classes)")
    set_global_seed(args.seed)
    model_b = build_regularized_bilstm_model(num_classes, normalization_adapt_data=X_train)
    results["regularized_bilstm"] = run_experiment(
        "regularized_bilstm", model_b, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )

    # ---- C: augmented_bilstm (training data only; val/test untouched) ----
    print("-" * 60)
    print(f"Experiment C: Augmented BiLSTM (215 classes, augment_factor={args.augment_factor})")
    X_train_aug, y_train_aug = build_augmented_training_set(X_train, y_train, args.seed, args.augment_factor)
    print(f"Augmented training set size: {X_train_aug.shape[0]} (from {X_train.shape[0]})")
    assert not np.any(np.isnan(X_train_aug)), "augmentation introduced NaN"
    assert not np.any(np.isinf(X_train_aug)), "augmentation introduced Inf"
    # Defensive re-check: val/test arrays must be bit-identical to what was
    # loaded from the manifests -- augmentation must never have touched them.
    X_val_check, y_val_check, _ = load_split_arrays(val_rows, gloss_to_id, config.PROJECT_ROOT)
    X_test_check, y_test_check, _ = load_split_arrays(test_rows, gloss_to_id, config.PROJECT_ROOT)
    assert np.array_equal(X_val, X_val_check), "validation data changed after augmentation step"
    assert np.array_equal(X_test, X_test_check), "test data changed after augmentation step"
    set_global_seed(args.seed)
    model_c = build_baseline_model(num_classes, normalization_adapt_data=X_train_aug)
    results["augmented_bilstm"] = run_experiment(
        "augmented_bilstm", model_c, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )
    results["augmented_bilstm"]["augment_factor"] = args.augment_factor
    results["augmented_bilstm"]["augmented_train_samples"] = int(X_train_aug.shape[0])

    # ---- D: gru ----
    print("-" * 60)
    print("Experiment D: GRU (215 classes)")
    set_global_seed(args.seed)
    model_d = build_gru_model(num_classes, normalization_adapt_data=X_train)
    results["gru"] = run_experiment(
        "gru", model_d, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.max_epochs, args.patience, args.batch_size,
    )

    # ---- Model selection: VALIDATION accuracy only ----
    print("-" * 60)
    print("Selecting best model by VALIDATION accuracy (test set not used for this decision)...")
    candidates = {name: r["val_accuracy"] for name, r in results.items()}
    for name, val_acc in candidates.items():
        print(f"  {name:25s} val_accuracy={val_acc:.4f}")
    selected_name = max(candidates, key=candidates.get)
    print(f"Selected: {selected_name} (val_accuracy={candidates[selected_name]:.4f})")

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
        "selected_experiment": selected_name,
    })
    with open(os.path.join(selected_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(selected["_test_classification_report"])
    np.savetxt(
        os.path.join(selected_dir, "confusion_matrix.csv"),
        selected["_test_confusion_matrix"], fmt="%d", delimiter=",",
    )
    from train_bilstm_baseline import save_confusion_matrix_plot
    save_confusion_matrix_plot(
        selected["_test_confusion_matrix"], class_names,
        os.path.join(selected_dir, "confusion_matrix.png"),
    )

    # ---- 82-class reference (read-only, for the report's comparison section) ----
    reference_82class = load_82class_reference()

    # ---- machine-readable results ----
    serializable_results = {}
    for key, r in results.items():
        serializable_results[key] = {k: v for k, v in r.items() if not k.startswith("_")}

    experiment_summary = {
        "seed": args.seed,
        "num_classes": num_classes,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "candidates_val_accuracy": candidates,
        "selected_model": selected_name,
        "selection_criterion": "validation accuracy only; test set was not used to choose between experiments",
        "experiments": serializable_results,
        "reference_82class_experiment": reference_82class,
        "comparison_note": (
            "82-class and 215-class test accuracies are NOT directly comparable -- "
            "different numbers of classes (different random-chance baselines: "
            "1/82=1.22% vs 1/215=0.47%) and different, non-overlapping class "
            "vocabularies and sample counts."
        ),
    }
    save_json(os.path.join(args.output_dir, "experiment_results.json"), experiment_summary)

    print("=" * 60)
    print(f"experiment_results.json written to: {os.path.join(args.output_dir, 'experiment_results.json')}")
    print("=" * 60)

    return experiment_summary


if __name__ == "__main__":
    main()
