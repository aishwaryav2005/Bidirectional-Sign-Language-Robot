"""
Phase 1 (isl_expansion_v2): evaluates an ISL-combined model (any number of
output classes >= 82, CISLR's 82 glosses always occupying class IDs 0-81
by construction -- see build_combined_dataset.py) on the EXACT SAME frozen
CISLR 82-class test set used by models/cislr_bilstm_baseline/
(data/splits/cislr_82class/test.csv, 82 samples), with NO retraining and
NO modification of that split.

Because the combined model has more output classes than the CISLR-only
model, a raw argmax over all classes would unfairly let the model "predict"
an INCLUDE-only class on a sample that is guaranteed (by construction of
this test set) to be one of the 82 CISLR glosses. To make this a fair,
apples-to-apples measurement against the CISLR-only baseline, the softmax
output is restricted to the 82 CISLR class indices (0-81) before argmax --
this is the "map predictions appropriately" step. Restricting is the
correct choice here because the evaluation question is "how well does this
model recognize CISLR's 82 glosses specifically," not "does it ever
mistake a CISLR video for an INCLUDE-only class" (that failure mode isn't
observable from this test set, which contains zero INCLUDE videos).

Reused, unmodified: src/train_bilstm_baseline.py's evaluate_predictions,
save_json, save_confusion_matrix_plot.

Usage:
    python src/evaluate_controlled_cislr.py --model-dir models/isl_combined/selected_model
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_bilstm_baseline import evaluate_predictions, save_json, save_confusion_matrix_plot

CISLR_TEST_CSV = os.path.join(config.DATA_DIR, "splits", "cislr_82class", "test.csv")
EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)


def load_cislr_test_set():
    with open(CISLR_TEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    glosses = sorted(set(r["gloss"] for r in rows))
    # Sanity: this must match the SAME 82 glosses / class ID assignment
    # build_combined_dataset.py used (sorted-alphabetical over the 82-class
    # split's train+val+test glosses) -- re-derive identically here rather
    # than trust an external file, so this script is self-contained.
    train_csv = os.path.join(config.DATA_DIR, "splits", "cislr_82class", "train.csv")
    val_csv = os.path.join(config.DATA_DIR, "splits", "cislr_82class", "val.csv")
    all_glosses = set()
    for p in (train_csv, val_csv, CISLR_TEST_CSV):
        with open(p, newline="", encoding="utf-8") as f:
            all_glosses.update(r["gloss"] for r in csv.DictReader(f))
    gloss_to_id = {g: i for i, g in enumerate(sorted(all_glosses))}
    return rows, gloss_to_id


def load_arrays(rows, gloss_to_id, project_root):
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
        y[i] = gloss_to_id[row["gloss"]]
        video_ids.append(row["video_id"])
    return X, y, video_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.join(config.PROJECT_ROOT, "models", "isl_combined", "selected_model"))
    parser.add_argument("--class-mapping", default=os.path.join(config.PROJECT_ROOT, "models", "isl_combined", "class_mapping.json"))
    parser.add_argument("--output-md", default=os.path.join(config.PROJECT_ROOT, "reports", "controlled_cislr_evaluation.md"))
    parser.add_argument("--output-json", default=os.path.join(config.PROJECT_ROOT, "reports", "controlled_cislr_evaluation.json"))
    parser.add_argument("--cm-png", default=os.path.join(config.PROJECT_ROOT, "reports", "controlled_cislr_evaluation_confusion_matrix.png"))
    args = parser.parse_args()

    import keras

    rows, gloss_to_id = load_cislr_test_set()
    num_cislr_classes = len(gloss_to_id)
    class_names = [None] * num_cislr_classes
    for g, i in gloss_to_id.items():
        class_names[i] = g
    print(f"CISLR 82-class test set: {len(rows)} samples, {num_cislr_classes} classes")

    X_test, y_test, video_ids = load_arrays(rows, gloss_to_id, config.PROJECT_ROOT)

    model_path = os.path.join(args.model_dir, "model.keras")
    model = keras.models.load_model(model_path)

    with open(args.class_mapping, encoding="utf-8") as f:
        combined_mapping = json.load(f)
    combined_num_classes = combined_mapping["num_classes"]
    combined_id_to_label = combined_mapping["class_id_to_label"]

    # Sanity: class IDs 0..num_cislr_classes-1 in the COMBINED model's
    # mapping must be the SAME 82 CISLR glosses in the SAME order as this
    # script's own gloss_to_id -- verified, not assumed.
    for i in range(num_cislr_classes):
        combined_label = combined_id_to_label[str(i)]
        if combined_label != class_names[i]:
            raise AssertionError(
                f"Class ID {i} mismatch: combined model has '{combined_label}', "
                f"this script's independently-derived CISLR ordering has '{class_names[i]}'. "
                f"Cannot proceed with a mismatched class mapping."
            )
    print(f"Verified: combined model's class IDs 0-{num_cislr_classes - 1} match the CISLR 82-class gloss ordering exactly.")
    print(f"Combined model has {combined_num_classes} total output classes ({combined_num_classes - num_cislr_classes} beyond CISLR's 82).")

    probs_full = model.predict(X_test, verbose=0)  # (N, combined_num_classes)
    # Restrict to the CISLR class columns before argmax -- see module
    # docstring for why this is the correct, fair mapping.
    probs_cislr_only = probs_full[:, :num_cislr_classes]
    y_pred = np.argmax(probs_cislr_only, axis=1)

    # Also record how often the model's UNRESTRICTED argmax would have
    # picked an INCLUDE-only class on a genuinely-CISLR sample -- informative,
    # reported separately, never substituted for the restricted accuracy.
    y_pred_unrestricted = np.argmax(probs_full, axis=1)
    n_would_predict_include_class = int(np.sum(y_pred_unrestricted >= num_cislr_classes))

    results = evaluate_predictions(y_test, y_pred, class_names)

    save_json(args.output_json, {
        "old_cislr_only_model": {
            "test_accuracy": 0.2561,
            "macro_f1": 0.1898,
            "source": "models/cislr_bilstm_baseline/evaluation_results.json",
        },
        "new_combined_model_on_cislr_test_set": {
            "test_accuracy": results["accuracy"],
            "macro_precision": results["macro_precision"],
            "macro_recall": results["macro_recall"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
            "test_samples": len(rows),
            "num_cislr_classes": num_cislr_classes,
            "combined_model_total_classes": combined_num_classes,
        },
        "prediction_mapping_method": "softmax restricted to the 82 CISLR class-ID columns (0..81) before argmax, since this test set is guaranteed by construction to contain only CISLR videos/glosses",
        "unrestricted_argmax_would_predict_include_class_count": n_would_predict_include_class,
        "unrestricted_argmax_would_predict_include_class_pct": round(100 * n_would_predict_include_class / len(rows), 2),
        "class_mapping_verified_identical_ordering": True,
        "test_set_path": os.path.relpath(CISLR_TEST_CSV, config.PROJECT_ROOT),
        "test_set_modified": False,
        "model_retrained": False,
    })

    with open(args.output_md.replace(".md", "_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(results["classification_report"])
    np.savetxt(args.output_md.replace(".md", "_confusion_matrix.csv"), results["confusion_matrix"], fmt="%d", delimiter=",")
    save_confusion_matrix_plot(results["confusion_matrix"], class_names, args.cm_png)

    improved = results["accuracy"] > 0.2561
    f1_improved = results["macro_f1"] > 0.1898

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(f"""# Controlled CISLR Evaluation: Combined Model on the Frozen 82-Class Test Set

## Method

The combined CISLR+INCLUDE model (`{args.model_dir}`, {combined_num_classes}
output classes) was evaluated on the **exact same, unmodified** 82-sample
CISLR test set used by `models/cislr_bilstm_baseline/`
(`data/splits/cislr_82class/test.csv`) -- no retraining, no split
modification, same `(60, 147)` feature representation and preprocessing as
every other CISLR evaluation in this project.

Because this model has {combined_num_classes - num_cislr_classes} more output
classes than the CISLR-only baseline, predictions were mapped by
**restricting the softmax to the 82 CISLR class-ID columns before argmax**
(verified to be class IDs 0-{num_cislr_classes - 1}, in the identical gloss
order the combined model's own class mapping uses -- checked programmatically,
not assumed). This is fair because every sample in this test set is
guaranteed to be a CISLR gloss; letting the model "choose" an INCLUDE-only
class would only ever be a guaranteed miss on this specific test set, and
would conflate two different questions (how well it knows CISLR's 82
glosses, vs. whether it ever confuses a CISLR video for an INCLUDE-only
class -- the latter is not observable here, since no INCLUDE video appears
in this test set).

## Results

| | OLD CISLR-only model | NEW combined model (this eval) |
|---|---:|---:|
| Test accuracy | 25.61% | **{results['accuracy']*100:.2f}%** |
| Macro F1 | 0.1898 | **{results['macro_f1']:.4f}** |
| Macro precision | -- | {results['macro_precision']:.4f} |
| Macro recall | -- | {results['macro_recall']:.4f} |
| Weighted F1 | -- | {results['weighted_f1']:.4f} |
| Test samples | 82 | 82 (identical set) |

**Accuracy {'IMPROVED' if improved else 'did NOT improve'}. Macro F1 {'IMPROVED' if f1_improved else 'did NOT improve'}.**

## Unrestricted-Argmax Diagnostic (informational only)

Without restricting to the 82 CISLR classes, the combined model's raw
argmax would have picked an INCLUDE-only class (guaranteed wrong on this
test set) for **{n_would_predict_include_class}/{len(rows)} samples
({100*n_would_predict_include_class/len(rows):.1f}%)**. This number is
reported for transparency but is **not** used in the headline accuracy
above -- see Method.

## Per-Class Results and Confusion Matrix

See `{os.path.basename(args.output_md).replace('.md', '_classification_report.txt')}`
and `{os.path.basename(args.cm_png)}` (full per-class precision/recall/F1
and the 82x82 confusion matrix).

## Important Distinction

This is **evaluation B** in the two-measurement framework requested for
this task: performance on the *original, frozen CISLR test distribution*,
as distinct from **evaluation A**, the combined dataset's own 98-class
test-set performance (46.55% accuracy, 0.3121 macro F1, reported
previously in `reports/combined_dataset_training_report.md`). These two
numbers measure different things and are not interchangeable.
""")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print("=" * 60)
    print(f"CONTROLLED CISLR TEST ACCURACY: {results['accuracy']*100:.2f}%  (old: 25.61%)")
    print(f"CONTROLLED CISLR MACRO F1:      {results['macro_f1']:.4f}  (old: 0.1898)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
