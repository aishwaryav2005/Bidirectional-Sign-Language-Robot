"""
Selects a controlled vocabulary for the Sign->Speech DEMO (isolated-sign
classifier) from the LOCALLY ALREADY-PROCESSED datasets, and builds a
train/val/test split for it.

Inspected first (see reports/sign_demo/dataset_inspection.json):
  - CISLR: data/public_processed/metadata/processing_metadata.csv
    -> max 13 samples for any single gloss (monday), long-tail, 71% of
       glosses have exactly 1 sample. Not suitable for a high-accuracy
       controlled-vocabulary classifier.
  - INCLUDE (Seasons/Electronics/People, already processed in this project):
    data/external_processed/include/metadata/processing_metadata.csv
    -> the "People" category alone has 23 classes with >=17 samples each
       (most at 20-21), dramatically better than CISLR or INCLUDE's other
       processed categories (Seasons/Electronics top out at 13-15).

Decision: build the demo vocabulary from INCLUDE's "People" category,
selecting the top-20-by-sample-count labels. This does NOT touch CISLR,
INCLUDE's raw zips/videos, or any existing split/model file -- it only
reads data/external_processed/include/metadata/processing_metadata.csv
(already local, 738 rows) and writes NEW files under
data/sign_demo/ and reports/sign_demo/.

Signer independence: INCLUDE's official metadata has no signer/participant
ID field (confirmed in prior project audits -- reports/include_dataset_analysis.md
Signer Metadata: "Not available"). True signer-independent evaluation is
therefore NOT POSSIBLE here, same documented limitation as CISLR
(reports/cislr_split_strategy_report.md). Each sample IS an independently
recorded video (unique video_id/original_filename per sample, verified
below -- not a sub-segment of a shared longer source video, unlike the
ISL-Fingerspelling dataset), so the "same source video in both splits"
risk does not apply the same way; the remaining, undocumented risk is
ordinary same-signer-appears-in-train-and-test, which cannot be measured
or avoided without a signer field. A seeded stratified random split by
label is used, matching this project's established practice for CISLR.
"""
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

INCLUDE_METADATA = os.path.join(config.DATA_DIR, "external_processed", "include", "metadata", "processing_metadata.csv")
CISLR_METADATA = os.path.join(config.PUBLIC_METADATA_PATH)

OUTPUT_DIR = os.path.join(config.DATA_DIR, "sign_demo")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")
REPORT_DIR = os.path.join(config.PROJECT_ROOT, "reports", "sign_demo")

SEED = 42
N_CLASSES = 20
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def inspect_sources():
    with open(CISLR_METADATA, newline="", encoding="utf-8") as f:
        cislr_rows = [r for r in csv.DictReader(f) if r["processing_status"] == "success"]
    cislr_counts = Counter(r["label"] for r in cislr_rows)

    with open(INCLUDE_METADATA, newline="", encoding="utf-8") as f:
        include_rows = [r for r in csv.DictReader(f) if r["processing_status"] == "success"]
    include_counts = Counter(r["label"] for r in include_rows)

    return {
        "cislr": {"total_rows": len(cislr_rows), "total_glosses": len(cislr_counts),
                  "max_samples_any_gloss": max(cislr_counts.values()),
                  "top_25": cislr_counts.most_common(25)},
        "include_people_seasons_electronics": {"total_rows": len(include_rows), "total_labels": len(include_counts),
                  "max_samples_any_label": max(include_counts.values()),
                  "top_40": include_counts.most_common(40)},
    }, include_rows, include_counts


def main():
    random.seed(SEED)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    inspection, include_rows, include_counts = inspect_sources()
    with open(os.path.join(REPORT_DIR, "dataset_inspection.json"), "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)

    print("=== CISLR: max samples/gloss =", inspection["cislr"]["max_samples_any_gloss"], "===")
    print("=== INCLUDE (processed subset): max samples/label =",
          inspection["include_people_seasons_electronics"]["max_samples_any_label"], "===")

    # Decision: select from the "People" category specifically (verified
    # separately to have the highest, most uniform per-class counts).
    people_rows = [r for r in include_rows if r["original_filename"].replace("/", "\\").split("\\")[0] == "People"]
    people_counts = Counter(r["label"] for r in people_rows)
    assert people_counts, "No People-category rows found -- inspect data/external_processed/include/ before proceeding"

    selected = [label for label, _ in people_counts.most_common(N_CLASSES)]
    selected_counts = {label: people_counts[label] for label in selected}

    print(f"\nSelected {len(selected)} classes from INCLUDE 'People' category (top {N_CLASSES} by sample count):")
    for label in sorted(selected, key=lambda l: -selected_counts[l]):
        print(f"  {label}: {selected_counts[label]} samples")

    # video_id uniqueness check (confirms these are independent recordings,
    # not segments of a shared source video -- see module docstring)
    selected_rows = [r for r in people_rows if r["label"] in selected]
    vids = [r["video_id"] for r in selected_rows]
    assert len(vids) == len(set(vids)), "Unexpected duplicate video_id within selected vocabulary"

    # --- stratified random split by label, seed=42 ---
    by_label = defaultdict(list)
    for r in selected_rows:
        by_label[r["label"]].append(r)

    train_rows, val_rows, test_rows = [], [], []
    split_summary = {}
    for label, rows in by_label.items():
        rows_sorted = sorted(rows, key=lambda r: r["video_id"])  # deterministic order before shuffling
        random.shuffle(rows_sorted)
        n = len(rows_sorted)
        n_val = max(1, round(n * VAL_FRAC))
        n_test = max(1, round(n * TEST_FRAC))
        n_train = n - n_val - n_test
        assert n_train >= 1, f"label {label} has too few samples ({n}) for a 3-way split"

        train_rows.extend(rows_sorted[:n_train])
        val_rows.extend(rows_sorted[n_train:n_train + n_val])
        test_rows.extend(rows_sorted[n_train + n_val:])
        split_summary[label] = {"total": n, "train": n_train, "val": n_val, "test": n_test}

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "label", "sequence_path"])
            for r in rows:
                writer.writerow([r["video_id"], r["label"], r["sequence_path"]])

    write_csv(os.path.join(SPLITS_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(SPLITS_DIR, "val.csv"), val_rows)
    write_csv(os.path.join(SPLITS_DIR, "test.csv"), test_rows)

    # overlap checks
    train_ids, val_ids, test_ids = (set(r["video_id"] for r in s) for s in (train_rows, val_rows, test_rows))
    checks = {
        "train_val_disjoint": len(train_ids & val_ids) == 0,
        "train_test_disjoint": len(train_ids & test_ids) == 0,
        "val_test_disjoint": len(val_ids & test_ids) == 0,
        "all_classes_in_all_splits": all(
            any(r["label"] == label for r in train_rows) and
            any(r["label"] == label for r in val_rows) and
            any(r["label"] == label for r in test_rows)
            for label in selected
        ),
    }
    assert all(checks.values()), f"split validation failed: {checks}"

    report = {
        "seed": SEED,
        "source_dataset": "INCLUDE (ai4bharat), 'People' category, already locally processed subset",
        "source_metadata_file": os.path.relpath(INCLUDE_METADATA, config.PROJECT_ROOT),
        "decision_rationale": "CISLR's best gloss has only 13 samples (71% of its vocabulary has exactly 1); "
                               "INCLUDE's People category has 23 labels with >=17 samples each -- selected "
                               "as the clearly superior source for a high-accuracy controlled-vocabulary demo.",
        "num_classes": len(selected),
        "selected_labels": sorted(selected),
        "samples_per_class": selected_counts,
        "total_samples": len(selected_rows),
        "split_method": "stratified random split by label, seed=42, grouped at the video level "
                         "(each sample is an independently recorded video, not a shared-source-video segment)",
        "split_per_class": split_summary,
        "split_totals": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "overlap_checks": checks,
        "signer_independence": {
            "possible": False,
            "reason": "INCLUDE's official metadata has no signer/participant ID field (confirmed in "
                      "reports/include_dataset_analysis.md). A random stratified split by label is used "
                      "instead, matching this project's established practice for CISLR "
                      "(reports/cislr_split_strategy_report.md). Reported accuracy may be optimistic "
                      "relative to a true signer-independent setting.",
        },
    }
    with open(os.path.join(REPORT_DIR, "vocabulary_and_split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nTotal samples: {len(selected_rows)}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")
    print("WROTE", os.path.join(SPLITS_DIR, "train.csv"))
    print("WROTE", os.path.join(SPLITS_DIR, "val.csv"))
    print("WROTE", os.path.join(SPLITS_DIR, "test.csv"))
    print("WROTE", os.path.join(REPORT_DIR, "vocabulary_and_split_report.json"))


if __name__ == "__main__":
    main()
