"""
Phase 3/4/5: builds the harmonized CISLR + INCLUDE dataset.

Read-only inputs (never modified):
  - data/splits/cislr_82class/{train,val,test}.csv  (existing, frozen split
    -- its train/val/test video assignment is REUSED EXACTLY AS-IS for the
    CISLR portion of the combined dataset, so no CISLR video ever moves
    split, and the frozen split files themselves are never touched)
  - data/external_processed/include/metadata/processing_metadata.csv
    (this run's newly-processed INCLUDE sequences)

New outputs only:
  - reports/isl_label_mapping.csv           (Phase 4)
  - data/splits/isl_combined/{train,val,test}.csv   (Phase 5, NEW directory,
    sibling to the frozen data/splits/cislr_82class -- does not touch it)
  - reports/isl_combined_split_report.md

Label harmonization rule (Phase 4): an INCLUDE label is merged into an
existing CISLR gloss ONLY on an exact, case-insensitive, whitespace-
normalized string match -- i.e. only when the label text is identical,
never an approximate/similarity merge. Every INCLUDE label that does not
exactly match a CISLR gloss becomes its own new class.

Split rule (Phase 5): CISLR keeps its existing per-video split assignment.
INCLUDE videos are newly split 70/15/15 (train/val/test), seeded 42,
grouped by video_id (each INCLUDE video file is already a single
source-video/single-take unit, so splitting by video_id cannot place the
same video in two splits) and stratified per label (each label's own
video list is shuffled and sliced so every split gets a share of every
label wherever the class has enough samples).

Signer-independence limitation (explicitly documented, not silently
assumed): neither CISLR nor the INCLUDE metadata used here (train/val/test
parquet files from Hugging Face) expose a signer/participant identifier,
so this split is NOT signer-independent for either dataset -- it is a
seeded, stratified split by video/source only. See reports/isl_combined_split_report.md.
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CISLR_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")
INCLUDE_METADATA_PATH = os.path.join(config.PROJECT_ROOT, "data", "external_processed", "include", "metadata", "processing_metadata.csv")
OUTPUT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined")
LABEL_MAPPING_REPORT = os.path.join(config.PROJECT_ROOT, "reports", "isl_label_mapping.csv")
SPLIT_REPORT_PATH = os.path.join(config.PROJECT_ROOT, "reports", "isl_combined_split_report.md")

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def normalize_label(label):
    return " ".join(label.strip().lower().split())


def load_cislr_split():
    rows = {"train": [], "val": [], "test": []}
    glosses = set()
    for split in ("train", "val", "test"):
        path = os.path.join(CISLR_SPLITS_DIR, f"{split}.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[split].append(row)
                glosses.add(row["gloss"])
    return rows, sorted(glosses)


def load_include_metadata():
    if not os.path.exists(INCLUDE_METADATA_PATH):
        raise FileNotFoundError(f"INCLUDE metadata not found: {INCLUDE_METADATA_PATH} -- run preprocess_public_dataset.py against the INCLUDE videos first")
    rows = []
    with open(INCLUDE_METADATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["processing_status"] == "success":
                rows.append(row)
    return rows


def build_label_mapping(cislr_glosses, include_rows):
    """Returns (mapping_rows_for_report, cislr_gloss_to_class_id, include_label_to_class_id)."""
    cislr_gloss_to_class_id = {gloss: i for i, gloss in enumerate(cislr_glosses)}
    cislr_norm_to_gloss = {normalize_label(g): g for g in cislr_glosses}

    include_labels = sorted(set(r["label"] for r in include_rows))

    report_rows = []
    for gloss in cislr_glosses:
        report_rows.append({
            "source_dataset": "CISLR",
            "original_label": gloss,
            "normalized_label": gloss,
            "class_id": cislr_gloss_to_class_id[gloss],
            "mapping_status": "CISLR_ORIGINAL",
            "notes": "existing 82-class CISLR gloss, unchanged",
        })

    include_label_to_class_id = {}
    next_new_id = len(cislr_glosses)
    for label in include_labels:
        norm = normalize_label(label)
        if norm in cislr_norm_to_gloss:
            matched_gloss = cislr_norm_to_gloss[norm]
            class_id = cislr_gloss_to_class_id[matched_gloss]
            include_label_to_class_id[label] = class_id
            report_rows.append({
                "source_dataset": "INCLUDE",
                "original_label": label,
                "normalized_label": matched_gloss,
                "class_id": class_id,
                "mapping_status": "MERGED",
                "notes": f"exact case/whitespace-insensitive match with CISLR gloss '{matched_gloss}'",
            })
        else:
            include_label_to_class_id[label] = next_new_id
            report_rows.append({
                "source_dataset": "INCLUDE",
                "original_label": label,
                "normalized_label": norm,
                "class_id": next_new_id,
                "mapping_status": "NEW_CLASS",
                "notes": "no exact match to any CISLR gloss; kept as a separate class (no approximate merging performed)",
            })
            next_new_id += 1

    return report_rows, cislr_gloss_to_class_id, include_label_to_class_id


def split_include_rows(include_rows, seed=SEED):
    """70/15/15 split by video_id, stratified per label, seeded. Returns
    {"train": [...], "val": [...], "test": [...]} of INCLUDE metadata rows."""
    rng = random.Random(seed)
    by_label = {}
    for row in include_rows:
        by_label.setdefault(row["label"], []).append(row)

    result = {"train": [], "val": [], "test": []}
    for label, rows in by_label.items():
        rows = rows[:]
        rng.shuffle(rows)
        n = len(rows)
        n_train = max(1, round(n * TRAIN_FRAC))
        n_val = max(1, round(n * VAL_FRAC)) if n >= 3 else 0
        n_train = min(n_train, n - (1 if n >= 2 else 0))  # leave room for at least val/test if possible
        remaining_after_train = n - n_train
        if remaining_after_train == 0:
            result["train"].extend(rows)
            continue
        n_val = min(n_val, remaining_after_train - 1) if remaining_after_train >= 2 else 0
        n_test = remaining_after_train - n_val
        result["train"].extend(rows[:n_train])
        result["val"].extend(rows[n_train:n_train + n_val])
        result["test"].extend(rows[n_train + n_val:n_train + n_val + n_test])

    return result


def write_combined_split_csv(path, combined_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["source_dataset", "video_id", "original_label", "normalized_label", "class_id", "sequence_path", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cislr_split_rows, cislr_glosses = load_cislr_split()
    include_rows = load_include_metadata()

    print(f"CISLR: {len(cislr_glosses)} glosses, "
          f"{len(cislr_split_rows['train'])}/{len(cislr_split_rows['val'])}/{len(cislr_split_rows['test'])} train/val/test")
    print(f"INCLUDE: {len(include_rows)} successfully-processed videos, "
          f"{len(set(r['label'] for r in include_rows))} unique labels")

    mapping_report_rows, cislr_gloss_to_class_id, include_label_to_class_id = build_label_mapping(cislr_glosses, include_rows)

    os.makedirs(os.path.dirname(LABEL_MAPPING_REPORT), exist_ok=True)
    with open(LABEL_MAPPING_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "original_label", "normalized_label", "class_id", "mapping_status", "notes"])
        writer.writeheader()
        writer.writerows(mapping_report_rows)
    print(f"Wrote label mapping: {LABEL_MAPPING_REPORT} ({len(mapping_report_rows)} rows)")

    n_merged = sum(1 for r in mapping_report_rows if r["mapping_status"] == "MERGED")
    n_new = sum(1 for r in mapping_report_rows if r["mapping_status"] == "NEW_CLASS")
    print(f"Label merges: {n_merged}  New INCLUDE classes: {n_new}")

    include_split = split_include_rows(include_rows, seed=args.seed)

    combined = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for row in cislr_split_rows[split]:
            combined[split].append({
                "source_dataset": "CISLR",
                "video_id": row["video_id"],
                "original_label": row["gloss"],
                "normalized_label": row["gloss"],
                "class_id": cislr_gloss_to_class_id[row["gloss"]],
                "sequence_path": row["sequence_path"],
                "split": split,
            })
        for row in include_split[split]:
            combined[split].append({
                "source_dataset": "INCLUDE",
                "video_id": row["video_id"],
                "original_label": row["label"],
                "normalized_label": normalize_label(row["label"]),
                "class_id": include_label_to_class_id[row["label"]],
                "sequence_path": row["sequence_path"],
                "split": split,
            })

    for split in ("train", "val", "test"):
        out_path = os.path.join(OUTPUT_SPLITS_DIR, f"{split}.csv")
        write_combined_split_csv(out_path, combined[split])
        print(f"Wrote {out_path}: {len(combined[split])} rows")

    # Leakage guard
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ids_a = set(r["video_id"] for r in combined[a])
        ids_b = set(r["video_id"] for r in combined[b])
        overlap = ids_a & ids_b
        assert not overlap, f"video_id overlap between {a} and {b}: {overlap}"
    print("Leakage guard passed: zero video_id overlap across train/val/test.")

    total_classes = len(cislr_glosses) + n_new
    with open(SPLIT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"""# ISL Combined Dataset (CISLR + INCLUDE) Split Report

## Summary

- CISLR classes (reused, existing 82-class vocabulary): {len(cislr_glosses)}
- INCLUDE labels processed this run: {len(set(r['label'] for r in include_rows))}
- INCLUDE labels merged into an existing CISLR class (exact match only): {n_merged}
- INCLUDE labels kept as new classes: {n_new}
- **Total combined classes: {total_classes}**

## Split sizes

| Split | CISLR rows | INCLUDE rows | Total |
|---|---:|---:|---:|
| train | {len(cislr_split_rows['train'])} | {len(include_split['train'])} | {len(combined['train'])} |
| val | {len(cislr_split_rows['val'])} | {len(include_split['val'])} | {len(combined['val'])} |
| test | {len(cislr_split_rows['test'])} | {len(include_split['test'])} | {len(combined['test'])} |

## Methodology

- **CISLR portion**: reuses the existing, frozen `data/splits/cislr_82class/{{train,val,test}}.csv`
  video-level split assignment EXACTLY as-is (read-only) -- no CISLR video
  changes split, and the frozen split files themselves are never modified.
- **INCLUDE portion**: newly split 70% / 15% / 15% by `video_id`, stratified
  per label (each label's own video list independently shuffled and sliced),
  seed=42. Each INCLUDE video file is already one distinct source-video/take,
  so grouping by `video_id` guarantees no video appears in more than one split.
- **Label harmonization**: an INCLUDE label was merged into an existing CISLR
  gloss only on an exact, case-insensitive, whitespace-normalized string
  match (see `reports/isl_label_mapping.csv`) -- no approximate/similarity
  merging was performed, per the explicit instruction not to merge
  approximate/similar words automatically.

## Limitations

- **Signer-independent splitting is NOT guaranteed for either dataset
  portion.** CISLR provides no signer/session identifier anywhere in its
  local metadata (documented in prior CISLR split reports). The INCLUDE
  metadata used here (the `train`/`val`/`test` parquet files distributed on
  Hugging Face) also does not expose a signer/participant identifier field
  usable for signer-independent grouping. This split is a seeded, stratified
  split by video/source only, not by signer.
- **INCLUDE portion of this run is a small subset of the full INCLUDE
  dataset** (see reports/include_dataset_analysis.md for which categories
  were processed and why), not the full 5,250-sample corpus -- chosen
  under an explicit time/storage budget for this task.
""")
    print(f"Wrote split report: {SPLIT_REPORT_PATH}")


if __name__ == "__main__":
    main()
