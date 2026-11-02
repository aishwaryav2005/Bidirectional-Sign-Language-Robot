"""
Phase 5 (isl_expansion_v2): rebuilds the harmonized CISLR + INCLUDE dataset
after additional INCLUDE categories have been processed (Seasons,
Electronics, People -- see reports/isl_expansion_v2_report.md).

Reuses, unmodified, every function from src/build_combined_dataset.py
(normalize_label, load_cislr_split, load_include_metadata,
build_label_mapping, split_include_rows, write_combined_split_csv) --
only the OUTPUT paths and the addition of a JSON split report are new.
Same methodology as v1: CISLR keeps its existing frozen per-video split
assignment; INCLUDE is newly split 70/15/15 by video_id, seed 42,
stratified per label.

New outputs only (does not touch v1's outputs or any frozen path):
  - reports/isl_label_mapping_v2.csv
  - data/splits/isl_combined_v2/{train,val,test}.csv
  - reports/isl_combined_split_report_v2.md
  - reports/isl_combined_split_report_v2.json

Signer-independence limitation: unchanged from v1 -- neither CISLR nor
INCLUDE's official metadata expose a signer/participant identifier, so
this split is NOT signer-independent for either dataset portion.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from build_combined_dataset import (
    load_cislr_split, load_include_metadata, build_label_mapping,
    split_include_rows, write_combined_split_csv, normalize_label, SEED,
)

OUTPUT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined_v2")
LABEL_MAPPING_REPORT = os.path.join(config.PROJECT_ROOT, "reports", "isl_label_mapping_v2.csv")
SPLIT_REPORT_MD = os.path.join(config.PROJECT_ROOT, "reports", "isl_combined_split_report_v2.md")
SPLIT_REPORT_JSON = os.path.join(config.PROJECT_ROOT, "reports", "isl_combined_split_report_v2.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cislr_split_rows, cislr_glosses = load_cislr_split()
    include_rows = load_include_metadata()

    include_categories = sorted(set(r.get("source_metadata", "") for r in include_rows))
    print(f"CISLR: {len(cislr_glosses)} glosses, "
          f"{len(cislr_split_rows['train'])}/{len(cislr_split_rows['val'])}/{len(cislr_split_rows['test'])} train/val/test")
    print(f"INCLUDE: {len(include_rows)} successfully-processed videos, "
          f"{len(set(r['label'] for r in include_rows))} unique labels")

    import csv as _csv
    mapping_report_rows, cislr_gloss_to_class_id, include_label_to_class_id = build_label_mapping(cislr_glosses, include_rows)

    os.makedirs(os.path.dirname(LABEL_MAPPING_REPORT), exist_ok=True)
    with open(LABEL_MAPPING_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["source_dataset", "original_label", "normalized_label", "class_id", "mapping_status", "notes"])
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
                "source_dataset": "CISLR", "video_id": row["video_id"],
                "original_label": row["gloss"], "normalized_label": row["gloss"],
                "class_id": cislr_gloss_to_class_id[row["gloss"]],
                "sequence_path": row["sequence_path"], "split": split,
            })
        for row in include_split[split]:
            combined[split].append({
                "source_dataset": "INCLUDE", "video_id": row["video_id"],
                "original_label": row["label"], "normalized_label": normalize_label(row["label"]),
                "class_id": include_label_to_class_id[row["label"]],
                "sequence_path": row["sequence_path"], "split": split,
            })

    for split in ("train", "val", "test"):
        out_path = os.path.join(OUTPUT_SPLITS_DIR, f"{split}.csv")
        write_combined_split_csv(out_path, combined[split])
        print(f"Wrote {out_path}: {len(combined[split])} rows")

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ids_a = set(r["video_id"] for r in combined[a])
        ids_b = set(r["video_id"] for r in combined[b])
        overlap = ids_a & ids_b
        assert not overlap, f"video_id overlap between {a} and {b}: {overlap}"
    print("Leakage guard passed: zero video_id overlap across train/val/test.")

    total_classes = len(cislr_glosses) + n_new
    include_by_category_counts = {}
    for r in include_rows:
        try:
            extra = json.loads(r.get("source_metadata", "{}"))
            cat = extra.get("category", "unknown")
        except Exception:
            cat = "unknown"
        include_by_category_counts[cat] = include_by_category_counts.get(cat, 0) + 1

    summary = {
        "cislr_classes": len(cislr_glosses),
        "include_labels_processed": len(set(r["label"] for r in include_rows)),
        "include_videos_processed": len(include_rows),
        "include_categories": include_by_category_counts,
        "label_merges": n_merged,
        "new_include_classes": n_new,
        "total_combined_classes": total_classes,
        "split_sizes": {
            "train": {"cislr": len(cislr_split_rows["train"]), "include": len(include_split["train"]), "total": len(combined["train"])},
            "val": {"cislr": len(cislr_split_rows["val"]), "include": len(include_split["val"]), "total": len(combined["val"])},
            "test": {"cislr": len(cislr_split_rows["test"]), "include": len(include_split["test"]), "total": len(combined["test"])},
        },
        "seed": args.seed,
        "cislr_split_reused_unchanged": True,
        "signer_independent": False,
        "signer_independent_note": "Neither CISLR nor INCLUDE's official metadata expose a signer/participant identifier; this split is NOT signer-independent for either dataset portion.",
    }
    with open(SPLIT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(SPLIT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(f"""# ISL Combined Dataset v2 (CISLR + INCLUDE) Split Report

## Summary

- CISLR classes (reused, existing 82-class vocabulary): {len(cislr_glosses)}
- INCLUDE videos processed (all categories so far): {len(include_rows)}
- INCLUDE categories included: {list(include_by_category_counts.keys())}
- INCLUDE labels merged into an existing CISLR class (exact match only): {n_merged}
- INCLUDE labels kept as new classes: {n_new}
- **Total combined classes: {total_classes}**

## Split sizes

| Split | CISLR rows | INCLUDE rows | Total |
|---|---:|---:|---:|
| train | {len(cislr_split_rows['train'])} | {len(include_split['train'])} | {len(combined['train'])} |
| val | {len(cislr_split_rows['val'])} | {len(include_split['val'])} | {len(combined['val'])} |
| test | {len(cislr_split_rows['test'])} | {len(include_split['test'])} | {len(combined['test'])} |

## Methodology (unchanged from v1)

- CISLR portion reuses the existing, frozen `data/splits/cislr_82class/` split exactly as-is.
- INCLUDE portion: newly split 70/15/15 by `video_id`, stratified per label, seed={args.seed}.
- Label harmonization: exact, case/whitespace-normalized string match only (see `reports/isl_label_mapping_v2.csv`).

## Limitations

- **Signer-independent splitting is NOT guaranteed** for either dataset portion (no signer/participant identifier in either CISLR's or INCLUDE's local metadata).
- This is still a partial subset of the full INCLUDE corpus ({sum(include_by_category_counts.values())} of 5,250 total INCLUDE videos, {len(include_by_category_counts)} of 15 categories).
""")
    print(f"Wrote {SPLIT_REPORT_MD}")
    print(f"Wrote {SPLIT_REPORT_JSON}")


if __name__ == "__main__":
    main()
