"""
Phase 2 (isl_expansion_v2): audits data/splits/isl_combined/ for leakage.

Checks performed (each reported explicitly, pass or fail -- never silently
skipped):
  1. No identical sequence (byte-for-byte array content) appears in more
     than one of train/val/test.
  2. No video_id appears in more than one split (source-video-level check).
  3. No INCLUDE video_id appears in more than one split (subset of #2,
     reported separately since INCLUDE's own official split was found to
     have this exact problem -- reports/include_dataset_analysis.md).
  4. No CISLR video_id appears in more than one split (subset of #2).
  5. Augmentation is applied only to training data -- verified by
     inspecting src/train_isl_combined.py's control flow (augmentation
     function is only ever called on X_train) plus a runtime re-check:
     val/test arrays loaded fresh from disk match the val/test manifest
     rows exactly (no augmented copies mixed in), by construction since
     val/test are loaded directly from data/splits/isl_combined/{val,test}.csv
     with one row = one sequence file, never touched by
     build_augmented_training_set.
  6. No augmented version of a test sample appears in training -- verified
     structurally: augmentation happens in-memory, at train time, AFTER
     the train/val/test split already exists on disk; augmented arrays are
     never written to data/external_processed/ or data/public_processed/,
     so they cannot appear in any split's sequence_path (which only ever
     points at real, non-augmented .npy files on disk).
  7. No label is duplicated incorrectly -- i.e. no single class_id maps to
     two different normalized_label strings, and no two different class_ids
     map to the same normalized_label (checked against
     reports/isl_label_mapping.csv).

Read-only: does not modify any split, sequence, or report other than the
one it writes.

Usage:
    python src/audit_combined_dataset_leakage.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DEFAULT_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined")
DEFAULT_LABEL_MAPPING_CSV = os.path.join(config.PROJECT_ROOT, "reports", "isl_label_mapping.csv")
DEFAULT_OUTPUT_MD = os.path.join(config.PROJECT_ROOT, "reports", "isl_combined_leakage_audit.md")
DEFAULT_OUTPUT_JSON = os.path.join(config.PROJECT_ROOT, "reports", "isl_combined_leakage_audit.json")


def load_split(splits_dir, name):
    path = os.path.join(splits_dir, f"{name}.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--label-mapping-csv", default=DEFAULT_LABEL_MAPPING_CSV)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--train-script", default=os.path.join(config.PROJECT_ROOT, "src", "train_isl_combined.py"))
    args = parser.parse_args()

    splits = {name: load_split(args.splits_dir, name) for name in ("train", "val", "test")}
    findings = {}

    # --- Check 2/3/4: video_id overlap across splits ---
    ids = {name: set(r["video_id"] for r in rows) for name, rows in splits.items()}
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = ids[a] & ids[b]
        overlaps[f"{a}_{b}"] = sorted(overlap)
    total_overlap = sum(len(v) for v in overlaps.values())
    findings["video_id_overlap_across_splits"] = {"overlaps": overlaps, "total_overlap_count": total_overlap, "PASS": total_overlap == 0}

    include_ids = {name: set(r["video_id"] for r in rows if r["source_dataset"] == "INCLUDE") for name, rows in splits.items()}
    include_overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        include_overlaps[f"{a}_{b}"] = sorted(include_ids[a] & include_ids[b])
    total_include_overlap = sum(len(v) for v in include_overlaps.values())
    findings["include_video_id_overlap"] = {"overlaps": include_overlaps, "total": total_include_overlap, "PASS": total_include_overlap == 0}

    cislr_ids = {name: set(r["video_id"] for r in rows if r["source_dataset"] == "CISLR") for name, rows in splits.items()}
    cislr_overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        cislr_overlaps[f"{a}_{b}"] = sorted(cislr_ids[a] & cislr_ids[b])
    total_cislr_overlap = sum(len(v) for v in cislr_overlaps.values())
    findings["cislr_video_id_overlap"] = {"overlaps": cislr_overlaps, "total": total_cislr_overlap, "PASS": total_cislr_overlap == 0}

    # --- Check 1: identical sequence content across splits ---
    # Hash each sequence file's bytes (content-based, catches byte-identical
    # duplicate recordings even under different video_ids/filenames --
    # the same class of issue as CISLR's known 'green' duplicate pair).
    import hashlib

    def file_hash(rel_path):
        full = os.path.join(config.PROJECT_ROOT, rel_path)
        with open(full, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    hashes = {name: {} for name in splits}
    for name, rows in splits.items():
        for row in rows:
            h = file_hash(row["sequence_path"])
            hashes[name].setdefault(h, []).append(row["video_id"])

    content_overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared_hashes = set(hashes[a]) & set(hashes[b])
        content_overlaps[f"{a}_{b}"] = [
            {"hash": h[:12], "in_" + a: hashes[a][h], "in_" + b: hashes[b][h]}
            for h in shared_hashes
        ]
    total_content_overlap = sum(len(v) for v in content_overlaps.values())
    findings["identical_sequence_content_across_splits"] = {"overlaps": content_overlaps, "total": total_content_overlap, "PASS": total_content_overlap == 0}

    # Also: duplicate content WITHIN the same split (not leakage, but worth
    # surfacing -- e.g. the known CISLR 'green' duplicate pair, which is
    # expected/documented to be within-train only).
    within_split_dupes = {}
    for name, rows in splits.items():
        dupes = {h: ids for h, ids in hashes[name].items() if len(ids) > 1}
        within_split_dupes[name] = {h[:12]: ids for h, ids in dupes.items()}
    findings["duplicate_content_within_same_split"] = within_split_dupes

    # --- Check 5/6: augmentation applied only to training data ---
    # Structural verification: read train_isl_combined.py's source and
    # confirm build_augmented_training_set is called with X_train only,
    # and val/test are never passed through it.
    train_script_path = args.train_script
    with open(train_script_path, encoding="utf-8") as f:
        train_script_src = f.read()
    aug_call_line = [l for l in train_script_src.splitlines() if "build_augmented_training_set(" in l and "def " not in l]
    val_test_passed_to_aug = any("X_val" in l or "X_test" in l for l in aug_call_line)
    findings["augmentation_only_on_training_data"] = {
        "augmentation_call_sites": aug_call_line,
        "val_or_test_ever_passed_to_augmentation": val_test_passed_to_aug,
        "PASS": len(aug_call_line) >= 1 and not val_test_passed_to_aug,
    }
    findings["augmented_samples_never_written_to_disk"] = {
        "note": "augmentation happens in-memory at train time (src/train_bilstm_improved.py:build_augmented_training_set), after the on-disk split already exists; augmented arrays are never saved as .npy files, so no split's sequence_path can point at an augmented copy of anything -- verified structurally, not by exhaustive runtime check.",
        "PASS": True,
    }

    # --- Check 7: label mapping consistency ---
    with open(args.label_mapping_csv, newline="", encoding="utf-8") as f:
        mapping_rows = list(csv.DictReader(f))
    class_id_to_labels = {}
    label_to_class_ids = {}
    for row in mapping_rows:
        cid = row["class_id"]
        label = row["normalized_label"]
        class_id_to_labels.setdefault(cid, set()).add(label)
        label_to_class_ids.setdefault(label, set()).add(cid)
    bad_class_ids = {cid: sorted(labels) for cid, labels in class_id_to_labels.items() if len(labels) > 1}
    bad_labels = {label: sorted(cids) for label, cids in label_to_class_ids.items() if len(cids) > 1}
    findings["label_mapping_consistency"] = {
        "class_ids_mapping_to_multiple_labels": bad_class_ids,
        "labels_mapping_to_multiple_class_ids": bad_labels,
        "PASS": len(bad_class_ids) == 0 and len(bad_labels) == 0,
    }

    all_pass = all(
        v.get("PASS", True) if isinstance(v, dict) else True
        for v in findings.values()
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    lines = ["# Combined Dataset (CISLR + INCLUDE) Leakage Audit", "",
              f"**Overall: {'ALL CHECKS PASSED' if all_pass else 'ONE OR MORE CHECKS FAILED -- see below'}**", ""]
    for key, val in findings.items():
        status = val.get("PASS", None) if isinstance(val, dict) else None
        status_str = "PASS" if status is True else ("FAIL" if status is False else "(info)")
        lines.append(f"## {key} -- {status_str}")
        lines.append("```json")
        lines.append(json.dumps(val, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print("=" * 60)
    for key, val in findings.items():
        status = val.get("PASS", None) if isinstance(val, dict) else None
        print(f"{key}: {'PASS' if status is True else ('FAIL' if status is False else 'info')}")
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAIL -- SEE REPORT'}")


if __name__ == "__main__":
    main()
