"""
Phase 9/10 validation + report generation for the processed ISL-Fingerspelling
dataset (data/fingerspelling_processed/). Read-only with respect to the
source dataset and to CISLR/INCLUDE; writes only its own report files under
data/fingerspelling_processed/reports/.

Run after src/stream_process_fingerspelling.py has finished (or stopped
safely due to low disk space).
"""

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SOURCE_ROOT = r"C:\Users\Elitebook 840 G9\OneDrive\ISL-Fingerspelling"
VIDEOS_DIR = os.path.join(SOURCE_ROOT, "videos")

OUTPUT_DIR = os.path.join(config.DATA_DIR, "fingerspelling_processed")
SEQUENCES_DIR = os.path.join(OUTPUT_DIR, "sequences")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata", "processing_metadata.csv")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)


def free_space_gb(path):
    import shutil
    return shutil.disk_usage(path).free / (1024 ** 3)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # A. videos discovered on disk (source of truth for "expected")
    source_videos = sorted(f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mp4"))
    source_uids = set(os.path.splitext(f)[0] for f in source_videos)
    n_discovered = len(source_videos)

    # metadata rows
    metadata_rows = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metadata_rows[row["video_id"]] = row

    n_success = sum(1 for r in metadata_rows.values() if r["processing_status"] == "success")
    n_failed = sum(1 for r in metadata_rows.values() if r["processing_status"] == "failed")
    failure_reasons = Counter(r["failure_reason"] for r in metadata_rows.values() if r["processing_status"] == "failed")

    # F. missing videos: metadata exists but underlying source video file gone (shouldn't happen)
    metadata_uids = set(metadata_rows.keys())
    missing_videos = sorted(metadata_uids - source_uids)

    # videos never attempted at all (present on disk, never produced ANY metadata row --
    # e.g. run was interrupted/stopped early due to low disk space)
    never_attempted = sorted(source_uids - metadata_uids)

    # E. missing annotations: success rows with empty full_text or 0 characters
    missing_annotations = [
        vid for vid, r in metadata_rows.items()
        if r["processing_status"] == "success" and (r.get("full_text_source") == "missing" or int(r.get("num_characters") or 0) == 0)
    ]

    # G/H/I. re-verify every claimed-success .npy on disk: shape, NaN, Inf
    invalid_shape, nan_sequences, inf_sequences, orphan_metadata, duplicate_ids = [], [], [], [], []
    seen_ids = set()
    valid_sequence_count = 0
    for vid, r in metadata_rows.items():
        if vid in seen_ids:
            duplicate_ids.append(vid)
        seen_ids.add(vid)

        if r["processing_status"] != "success":
            continue
        seq_path = os.path.join(config.PROJECT_ROOT, r["sequence_path"]) if r["sequence_path"] else None
        if not seq_path or not os.path.exists(seq_path):
            orphan_metadata.append(vid)
            continue
        arr = np.load(seq_path)
        if arr.shape != EXPECTED_SHAPE:
            invalid_shape.append(vid)
            continue
        if np.isnan(arr).any():
            nan_sequences.append(vid)
            continue
        if np.isinf(arr).any():
            inf_sequences.append(vid)
            continue
        valid_sequence_count += 1

    # orphan sequence files: .npy on disk with no metadata row
    sequence_files = set(os.path.splitext(f)[0] for f in os.listdir(SEQUENCES_DIR) if f.endswith(".npy")) if os.path.exists(SEQUENCES_DIR) else set()
    orphan_sequence_files = sorted(sequence_files - metadata_uids)

    # splits: read the 4 split files written by the processing script
    def read_split(name):
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [row["video_id"] for row in csv.DictReader(f)]

    train_ids = set(read_split("train"))
    test_ids = set(read_split("test"))
    standard_train_ids = set(read_split("standard_train"))
    standard_test_ids = set(read_split("standard_test"))

    train_test_overlap = sorted(train_ids & test_ids)
    standard_train_test_overlap = sorted(standard_train_ids & standard_test_ids)
    duplicate_uid_across_split = train_test_overlap  # same check, named per task wording

    # signer leakage: does any signer appear in both train and test (signer-independent split)?
    signer_by_uid = {vid: r.get("signer", "") for vid, r in metadata_rows.items()}
    train_signers = set(signer_by_uid.get(v, "") for v in train_ids)
    test_signers = set(signer_by_uid.get(v, "") for v in test_ids)
    signer_overlap = sorted(train_signers & test_signers)

    # character distribution (from char_sequence_json over all successfully processed videos)
    char_counter = Counter()
    for r in metadata_rows.values():
        if r["processing_status"] != "success":
            continue
        try:
            chars = json.loads(r["char_sequence_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        for c in chars:
            char_counter[c["letter"]] += 1

    # signer distribution among successfully processed videos
    signer_counts = Counter(r.get("signer", "") for r in metadata_rows.values() if r["processing_status"] == "success")

    # storage size of output
    total_output_bytes = 0
    if os.path.exists(SEQUENCES_DIR):
        for f in os.listdir(SEQUENCES_DIR):
            if f.endswith(".npy"):
                total_output_bytes += os.path.getsize(os.path.join(SEQUENCES_DIR, f))

    current_free_gb = free_space_gb("C:\\")

    results = {
        "dataset": "ISL-Fingerspelling",
        "source_root": SOURCE_ROOT,
        "output_dir": OUTPUT_DIR,
        "expected_videos": 1308,
        "A_source_videos_discovered": n_discovered,
        "B_successfully_processed": n_success,
        "C_failed": n_failed,
        "D_skipped_already_valid": None,  # reported by the processing script's own stdout, not recomputable after the fact
        "E_missing_annotations": len(missing_annotations),
        "E_missing_annotations_list": missing_annotations[:50],
        "F_missing_videos_referenced_in_metadata_but_absent_on_disk": len(missing_videos),
        "F_missing_videos_list": missing_videos[:50],
        "never_attempted_count": len(never_attempted),
        "never_attempted_list": never_attempted[:50],
        "G_invalid_shape_count": len(invalid_shape),
        "G_invalid_shape_list": invalid_shape[:50],
        "H_nan_count": len(nan_sequences),
        "H_nan_list": nan_sequences[:50],
        "I_inf_count": len(inf_sequences),
        "I_inf_list": inf_sequences[:50],
        "valid_sequence_count_reverified": valid_sequence_count,
        "duplicate_metadata_ids": duplicate_ids,
        "orphan_metadata_rows_missing_npy": orphan_metadata,
        "orphan_sequence_files_missing_metadata": orphan_sequence_files[:50],
        "orphan_sequence_files_count": len(orphan_sequence_files),
        "failure_reasons": dict(failure_reasons),
        "splits": {
            "signer_independent_train_count": len(train_ids),
            "signer_independent_test_count": len(test_ids),
            "standard_train_count": len(standard_train_ids),
            "standard_test_count": len(standard_test_ids),
            "signer_independent_train_test_overlap": train_test_overlap,
            "standard_train_test_overlap": standard_train_test_overlap,
            "duplicate_uid_across_signer_independent_split": duplicate_uid_across_split,
            "signer_overlap_between_train_and_test": signer_overlap,
            "train_signers": sorted(train_signers),
            "test_signers": sorted(test_signers),
            "validation_split_present": False,
            "validation_split_note": "No validation partition exists in either split_info.csv or signer_split.csv -- documented as absent, not invented.",
        },
        "character_distribution": dict(sorted(char_counter.items())),
        "character_distribution_unique_characters": len(char_counter),
        "signer_distribution_processed": dict(signer_counts),
        "output_storage_bytes": total_output_bytes,
        "output_storage_mb": round(total_output_bytes / (1024 ** 2), 2),
        "current_c_free_gb": round(current_free_gb, 2),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    json_path = os.path.join(REPORTS_DIR, "fingerspelling_preprocessing_validation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    char_csv_path = os.path.join(REPORTS_DIR, "fingerspelling_character_distribution.csv")
    with open(char_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["character", "count"])
        for ch, count in sorted(char_counter.items(), key=lambda kv: -kv[1]):
            label = "SPACE" if ch == " " else ch
            writer.writerow([label, count])

    print(json.dumps(results, indent=2)[:4000])
    print("...")
    print("WROTE:", json_path)
    print("WROTE:", char_csv_path)


if __name__ == "__main__":
    main()
