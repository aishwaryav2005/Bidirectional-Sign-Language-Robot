"""
Lightweight, streaming, read-only validator for data/public_processed/.

Distinct from src/validate_public_dataset.py (the original per-label
summary report): this script is built for full-dataset scale (thousands
of files) --
  - iterates the .npy files on disk one at a time, never holding more
    than one array in memory,
  - loads each with numpy.load(..., mmap_mode="r") so the file's data is
    memory-mapped rather than fully read into RAM up front,
  - cross-checks the filesystem against processing_metadata.csv in both
    directions (files with no metadata row, and successful metadata rows
    with no file -- "missing" vs. "orphan"),
  - prints progress periodically so a multi-thousand-file run is never
    silent long enough to look frozen.

This script NEVER writes, deletes, or regenerates anything -- it only
reads data/public_processed/ and data/public_raw/dataset.csv (for the
expected total video count) and prints a report.

Usage:
    python src/validate_public_dataset_streaming.py
    python src/validate_public_dataset_streaming.py --progress-every 250
    python src/validate_public_dataset_streaming.py --expected-total 7050
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def find_npy_files(sequences_dir):
    """Yields (video_id, path) for every .npy file under sequences_dir
    (one subfolder per label), without loading any file contents."""
    for root, _dirnames, filenames in os.walk(sequences_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                video_id = os.path.splitext(filename)[0]
                yield video_id, os.path.join(root, filename)


def load_metadata_rows(metadata_path):
    with open(metadata_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_expected_total(dataset_csv_path):
    """Authoritative expected video count, read from dataset.csv rather
    than hardcoded, so this script stays correct if the dataset changes."""
    if not os.path.exists(dataset_csv_path):
        return None
    with open(dataset_csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def validate(sequences_dir, metadata_path, dataset_csv_path, progress_every, expected_total_override=None):
    report = {
        "total_files": 0,
        "correct_shape": 0,
        "wrong_shape": 0,
        "nan_count": 0,
        "inf_count": 0,
        "missing_metadata": 0,     # .npy on disk with no metadata row at all
        "orphan_metadata": 0,      # metadata row says success, but no .npy file exists
        "duplicate_video_ids_on_disk": [],
        "duplicate_video_ids_in_metadata": [],
        "load_errors": [],
        "wrong_shape_details": [],
        "metadata_row_count": 0,
        "expected_total": None,
    }

    expected_total = expected_total_override
    if expected_total is None:
        expected_total = read_expected_total(dataset_csv_path)
    report["expected_total"] = expected_total

    # ---- discover files on disk (metadata only, no array data read yet) ----
    npy_files = list(find_npy_files(sequences_dir)) if os.path.isdir(sequences_dir) else []
    disk_id_counts = Counter(vid for vid, _ in npy_files)
    report["duplicate_video_ids_on_disk"] = sorted(vid for vid, c in disk_id_counts.items() if c > 1)
    report["total_files"] = len(npy_files)

    # ---- read metadata (small text file, fine to hold fully in memory) ----
    metadata_rows = load_metadata_rows(metadata_path) if os.path.exists(metadata_path) else []
    report["metadata_row_count"] = len(metadata_rows)

    meta_id_counts = Counter(r["video_id"] for r in metadata_rows)
    report["duplicate_video_ids_in_metadata"] = sorted(vid for vid, c in meta_id_counts.items() if c > 1)

    metadata_ids = set(meta_id_counts.keys())
    success_ids_in_metadata = {r["video_id"] for r in metadata_rows if r["processing_status"] == "success"}
    npy_ids = set(disk_id_counts.keys())

    report["missing_metadata"] = len(npy_ids - metadata_ids)
    report["orphan_metadata"] = len(success_ids_in_metadata - npy_ids)

    # ---- stream-check every .npy file, one at a time, memory-mapped ----
    total = len(npy_files)
    print(f"Validating {total} sequence files (streaming, mmap)...")
    for i, (video_id, path) in enumerate(npy_files, start=1):
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            report["load_errors"].append((video_id, str(e)))
        else:
            if arr.shape != (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH):
                report["wrong_shape"] += 1
                report["wrong_shape_details"].append((video_id, arr.shape))
            else:
                report["correct_shape"] += 1

            if np.isnan(arr).any():
                report["nan_count"] += 1
            if np.isinf(arr).any():
                report["inf_count"] += 1

        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  ...{i}/{total} checked")

    return report


def _all_checks_passed(report):
    if report["expected_total"] is not None:
        if report["total_files"] != report["expected_total"]:
            return False
        if report["metadata_row_count"] != report["expected_total"]:
            return False
    return (
        report["wrong_shape"] == 0
        and report["nan_count"] == 0
        and report["inf_count"] == 0
        and report["missing_metadata"] == 0
        and report["orphan_metadata"] == 0
        and not report["duplicate_video_ids_on_disk"]
        and not report["duplicate_video_ids_in_metadata"]
        and not report["load_errors"]
        and report["total_files"] > 0
    )


def print_report(report):
    passed = _all_checks_passed(report)

    print("=" * 60)
    print("Streaming Validation Report -- data/public_processed/")
    print("=" * 60)
    if report["expected_total"] is not None:
        print(f"Expected total (from dataset.csv):  {report['expected_total']}")
    else:
        print("Expected total: unknown (dataset.csv not found; count checks skipped)")
    print(f"Total .npy files found:             {report['total_files']}")
    print(f"Metadata rows (excl. header):        {report['metadata_row_count']}")
    print("-" * 60)
    print(f"Correct shape ({config.SEQUENCE_LENGTH},{config.FEATURE_VECTOR_LENGTH}): {report['correct_shape']}")
    print(f"Wrong shape count:                   {report['wrong_shape']}")
    print(f"NaN count:                           {report['nan_count']}")
    print(f"Inf count:                           {report['inf_count']}")
    print(f"Load errors (corrupted files):       {len(report['load_errors'])}")
    print("-" * 60)
    print(f"Missing metadata (.npy w/ no row):   {report['missing_metadata']}")
    print(f"Orphan metadata (row w/ no .npy):    {report['orphan_metadata']}")
    print(f"Duplicate video_id on disk:          {len(report['duplicate_video_ids_on_disk'])}")
    print(f"Duplicate video_id in metadata:      {len(report['duplicate_video_ids_in_metadata'])}")
    print("-" * 60)

    if report["wrong_shape_details"]:
        print(f"Wrong-shape files ({len(report['wrong_shape_details'])}):")
        for vid, shape in report["wrong_shape_details"][:20]:
            print(f"  {vid}: {shape}")
        if len(report["wrong_shape_details"]) > 20:
            print(f"  ... and {len(report['wrong_shape_details']) - 20} more")

    if report["load_errors"]:
        print(f"Load errors ({len(report['load_errors'])}):")
        for vid, err in report["load_errors"][:20]:
            print(f"  {vid}: {err}")

    if report["duplicate_video_ids_on_disk"]:
        print(f"Duplicate video_ids on disk: {report['duplicate_video_ids_on_disk'][:20]}")
    if report["duplicate_video_ids_in_metadata"]:
        print(f"Duplicate video_ids in metadata: {report['duplicate_video_ids_in_metadata'][:20]}")

    print("=" * 60)
    print("RESULT: " + ("PASS -- all checks passed" if passed else "FAIL -- see counts above"))
    print("=" * 60)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Lightweight streaming, read-only validator for data/public_processed/.")
    parser.add_argument("--sequences-dir", default=config.PUBLIC_SEQUENCES_DIR)
    parser.add_argument("--metadata", default=config.PUBLIC_METADATA_PATH)
    parser.add_argument("--dataset-csv", default=os.path.join(config.PUBLIC_RAW_DIR, "dataset.csv"))
    parser.add_argument("--expected-total", type=int, default=None, help="Override the expected file/row count instead of deriving it from dataset.csv")
    parser.add_argument("--progress-every", type=int, default=500, help="Print a progress line every N files (0 to disable)")
    args = parser.parse_args()

    report = validate(
        sequences_dir=args.sequences_dir,
        metadata_path=args.metadata,
        dataset_csv_path=args.dataset_csv,
        progress_every=args.progress_every,
        expected_total_override=args.expected_total,
    )
    passed = print_report(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
