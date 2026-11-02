"""
Inspects data/public_processed/ (produced by preprocess_public_dataset.py)
and reports: total videos processed, successful/failed counts, number of
labels, samples per label, sequence length / feature dimension
consistency, average per-hand/pose detection rates, and any corrupted
.npy files (rows marked successful whose file is missing or malformed).

Run: python src/validate_public_dataset.py
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def validate_public_dataset(metadata_path=config.PUBLIC_METADATA_PATH):
    report = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "labels": defaultdict(int),
        "sequence_lengths": set(),
        "feature_dimensions": set(),
        "left_rates": [],
        "right_rates": [],
        "pose_rates": [],
        "corrupted_files": [],
        "failure_reasons": [],
    }

    if not os.path.exists(metadata_path):
        print(f"No metadata file found at: {metadata_path}")
        print("Run src/preprocess_public_dataset.py first.")
        return report

    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    report["total_processed"] = len(rows)

    for row in rows:
        if row["processing_status"] != "success":
            report["failed"] += 1
            report["failure_reasons"].append((row["video_id"], row.get("failure_reason", "")))
            continue

        report["successful"] += 1
        report["labels"][row["label"]] += 1

        seq_path = os.path.join(config.PROJECT_ROOT, row["sequence_path"])
        try:
            arr = np.load(seq_path)
        except Exception as e:
            report["corrupted_files"].append((seq_path, str(e)))
            continue

        if arr.ndim != 2:
            report["corrupted_files"].append((seq_path, f"expected 2D array, got shape {arr.shape}"))
            continue

        report["sequence_lengths"].add(arr.shape[0])
        report["feature_dimensions"].add(arr.shape[1])
        report["left_rates"].append(float(row["left_hand_detection_rate"]))
        report["right_rates"].append(float(row["right_hand_detection_rate"]))
        report["pose_rates"].append(float(row["pose_detection_rate"]))

    return report


def print_report(report):
    print("=" * 60)
    print("Public Dataset Validation Report")
    print("=" * 60)
    print(f"Total videos processed: {report['total_processed']}")
    print(f"Successful sequences:   {report['successful']}")
    print(f"Failed sequences:       {report['failed']}")
    print("-" * 60)

    print(f"Number of labels: {len(report['labels'])}")
    for label, count in sorted(report["labels"].items()):
        print(f"  {label}: {count} samples")
    print("-" * 60)

    seq_lens = report["sequence_lengths"]
    feat_dims = report["feature_dimensions"]
    if seq_lens:
        if len(seq_lens) == 1:
            print(f"Sequence length: {next(iter(seq_lens))} (consistent)")
        else:
            print(f"WARNING: inconsistent sequence lengths found: {sorted(seq_lens)}")
    if feat_dims:
        if len(feat_dims) == 1:
            print(f"Feature dimension: {next(iter(feat_dims))} (consistent)")
        else:
            print(f"WARNING: inconsistent feature dimensions found: {sorted(feat_dims)}")
    print("-" * 60)

    if report["left_rates"]:
        print(f"Average left-hand detection rate:  {100 * sum(report['left_rates']) / len(report['left_rates']):.1f}%")
        print(f"Average right-hand detection rate: {100 * sum(report['right_rates']) / len(report['right_rates']):.1f}%")
        print(f"Average pose detection rate:       {100 * sum(report['pose_rates']) / len(report['pose_rates']):.1f}%")
    else:
        print("No successful sequences to compute detection rates from.")
    print("-" * 60)

    if report["corrupted_files"]:
        print(f"Corrupted files ({len(report['corrupted_files'])}):")
        for fpath, err in report["corrupted_files"]:
            print(f"  {fpath}: {err}")
    else:
        print("Corrupted files: none")

    if report["failure_reasons"]:
        print(f"\nFailed videos ({len(report['failure_reasons'])}):")
        for video_id, reason in report["failure_reasons"]:
            print(f"  {video_id}: {reason}")
    print("=" * 60)


if __name__ == "__main__":
    report = validate_public_dataset()
    print_report(report)
