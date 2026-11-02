"""
Inspects collected data/sequences/ and reports dataset statistics:
number of samples, labels, samples per label, sequence length / feature
dimension consistency, missing-hand statistics, and corrupted files.

Run: python src/validate_dataset.py
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def validate_dataset(sequences_dir=config.SEQUENCES_DIR, metadata_path=config.METADATA_PATH):
    report = {
        "labels": {},
        "total_samples": 0,
        "sequence_lengths": set(),
        "feature_dimensions": set(),
        "corrupted_files": [],
        "missing_hand_stats": defaultdict(lambda: {"left_missing_frames": 0,
                                                     "right_missing_frames": 0,
                                                     "pose_missing_frames": 0,
                                                     "total_frames": 0}),
    }

    if not os.path.isdir(sequences_dir):
        print(f"No sequences directory found at: {sequences_dir}")
        return report

    label_dirs = sorted(
        d for d in os.listdir(sequences_dir)
        if os.path.isdir(os.path.join(sequences_dir, d))
    )

    for label in label_dirs:
        label_path = os.path.join(sequences_dir, label)
        npy_files = sorted(f for f in os.listdir(label_path) if f.endswith(".npy"))
        valid_count = 0

        for fname in npy_files:
            fpath = os.path.join(label_path, fname)
            try:
                arr = np.load(fpath)
            except Exception as e:
                report["corrupted_files"].append((fpath, str(e)))
                continue

            if arr.ndim != 2:
                report["corrupted_files"].append((fpath, f"expected 2D array, got shape {arr.shape}"))
                continue

            seq_len, feat_dim = arr.shape
            report["sequence_lengths"].add(seq_len)
            report["feature_dimensions"].add(feat_dim)
            valid_count += 1

            stats = report["missing_hand_stats"][label]
            stats["total_frames"] += seq_len
            # detection flags are the last 3 values of each frame's feature
            # vector: [left_hand_detected, right_hand_detected, pose_detected]
            if feat_dim >= config.NUM_DETECTION_FLAGS:
                flags = arr[:, -config.NUM_DETECTION_FLAGS:]
                stats["left_missing_frames"] += int(np.sum(flags[:, 0] == 0))
                stats["right_missing_frames"] += int(np.sum(flags[:, 1] == 0))
                stats["pose_missing_frames"] += int(np.sum(flags[:, 2] == 0))

        report["labels"][label] = valid_count
        report["total_samples"] += valid_count

    return report


def print_report(report):
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)

    if not report["labels"]:
        print("No samples found.")
        return

    for label, count in sorted(report["labels"].items()):
        print(f"{label}: {count} samples")

    print("-" * 60)
    print(f"Total: {report['total_samples']} sequences across {len(report['labels'])} labels")

    seq_lens = report["sequence_lengths"]
    feat_dims = report["feature_dimensions"]
    if len(seq_lens) == 1:
        print(f"Sequence length: {next(iter(seq_lens))} (consistent)")
    else:
        print(f"WARNING: inconsistent sequence lengths found: {sorted(seq_lens)}")

    if len(feat_dims) == 1:
        print(f"Feature dimension: {next(iter(feat_dims))} (consistent)")
    else:
        print(f"WARNING: inconsistent feature dimensions found: {sorted(feat_dims)}")

    print("-" * 60)
    print("Missing-hand / pose statistics (frames where not detected):")
    for label, stats in sorted(report["missing_hand_stats"].items()):
        total = stats["total_frames"] or 1
        print(
            f"  {label}: left missing {stats['left_missing_frames']}/{total} "
            f"({100 * stats['left_missing_frames'] / total:.1f}%), "
            f"right missing {stats['right_missing_frames']}/{total} "
            f"({100 * stats['right_missing_frames'] / total:.1f}%), "
            f"pose missing {stats['pose_missing_frames']}/{total} "
            f"({100 * stats['pose_missing_frames'] / total:.1f}%)"
        )

    print("-" * 60)
    if report["corrupted_files"]:
        print(f"Corrupted files ({len(report['corrupted_files'])}):")
        for fpath, err in report["corrupted_files"]:
            print(f"  {fpath}: {err}")
    else:
        print("Corrupted files: none")
    print("=" * 60)


if __name__ == "__main__":
    report = validate_dataset()
    print_report(report)
