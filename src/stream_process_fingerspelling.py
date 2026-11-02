"""
Streaming preprocessing for ISL-Fingerspelling (separate fingerspelling model).

Source videos are already local, individual MP4 files at
C:\\Users\\Elitebook 840 G9\\OneDrive\\ISL-Fingerspelling\\videos\\<uid>.mp4
-- unlike the INCLUDE streaming script (stream_process_include_v6.py), there
is no ZIP to read and therefore NO temporary video copy is ever created:
each video is opened directly, read-only, from its original location.

Pipeline per video (one at a time, resumable):
    original mp4 (opened read-only, never copied/moved/modified)
    -> existing, unmodified feature_extractor.py / landmark_detector.py /
       temporal_sampling.py (same functions used for CISLR and INCLUDE)
    -> (60, 147) feature array
    -> shape / NaN / Inf validation
    -> save permanent .npy + metadata row (with character-level annotations
       preserved, since this dataset is NOT a single-label classification
       problem -- see reports/isl_label_mapping notes and the dataset
       strategy report)

This script does NOT merge fingerspelling data into CISLR/INCLUDE in any
way: separate output directory, separate metadata schema, separate splits.
It does NOT train anything.

Output (NEW directory, does not touch any existing/frozen path):
    data/fingerspelling_processed/sequences/<uid>.npy
    data/fingerspelling_processed/metadata/processing_metadata.csv
    data/fingerspelling_processed/splits/{train,test}.csv   (signer-independent,
        from signer_split.csv -- test signer never appears in train)
    data/fingerspelling_processed/splits/{standard_train,standard_test}.csv
        (from split_info.csv, kept for reference; NOT signer-independent)
    data/fingerspelling_processed/reports/  (written by a separate validation
        script after this one finishes)

Usage:
    python src/stream_process_fingerspelling.py [--limit N] [--min-free-gb 7]
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector
from temporal_sampling import sample_frame_indices

SOURCE_ROOT = r"C:\Users\Elitebook 840 G9\OneDrive\ISL-Fingerspelling"
VIDEOS_DIR = os.path.join(SOURCE_ROOT, "videos")
FS_ANNOTATIONS_CSV = os.path.join(SOURCE_ROOT, "fingerspelling_annotations.csv")
FS_ANNOTATIONS_UPDATED_CSV = os.path.join(SOURCE_ROOT, "fingerspelling_annotations_updated.csv")
LETTER_ANNOTATIONS_CSV = os.path.join(SOURCE_ROOT, "letter_annotations.csv")
SIGNER_SPLIT_CSV = os.path.join(SOURCE_ROOT, "signer_split.csv")
SPLIT_INFO_CSV = os.path.join(SOURCE_ROOT, "split_info.csv")

OUTPUT_DIR = os.path.join(config.DATA_DIR, "fingerspelling_processed")
SEQUENCES_DIR = os.path.join(OUTPUT_DIR, "sequences")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
METADATA_PATH = os.path.join(METADATA_DIR, "processing_metadata.csv")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")

_METADATA_FIELDS = [
    "video_id", "signer", "signer_independent_split", "standard_split",
    "full_text", "full_text_source", "num_characters", "char_sequence_json",
    "original_video_length_frames", "sampled_frame_indices",
    "processing_status", "failure_reason",
    "sequence_length", "feature_dimension",
    "left_hand_detection_rate", "right_hand_detection_rate", "pose_detection_rate",
    "sequence_path", "processing_time_seconds",
]


def load_csv_dict(path, key_col):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row[key_col]] = row
    return rows


def load_letter_sequences(path):
    """uid -> sorted list of {letter, start_frame, end_frame} by char_index."""
    by_uid = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_uid[row["uid"]].append(
                {
                    "char_index": int(row["char_index"]),
                    "letter": row["letter"],
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                }
            )
    for uid in by_uid:
        by_uid[uid].sort(key=lambda r: r["char_index"])
    return by_uid


def load_existing_metadata():
    existing = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["video_id"]] = row
    return existing


def write_metadata_csv(rows):
    os.makedirs(METADATA_DIR, exist_ok=True)
    tmp = METADATA_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    os.replace(tmp, METADATA_PATH)


def save_sequence_atomic(path, sequence):
    tmp = path + ".tmp.npy"
    np.save(tmp, sequence)
    os.replace(tmp, path)


def is_valid_existing_sequence(seq_path, row):
    """Resumability check: trust the metadata row only if the .npy file
    actually exists and has the expected shape -- do not trust the CSV
    flag alone (it could be stale from an interrupted/corrupted write)."""
    if row.get("processing_status") != "success":
        return False
    if not os.path.exists(seq_path):
        return False
    try:
        arr = np.load(seq_path)
    except Exception:
        return False
    if arr.shape != (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH):
        return False
    if np.isnan(arr).any() or np.isinf(arr).any():
        return False
    return True


def free_space_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def process_one_video(video_path, detector, next_ts):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, next_ts, "could not open video"
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None, None, next_ts, "zero decodable frames"

    indices = sample_frame_indices(len(frames), config.SEQUENCE_LENGTH)
    vectors, l_flags, r_flags, p_flags = [], [], [], []
    for idx in indices:
        result = detector.detect(frames[int(idx)], next_ts)
        next_ts += 1
        fv, info = extract_features(result["hand_result"], result["pose_result"])
        vectors.append(fv)
        l_flags.append(info["left_hand"])
        r_flags.append(info["right_hand"])
        p_flags.append(info["pose"])

    sequence = np.stack(vectors, axis=0).astype(np.float32)

    if sequence.shape != (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH):
        return None, None, next_ts, f"invalid shape {sequence.shape}"
    if np.isnan(sequence).any():
        return None, None, next_ts, "NaN in sequence"
    if np.isinf(sequence).any():
        return None, None, next_ts, "Inf in sequence"

    stats = {
        "source_length": len(frames),
        "sampled_indices": indices.tolist(),
        "left_rate": float(np.mean(l_flags)),
        "right_rate": float(np.mean(r_flags)),
        "pose_rate": float(np.mean(p_flags)),
    }
    return sequence, stats, next_ts, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: process only the first N videos")
    parser.add_argument("--min-free-gb", type=float, default=7.0, help="Stop safely if free space on C: falls below this many GB")
    parser.add_argument("--check-every", type=int, default=20, help="Check free space every N videos")
    args = parser.parse_args()

    os.makedirs(SEQUENCES_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    start_free_gb = free_space_gb("C:\\")
    print(f"Free space on C: at start: {start_free_gb:.2f} GB")
    if start_free_gb < args.min_free_gb:
        print(f"ABORT: free space already below --min-free-gb={args.min_free_gb}. Nothing processed.")
        return

    fs_annot = load_csv_dict(FS_ANNOTATIONS_CSV, "uid")
    fs_annot_updated = load_csv_dict(FS_ANNOTATIONS_UPDATED_CSV, "uid")
    signer_split = load_csv_dict(SIGNER_SPLIT_CSV, "uid")
    split_info = load_csv_dict(SPLIT_INFO_CSV, "uid")
    letter_sequences = load_letter_sequences(LETTER_ANNOTATIONS_CSV)

    video_files = sorted(f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mp4"))
    if args.limit:
        video_files = video_files[: args.limit]

    existing_metadata = load_existing_metadata()

    detector = LandmarkDetector()
    next_ts = 0
    n_success, n_failed, n_skipped = 0, 0, 0
    start_time = time.perf_counter()
    stopped_early = False

    try:
        for i, fname in enumerate(video_files):
            uid = os.path.splitext(fname)[0]
            video_path = os.path.join(VIDEOS_DIR, fname)
            seq_path = os.path.join(SEQUENCES_DIR, f"{uid}.npy")

            if i > 0 and i % args.check_every == 0:
                free_gb = free_space_gb("C:\\")
                if free_gb < args.min_free_gb:
                    print(f"STOPPING SAFELY: free space {free_gb:.2f} GB < --min-free-gb={args.min_free_gb} "
                          f"after {i} videos considered.")
                    stopped_early = True
                    break

            existing_row = existing_metadata.get(uid, {})
            if is_valid_existing_sequence(seq_path, existing_row):
                n_skipped += 1
                continue

            letters = letter_sequences.get(uid, [])
            full_text = None
            full_text_source = ""
            if uid in fs_annot_updated:
                full_text = fs_annot_updated[uid]["text"]
                full_text_source = "fingerspelling_annotations_updated.csv (letter-derived, authoritative)"
            elif uid in fs_annot:
                full_text = fs_annot[uid]["text"]
                full_text_source = "fingerspelling_annotations.csv"
            else:
                full_text = ""
                full_text_source = "missing"

            signer_row = signer_split.get(uid)
            standard_row = split_info.get(uid)

            row = {
                "video_id": uid,
                "signer": signer_row["signer"] if signer_row else "",
                "signer_independent_split": signer_row["split"] if signer_row else "",
                "standard_split": standard_row["split"] if standard_row else "",
                "full_text": full_text,
                "full_text_source": full_text_source,
                "num_characters": len(letters),
                "char_sequence_json": json.dumps(letters),
            }

            vstart = time.perf_counter()
            sequence, stats, next_ts, error = process_one_video(video_path, detector, next_ts)
            velapsed = time.perf_counter() - vstart

            if error is None:
                save_sequence_atomic(seq_path, sequence)
                row.update({
                    "original_video_length_frames": stats["source_length"],
                    "sampled_frame_indices": json.dumps(stats["sampled_indices"]),
                    "processing_status": "success",
                    "failure_reason": "",
                    "sequence_length": sequence.shape[0],
                    "feature_dimension": sequence.shape[1],
                    "left_hand_detection_rate": round(stats["left_rate"], 4),
                    "right_hand_detection_rate": round(stats["right_rate"], 4),
                    "pose_detection_rate": round(stats["pose_rate"], 4),
                    "sequence_path": os.path.relpath(seq_path, config.PROJECT_ROOT),
                    "processing_time_seconds": round(velapsed, 3),
                })
                n_success += 1
                if n_success % 50 == 0:
                    print(f"{n_success} succeeded so far... ({uid}: OK, {velapsed:.2f}s)")
            else:
                row.update({
                    "original_video_length_frames": "", "sampled_frame_indices": "",
                    "processing_status": "failed", "failure_reason": error,
                    "sequence_length": "", "feature_dimension": "",
                    "left_hand_detection_rate": "", "right_hand_detection_rate": "", "pose_detection_rate": "",
                    "sequence_path": "", "processing_time_seconds": round(velapsed, 3),
                })
                n_failed += 1
                print(f"{uid}: FAILED ({error})")

            existing_metadata[uid] = row
            write_metadata_csv(existing_metadata)
    finally:
        detector.close()

    # --- write signer-independent and standard split files from metadata ---
    def write_split_csv(name, split_field, split_value):
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "signer", "full_text", "sequence_path"])
            for vid, row in sorted(existing_metadata.items()):
                if row.get(split_field) == split_value and row.get("processing_status") == "success":
                    writer.writerow([vid, row.get("signer", ""), row.get("full_text", ""), row.get("sequence_path", "")])

    write_split_csv("train", "signer_independent_split", "train")
    write_split_csv("test", "signer_independent_split", "test")
    write_split_csv("standard_train", "standard_split", "train")
    write_split_csv("standard_test", "standard_split", "test")

    elapsed = time.perf_counter() - start_time
    end_free_gb = free_space_gb("C:\\")
    print("=" * 60)
    print(f"Done. Success: {n_success}  Failed: {n_failed}  Skipped (already valid): {n_skipped}")
    print(f"Stopped early due to low free space: {stopped_early}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Free space: start {start_free_gb:.2f} GB -> end {end_free_gb:.2f} GB")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Splits dir: {SPLITS_DIR}")
    print("No validation split exists yet -- signer_independent_split and "
          "standard_split are both train/test only, documented as such.")
    print("=" * 60)


if __name__ == "__main__":
    main()
