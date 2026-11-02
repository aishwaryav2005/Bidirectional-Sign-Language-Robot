"""
Converts a public isolated-sign video dataset (e.g. CISLR — see
DATASET_INFO.md for why CISLR and not the full continuous iSign corpus)
into the SAME (SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH) representation used
by our own webcam recordings.

PUBLIC ISL VIDEOS -> FRAME SAMPLING -> MEDIAPIPE -> LANDMARKS
                   -> 147 FEATURES -> 60-FRAME SEQUENCE -> PROCESSED DATA

This script never downloads anything. It only processes video files that
already exist locally (see DATASET_INFO.md section 14 for how to obtain
them). It is dataset-agnostic: point it at any folder of videos plus a
two-column `video_filename,label` CSV.

Input layout expected under --videos-dir (default data/public_raw/):
    <videos-dir>/
        labels.csv              # columns: video_filename,label[,extra columns...]
        <video files...>

Any columns beyond video_filename,label (e.g. a source dataset's
`category`/`duration`) are preserved verbatim as a JSON blob in each
output row's `source_metadata` field -- nothing dataset-specific is
hardcoded here.

Output layout (default data/public_processed/):
    sequences/<label>/<video_id>.npy   # shape (SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH)
    metadata/processing_metadata.csv   # one row per attempted video

Also updates data/mappings/label_mapping.json (original label -> numeric
class ID, preserving previously-assigned IDs across runs).

RESUME / CRASH SAFETY (important for a multi-hour full-dataset run):
metadata is checkpointed to disk after EVERY video (not batched until the
end), and both the metadata CSV and each .npy file are written via a
temp-file-then-os.replace pattern, which is atomic on both Windows and
POSIX. If the process is killed at any point, everything already
completed is safely on disk; rerunning the exact same command skips every
video whose video_id is already recorded with processing_status=success
in metadata/processing_metadata.csv and continues from there -- no
progress is lost, and no already-valid sequence is redundantly
reprocessed or overwritten.

Usage:
    python src/preprocess_public_dataset.py --limit 10
    python src/preprocess_public_dataset.py --labels-csv path\\to\\labels.csv --videos-dir path\\to\\videos
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector
from temporal_sampling import sample_frame_indices

_METADATA_FIELDS = [
    "video_id",
    "original_filename",
    "label",
    "class_id",
    "original_video_length_frames",
    "sampled_frame_indices",
    "processing_status",
    "failure_reason",
    "sequence_length",
    "feature_dimension",
    "left_hand_detection_rate",
    "right_hand_detection_rate",
    "pose_detection_rate",
    "sequence_path",
    "source_metadata",
    "processing_time_seconds",
]


def _video_id_from_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def load_label_csv(labels_csv_path):
    """Reads a `video_filename,label` CSV (any additional columns, e.g. a
    source dataset's `category`/`duration`, are preserved verbatim and
    passed through into each row's `source_metadata`). Returns a list of
    dicts with keys "video_filename", "label", and "extra" (a dict of any
    remaining columns), in file order."""
    rows = []
    with open(labels_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{labels_csv_path} must have columns 'video_filename,label'; "
                f"found {reader.fieldnames}"
            )
        extra_cols = [c for c in reader.fieldnames if c not in ("video_filename", "label")]
        for row in reader:
            rows.append({
                "video_filename": row["video_filename"].strip(),
                "label": row["label"].strip(),
                "extra": {c: row[c] for c in extra_cols},
            })
    return rows


def load_existing_metadata(metadata_path):
    """Returns {video_id: row_dict} for any previously processed videos, so
    reruns are idempotent and cumulative rather than duplicating rows."""
    existing = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["video_id"]] = row
    return existing


def load_label_mapping(mapping_path):
    if os.path.exists(mapping_path):
        with open(mapping_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_label_mapping(mapping_path, mapping):
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True, ensure_ascii=False)


def write_metadata_csv(metadata_path, existing_metadata):
    """Atomically (over)writes the full metadata CSV from `existing_metadata`
    (write to a temp file, then os.replace -- an atomic rename on both
    Windows and POSIX, so a crash mid-write never leaves a truncated or
    corrupted metadata file). Called after every video during a run (not
    just once at the end) so a long run can be safely interrupted and
    resumed without losing progress already made."""
    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        for row in existing_metadata.values():
            writer.writerow(row)
    os.replace(tmp_path, metadata_path)


def save_sequence_atomic(sequence_path, sequence):
    """Writes the .npy file via a temp-file-then-rename so an interruption
    mid-write can never leave a truncated/corrupted sequence file on disk."""
    tmp_path = sequence_path + ".tmp.npy"  # keep the .npy suffix so numpy doesn't append its own
    np.save(tmp_path, sequence)
    os.replace(tmp_path, sequence_path)


def assign_class_ids(mapping, labels):
    """Adds any new labels to `mapping` (original_label -> {class_id,
    display_name}), preserving IDs already assigned in earlier runs."""
    next_id = max((entry["class_id"] for entry in mapping.values()), default=-1) + 1
    for label in labels:
        if label not in mapping:
            mapping[label] = {"class_id": next_id, "display_name": label}
            next_id += 1
    return mapping


def _read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def process_video(video_path, detector, next_timestamp_ms):
    """Runs the full extraction pipeline on one video.

    Returns (sequence_array, stats, next_timestamp_ms) on success, where
    stats = {"source_length", "sampled_indices", "left_rate", "right_rate",
    "pose_rate"}. Raises ValueError/RuntimeError with a clear message on
    any failure (unreadable file, zero frames, etc.) so the caller can
    record processing_status="failed" without crashing the batch.
    """
    frames = _read_all_frames(video_path)
    if frames is None:
        raise RuntimeError(f"could not open video file: {video_path}")
    if len(frames) == 0:
        raise RuntimeError(f"video contains zero decodable frames: {video_path}")

    source_length = len(frames)
    indices = sample_frame_indices(source_length, config.SEQUENCE_LENGTH)

    feature_vectors = []
    left_flags, right_flags, pose_flags = [], [], []

    for idx in indices:
        frame = frames[int(idx)]
        result = detector.detect(frame, next_timestamp_ms)
        next_timestamp_ms += 1
        fv, info = extract_features(result["hand_result"], result["pose_result"])
        feature_vectors.append(fv)
        left_flags.append(info["left_hand"])
        right_flags.append(info["right_hand"])
        pose_flags.append(info["pose"])

    sequence = np.stack(feature_vectors, axis=0).astype(np.float32)
    stats = {
        "source_length": source_length,
        "sampled_indices": indices.tolist(),
        "left_rate": float(np.mean(left_flags)),
        "right_rate": float(np.mean(right_flags)),
        "pose_rate": float(np.mean(pose_flags)),
    }
    return sequence, stats, next_timestamp_ms


def _print_dry_run_report(rows, labels_csv_path, videos_dir, metadata_path):
    """Read-only report: what a real run against this labels CSV would do,
    without opening MediaPipe, creating any directory, or writing any file."""
    from collections import Counter

    video_ids = [_video_id_from_filename(r["video_filename"]) for r in rows]
    id_counts = Counter(video_ids)
    duplicates = {vid: c for vid, c in id_counts.items() if c > 1}

    existing_metadata = load_existing_metadata(metadata_path)
    already_success = sum(1 for vid in video_ids if existing_metadata.get(vid, {}).get("processing_status") == "success")
    already_failed = sum(1 for vid in video_ids if existing_metadata.get(vid, {}).get("processing_status") == "failed")
    remaining = len(rows) - already_success

    print("=" * 60)
    print("DRY RUN -- no video will be processed, no file will be written")
    print("=" * 60)
    print(f"Labels CSV:                       {labels_csv_path}")
    print(f"Videos dir:                       {videos_dir}")
    print(f"Existing metadata file:           {metadata_path}"
          f" ({'found' if os.path.exists(metadata_path) else 'not found -- nothing processed yet'})")
    print("-" * 60)
    print(f"Total videos in labels CSV:            {len(rows)}")
    print(f"Already processed (success, will skip): {already_success}")
    print(f"Previously failed (will retry):         {already_failed}")
    print(f"Remaining to process this run:          {remaining}")
    print(f"Expected new sequences if all succeed:  {remaining}")
    print("-" * 60)
    print(f"Duplicate video_id within this labels CSV: {len(duplicates)}")
    if duplicates:
        for vid, c in duplicates.items():
            print(f"  {vid}: appears {c} times")
    print("=" * 60)
    print("Note: this command does not check video_filename<->UID matching "
          "against dataset.csv -- that verification happens when "
          "select_cislr_sample.py builds the labels CSV (run that first "
          "and check its 'MISMATCHED' count).")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Preprocess a public ISL video dataset into (SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH) sequences.")
    parser.add_argument("--videos-dir", default=config.PUBLIC_RAW_DIR, help="Directory containing the video files (default: data/public_raw)")
    parser.add_argument("--labels-csv", default=None, help="Path to a video_filename,label CSV (default: <videos-dir>/labels.csv)")
    parser.add_argument("--output-dir", default=config.PUBLIC_PROCESSED_DIR, help="Where to write sequences/ and metadata/ (default: data/public_processed)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows of the labels CSV (test mode)")
    parser.add_argument("--label-mapping", default=config.LABEL_MAPPING_PATH, help="Path to label_mapping.json (default: data/mappings/label_mapping.json)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen (already processed / remaining / duplicate video_id check) without opening MediaPipe, writing any file, or processing any video")
    args = parser.parse_args()

    labels_csv_path = args.labels_csv or os.path.join(args.videos_dir, "labels.csv")
    if not os.path.exists(labels_csv_path):
        print(f"[preprocess_public_dataset.py] FATAL: labels CSV not found: {labels_csv_path}")
        print("See DATASET_INFO.md section 14 for how to obtain videos + build this CSV.")
        return 1

    rows = load_label_csv(labels_csv_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    sequences_dir = os.path.join(args.output_dir, "sequences")
    metadata_dir = os.path.join(args.output_dir, "metadata")
    metadata_path = os.path.join(metadata_dir, "processing_metadata.csv")

    if args.dry_run:
        return _print_dry_run_report(rows, labels_csv_path, args.videos_dir, metadata_path)

    os.makedirs(sequences_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    mapping = load_label_mapping(args.label_mapping)
    mapping = assign_class_ids(mapping, [row["label"] for row in rows])
    save_label_mapping(args.label_mapping, mapping)

    existing_metadata = load_existing_metadata(metadata_path)

    print("=" * 60)
    print("Public ISL Dataset Preprocessing")
    print("=" * 60)
    print(f"Videos dir:  {args.videos_dir}")
    print(f"Labels CSV:  {labels_csv_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Videos to process this run: {len(rows)}" + (f" (--limit {args.limit})" if args.limit else ""))
    print("=" * 60)

    detector = LandmarkDetector()
    next_timestamp_ms = 0
    n_success, n_failed, n_skipped = 0, 0, 0
    run_start_time = time.perf_counter()

    try:
        for i, row_in in enumerate(rows, start=1):
            video_filename = row_in["video_filename"]
            label = row_in["label"]
            source_metadata_json = json.dumps(row_in["extra"], ensure_ascii=False)
            video_id = _video_id_from_filename(video_filename)

            if existing_metadata.get(video_id, {}).get("processing_status") == "success":
                print(f"[{i}/{len(rows)}] {video_id}: already processed, skipping")
                n_skipped += 1
                continue

            video_path = os.path.join(args.videos_dir, video_filename)
            class_id = mapping[label]["class_id"]
            video_start_time = time.perf_counter()

            try:
                if not os.path.exists(video_path):
                    raise RuntimeError(f"video file not found: {video_path}")

                sequence, stats, next_timestamp_ms = process_video(video_path, detector, next_timestamp_ms)
                video_elapsed = time.perf_counter() - video_start_time

                label_dir = os.path.join(sequences_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                sequence_path = os.path.join(label_dir, f"{video_id}.npy")
                save_sequence_atomic(sequence_path, sequence)

                existing_metadata[video_id] = {
                    "video_id": video_id,
                    "original_filename": video_filename,
                    "label": label,
                    "class_id": class_id,
                    "original_video_length_frames": stats["source_length"],
                    "sampled_frame_indices": json.dumps(stats["sampled_indices"]),
                    "processing_status": "success",
                    "failure_reason": "",
                    "sequence_length": sequence.shape[0],
                    "feature_dimension": sequence.shape[1],
                    "left_hand_detection_rate": round(stats["left_rate"], 4),
                    "right_hand_detection_rate": round(stats["right_rate"], 4),
                    "pose_detection_rate": round(stats["pose_rate"], 4),
                    "sequence_path": os.path.relpath(sequence_path, config.PROJECT_ROOT),
                    "source_metadata": source_metadata_json,
                    "processing_time_seconds": round(video_elapsed, 3),
                }
                n_success += 1
                print(f"[{i}/{len(rows)}] {video_id}: OK "
                      f"(src_len={stats['source_length']}, "
                      f"L={stats['left_rate']:.0%} R={stats['right_rate']:.0%} P={stats['pose_rate']:.0%}, "
                      f"{video_elapsed:.2f}s)")

            except Exception as e:
                video_elapsed = time.perf_counter() - video_start_time
                existing_metadata[video_id] = {
                    "video_id": video_id,
                    "original_filename": video_filename,
                    "label": label,
                    "class_id": class_id,
                    "original_video_length_frames": "",
                    "sampled_frame_indices": "",
                    "processing_status": "failed",
                    "failure_reason": str(e),
                    "sequence_length": "",
                    "feature_dimension": "",
                    "left_hand_detection_rate": "",
                    "right_hand_detection_rate": "",
                    "pose_detection_rate": "",
                    "sequence_path": "",
                    "source_metadata": source_metadata_json,
                    "processing_time_seconds": round(video_elapsed, 3),
                }
                n_failed += 1
                print(f"[{i}/{len(rows)}] {video_id}: FAILED ({e})")

            # Checkpoint after every video (not just at the end of the run):
            # if the process is interrupted, everything up to and including
            # the most recently completed video is preserved on disk, and a
            # resumed run picks up from there via the skip-if-success check
            # above -- instead of losing an entire multi-hour run's progress.
            write_metadata_csv(metadata_path, existing_metadata)
    finally:
        detector.close()

    total_run_time = time.perf_counter() - run_start_time
    attempted = n_success + n_failed  # excludes skipped (no work done for those)
    avg_time = total_run_time / attempted if attempted else 0.0

    print("=" * 60)
    print(f"Done. Success: {n_success}  Failed: {n_failed}  Skipped (already done): {n_skipped}")
    print(f"Total processing time: {total_run_time:.2f}s")
    print(f"Average time per video attempted: {avg_time:.2f}s")
    print(f"Metadata written to: {metadata_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
