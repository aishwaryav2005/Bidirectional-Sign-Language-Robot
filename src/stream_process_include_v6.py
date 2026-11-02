"""
Streaming INCLUDE preprocessing for Greetings/Home/Jobs (v6).

Reads video bytes directly from the source ZIP archives in Downloads
(confirmed locally-hydrated, unaffected by the OneDrive dehydration
found in data/external/INCLUDE -- see reports/onedrive_dehydration_notice.md),
processes ONE video at a time, and writes ONLY the small permanent
outputs (feature array + metadata row) to a NEW, clean output directory.
No bulk extraction. No second permanent raw-video copy.

Pipeline per video:
    zip member (read into memory)
    -> ONE temp video file
    -> existing, unmodified feature_extractor.py / landmark_detector.py / temporal_sampling.py
    -> (60, 147) feature array
    -> save permanent .npy + metadata row
    -> delete temp video file immediately

video_id scheme: "<Category>__<Label>__<original_basename>" -- globally
unique by construction. A category-only prefix was tried first and found
INSUFFICIENT: a pre-processing collision check found 29 basenames
repeated WITHIN Greetings alone, across different gloss labels (e.g.
"Greetings/48. Hello/MVI_0037.MOV" and "Greetings/50. Alright/MVI_0037.MOV")
-- the exact same root cause (camera-sequential filenames reused across
recording sessions) as the 41-video collision bug found and fixed in a
prior session (see reports/isl_expansion_v2_report.md section 4, and
data/external/INCLUDE/raw_videos_dedup/, which used this same
Category__Label__filename scheme). Including the label in the ID
resolves both the within-category and cross-category collisions; see
reports/include_filename_collision_check.json for the full verification.

Output (NEW, does not touch any existing/frozen path):
    data/include_streamed_processed/sequences/<label>/<video_id>.npy
    data/include_streamed_processed/metadata/processing_metadata.csv

Usage:
    python src/stream_process_include_v6.py [--limit-per-category N]
"""

import argparse
import csv
import json
import os
import sys
import time
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector
from temporal_sampling import sample_frame_indices

DOWNLOADS = r"C:\Users\Elitebook 840 G9\Downloads"
OUTPUT_DIR = os.path.join(config.DATA_DIR, "include_streamed_processed")
SEQUENCES_DIR = os.path.join(OUTPUT_DIR, "sequences")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
METADATA_PATH = os.path.join(METADATA_DIR, "processing_metadata.csv")
TEMP_DIR = os.path.join(OUTPUT_DIR, "_tmp")  # single reusable temp slot, cleared continuously

CATEGORY_ZIPS = {
    "Greetings": ["Greetings_1of2.zip", "Greetings_2of2.zip"],
    "Home": ["Home_1of4.zip", "Home_2of4.zip", "Home_3of4.zip", "Home_4of4.zip"],
    "Jobs": ["Jobs_1of2.zip", "Jobs_2of2.zip"],
}

_METADATA_FIELDS = [
    "video_id", "category", "label", "source_archive", "source_filename",
    "original_video_length_frames", "sampled_frame_indices",
    "processing_status", "failure_reason", "sequence_length", "feature_dimension",
    "left_hand_detection_rate", "right_hand_detection_rate", "pose_detection_rate",
    "sequence_path", "processing_time_seconds",
]


def clean_label(gloss_folder_name):
    import re
    return re.sub(r"^\s*\d+\.\s*", "", gloss_folder_name).strip()


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


def process_one(zf, member, category, label, video_id, detector, next_ts, temp_dir):
    ext = os.path.splitext(member)[1]
    temp_path = os.path.join(temp_dir, f"stream_temp{ext}")
    data = zf.read(member)
    with open(temp_path, "wb") as f:
        f.write(data)

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise RuntimeError("could not open temp video")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("zero decodable frames")

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
        stats = {
            "source_length": len(frames), "sampled_indices": indices.tolist(),
            "left_rate": float(np.mean(l_flags)), "right_rate": float(np.mean(r_flags)),
            "pose_rate": float(np.mean(p_flags)),
        }
        return sequence, stats, next_ts, None
    except Exception as e:
        return None, None, next_ts, str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-category", type=int, default=None, help="For smoke testing: only process the first N videos of each category")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_ZIPS.keys()))
    args = parser.parse_args()

    os.makedirs(SEQUENCES_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    existing_metadata = load_existing_metadata()

    detector = LandmarkDetector()
    next_ts = 0
    n_success, n_failed, n_skipped = 0, 0, 0
    start_time = time.perf_counter()

    try:
        for category in args.categories:
            zip_names = CATEGORY_ZIPS[category]
            count_this_category = 0
            for zname in zip_names:
                zpath = os.path.join(DOWNLOADS, zname)
                with zipfile.ZipFile(zpath) as zf:
                    members = [m for m in zf.namelist() if m.lower().endswith((".mov", ".mp4"))]
                    for member in members:
                        if args.limit_per_category and count_this_category >= args.limit_per_category:
                            break
                        count_this_category += 1

                        basename = os.path.splitext(os.path.basename(member))[0]
                        gloss_folder = member.split("/")[1] if "/" in member else "unknown"
                        label = clean_label(gloss_folder)
                        label_slug = label.replace(" ", "_").replace("/", "_")
                        video_id = f"{category}__{label_slug}__{basename}"

                        if existing_metadata.get(video_id, {}).get("processing_status") == "success":
                            n_skipped += 1
                            continue

                        row = {
                            "video_id": video_id, "category": category, "label": label,
                            "source_archive": zname, "source_filename": member,
                        }
                        vstart = time.perf_counter()
                        sequence, stats, next_ts, error = process_one(zf, member, category, label, video_id, detector, next_ts, TEMP_DIR)
                        velapsed = time.perf_counter() - vstart

                        if error is None:
                            label_dir = os.path.join(SEQUENCES_DIR, label)
                            os.makedirs(label_dir, exist_ok=True)
                            seq_path = os.path.join(label_dir, f"{video_id}.npy")
                            save_sequence_atomic(seq_path, sequence)

                            row.update({
                                "original_video_length_frames": stats["source_length"],
                                "sampled_frame_indices": json.dumps(stats["sampled_indices"]),
                                "processing_status": "success", "failure_reason": "",
                                "sequence_length": sequence.shape[0], "feature_dimension": sequence.shape[1],
                                "left_hand_detection_rate": round(stats["left_rate"], 4),
                                "right_hand_detection_rate": round(stats["right_rate"], 4),
                                "pose_detection_rate": round(stats["pose_rate"], 4),
                                "sequence_path": os.path.relpath(seq_path, config.PROJECT_ROOT),
                                "processing_time_seconds": round(velapsed, 3),
                            })
                            n_success += 1
                            if n_success % 25 == 0 or args.limit_per_category:
                                print(f"[{category}] {n_success} succeeded so far... ({video_id}: OK, {velapsed:.2f}s)")
                        else:
                            row.update({
                                "original_video_length_frames": "", "sampled_frame_indices": "",
                                "processing_status": "failed", "failure_reason": error,
                                "sequence_length": "", "feature_dimension": "",
                                "left_hand_detection_rate": "", "right_hand_detection_rate": "", "pose_detection_rate": "",
                                "sequence_path": "", "processing_time_seconds": round(velapsed, 3),
                            })
                            n_failed += 1
                            print(f"[{category}] {video_id}: FAILED ({error})")

                        existing_metadata[video_id] = row
                        write_metadata_csv(existing_metadata)
                    if args.limit_per_category and count_this_category >= args.limit_per_category:
                        continue
    finally:
        detector.close()
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass

    elapsed = time.perf_counter() - start_time
    print("=" * 60)
    print(f"Done. Success: {n_success}  Failed: {n_failed}  Skipped: {n_skipped}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Metadata: {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
