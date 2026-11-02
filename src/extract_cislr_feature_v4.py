"""
ISL Feature Representation v4: reprocesses ONLY the 492 CISLR videos used
by the frozen data/splits/cislr_82class/ split (328 train + 82 val + 82
test) through MediaPipe ONCE per video, computing a "master" per-frame
feature vector that contains BOTH hand-normalization variants needed by
train_cislr_feature_v4.py's experiments, so no video is ever processed
through MediaPipe more than once:

    [ wrist_relative_hands (126) | body_relative_hands (126)
      | pose (18) | detection_flags (3) ]                    = 273-D/frame

- wrist_relative_hands: EXACTLY src/feature_extractor.py's existing
  per-hand normalization (wrist-centered, scaled by wrist->middle-MCP
  distance) -- i.e. the SAME transformation already baked into every
  sequence under data/public_processed/sequences/. Reused via import,
  not reimplemented, so Experiment 1 (below) is byte-for-byte
  reproducible from EITHER this master file's first 126+18+3 columns OR
  the original frozen data/public_processed/ sequences.
- body_relative_hands: NEW. Each hand landmark is expressed relative to
  the shoulder midpoint and scaled by shoulder width -- the SAME
  reference frame src/feature_extractor.py already uses for pose (see
  _normalize_pose), just applied to hand landmarks too instead of each
  hand's own wrist. Zero-filled whenever EITHER that hand OR pose was not
  detected in a frame (a body-relative position is meaningless without a
  body reference), independent of the wrist-relative block's own
  detection state.
- pose, detection_flags: byte-for-byte identical to
  src/feature_extractor.py's existing output (reused via import).

IMPORTANT DISCOVERY (documented in reports/cislr_feature_v4_report.md):
the project's EXISTING "raw" 147-D pipeline already applies wrist-relative
+ scale normalization to hands and shoulder-relative + scale
normalization to pose -- it is NOT unnormalized/absolute-position data.
This master file's wrist_relative_hands block therefore reproduces the
EXISTING representation exactly, not a new one; body_relative_hands is
the genuinely new spatial representation this experiment introduces.

Writes to a NEW directory only:
    data/cislr_feature_v4_processed/sequences/<label>/<video_id>.npy   (60, 273)
    data/cislr_feature_v4_processed/metadata/processing_metadata.csv

Never touches data/public_processed/, data/public_raw/ (read-only
source), or any frozen split/model/report.

Usage:
    python src/extract_cislr_feature_v4.py
"""

import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from feature_extractor import _landmarks_to_array, _normalize_hand, _normalize_pose, _EPS
from landmark_detector import LandmarkDetector
from temporal_sampling import sample_frame_indices

RAW_VIDEOS_ROOT = os.path.join(config.DATA_DIR, "public_raw", "CISLR_v1.5-a_videos", "extracted", "CISLR_v1.5-a_videos")
OUTPUT_DIR = os.path.join(config.DATA_DIR, "cislr_feature_v4_processed")
SEQUENCES_DIR = os.path.join(OUTPUT_DIR, "sequences")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata", "processing_metadata.csv")

CISLR_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")

MASTER_FEATURE_LEN = config.HAND_FEATURE_LEN * 2 * 2 + config.POSE_FEATURE_LEN + config.NUM_DETECTION_FLAGS  # 126+126+18+3=273

_METADATA_FIELDS = [
    "video_id", "gloss", "split", "original_video_length_frames",
    "processing_status", "failure_reason", "sequence_path",
    "left_hand_detection_rate", "right_hand_detection_rate", "pose_detection_rate",
]


def find_split_video_ids():
    """Returns {video_id: (gloss, split)} for exactly the 492 videos in
    the frozen 82-class split -- read-only."""
    result = {}
    for split in ("train", "val", "test"):
        path = os.path.join(CISLR_SPLITS_DIR, f"{split}.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                result[row["video_id"]] = (row["gloss"], split)
    return result


def locate_video_path(video_id):
    matches = glob.glob(os.path.join(RAW_VIDEOS_ROOT, f"{video_id}.*"))
    return matches[0] if matches else None


def _normalize_body_relative(points_xyz, shoulder_mid, scale):
    """points_xyz: (21,3) hand landmarks in the SAME raw coordinate space
    as the pose landmarks passed alongside them. Shoulder-midpoint
    relative, shoulder-width scaled -- the identical reference frame
    src/feature_extractor.py's _normalize_pose already uses."""
    return (points_xyz - shoulder_mid) / scale


def extract_master_features(hand_result, pose_result):
    """One MediaPipe detection -> the 273-D master vector for one frame.
    Returns (vector, detection_info) where detection_info additionally
    reports whether each hand's body-relative block was computable
    (requires BOTH that hand and pose detected)."""
    wrist_rel_left = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    wrist_rel_right = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    body_rel_left = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    body_rel_right = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    left_detected = False
    right_detected = False

    pose_vec = np.zeros(config.POSE_FEATURE_LEN, dtype=np.float32)
    pose_detected = False
    shoulder_mid = None
    shoulder_scale = None

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        indices = [config.POSE_LANDMARK_INDICES[name] for name in config.POSE_LANDMARK_NAMES]
        selected = [landmarks[i] for i in indices]
        points = _landmarks_to_array(selected)
        normalized = _normalize_pose(points)
        pose_vec = normalized.flatten()
        pose_detected = True

        left_shoulder = points[0]
        right_shoulder = points[1]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        shoulder_scale = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_scale < _EPS:
            shoulder_scale = 1.0

    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            points = _landmarks_to_array(landmarks)
            wrist_rel = _normalize_hand(points).flatten()
            label = handedness[0].category_name if handedness else None

            body_rel = None
            if pose_detected:
                body_rel = _normalize_body_relative(points, shoulder_mid, shoulder_scale).flatten()

            if label == "Left":
                wrist_rel_left = wrist_rel
                left_detected = True
                if body_rel is not None:
                    body_rel_left = body_rel
            elif label == "Right":
                wrist_rel_right = wrist_rel
                right_detected = True
                if body_rel is not None:
                    body_rel_right = body_rel

    flags = np.array([float(left_detected), float(right_detected), float(pose_detected)], dtype=np.float32)

    vector = np.concatenate([wrist_rel_left, wrist_rel_right, body_rel_left, body_rel_right, pose_vec, flags])
    assert vector.shape[0] == MASTER_FEATURE_LEN, f"expected {MASTER_FEATURE_LEN}, got {vector.shape[0]}"

    detection_info = {
        "left_hand": left_detected, "right_hand": right_detected, "pose": pose_detected,
        "left_body_relative_valid": left_detected and pose_detected,
        "right_body_relative_valid": right_detected and pose_detected,
    }
    return vector, detection_info


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


def save_sequence_atomic(path, sequence):
    tmp = path + ".tmp.npy"
    np.save(tmp, sequence)
    os.replace(tmp, path)


def write_metadata_csv(rows):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    tmp = METADATA_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    os.replace(tmp, METADATA_PATH)


def main():
    video_ids = find_split_video_ids()
    print(f"Target: {len(video_ids)} videos from the frozen 82-class split")

    os.makedirs(SEQUENCES_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["video_id"]] = row

    detector = LandmarkDetector()
    next_ts = 0
    n_success, n_failed, n_skipped = 0, 0, 0
    start = time.perf_counter()

    try:
        for i, (video_id, (gloss, split)) in enumerate(sorted(video_ids.items()), start=1):
            if existing.get(video_id, {}).get("processing_status") == "success":
                n_skipped += 1
                continue

            video_path = locate_video_path(video_id)
            row = {"video_id": video_id, "gloss": gloss, "split": split}
            try:
                if video_path is None:
                    raise RuntimeError(f"raw video not found for video_id={video_id}")
                frames = _read_all_frames(video_path)
                if not frames:
                    raise RuntimeError(f"unreadable/empty video: {video_path}")

                indices = sample_frame_indices(len(frames), config.SEQUENCE_LENGTH)
                vectors, l_flags, r_flags, p_flags = [], [], [], []
                for idx in indices:
                    result = detector.detect(frames[int(idx)], next_ts)
                    next_ts += 1
                    vec, info = extract_master_features(result["hand_result"], result["pose_result"])
                    vectors.append(vec)
                    l_flags.append(info["left_hand"])
                    r_flags.append(info["right_hand"])
                    p_flags.append(info["pose"])

                sequence = np.stack(vectors, axis=0).astype(np.float32)
                label_dir = os.path.join(SEQUENCES_DIR, gloss)
                os.makedirs(label_dir, exist_ok=True)
                seq_path = os.path.join(label_dir, f"{video_id}.npy")
                save_sequence_atomic(seq_path, sequence)

                row.update({
                    "original_video_length_frames": len(frames), "processing_status": "success",
                    "failure_reason": "", "sequence_path": os.path.relpath(seq_path, config.PROJECT_ROOT),
                    "left_hand_detection_rate": round(float(np.mean(l_flags)), 4),
                    "right_hand_detection_rate": round(float(np.mean(r_flags)), 4),
                    "pose_detection_rate": round(float(np.mean(p_flags)), 4),
                })
                n_success += 1
                print(f"[{i}/{len(video_ids)}] {video_id} ({split}): OK")
            except Exception as e:
                row.update({
                    "original_video_length_frames": "", "processing_status": "failed",
                    "failure_reason": str(e), "sequence_path": "",
                    "left_hand_detection_rate": "", "right_hand_detection_rate": "", "pose_detection_rate": "",
                })
                n_failed += 1
                print(f"[{i}/{len(video_ids)}] {video_id} ({split}): FAILED ({e})")

            existing[video_id] = row
            write_metadata_csv(existing)
    finally:
        detector.close()

    elapsed = time.perf_counter() - start
    print("=" * 60)
    print(f"Done. Success: {n_success}  Failed: {n_failed}  Skipped: {n_skipped}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Master feature dimension: {MASTER_FEATURE_LEN}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
