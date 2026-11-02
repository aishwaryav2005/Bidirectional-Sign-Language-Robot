"""
Debug utility: re-runs detection on a handful of a video's already-sampled
frames and saves a grid image with hand/pose landmarks overlaid, so
detection quality can be inspected visually rather than only numerically.

Usage:
    python src/visualize_sequence.py --video path\\to\\video.mp4 --output outputs\\debug.png
    python src/visualize_sequence.py --video-id --7-rCNOiK0_1 --output outputs\\debug.png
        (looks up the video via data/public_processed/metadata/processing_metadata.csv
        and data/public_raw/cislr_sample_10_labels.csv)
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from landmark_detector import LandmarkDetector, draw_hand_landmarks, draw_pose_landmarks


def _find_video_by_id(video_id, metadata_path, labels_csv_path, videos_root):
    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = {r["video_id"]: r for r in csv.DictReader(f)}
    if video_id not in rows:
        raise ValueError(f"{video_id} not found in {metadata_path}")
    original_filename = rows[video_id]["original_filename"]
    return os.path.join(videos_root, original_filename), rows[video_id]


def make_debug_grid(video_path, output_path, n_panels=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"video has zero frames: {video_path}")

    panel_indices = np.linspace(0, len(frames) - 1, num=min(n_panels, len(frames))).astype(int)

    detector = LandmarkDetector()
    panels = []
    try:
        for i, idx in enumerate(panel_indices):
            frame = frames[idx].copy()
            result = detector.detect(frame, i)
            frame = draw_hand_landmarks(frame, result["hand_result"])
            frame = draw_pose_landmarks(frame, result["pose_result"])
            cv2.putText(frame, f"frame {idx}/{len(frames)-1}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"frame {idx}/{len(frames)-1}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(frame)
    finally:
        detector.close()

    h, w = panels[0].shape[:2]
    cols = 3
    rows_n = (len(panels) + cols - 1) // cols
    grid = np.zeros((h * rows_n, w * cols, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = panel

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize hand/pose landmark detection on a few frames of a video.")
    parser.add_argument("--video", default=None, help="Direct path to a video file")
    parser.add_argument("--video-id", default=None, help="video_id to look up in processing_metadata.csv")
    parser.add_argument("--videos-root", default=os.path.join(config.PUBLIC_RAW_DIR, "CISLR_v1.5-a_videos", "extracted"))
    parser.add_argument("--metadata", default=config.PUBLIC_METADATA_PATH)
    parser.add_argument("--output", default=os.path.join(config.OUTPUTS_DIR, "landmark_debug_grid.png"))
    parser.add_argument("--panels", type=int, default=6)
    args = parser.parse_args()

    if args.video:
        video_path = args.video
    elif args.video_id:
        video_path, row = _find_video_by_id(args.video_id, args.metadata, None, args.videos_root)
        print(f"Resolved {args.video_id} -> {video_path} (gloss={row['label']!r})")
    else:
        print("Must pass --video or --video-id")
        return 1

    out = make_debug_grid(video_path, args.output, n_panels=args.panels)
    print(f"Wrote debug grid: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
