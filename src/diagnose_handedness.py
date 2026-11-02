"""
Diagnostic (read-only, does not touch any processed dataset).

HISTORY: this script was used to empirically determine the correct
mirroring convention, by running the SAME frames through detection twice
(once unmirrored, once mirrored via cv2.flip) and comparing what
MediaPipe reports for the SAME physical hand in each case. Conclusion
(confirmed both on a single-hand-dominant CISLR video and live, on a
known physical right hand): UNMIRRORED input gives anatomically correct
Left/Right labels for this camera + model. main.py was updated
accordingly to no longer mirror frames before detection (see
DATASET_INFO.md and README.md for the full writeup).

CURRENT STATE: both pipelines now feed MediaPipe unmirrored frames --
preprocess_public_dataset.py (never mirrored CISLR) and main.py (mirror
removed). The "mirrored" pass here is kept only as a regression check:
if it ever again matches what main.py actually does, that means the flip
was reintroduced and handedness will be wrong again.

MediaPipe's own documentation (ai.google.dev/edge/mediapipe/solutions/
vision/hand_landmarker): "Handedness is determined assuming the input
image is mirrored, i.e., taken with a front-facing/selfie camera with
images flipped horizontally. If it is not the case, please swap the
handedness output in the application." Our empirical result differs from
a literal reading of this note -- see DATASET_INFO.md for why (the
model's actual behavior on our footage was verified directly rather than
assumed from the docs).

Ground-truth check used for CISLR (no live human needed): for a normal,
non-selfie frontal recording, a person's own RIGHT hand appears on
SCREEN-LEFT (small x-fraction) and their own LEFT hand appears on
SCREEN-RIGHT (large x-fraction) -- ordinary mirror geometry for someone
facing the camera. Each panel prints the wrist's x-fraction alongside
MediaPipe's label so this can be checked directly against the image.

Usage:
    python src/diagnose_handedness.py --cislr-video-id -0XujLX2_9I_2
    python src/diagnose_handedness.py --cislr-video path\\to\\video.mp4 --webcam-seconds 4
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from camera import Camera, CameraError
from landmark_detector import LandmarkDetector, HAND_CONNECTIONS


def _read_all_frames(video_path):
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
    return frames


def _annotate_frame(frame_bgr, hand_result, frame_label):
    h, w = frame_bgr.shape[:2]
    frame_bgr = frame_bgr.copy()
    records = []

    for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (0, 255, 0), 2)
        for p in points:
            cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)

        label = handedness[0].category_name
        score = handedness[0].score
        wrist_x_fraction = hand_landmarks[0].x
        wrist_px = points[0]

        big_text = f"{label.upper()} HAND"
        color = (255, 0, 255) if label == "Left" else (0, 200, 255)
        cv2.putText(frame_bgr, big_text, (wrist_px[0] - 40, wrist_px[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, big_text, (wrist_px[0] - 40, wrist_px[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        records.append({"label": label, "score": score, "wrist_x_fraction": wrist_x_fraction})

    cv2.putText(frame_bgr, frame_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, frame_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr, records


def run_diagnostic_pass(frames, mirror, n_panels, pass_label):
    """Runs detection on n_panels evenly-spaced frames, optionally mirrored
    first. Returns (annotated_panels, per_frame_records). Uses its own
    fresh LandmarkDetector (VIDEO mode requires strictly increasing
    timestamps within one detector's lifetime; a dedicated detector per
    independent pass sidesteps that instead of threading a shared counter
    through every call)."""
    indices = np.linspace(0, len(frames) - 1, num=min(n_panels, len(frames))).astype(int)
    panels = []
    all_records = []
    detector = LandmarkDetector()
    try:
        for i, idx in enumerate(indices):
            frame = frames[int(idx)]
            if mirror:
                frame = cv2.flip(frame, 1)
            result = detector.detect(frame, i)
            frame_label = f"{pass_label} | frame {idx}/{len(frames)-1}"
            annotated, records = _annotate_frame(frame, result["hand_result"], frame_label)
            panels.append(annotated)
            all_records.append({"frame_index": int(idx), "hands": records})
    finally:
        detector.close()
    return panels, all_records


def make_grid(panels, cols=3):
    if not panels:
        return None
    h, w = panels[0].shape[:2]
    rows_n = (len(panels) + cols - 1) // cols
    grid = np.zeros((h * rows_n, w * cols, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = panel
    return grid


def print_records_table(title, all_records):
    print(f"\n--- {title} ---")
    print(f"{'frame':>6}  {'hand':<7} {'label':<7} {'score':>6}  {'wrist_x_frac':>13}  {'screen side':<12}")
    for frame_rec in all_records:
        idx = frame_rec["frame_index"]
        if not frame_rec["hands"]:
            print(f"{idx:>6}  {'(none)':<7}")
            continue
        for h_i, hand in enumerate(frame_rec["hands"]):
            side = "LEFT half" if hand["wrist_x_fraction"] < 0.5 else "RIGHT half"
            print(f"{idx:>6}  hand{h_i:<3} {hand['label']:<7} {hand['score']:>6.2f}  "
                  f"{hand['wrist_x_fraction']:>13.3f}  {side:<12}")


def resolve_cislr_video_path(video_id, metadata_path, videos_root):
    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = {r["video_id"]: r for r in csv.DictReader(f)}
    if video_id not in rows:
        raise ValueError(f"{video_id} not in {metadata_path}")
    return os.path.join(videos_root, rows[video_id]["original_filename"]), rows[video_id]["label"]


def capture_webcam_clip(seconds):
    cam = Camera()
    frames = []
    start = time.time()
    try:
        while time.time() - start < seconds:
            ok, frame = cam.read()
            if ok:
                frames.append(frame)
    finally:
        cam.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Diagnose MediaPipe handedness convention consistency between the CISLR (unmirrored) and webcam (mirrored) pipelines.")
    parser.add_argument("--cislr-video-id", default=None, help="video_id already in data/public_processed/metadata/processing_metadata.csv")
    parser.add_argument("--cislr-video", default=None, help="Direct path to a CISLR video (overrides --cislr-video-id)")
    parser.add_argument("--videos-root", default=os.path.join(config.PUBLIC_RAW_DIR, "CISLR_v1.5-a_videos", "extracted"))
    parser.add_argument("--metadata", default=config.PUBLIC_METADATA_PATH)
    parser.add_argument("--webcam-seconds", type=float, default=4.0)
    parser.add_argument("--skip-webcam", action="store_true")
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUTS_DIR, "handedness_diagnostic"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- CISLR: same frames, two passes (unmirrored = real pipeline behavior; mirrored = comparison) ----
    if args.cislr_video:
        cislr_path, gloss = args.cislr_video, "(direct path)"
    elif args.cislr_video_id:
        cislr_path, gloss = resolve_cislr_video_path(args.cislr_video_id, args.metadata, args.videos_root)
    else:
        print("No --cislr-video or --cislr-video-id given; skipping CISLR pass.")
        cislr_path, gloss = None, None

    if cislr_path:
        print(f"\n=== CISLR video: {cislr_path} (gloss={gloss!r}) ===")
        frames = _read_all_frames(cislr_path)
        print(f"Total frames in source video: {len(frames)}")

        unmirrored_panels, unmirrored_records = run_diagnostic_pass(
            frames, mirror=False, n_panels=args.panels,
            pass_label="CISLR UNMIRRORED (matches preprocess_public_dataset.py)")
        mirrored_panels, mirrored_records = run_diagnostic_pass(
            frames, mirror=True, n_panels=args.panels,
            pass_label="CISLR MIRRORED (comparison only)")

        print_records_table("CISLR - UNMIRRORED pass (actual preprocessing pipeline behavior)", unmirrored_records)
        print_records_table("CISLR - MIRRORED pass (comparison: what main.py's convention would report)", mirrored_records)

        grid_unmirrored = make_grid(unmirrored_panels)
        grid_mirrored = make_grid(mirrored_panels)
        path_um = os.path.join(args.output_dir, "cislr_unmirrored.png")
        path_m = os.path.join(args.output_dir, "cislr_mirrored.png")
        cv2.imwrite(path_um, grid_unmirrored)
        cv2.imwrite(path_m, grid_mirrored)
        print(f"\nSaved: {path_um}")
        print(f"Saved: {path_m}")

    # ---- Webcam: real live capture, both passes for comparison; UNMIRRORED = actual main.py behavior ----
    if not args.skip_webcam:
        print(f"\n=== Live webcam capture ({args.webcam_seconds}s) ===")
        try:
            webcam_frames = capture_webcam_clip(args.webcam_seconds)
        except CameraError as e:
            print(f"Camera unavailable: {e}")
            webcam_frames = []

        if webcam_frames:
            print(f"Captured {len(webcam_frames)} raw frames.")
            wc_unmirrored_panels, wc_unmirrored_records = run_diagnostic_pass(
                webcam_frames, mirror=False, n_panels=args.panels,
                pass_label="WEBCAM UNMIRRORED (matches main.py)")
            wc_mirrored_panels, wc_mirrored_records = run_diagnostic_pass(
                webcam_frames, mirror=True, n_panels=args.panels,
                pass_label="WEBCAM MIRRORED (regression check only -- should NOT match main.py)")

            print_records_table("WEBCAM - UNMIRRORED pass (actual main.py pipeline behavior)", wc_unmirrored_records)
            print_records_table("WEBCAM - MIRRORED pass (regression check: this is what main.py would report if the flip were reintroduced)", wc_mirrored_records)

            path_wm = os.path.join(args.output_dir, "webcam_mirrored.png")
            path_wu = os.path.join(args.output_dir, "webcam_unmirrored.png")
            cv2.imwrite(path_wm, make_grid(wc_mirrored_panels))
            cv2.imwrite(path_wu, make_grid(wc_unmirrored_panels))
            print(f"\nSaved: {path_wm}")
            print(f"Saved: {path_wu}")

            any_hand = any(r["hands"] for r in wc_mirrored_records + wc_unmirrored_records)
            if not any_hand:
                print("\nNOTE: no hand was detected in the live capture window "
                      "(nobody in frame with a hand raised). Webcam-side evidence "
                      "is inconclusive from this capture; see report for what the "
                      "CISLR experiment + code inspection alone already establish.")
        else:
            print("No frames captured (camera unavailable or no frames read).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
