"""
Phase 1 entry point: live perception + data-collection loop.

CAMERA -> HAND+POSE DETECTION -> LANDMARK EXTRACTION -> FEATURE VECTOR
       -> SEQUENCE_LENGTH-FRAME TEMPORAL SEQUENCE -> SAVED DATASET

Controls:
    L  - set/change the current label (typed in the terminal)
    R  - start a COUNTDOWN_SECONDS countdown, then record a
         SEQUENCE_LENGTH-frame sample for the current label
    C  - cancel an in-progress countdown or recording
    Q  - quit

Recording flow:
    Press R -> 3, 2, 1 countdown -> RECORDING (Frame 1/N ... N/N) -> saved
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from camera import Camera, CameraError
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector, draw_hand_landmarks, draw_pose_landmarks
from sequence_collector import SequenceCollector


def _prompt_for_label(current_label):
    print()
    new_label = input(f"Enter sign label [{current_label or 'none'}]: ").strip()
    if not new_label:
        return current_label
    return new_label.upper().replace(" ", "_")


def _draw_hud(frame, fps, detection_info, collector, label, countdown_seconds_left):
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Label: {label or '(none - press L)'}",
        f"Left hand detected: {'YES' if detection_info['left_hand'] else 'NO'}",
        f"Right hand detected: {'YES' if detection_info['right_hand'] else 'NO'}",
        f"Pose detected: {'YES' if detection_info['pose'] else 'NO'}",
    ]
    if countdown_seconds_left is not None:
        lines.append(f"Get ready... starting in {countdown_seconds_left}")
    elif collector.is_recording:
        lines.append(f"RECORDING  Frame: {collector.current_frame_count}/{collector.sequence_length}")
    else:
        lines.append("Status: idle (press R to record)")

    y = 25
    for line in lines:
        color = (0, 0, 255) if line.startswith("RECORDING") else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 24

    if collector.is_recording:
        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)

    if countdown_seconds_left is not None:
        _draw_countdown_overlay(frame, countdown_seconds_left)

    return frame


def _draw_countdown_overlay(frame, seconds_left):
    """Big centered countdown number (3, 2, 1) shown before recording starts."""
    h, w = frame.shape[:2]
    text = str(seconds_left)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 6
    thickness = 8
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)


def main():
    try:
        cam = Camera()
    except CameraError as e:
        print(f"[main.py] FATAL: {e}")
        return 1

    try:
        detector = LandmarkDetector()
    except Exception as e:
        print(f"[main.py] FATAL: could not initialize MediaPipe landmarkers: {e}")
        cam.release()
        return 1

    collector = SequenceCollector()
    resolution = cam.get_resolution()
    label = None

    print("=" * 60)
    print("Sign Language Robot - Phase 1 Perception Pipeline")
    print("=" * 60)
    print(f"Camera resolution: {resolution}")
    print(f"Sequence length: {config.SEQUENCE_LENGTH} frames")
    print(f"Countdown before recording: {config.COUNTDOWN_SECONDS} seconds")
    print(f"Feature vector length: {config.FEATURE_VECTOR_LENGTH}")
    print()
    print("Controls: [L] set label   [R] record sample (after countdown)   [C] cancel   [Q] quit")
    print("=" * 60)

    prev_time = time.time()
    fps = 0.0
    frame_index = 0
    last_detection_info = {"left_hand": False, "right_hand": False, "pose": False}

    countdown_active = False
    countdown_start_time = None

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[main.py] WARNING: failed to read frame from camera, stopping.")
                break

            now = time.time()
            # Intentionally NOT mirrored (no cv2.flip). MediaPipe's handedness
            # assumes mirrored/selfie input; empirically verified (see
            # src/diagnose_handedness.py and DATASET_INFO.md) that raw,
            # unmirrored frames give anatomically correct Left/Right labels
            # for this camera, matching CISLR's unmirrored convention.

            timestamp_ms = int(now * 1000)
            result = detector.detect(frame, timestamp_ms)
            feature_vector, detection_info = extract_features(
                result["hand_result"], result["pose_result"]
            )
            last_detection_info = detection_info

            display = draw_hand_landmarks(frame, result["hand_result"])
            display = draw_pose_landmarks(display, result["pose_result"])

            countdown_seconds_left = None
            if countdown_active:
                elapsed = now - countdown_start_time
                remaining = config.COUNTDOWN_SECONDS - int(elapsed)
                if remaining <= 0:
                    countdown_active = False
                    collector.start_recording()
                    print(f"[main.py] Recording sample for label '{label}'...")
                else:
                    countdown_seconds_left = remaining

            if collector.is_recording:
                just_completed = collector.add_frame(feature_vector, detection_info)
                if just_completed:
                    if label:
                        saved_path = collector.save_sequence(label, camera_resolution=resolution)
                        print(f"[main.py] Sample saved successfully: {saved_path}")
                    else:
                        print("[main.py] WARNING: no label set, discarding recorded sequence.")
                        collector.cancel_recording()

            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            display = _draw_hud(display, fps, last_detection_info, collector, label, countdown_seconds_left)
            cv2.imshow(config.WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l") and not collector.is_recording and not countdown_active:
                label = _prompt_for_label(label)
            elif key == ord("r"):
                if not label:
                    print("[main.py] Set a label first (press L).")
                elif collector.is_recording:
                    print("[main.py] Already recording.")
                elif countdown_active:
                    print("[main.py] Already counting down.")
                else:
                    countdown_active = True
                    countdown_start_time = now
                    print(f"[main.py] Get ready! Recording starts in {config.COUNTDOWN_SECONDS}...")
            elif key == ord("c") and (collector.is_recording or countdown_active):
                if collector.is_recording:
                    collector.cancel_recording()
                countdown_active = False
                print("[main.py] Recording cancelled.")

            frame_index += 1
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

    print("[main.py] Shutdown complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
