"""
Phase 9: real-time webcam inference using the trained ISL combined model
(models/isl_combined/selected_model/, BiLSTM, 98 classes -- CISLR's 82
existing glosses + 16 new INCLUDE glosses; see
reports/combined_dataset_training_report.md for how it was selected).

RGB webcam -> MediaPipe (existing LandmarkDetector) -> 147-D features
(existing extract_features) -> 60-frame rolling buffer -> trained model
-> predicted gloss -> confidence threshold -> temporal smoothing (majority
vote over the last few predictions) -> displayed text.

Below the confidence threshold, or without majority agreement, displays
UNKNOWN rather than forcing a class.

Reuses config.py, feature_extractor.py, landmark_detector.py, camera.py
unmodified. Does not touch any CISLR-only model or the webcam recording
pipeline in main.py.

Usage:
    python src/realtime_inference.py --confidence-threshold 0.5
"""

import argparse
import collections
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from camera import Camera, CameraError
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector, draw_hand_landmarks, draw_pose_landmarks

MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models", "isl_combined", "selected_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CLASS_MAPPING_PATH = os.path.join(config.PROJECT_ROOT, "models", "isl_combined", "class_mapping.json")

UNKNOWN_LABEL = "UNKNOWN"


def load_class_names():
    with open(CLASS_MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    id_to_label = mapping["class_id_to_label"]
    num_classes = mapping["num_classes"]
    return [id_to_label[str(i)] for i in range(num_classes)]


class PredictionSmoother:
    """Majority vote over the last `window` predictions (including
    UNKNOWN, which counts as its own outcome -- a run of low-confidence
    frames correctly displays UNKNOWN rather than an averaged-away guess)."""

    def __init__(self, window=8, min_agreement=0.5):
        self.window = window
        self.min_agreement = min_agreement
        self.history = collections.deque(maxlen=window)

    def update(self, label):
        self.history.append(label)
        if not self.history:
            return UNKNOWN_LABEL
        counts = collections.Counter(self.history)
        top_label, top_count = counts.most_common(1)[0]
        if top_count / len(self.history) >= self.min_agreement:
            return top_label
        return UNKNOWN_LABEL


def _draw_hud(frame, fps, buffer_fill, buffer_size, predicted_label, confidence, detection_info):
    lines = [
        f"FPS: {fps:.1f}",
        f"Buffer: {buffer_fill}/{buffer_size}",
        f"Left hand: {'YES' if detection_info['left_hand'] else 'NO'}  Right hand: {'YES' if detection_info['right_hand'] else 'NO'}  Pose: {'YES' if detection_info['pose'] else 'NO'}",
        f"Prediction: {predicted_label}  (confidence: {confidence:.2f})",
    ]
    y = 25
    for line in lines:
        color = (0, 255, 0) if line.startswith("Prediction") and predicted_label != UNKNOWN_LABEL else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 24
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time ISL sign recognition using the trained combined model.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum softmax probability to accept a prediction (else UNKNOWN)")
    parser.add_argument("--smoothing-window", type=int, default=8, help="Number of recent predictions to majority-vote over")
    parser.add_argument("--predict-every", type=int, default=5, help="Run inference every N frames (buffer still updates every frame)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[realtime_inference.py] FATAL: model not found at {MODEL_PATH}. Run src/train_isl_combined.py first.")
        return 1

    import keras
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    print(f"Loaded model with {len(class_names)} classes from {MODEL_PATH}")

    try:
        cam = Camera()
    except CameraError as e:
        print(f"[realtime_inference.py] FATAL: {e}")
        return 1

    try:
        detector = LandmarkDetector()
    except Exception as e:
        print(f"[realtime_inference.py] FATAL: could not initialize MediaPipe landmarkers: {e}")
        cam.release()
        return 1

    seq_len = config.SEQUENCE_LENGTH
    feat_dim = config.FEATURE_VECTOR_LENGTH
    frame_buffer = collections.deque(maxlen=seq_len)
    smoother = PredictionSmoother(window=args.smoothing_window)

    predicted_label = UNKNOWN_LABEL
    confidence = 0.0
    prev_time = time.time()
    fps = 0.0
    frame_index = 0

    print("=" * 60)
    print("ISL Combined Model - Real-Time Inference")
    print("=" * 60)
    print(f"Classes: {len(class_names)}  Confidence threshold: {args.confidence_threshold}")
    print("Press Q to quit.")
    print("=" * 60)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[realtime_inference.py] WARNING: failed to read frame from camera, stopping.")
                break

            now = time.time()
            timestamp_ms = int(now * 1000)
            result = detector.detect(frame, timestamp_ms)
            feature_vector, detection_info = extract_features(result["hand_result"], result["pose_result"])
            frame_buffer.append(feature_vector)

            if len(frame_buffer) == seq_len and frame_index % args.predict_every == 0:
                sequence = np.stack(frame_buffer, axis=0).astype(np.float32)[None, ...]  # (1, 60, 147)
                probs = model.predict(sequence, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                raw_label = class_names[top_idx] if confidence >= args.confidence_threshold else UNKNOWN_LABEL
                predicted_label = smoother.update(raw_label)

            display = draw_hand_landmarks(frame, result["hand_result"])
            display = draw_pose_landmarks(display, result["pose_result"])

            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            display = _draw_hud(display, fps, len(frame_buffer), seq_len, predicted_label, confidence, detection_info)
            cv2.imshow("ISL Combined Model - Real-Time Inference", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_index += 1
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

    print("[realtime_inference.py] Shutdown complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
