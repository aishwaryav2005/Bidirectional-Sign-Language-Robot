"""
Live Sign -> Speech demo.

    Webcam -> MediaPipe -> 147-D feature extraction -> 60-frame temporal
    buffer -> sign_demo classifier -> softmax probabilities -> temporal
    smoothing -> confidence threshold -> stable prediction -> English text
    -> TTS (pyttsx3, offline, no internet required).

Reuses the existing, unmodified feature pipeline (src/feature_extractor.py,
src/landmark_detector.py) and loads a model trained by src/train_sign_demo.py
(models/sign_demo/best_model_<cell>.keras) -- completely isolated from the
CISLR/INCLUDE/iSign/fingerspelling branches; does not modify or read any of
their model files.

Live prediction logic (per the task spec):
  - Collect a rolling 60-frame buffer; predict once the buffer is full.
  - Maintain the last N window predictions; require the SAME class to be
    the argmax for `stable_windows_required` consecutive windows before
    treating it as a stable, speakable prediction (reduces flicker from
    single noisy windows -- a lightweight alternative to probability
    averaging that also works, which is implemented as a config option).
  - Only speak when confidence >= `confidence_threshold` (default 0.80).
  - Never re-speak the same word while on cooldown (default 2.5s) to avoid
    repeatedly announcing a held sign every frame.
  - Below threshold: display "Sign not recognized. Please try again." and
    stay silent.

Usage:
    python src/demo_sign_to_speech.py --cell bilstm
    python src/demo_sign_to_speech.py --cell bilstm --no-tts   # text-only, for headless/no-speaker testing
    python src/demo_sign_to_speech.py --cell bilstm --camera-index 0
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector, draw_hand_landmarks, draw_pose_landmarks

MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models", "sign_demo")


def load_demo_model(cell):
    import keras
    suffix = f"_{cell}"
    model_path = os.path.join(MODEL_DIR, f"best_model{suffix}.keras")
    class_mapping_path = os.path.join(MODEL_DIR, f"class_mapping{suffix}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found -- train it first with "
            f"`python src/train_sign_demo.py --cell {cell}`"
        )
    model = keras.models.load_model(model_path)
    with open(class_mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}
    return model, id_to_label


class TTSEngine:
    """Thin wrapper so the demo degrades gracefully to text-only if pyttsx3
    or a speech backend isn't available (e.g. no audio device)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._engine = None
        if not enabled:
            return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 170)
        except Exception as e:
            print(f"[TTS] pyttsx3 unavailable ({e}) -- falling back to text-only output.")
            self.enabled = False

    def speak(self, text):
        print(f"\n>>> SPEAKING: \"{text}\"\n")
        if not self.enabled or self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[TTS] speak failed non-fatally: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["bilstm", "gru"], default="bilstm")
    parser.add_argument("--camera-index", type=int, default=None, help="None = auto-detect, matching config.MAX_CAMERA_INDEX_TO_PROBE")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--stable-windows-required", type=int, default=2,
                         help="number of consecutive 60-frame windows that must agree before announcing")
    parser.add_argument("--cooldown-seconds", type=float, default=2.5)
    parser.add_argument("--window-stride", type=int, default=10,
                         help="predict every N new frames (sliding window), not only every full 60")
    parser.add_argument("--no-tts", action="store_true", help="text-only mode, no audio")
    parser.add_argument("--no-display", action="store_true", help="headless: print to console only, no cv2 window")
    args = parser.parse_args()

    model, id_to_label = load_demo_model(args.cell)
    num_classes = len(id_to_label)
    print(f"Loaded sign_demo model ({args.cell}), {num_classes} classes: {sorted(id_to_label.values())}")

    tts = TTSEngine(enabled=not args.no_tts)

    cap = None
    indices_to_try = [args.camera_index] if args.camera_index is not None else list(range(config.MAX_CAMERA_INDEX_TO_PROBE))
    for idx in indices_to_try:
        candidate = cv2.VideoCapture(idx)
        if candidate.isOpened():
            cap = candidate
            print(f"Opened camera index {idx}.")
            break
        candidate.release()
    if cap is None:
        print("ERROR: no working camera found. Checked indices:", indices_to_try)
        print("If running headlessly (no physical webcam attached), this demo cannot proceed past this point --")
        print("re-run on a machine with a webcam, or use --camera-index to point at a specific device.")
        return

    detector = LandmarkDetector()
    frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    recent_predictions = deque(maxlen=args.stable_windows_required)
    last_spoken_label = None
    last_spoken_time = 0.0
    frames_since_last_predict = 0
    next_ts = 0
    window_name = "Sign -> Speech Demo"

    print("=" * 60)
    print("Sign -> Speech DEMO running. Press 'q' to quit.")
    print(f"Confidence threshold: {args.confidence_threshold}  Stable windows required: {args.stable_windows_required}  Cooldown: {args.cooldown_seconds}s")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            result = detector.detect(frame, next_ts)
            next_ts += 1
            feature_vec, info = extract_features(result["hand_result"], result["pose_result"])
            frame_buffer.append(feature_vec)
            frames_since_last_predict += 1

            display_text = "Collecting frames..."
            confidence_text = ""

            if len(frame_buffer) == config.SEQUENCE_LENGTH and frames_since_last_predict >= args.window_stride:
                frames_since_last_predict = 0
                sequence = np.stack(frame_buffer, axis=0)[np.newaxis, ...].astype(np.float32)
                probs = model.predict(sequence, verbose=0)[0]
                pred_id = int(np.argmax(probs))
                pred_label = id_to_label[pred_id]
                pred_conf = float(probs[pred_id])

                recent_predictions.append((pred_label, pred_conf))
                display_text = f"SIGN: {pred_label.upper()}"
                confidence_text = f"CONFIDENCE: {pred_conf * 100:.1f}%"

                stable = (
                    len(recent_predictions) == args.stable_windows_required
                    and len({p[0] for p in recent_predictions}) == 1
                    and pred_conf >= args.confidence_threshold
                )

                if pred_conf < args.confidence_threshold:
                    display_text = "Sign not recognized. Please try again."
                    confidence_text = f"(best guess: {pred_label}, {pred_conf * 100:.1f}%)"
                elif stable:
                    now = time.perf_counter()
                    cooldown_elapsed = (now - last_spoken_time) >= args.cooldown_seconds
                    if pred_label != last_spoken_label or cooldown_elapsed:
                        tts.speak(pred_label.replace("_", " "))
                        last_spoken_label = pred_label
                        last_spoken_time = now
                else:
                    display_text += "  (Hold sign...)"

                print(f"{display_text}  {confidence_text}")

            if not args.no_display:
                frame = draw_hand_landmarks(frame, result["hand_result"])
                frame = draw_pose_landmarks(frame, result["pose_result"])
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, confidence_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                buffer_fill = f"Buffer: {len(frame_buffer)}/{config.SEQUENCE_LENGTH}"
                cv2.putText(frame, buffer_fill, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        detector.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Demo stopped.")


if __name__ == "__main__":
    main()
