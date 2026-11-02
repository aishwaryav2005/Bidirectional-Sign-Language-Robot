"""
Hand + pose landmark detection using the MediaPipe Tasks API.

NOTE: the installed mediapipe version (1.0.1) removed the old
`mp.solutions.hands` / `mp.solutions.pose` API that most tutorials use.
This module targets the current Tasks API (HandLandmarker / PoseLandmarker)
instead.

Two separate task models are used (rather than the combined
HolisticLandmarker) so hand and pose detection can be reasoned about,
tuned, and disabled independently, and because HandLandmarker gives an
explicit left/right handedness classification per hand.
"""

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LandmarkDetector:
    """Runs MediaPipe HandLandmarker + PoseLandmarker over a video stream.

    Usage: call `detect(frame_bgr, timestamp_ms)` once per frame, in
    increasing timestamp order (required by VIDEO running mode).
    """

    def __init__(
        self,
        hand_model_path=config.HAND_LANDMARKER_MODEL_PATH,
        pose_model_path=config.POSE_LANDMARKER_MODEL_PATH,
    ):
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.HAND_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    def detect(self, frame_bgr, timestamp_ms):
        """Run both landmarkers on one BGR frame.

        Returns a dict:
            {
                "hand_result": HandLandmarkerResult,
                "pose_result": PoseLandmarkerResult,
            }
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        return {"hand_result": hand_result, "pose_result": pose_result}

    def close(self):
        self._hand_landmarker.close()
        self._pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# Drawing helpers (kept here since they operate directly on Tasks API output)
# ---------------------------------------------------------------------------

# HandLandmarker connections mirror the classic MediaPipe hand topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

POSE_CONNECTIONS = [
    (11, 12),  # shoulder to shoulder
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
]


def draw_hand_landmarks(frame_bgr, hand_result):
    h, w = frame_bgr.shape[:2]
    if not hand_result.hand_landmarks:
        return frame_bgr

    for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (0, 255, 0), 2)
        for p in points:
            cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)

        label = handedness[0].category_name if handedness else "?"
        score = handedness[0].score if handedness else 0.0
        wrist = points[0]
        cv2.putText(
            frame_bgr,
            f"{label} ({score:.2f})",
            (wrist[0] - 20, wrist[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    return frame_bgr


def draw_pose_landmarks(frame_bgr, pose_result):
    h, w = frame_bgr.shape[:2]
    if not pose_result.pose_landmarks:
        return frame_bgr

    landmarks = pose_result.pose_landmarks[0]  # single person
    used_indices = set(config.POSE_LANDMARK_INDICES.values())
    points = {}
    for idx in used_indices:
        lm = landmarks[idx]
        points[idx] = (int(lm.x * w), int(lm.y * h))

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame_bgr, points[start], points[end], (255, 165, 0), 2)
    for idx, p in points.items():
        cv2.circle(frame_bgr, p, 4, (0, 165, 255), -1)

    return frame_bgr


if __name__ == "__main__":
    # Standalone smoke test: run detector on a few synthetic-free live frames
    # and report whether hands/pose are detected, without opening a GUI window.
    import numpy as np

    detector = LandmarkDetector()
    blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    result = detector.detect(blank, 0)
    print("Detector initialized OK.")
    print("Hand result on blank frame:", result["hand_result"].hand_landmarks)
    print("Pose result on blank frame:", result["pose_result"].pose_landmarks)
    detector.close()
    print("PASS")
