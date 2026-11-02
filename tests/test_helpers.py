"""Shared test scaffolding: path bootstrap + lightweight fakes that mimic the
MediaPipe Tasks result objects (HandLandmarkerResult / PoseLandmarkerResult)
without requiring a real model inference call."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeCategory:
    def __init__(self, category_name, score=0.95):
        self.category_name = category_name
        self.score = score


class FakeHandResult:
    """Mimics mediapipe.tasks.python.vision.HandLandmarkerResult."""

    def __init__(self, hand_landmarks=None, handedness=None):
        self.hand_landmarks = hand_landmarks or []
        self.handedness = handedness or []


class FakePoseResult:
    """Mimics mediapipe.tasks.python.vision.PoseLandmarkerResult."""

    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = pose_landmarks or []


def make_hand_landmarks(seed=0.0):
    """21 fake landmarks with distinct, deterministic coordinates."""
    return [FakeLandmark(0.1 * i + seed, 0.05 * i + seed, 0.01 * i) for i in range(21)]


def make_pose_landmarks():
    """33 fake landmarks (full MediaPipe Pose topology), only indices used
    by config.POSE_LANDMARK_INDICES need realistic values."""
    landmarks = [FakeLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    landmarks[11] = FakeLandmark(0.3, 0.4, 0.0)   # left_shoulder
    landmarks[12] = FakeLandmark(0.7, 0.4, 0.0)   # right_shoulder
    landmarks[13] = FakeLandmark(0.2, 0.6, 0.0)   # left_elbow
    landmarks[14] = FakeLandmark(0.8, 0.6, 0.0)   # right_elbow
    landmarks[15] = FakeLandmark(0.15, 0.8, 0.0)  # left_wrist
    landmarks[16] = FakeLandmark(0.85, 0.8, 0.0)  # right_wrist
    return landmarks


def make_two_hand_result():
    left = make_hand_landmarks(seed=0.1)
    right = make_hand_landmarks(seed=0.5)
    return FakeHandResult(
        hand_landmarks=[left, right],
        handedness=[[FakeCategory("Left")], [FakeCategory("Right")]],
    )


def make_left_only_hand_result():
    left = make_hand_landmarks(seed=0.1)
    return FakeHandResult(
        hand_landmarks=[left],
        handedness=[[FakeCategory("Left")]],
    )


def make_no_hand_result():
    return FakeHandResult()


def make_pose_result():
    return FakePoseResult(pose_landmarks=[make_pose_landmarks()])


def make_no_pose_result():
    return FakePoseResult()
