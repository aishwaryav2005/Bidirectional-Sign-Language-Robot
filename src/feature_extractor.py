"""
Converts raw MediaPipe HandLandmarker/PoseLandmarker results into a single
fixed-size NumPy feature vector per frame.

Feature vector layout (config.FEATURE_VECTOR_LENGTH = 147 values):

    [ left_hand (63) | right_hand (63) | pose (18) | detection_flags (3) ]

    left_hand / right_hand : 21 landmarks * (x, y, z), wrist-relative and
                              scale-normalized (see _normalize_hand below).
                              All zeros if that hand was not detected.
    pose                   : 6 upper-body landmarks (shoulders, elbows,
                              wrists) * (x, y, z), normalized relative to
                              the shoulder midpoint and shoulder width.
                              All zeros if pose was not detected.
    detection_flags        : [left_hand_detected, right_hand_detected,
                               pose_detected] as 1.0 / 0.0.

Normalization (documented in detail in README.md):
  - Each hand is translated so its wrist (landmark 0) sits at the origin,
    then scaled by the distance from the wrist to the middle-finger MCP
    (landmark 9). This makes the hand shape roughly invariant to the
    signer's distance from the camera and position in frame.
  - Pose landmarks are translated so the shoulder midpoint sits at the
    origin, then scaled by the shoulder-to-shoulder distance, for the
    same reason.
  - If the scale reference distance is ~0 (degenerate detection), a scale
    of 1.0 is used instead of dividing by zero.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_EPS = 1e-6


def _landmarks_to_array(landmarks):
    """List of MediaPipe NormalizedLandmark -> (N, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def _normalize_hand(points_xyz):
    """points_xyz: (21, 3) array. Wrist-relative, scale-normalized."""
    wrist = points_xyz[0].copy()
    centered = points_xyz - wrist
    scale = np.linalg.norm(centered[9])  # distance wrist -> middle MCP
    if scale < _EPS:
        scale = 1.0
    return centered / scale


def _normalize_pose(points_xyz):
    """points_xyz: (6, 3) array in POSE_LANDMARK_NAMES order
    (left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
    right_wrist). Shoulder-midpoint-relative, shoulder-width-scaled."""
    left_shoulder = points_xyz[0]
    right_shoulder = points_xyz[1]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    centered = points_xyz - mid_shoulder
    scale = np.linalg.norm(right_shoulder - left_shoulder)
    if scale < _EPS:
        scale = 1.0
    return centered / scale


def extract_features(hand_result, pose_result):
    """Build the fixed-size feature vector for one frame.

    Args:
        hand_result: mediapipe HandLandmarkerResult
        pose_result: mediapipe PoseLandmarkerResult

    Returns:
        (feature_vector, detection_info) where:
          - feature_vector is a float32 NumPy array of length
            config.FEATURE_VECTOR_LENGTH
          - detection_info is a dict {"left_hand": bool, "right_hand": bool,
            "pose": bool}
    """
    left_hand_vec = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    right_hand_vec = np.zeros(config.HAND_FEATURE_LEN, dtype=np.float32)
    left_detected = False
    right_detected = False

    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            points = _landmarks_to_array(landmarks)
            normalized = _normalize_hand(points)
            flat = normalized.flatten()

            # category_name is used as-is (no swap): both the webcam
            # pipeline (main.py, unmirrored frames) and the CISLR
            # preprocessing pipeline feed MediaPipe raw, unmirrored frames,
            # which was empirically verified to give anatomically correct
            # Left/Right labels for both (see src/diagnose_handedness.py
            # and DATASET_INFO.md) -- so this same convention is correct
            # for either caller.
            label = handedness[0].category_name if handedness else None
            if label == "Left":
                left_hand_vec = flat
                left_detected = True
            elif label == "Right":
                right_hand_vec = flat
                right_detected = True

    pose_vec = np.zeros(config.POSE_FEATURE_LEN, dtype=np.float32)
    pose_detected = False

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]  # single signer assumed
        indices = [config.POSE_LANDMARK_INDICES[name] for name in config.POSE_LANDMARK_NAMES]
        selected = [landmarks[i] for i in indices]
        points = _landmarks_to_array(selected)
        normalized = _normalize_pose(points)
        pose_vec = normalized.flatten()
        pose_detected = True

    flags = np.array(
        [float(left_detected), float(right_detected), float(pose_detected)],
        dtype=np.float32,
    )

    feature_vector = np.concatenate([left_hand_vec, right_hand_vec, pose_vec, flags])
    assert feature_vector.shape[0] == config.FEATURE_VECTOR_LENGTH, (
        f"Feature vector length mismatch: got {feature_vector.shape[0]}, "
        f"expected {config.FEATURE_VECTOR_LENGTH}"
    )

    detection_info = {
        "left_hand": left_detected,
        "right_hand": right_detected,
        "pose": pose_detected,
    }
    return feature_vector, detection_info


if __name__ == "__main__":
    # Smoke test with a fake result-like structure would require constructing
    # MediaPipe result objects, which are not trivial to build by hand.
    # Instead we test the pure-numpy normalization helpers directly.
    fake_hand = np.random.rand(21, 3).astype(np.float32)
    normalized = _normalize_hand(fake_hand)
    assert normalized.shape == (21, 3)
    assert np.allclose(normalized[0], 0.0), "wrist should be at origin after normalization"

    fake_pose = np.random.rand(6, 3).astype(np.float32)
    normalized_pose = _normalize_pose(fake_pose)
    assert normalized_pose.shape == (6, 3)

    print("Normalization helpers OK.")
    print("Expected feature vector length:", config.FEATURE_VECTOR_LENGTH)
    print("PASS")
