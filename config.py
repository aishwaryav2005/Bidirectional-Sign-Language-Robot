"""
Central configuration for the Sign Language Robot - Phase 1 perception pipeline.

Every tunable knob (camera, model paths, sequence length, feature layout)
lives here so the rest of the codebase never hardcodes these values.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HAND_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
METADATA_PATH = os.path.join(SEQUENCES_DIR, "metadata.csv")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Phase 2: public-dataset preprocessing (see DATASET_INFO.md)
PUBLIC_RAW_DIR = os.path.join(DATA_DIR, "public_raw")
PUBLIC_PROCESSED_DIR = os.path.join(DATA_DIR, "public_processed")
PUBLIC_SEQUENCES_DIR = os.path.join(PUBLIC_PROCESSED_DIR, "sequences")
PUBLIC_METADATA_DIR = os.path.join(PUBLIC_PROCESSED_DIR, "metadata")
PUBLIC_METADATA_PATH = os.path.join(PUBLIC_METADATA_DIR, "processing_metadata.csv")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")
LABEL_MAPPING_PATH = os.path.join(MAPPINGS_DIR, "label_mapping.json")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = None          # None = auto-detect first working camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS_TARGET = 30
MAX_CAMERA_INDEX_TO_PROBE = 5  # how many device indices to try during auto-detect

# ---------------------------------------------------------------------------
# MediaPipe Tasks detection thresholds
# ---------------------------------------------------------------------------
NUM_HANDS = 2
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Upper-body pose landmarks relevant to signing (MediaPipe Pose indices).
# Full landmark list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
POSE_LANDMARK_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
POSE_LANDMARK_NAMES = list(POSE_LANDMARK_INDICES.keys())
NUM_POSE_LANDMARKS_USED = len(POSE_LANDMARK_INDICES)

# ---------------------------------------------------------------------------
# Feature vector layout
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z

HAND_FEATURE_LEN = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK          # 63
POSE_FEATURE_LEN = NUM_POSE_LANDMARKS_USED * COORDS_PER_LANDMARK     # 18

# Order: [left_hand(63), right_hand(63), pose(18), detection_flags(3)]
# detection_flags = [left_hand_detected, right_hand_detected, pose_detected]
NUM_DETECTION_FLAGS = 3
FEATURE_VECTOR_LENGTH = (
    HAND_FEATURE_LEN * 2 + POSE_FEATURE_LEN + NUM_DETECTION_FLAGS
)  # 63 + 63 + 18 + 3 = 147

# ---------------------------------------------------------------------------
# Temporal sequences
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 60  # frames per recorded sign sample
COUNTDOWN_SECONDS = 3  # on-screen countdown before recording starts

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_NAME = "Sign Language Robot - Phase 1 Perception"
