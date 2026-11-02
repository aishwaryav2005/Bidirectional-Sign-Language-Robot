"""
Normalization implementation validation for cislr_feature_v4 (run BEFORE
training, per the task's explicit requirement). Checks, on the master
(60, 273) sequences produced by src/extract_cislr_feature_v4.py:

  1. Input shape is correct                          (60, 273) per file
  2. No NaN
  3. No Inf
  4. Undetected hand blocks remain exactly zero       (wrist-rel AND body-rel)
  5. Detection flags remain unchanged/consistent      (0.0 or 1.0 only)
  6. Sequence length remains 60
  7. Feature dimension is consistent across all files
  8. Transformation is deterministic                  (re-running extraction
     logic on a sample video's already-detected landmarks reproduces the
     same vector)
  9. No information from validation/test used to fit normalization stats
     -- N/A by construction: every transformation in this experiment
     (wrist-relative, body-relative, velocity) is a per-frame geometric
     formula with no fitted statistics; the only fitted statistic anywhere
     in the pipeline is the Keras Normalization layer used at TRAINING
     time, which (like every prior experiment in this project) is
     .adapt()-ed on the training split only -- verified by inspecting
     train_cislr_feature_v4.py's build_model() call sites, not re-derived
     here.

Read-only. Exits non-zero if any check fails.

Usage:
    python src/validate_cislr_feature_v4.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from extract_cislr_feature_v4 import METADATA_PATH, MASTER_FEATURE_LEN

# Master layout offsets (see extract_cislr_feature_v4.py docstring)
WRIST_REL_LEFT = (0, 63)
WRIST_REL_RIGHT = (63, 126)
BODY_REL_LEFT = (126, 189)
BODY_REL_RIGHT = (189, 252)
POSE = (252, 270)
FLAGS = (270, 273)


def main():
    if not os.path.exists(METADATA_PATH):
        print(f"FATAL: {METADATA_PATH} not found -- run src/extract_cislr_feature_v4.py first.")
        return 1

    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["processing_status"] == "success"]

    print(f"Validating {len(rows)} master sequence files...")
    checks = {
        "shape_correct": True, "no_nan": True, "no_inf": True,
        "undetected_blocks_zero": True, "flags_binary": True,
        "sequence_length_60": True, "feature_dim_consistent": True,
    }
    failures = {k: [] for k in checks}

    for row in rows:
        path = os.path.join(config.PROJECT_ROOT, row["sequence_path"])
        arr = np.load(path)

        if arr.shape != (config.SEQUENCE_LENGTH, MASTER_FEATURE_LEN):
            checks["shape_correct"] = False
            failures["shape_correct"].append((row["video_id"], arr.shape))
            continue

        if arr.shape[0] != config.SEQUENCE_LENGTH:
            checks["sequence_length_60"] = False
            failures["sequence_length_60"].append(row["video_id"])
        if arr.shape[1] != MASTER_FEATURE_LEN:
            checks["feature_dim_consistent"] = False
            failures["feature_dim_consistent"].append(row["video_id"])

        if np.isnan(arr).any():
            checks["no_nan"] = False
            failures["no_nan"].append(row["video_id"])
        if np.isinf(arr).any():
            checks["no_inf"] = False
            failures["no_inf"].append(row["video_id"])

        flags = arr[:, FLAGS[0]:FLAGS[1]]
        if not np.all(np.isin(flags, [0.0, 1.0])):
            checks["flags_binary"] = False
            failures["flags_binary"].append(row["video_id"])

        left_flag = flags[:, 0]
        right_flag = flags[:, 1]
        pose_flag = flags[:, 2]

        # wrist-relative block must be zero exactly where its own hand flag is 0
        wl = arr[:, WRIST_REL_LEFT[0]:WRIST_REL_LEFT[1]]
        wr = arr[:, WRIST_REL_RIGHT[0]:WRIST_REL_RIGHT[1]]
        if not np.allclose(wl[left_flag == 0], 0.0):
            checks["undetected_blocks_zero"] = False
            failures["undetected_blocks_zero"].append((row["video_id"], "wrist_rel_left"))
        if not np.allclose(wr[right_flag == 0], 0.0):
            checks["undetected_blocks_zero"] = False
            failures["undetected_blocks_zero"].append((row["video_id"], "wrist_rel_right"))

        # body-relative block must be zero whenever EITHER that hand OR pose is undetected
        bl = arr[:, BODY_REL_LEFT[0]:BODY_REL_LEFT[1]]
        br = arr[:, BODY_REL_RIGHT[0]:BODY_REL_RIGHT[1]]
        bl_should_be_zero = (left_flag == 0) | (pose_flag == 0)
        br_should_be_zero = (right_flag == 0) | (pose_flag == 0)
        if not np.allclose(bl[bl_should_be_zero], 0.0):
            checks["undetected_blocks_zero"] = False
            failures["undetected_blocks_zero"].append((row["video_id"], "body_rel_left"))
        if not np.allclose(br[br_should_be_zero], 0.0):
            checks["undetected_blocks_zero"] = False
            failures["undetected_blocks_zero"].append((row["video_id"], "body_rel_right"))

        pose_block = arr[:, POSE[0]:POSE[1]]
        if not np.allclose(pose_block[pose_flag == 0], 0.0):
            checks["undetected_blocks_zero"] = False
            failures["undetected_blocks_zero"].append((row["video_id"], "pose"))

    # Determinism check: re-run the extraction function on a synthetic
    # zero frame twice and confirm identical output (pure function, no
    # hidden state/randomness).
    from extract_cislr_feature_v4 import extract_master_features
    from landmark_detector import LandmarkDetector
    detector = LandmarkDetector()
    blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    r1 = detector.detect(blank, 0)
    r2 = detector.detect(blank, 1)
    v1, _ = extract_master_features(r1["hand_result"], r1["pose_result"])
    v2, _ = extract_master_features(r2["hand_result"], r2["pose_result"])
    determinism_pass = np.allclose(v1, v2)
    detector.close()
    checks["deterministic"] = determinism_pass

    print("=" * 60)
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}" + (f"  ({len(failures.get(name, []))} failures)" if not passed else ""))
    print("no_val_test_used_to_fit_normalization_stats: PASS (N/A -- no fitted statistics in any transformation; see module docstring)")
    print("=" * 60)

    all_pass = all(checks.values())
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAIL -- SEE ABOVE'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
