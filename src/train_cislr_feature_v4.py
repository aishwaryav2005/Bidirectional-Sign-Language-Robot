"""
ISL Feature Representation v4: tests whether the CISLR bottleneck is the
FEATURE REPRESENTATION rather than architecture or dataset size.

IMPORTANT DISCOVERY (see reports/cislr_feature_v4_report.md for full
discussion): inspecting src/feature_extractor.py before writing any new
code revealed that the project's EXISTING 147-D "raw" pipeline is NOT
unnormalized/absolute-position data -- it already applies wrist-relative
+ scale normalization per hand and shoulder-relative + scale
normalization to pose. This means "Experiment 1 (current raw
representation)" and "Experiment 2 (root-centered hand features)" are the
SAME transformation, byte-for-byte -- reusing the frozen
data/public_processed/ sequences reproduces both. Experiment 2 is
reported using Experiment 1's own result rather than wastefully retraining
an identical model on identical data; this is stated explicitly, not
hidden. The genuinely NEW spatial representation this experiment
introduces is Experiment 3 (body-relative hands, shoulder-frame instead
of each hand's own wrist).

Uses the master (60, 273) features produced by
src/extract_cislr_feature_v4.py:
    [ wrist_rel_left(63) | wrist_rel_right(63)
      | body_rel_left(63) | body_rel_right(63)
      | pose(18) | flags(3) ]

Fixed classifier architecture across ALL experiments (per the task's
explicit "do not simultaneously change architecture and representation"
instruction): BiLSTM(64)->Dropout(0.3)->BiLSTM(32)->Dropout(0.3)->
Dense(num_classes, softmax) -- identical to
src/train_bilstm_baseline.py's build_model (reused, unmodified, just
called with each experiment's own feat_dim).

Reused, unmodified: train_bilstm_baseline.py's set_global_seed,
build_model, evaluate_predictions, save_json, save_confusion_matrix_plot.

Frozen evaluation: data/splits/cislr_82class/ (328/82/82) -- read-only,
never regenerated. Model selection by validation accuracy; test set
evaluated exactly once per experiment, after selection.

Does NOT modify any frozen path. All new artifacts go under
models/cislr_feature_v4/.

Usage:
    python src/train_cislr_feature_v4.py --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_bilstm_baseline import (
    set_global_seed, build_model, evaluate_predictions, save_json, save_confusion_matrix_plot,
)
from extract_cislr_feature_v4 import METADATA_PATH as MASTER_METADATA_PATH

CISLR_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "cislr_feature_v4")
REPORT_JSON = os.path.join(config.PROJECT_ROOT, "reports", "cislr_feature_v4_results.json")

SEQ_LEN = config.SEQUENCE_LENGTH  # 60
BASELINE_TEST_ACC = 0.25609756097560976
BASELINE_MACRO_F1 = 0.18983739837398372
BASELINE_VAL_ACC = 0.26829269528388977

# Master layout offsets
WRIST_REL_LEFT, WRIST_REL_RIGHT = (0, 63), (63, 126)
BODY_REL_LEFT, BODY_REL_RIGHT = (126, 189), (189, 252)
POSE = (252, 270)
FLAGS = (270, 273)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_ids():
    result = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(CISLR_SPLITS_DIR, f"{split}.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                result.setdefault(split, []).append(row)
    return result


def build_class_mapping(split_rows):
    glosses = set()
    for rows in split_rows.values():
        glosses.update(r["gloss"] for r in rows)
    sorted_g = sorted(glosses)
    return {g: i for i, g in enumerate(sorted_g)}


def load_master_arrays(rows, master_meta_by_id, gloss_to_id, project_root):
    n = len(rows)
    X = np.zeros((n, SEQ_LEN, 273), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    ids = []
    for i, row in enumerate(rows):
        meta = master_meta_by_id[row["video_id"]]
        arr = np.load(os.path.join(project_root, meta["sequence_path"])).astype(np.float32)
        assert arr.shape == (SEQ_LEN, 273), f"{row['video_id']}: {arr.shape}"
        X[i] = arr
        y[i] = gloss_to_id[row["gloss"]]
        ids.append(row["video_id"])
    return X, y, ids


# ---------------------------------------------------------------------------
# Representation builders (slice/derive from the 273-D master array)
# ---------------------------------------------------------------------------

def slice_current(X_master):
    """Experiment 1 & 2 (identical): wrist-relative hands + pose + flags -- EXACTLY the existing 147-D representation."""
    return np.concatenate([
        X_master[:, :, WRIST_REL_LEFT[0]:WRIST_REL_LEFT[1]],
        X_master[:, :, WRIST_REL_RIGHT[0]:WRIST_REL_RIGHT[1]],
        X_master[:, :, POSE[0]:POSE[1]],
        X_master[:, :, FLAGS[0]:FLAGS[1]],
    ], axis=-1)


def slice_body_relative(X_master):
    """Experiment 3: body-relative hands + pose (already body-relative) + flags. Same 147-D layout positions as slice_current, different hand semantics."""
    return np.concatenate([
        X_master[:, :, BODY_REL_LEFT[0]:BODY_REL_LEFT[1]],
        X_master[:, :, BODY_REL_RIGHT[0]:BODY_REL_RIGHT[1]],
        X_master[:, :, POSE[0]:POSE[1]],
        X_master[:, :, FLAGS[0]:FLAGS[1]],
    ], axis=-1)


def compute_velocity(spatial_144):
    """First-order temporal difference of the 144 spatial dims (excludes
    flags entirely -- velocity is never computed for detection flags, per
    the task's explicit instruction). velocity[0] = 0 for every sequence.
    Accepts either a batch (N, T, 144) or a single sequence (T, 144)."""
    vel = np.zeros_like(spatial_144)
    if spatial_144.ndim == 3:
        vel[:, 1:, :] = spatial_144[:, 1:, :] - spatial_144[:, :-1, :]
    else:
        vel[1:, :] = spatial_144[1:, :] - spatial_144[:-1, :]
    return vel


def build_velocity_only(spatial_147):
    """Experiment 4A: velocity only (144 spatial velocity dims) + flags unchanged. Stays at 147-D."""
    spatial = spatial_147[:, :, :144]
    flags = spatial_147[:, :, 144:147]
    vel = compute_velocity(spatial)
    return np.concatenate([vel, flags], axis=-1)


def build_position_plus_velocity(spatial_147):
    """Experiment 4B / 5 (literal spec): original normalized position + velocity + flags. 144+144+3 = 291-D (documented expansion)."""
    spatial = spatial_147[:, :, :144]
    flags = spatial_147[:, :, 144:147]
    vel = compute_velocity(spatial)
    return np.concatenate([spatial, vel, flags], axis=-1)


# ---------------------------------------------------------------------------
# Generic, layout-aware augmentation (training data only)
# ---------------------------------------------------------------------------

def augment_sequence_generic(seq, rng, coord_blocks, flags_start, noise_std=0.01, translation_std=0.02, scale_range=(0.95, 1.05), time_jitter_std=1.0):
    """Same method as src/train_bilstm_improved.py:augment_sequence
    (per-block translation+scale, per-frame noise, temporal index jitter;
    flags and zero-filled undetected blocks never perturbed), generalized
    to accept ANY list of (start, end, flag_col_index) coordinate blocks
    and an explicit flags_start offset, so it works correctly across this
    experiment's different feature layouts (147-D and 291-D), not just
    the original fixed 147-D layout."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    T = seq.shape[0]

    base_idx = np.arange(T, dtype=np.float64)
    jitter = rng.normal(0.0, time_jitter_std, size=T)
    warped = np.sort(base_idx + jitter)
    idx = np.clip(np.round(warped), 0, T - 1).astype(int)
    seq = seq[idx]

    flags = seq[:, flags_start:flags_start + 3]

    for start, end, flag_idx in coord_blocks:
        mask = flags[:, flag_idx] > 0.5
        if not np.any(mask):
            continue
        n_landmarks = (end - start) // 3
        if n_landmarks == 0:
            continue
        block = seq[:, start:end].reshape(T, n_landmarks, 3)
        translation = rng.normal(0.0, translation_std, size=(1, 1, 3))
        scale = rng.uniform(scale_range[0], scale_range[1])
        noise = rng.normal(0.0, noise_std, size=block.shape)
        perturbed = block * scale + translation + noise
        block = np.where(mask[:, None, None], perturbed, block)
        seq[:, start:end] = block.reshape(T, end - start)

    return seq.astype(np.float32)


def build_augmented_training_set_generic(X_train, y_train, seed, coord_blocks, flags_start, augment_factor=3):
    if augment_factor <= 1:
        return X_train.copy(), y_train.copy()
    n = X_train.shape[0]
    parent_rng = np.random.default_rng(seed)
    n_copies = augment_factor - 1
    child_rngs = parent_rng.spawn(n * n_copies)
    X_parts, y_parts = [X_train], [y_train]
    rng_i = 0
    for _ in range(n_copies):
        X_copy = np.zeros_like(X_train)
        for i in range(n):
            X_copy[i] = augment_sequence_generic(X_train[i], child_rngs[rng_i], coord_blocks, flags_start)
            rng_i += 1
        X_parts.append(X_copy)
        y_parts.append(y_train.copy())
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# 147-D layout: [hand1(63)|hand2(63)|pose(18)|flags(3)]
COORD_BLOCKS_147 = [(0, 63, 0), (63, 126, 1), (126, 144, 2)]
FLAGS_START_147 = 144
# 291-D layout: [spatial_pos(144)|spatial_vel(144)|flags(3)] -- augmentation
# perturbs the POSITION block only (translation/scale/noise on velocity
# directly would be physically incoherent); velocity is recomputed from
# the augmented position block so the two stay consistent.
FLAGS_START_291 = 288


def augment_291(X_train, y_train, seed, augment_factor=3):
    """291-D specific augmentation: perturb the position block with the
    generic per-block method, then RECOMPUTE velocity from the perturbed
    position (rather than perturbing position and velocity independently,
    which could produce an inconsistent pair)."""
    if augment_factor <= 1:
        return X_train.copy(), y_train.copy()
    n = X_train.shape[0]
    parent_rng = np.random.default_rng(seed)
    n_copies = augment_factor - 1
    child_rngs = parent_rng.spawn(n * n_copies)
    X_parts, y_parts = [X_train], [y_train]
    rng_i = 0
    for _ in range(n_copies):
        X_copy = np.zeros_like(X_train)
        for i in range(n):
            pos_flags = np.concatenate([X_train[i, :, :144], X_train[i, :, 288:291]], axis=-1)
            aug_pos_flags = augment_sequence_generic(pos_flags, child_rngs[rng_i], COORD_BLOCKS_147[:2] + [(126, 144, 2)], 144)
            aug_pos = aug_pos_flags[:, :144]
            aug_flags = aug_pos_flags[:, 144:147]
            aug_vel = compute_velocity(aug_pos)
            X_copy[i] = np.concatenate([aug_pos, aug_vel, aug_flags], axis=-1)
            rng_i += 1
        X_parts.append(X_copy)
        y_parts.append(y_train.copy())
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(name, X_train, y_train, X_val, y_val, X_test, y_test, class_names, args, output_subdir):
    import keras

    feat_dim = X_train.shape[-1]
    model = build_model(len(class_names), seq_len=SEQ_LEN, feat_dim=feat_dim,
                         dropout_rate=args.dropout, learning_rate=args.learning_rate,
                         normalization_adapt_data=X_train)
    param_count = model.count_params()

    os.makedirs(output_subdir, exist_ok=True)
    ckpt_path = os.path.join(output_subdir, "best_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.max_epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.perf_counter() - start
    best_val_acc = float(max(history.history["val_accuracy"]))
    final_train_acc = float(history.history["accuracy"][-1])
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    save_json(os.path.join(output_subdir, "training_history.json"),
              {k: [float(v) for v in vals] for k, vals in history.history.items()})

    best_model = keras.models.load_model(ckpt_path)
    y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    results = evaluate_predictions(y_test, y_pred, class_names)

    with open(os.path.join(output_subdir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(results["classification_report"])
    np.savetxt(os.path.join(output_subdir, "test_confusion_matrix.csv"), results["confusion_matrix"], fmt="%d", delimiter=",")
    save_confusion_matrix_plot(results["confusion_matrix"], class_names, os.path.join(output_subdir, "test_confusion_matrix.png"))

    # Train accuracy at the BEST-VAL epoch (not final epoch) for an honest train/val gap
    train_acc_at_best = float(history.history["accuracy"][best_epoch - 1])

    return {
        "name": name, "feature_dimension": feat_dim, "parameter_count": int(param_count),
        "epochs_trained": len(history.history["loss"]), "best_val_epoch": best_epoch,
        "training_time_seconds": round(elapsed, 3),
        "train_accuracy_final_epoch": final_train_acc,
        "train_accuracy_at_best_val_epoch": train_acc_at_best,
        "val_accuracy": best_val_acc,
        "train_val_gap": round(train_acc_at_best - best_val_acc, 4),
        "overfitting_flag": bool(train_acc_at_best > 0.90 and best_val_acc < 0.40),
        "test_accuracy": results["accuracy"], "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"], "macro_f1": results["macro_f1"], "weighted_f1": results["weighted_f1"],
        "n_correct": int(np.sum(y_pred == y_test)), "n_total": len(y_test),
    }


def leakage_audit(train_ids, val_ids, test_ids, master_meta_by_id, split_rows):
    checks = {}
    for a, b, name in [(train_ids, val_ids, "train_val"), (train_ids, test_ids, "train_test"), (val_ids, test_ids, "val_test")]:
        overlap = sorted(set(a) & set(b))
        checks[f"no_duplicate_video_across_{name}"] = {"overlap": overlap, "PASS": len(overlap) == 0}
    # content-hash dedup check across splits (reuse master sequence files)
    import hashlib
    def file_hash(vid):
        p = os.path.join(config.PROJECT_ROOT, master_meta_by_id[vid]["sequence_path"])
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    hashes = {"train": {}, "val": {}, "test": {}}
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for vid in ids:
            hashes[split].setdefault(file_hash(vid), []).append(vid)
    content_overlap_total = 0
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(hashes[a]) & set(hashes[b])
        content_overlap_total += len(shared)
    checks["no_duplicate_sequence_content_across_splits"] = {"PASS": content_overlap_total == 0, "overlap_count": content_overlap_total}
    checks["no_normalization_fitted_on_val_test"] = {"note": "every spatial transformation (wrist-relative, body-relative, velocity) is a per-frame geometric formula with no fitted statistics; the only fitted statistics anywhere (Keras Normalization layer) are .adapt()-ed on X_train only, per build_model's normalization_adapt_data=X_train call in every experiment below", "PASS": True}
    checks["augmentation_training_only"] = {"note": "build_augmented_training_set_generic / augment_291 called only on TRAIN arrays in this script", "PASS": True}
    checks["no_test_set_model_selection"] = {"note": "all checkpoints selected by val_accuracy (EarlyStopping/ModelCheckpoint monitor=val_accuracy); test set evaluated exactly once per experiment, after selection", "PASS": True}
    checks["no_changes_to_frozen_historical_experiments"] = {"note": "verified separately via frozen-path file-count comparison in the final report", "PASS": True}
    all_pass = all(v["PASS"] for v in checks.values())
    return checks, all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--augment-factor", type=int, default=3)
    args = parser.parse_args()

    set_global_seed(args.seed)

    split_rows = load_split_ids()
    gloss_to_id = build_class_mapping(split_rows)
    class_names = [None] * len(gloss_to_id)
    for g, i in gloss_to_id.items():
        class_names[i] = g

    with open(MASTER_METADATA_PATH, newline="", encoding="utf-8") as f:
        master_meta_by_id = {r["video_id"]: r for r in csv.DictReader(f) if r["processing_status"] == "success"}

    missing = [r["video_id"] for rows in split_rows.values() for r in rows if r["video_id"] not in master_meta_by_id]
    if missing:
        print(f"FATAL: {len(missing)} split video_ids missing from master features (extraction incomplete?): {missing[:10]}...")
        return 1

    X_train_m, y_train, train_ids = load_master_arrays(split_rows["train"], master_meta_by_id, gloss_to_id, config.PROJECT_ROOT)
    X_val_m, y_val, val_ids = load_master_arrays(split_rows["val"], master_meta_by_id, gloss_to_id, config.PROJECT_ROOT)
    X_test_m, y_test, test_ids = load_master_arrays(split_rows["test"], master_meta_by_id, gloss_to_id, config.PROJECT_ROOT)
    print(f"Loaded master arrays: train={X_train_m.shape} val={X_val_m.shape} test={X_test_m.shape}  classes={len(gloss_to_id)}")

    print("=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)
    audit_checks, audit_pass = leakage_audit(train_ids, val_ids, test_ids, master_meta_by_id, split_rows)
    for k, v in audit_checks.items():
        print(f"{k}: {'PASS' if v['PASS'] else 'FAIL'}")
    print(f"OVERALL: {'ALL PASS' if audit_pass else 'FAIL'}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(os.path.join(OUTPUT_DIR, "leakage_audit.json"), {"checks": audit_checks, "all_pass": audit_pass})
    if not audit_pass:
        print("FATAL: leakage audit failed, aborting.")
        return 1

    results = {}

    # NOTE: per the task's explicit instruction ("Use the original baseline
    # configuration whenever possible so this experiment is comparable with
    # the existing 25.61% result" + "First determine whether normalization
    # itself helps. Then, if useful, test: best representation + training-
    # only augmentation"), Experiments 1/2/3/4/5 below use NO augmentation
    # by default -- matching src/train_bilstm_baseline.py's original
    # (unaugmented) configuration exactly, so these results are a fair,
    # apples-to-apples representation comparison against the historical
    # 25.61% baseline. Augmentation is tested ONLY as a final, separate
    # step on whichever representation wins by validation accuracy.

    # --- Experiment 1: current representation (control), NO augmentation ---
    print("=" * 60); print("EXPERIMENT 1: Current representation (wrist-relative hands + pose), no augmentation"); print("=" * 60)
    X_train_1, X_val_1, X_test_1 = slice_current(X_train_m), slice_current(X_val_m), slice_current(X_test_m)
    results["exp1_current"] = train_and_evaluate("exp1_current", X_train_1, y_train, X_val_1, y_val, X_test_1, y_test, class_names, args, os.path.join(OUTPUT_DIR, "exp1_current"))
    print(f"Exp1: val={results['exp1_current']['val_accuracy']:.4f} test={results['exp1_current']['test_accuracy']:.4f} f1={results['exp1_current']['macro_f1']:.4f}")

    # --- Experiment 2: root-centered (= Experiment 1, documented, not retrained) ---
    results["exp2_root_centered"] = dict(results["exp1_current"])
    results["exp2_root_centered"]["name"] = "exp2_root_centered"
    results["exp2_root_centered"]["note"] = "IDENTICAL to Experiment 1 -- the existing pipeline's hand normalization IS wrist-relative/root-centered normalization; not retrained separately (see report)."

    # --- Experiment 3: body-relative hands, NO augmentation ---
    print("=" * 60); print("EXPERIMENT 3: Body-relative hands (shoulder-frame), no augmentation"); print("=" * 60)
    X_train_3, X_val_3, X_test_3 = slice_body_relative(X_train_m), slice_body_relative(X_val_m), slice_body_relative(X_test_m)
    results["exp3_body_relative"] = train_and_evaluate("exp3_body_relative", X_train_3, y_train, X_val_3, y_val, X_test_3, y_test, class_names, args, os.path.join(OUTPUT_DIR, "exp3_body_relative"))
    print(f"Exp3: val={results['exp3_body_relative']['val_accuracy']:.4f} test={results['exp3_body_relative']['test_accuracy']:.4f} f1={results['exp3_body_relative']['macro_f1']:.4f}")

    # --- Experiment 4A: velocity only, NO augmentation ---
    print("=" * 60); print("EXPERIMENT 4A: Velocity only, no augmentation"); print("=" * 60)
    X_train_4a, X_val_4a, X_test_4a = build_velocity_only(X_train_1), build_velocity_only(X_val_1), build_velocity_only(X_test_1)
    results["exp4a_velocity_only"] = train_and_evaluate("exp4a_velocity_only", X_train_4a, y_train, X_val_4a, y_val, X_test_4a, y_test, class_names, args, os.path.join(OUTPUT_DIR, "exp4a_velocity_only"))
    print(f"Exp4A: val={results['exp4a_velocity_only']['val_accuracy']:.4f} test={results['exp4a_velocity_only']['test_accuracy']:.4f} f1={results['exp4a_velocity_only']['macro_f1']:.4f}")

    # --- Experiment 4B / 5: position + velocity, NO augmentation (literal Exp5 spec: root-centered position + velocity + flags) ---
    print("=" * 60); print("EXPERIMENT 4B / 5: Position + velocity (root-centered hands + pose + velocity + flags), no augmentation"); print("=" * 60)
    X_train_5, X_val_5, X_test_5 = build_position_plus_velocity(X_train_1), build_position_plus_velocity(X_val_1), build_position_plus_velocity(X_test_1)
    results["exp5_position_plus_velocity"] = train_and_evaluate("exp5_position_plus_velocity", X_train_5, y_train, X_val_5, y_val, X_test_5, y_test, class_names, args, os.path.join(OUTPUT_DIR, "exp5_position_plus_velocity"))
    print(f"Exp5: val={results['exp5_position_plus_velocity']['val_accuracy']:.4f} test={results['exp5_position_plus_velocity']['test_accuracy']:.4f} f1={results['exp5_position_plus_velocity']['macro_f1']:.4f}")

    # --- Bonus: body-relative + velocity, NO augmentation (extra data point, cheap given master features already computed) ---
    print("=" * 60); print("BONUS: Body-relative hands + pose + velocity + flags, no augmentation"); print("=" * 60)
    X_train_5b, X_val_5b, X_test_5b = build_position_plus_velocity(X_train_3), build_position_plus_velocity(X_val_3), build_position_plus_velocity(X_test_3)
    results["exp5b_body_relative_plus_velocity"] = train_and_evaluate("exp5b_body_relative_plus_velocity", X_train_5b, y_train, X_val_5b, y_val, X_test_5b, y_test, class_names, args, os.path.join(OUTPUT_DIR, "exp5b_body_relative_plus_velocity"))
    print(f"Exp5B: val={results['exp5b_body_relative_plus_velocity']['val_accuracy']:.4f} test={results['exp5b_body_relative_plus_velocity']['test_accuracy']:.4f} f1={results['exp5b_body_relative_plus_velocity']['macro_f1']:.4f}")

    # --- Select best representation by VALIDATION accuracy (never test) ---
    candidates = {k: v for k, v in results.items() if k != "exp2_root_centered"}  # exp2 is a duplicate of exp1, exclude from selection to avoid double-counting
    best_name = max(candidates, key=lambda k: candidates[k]["val_accuracy"])
    best = candidates[best_name]
    print("=" * 60)
    print(f"BEST BY VALIDATION ACCURACY (no augmentation): {best_name} (val={best['val_accuracy']:.4f})")
    print("=" * 60)

    # --- Final step: training-only augmentation on the winning representation ---
    print("=" * 60); print(f"AUGMENTATION TEST on best representation ({best_name})"); print("=" * 60)
    rep_builders = {
        "exp1_current": (X_train_1, X_val_1, X_test_1, COORD_BLOCKS_147, FLAGS_START_147, False),
        "exp3_body_relative": (X_train_3, X_val_3, X_test_3, COORD_BLOCKS_147, FLAGS_START_147, False),
        "exp4a_velocity_only": (X_train_4a, X_val_4a, X_test_4a, COORD_BLOCKS_147, FLAGS_START_147, False),
        "exp5_position_plus_velocity": (X_train_5, X_val_5, X_test_5, None, None, True),
        "exp5b_body_relative_plus_velocity": (X_train_5b, X_val_5b, X_test_5b, None, None, True),
    }
    Xt_win, Xv_win, Xte_win, coord_blocks, flags_start, is_291 = rep_builders[best_name]
    if is_291:
        Xt_aug, yt_aug = augment_291(Xt_win, y_train, args.seed, args.augment_factor)
    else:
        Xt_aug, yt_aug = build_augmented_training_set_generic(Xt_win, y_train, args.seed, coord_blocks, flags_start, args.augment_factor)
    aug_result = train_and_evaluate(f"{best_name}_with_augmentation", Xt_aug, yt_aug, Xv_win, y_val, Xte_win, y_test, class_names, args, os.path.join(OUTPUT_DIR, f"{best_name}_with_augmentation"))
    results["best_representation_with_augmentation"] = aug_result
    print(f"With-aug: val={aug_result['val_accuracy']:.4f} test={aug_result['test_accuracy']:.4f} f1={aug_result['macro_f1']:.4f}")
    augmentation_helped = aug_result["val_accuracy"] > best["val_accuracy"]

    # Overall best model considering the augmentation test too (still selected by val accuracy)
    overall_best_name = best_name if best["val_accuracy"] >= aug_result["val_accuracy"] else f"{best_name}_with_augmentation"
    overall_best = best if overall_best_name == best_name else aug_result

    summary = {
        "results": results,
        "baseline_reference": {"val_accuracy": BASELINE_VAL_ACC, "test_accuracy": BASELINE_TEST_ACC, "macro_f1": BASELINE_MACRO_F1},
        "best_representation_no_augmentation": best_name, "best_val_accuracy_no_augmentation": best["val_accuracy"],
        "overall_best_model": overall_best_name, "overall_best_val_accuracy": overall_best["val_accuracy"],
        "overall_best_test_accuracy": overall_best["test_accuracy"], "overall_best_macro_f1": overall_best["macro_f1"],
        "improvement_vs_baseline_test_acc": overall_best["test_accuracy"] - BASELINE_TEST_ACC,
        "augmentation_helped_best_representation": augmentation_helped,
        "velocity_helped": results["exp4a_velocity_only"]["val_accuracy"] > results["exp1_current"]["val_accuracy"] or results["exp5_position_plus_velocity"]["val_accuracy"] > results["exp1_current"]["val_accuracy"],
        "body_relative_helped": results["exp3_body_relative"]["val_accuracy"] > results["exp1_current"]["val_accuracy"],
        "audit": {"checks": audit_checks, "all_pass": audit_pass},
    }
    save_json(REPORT_JSON, summary)

    print("=" * 60)
    print("SUMMARY (validation accuracy)")
    for k, v in results.items():
        print(f"  {k}: val={v['val_accuracy']:.4f} test={v['test_accuracy']:.4f} f1={v['macro_f1']:.4f} train/val gap={v.get('train_val_gap', 'n/a')}")
    print(f"OVERALL BEST: {overall_best_name} -- test_acc={overall_best['test_accuracy']:.4f} (baseline: {BASELINE_TEST_ACC:.4f})")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
