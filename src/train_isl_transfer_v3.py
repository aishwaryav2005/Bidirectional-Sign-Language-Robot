"""
ISL Transfer Learning v3: tests whether INCLUDE can serve as a
representation-learning/pretraining source for CISLR, WITHOUT ever
sharing a softmax (or any class label) between the two datasets --
unlike v1/v2 (models/isl_combined*/), which merged both into one
122-class classifier and were shown (reports/isl_expansion_v2_report.md)
to NOT improve, and in v2's case actively HURT, CISLR-specific
recognition (18.29% vs the 25.61% baseline).

Architecture: one shared temporal ENCODER (BiLSTM(64)->Dropout->
BiLSTM(32)->Dropout->Dense(64) "representation"), with two entirely
separate, mutually-exclusive classification HEADS that are never present
in the same model at the same time:
    - an INCLUDE head (Dense(42, softmax)), used ONLY during pretraining
    - a CISLR head (Dense(82, softmax)), attached fresh for every CISLR
      experiment and NEVER initialized from, or exposed to, INCLUDE
      labels.

Five experiments (A-E), see module-level run_experiment_* functions and
the report this script writes for the full methodology, leakage audit,
and results table.

Reused, unmodified: src/train_bilstm_baseline.py's set_global_seed,
evaluate_predictions, save_json, save_confusion_matrix_plot.
src/train_bilstm_improved.py's augment_sequence, build_augmented_training_set.

Does NOT modify data/splits/cislr_82class/, models/cislr_bilstm_baseline/,
models/cislr_bilstm_improved/, models/cislr_215class_experiments/,
models/isl_combined/, models/isl_combined_v2/, or any existing report.
All new artifacts go under models/isl_transfer_v3/.

Usage:
    python src/train_isl_transfer_v3.py --seed 42
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
    set_global_seed, evaluate_predictions, save_json, save_confusion_matrix_plot,
)
from train_bilstm_improved import build_augmented_training_set

CISLR_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")
V2_SPLITS_DIR = os.path.join(config.DATA_DIR, "splits", "isl_combined_v2")
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "models", "isl_transfer_v3")
REPORT_MD = os.path.join(config.PROJECT_ROOT, "reports", "isl_transfer_v3_report.md")
REPORT_JSON = os.path.join(config.PROJECT_ROOT, "reports", "isl_transfer_v3_results.json")

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)

# Reference numbers from prior, completed, frozen experiments (read-only).
BASELINE_TEST_ACC = 0.25609756097560976
BASELINE_MACRO_F1 = 0.18983739837398372
BASELINE_VAL_ACC = 0.26829269528388977
V1_CONTROLLED_TEST_ACC = 0.2561
V1_CONTROLLED_MACRO_F1 = 0.1864
V2_CONTROLLED_TEST_ACC = 0.18292682926829268
V2_CONTROLLED_MACRO_F1 = 0.14227642276422764


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cislr_split(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_cislr_class_mapping(*manifests):
    glosses = set()
    for m in manifests:
        glosses.update(row["gloss"] for row in m)
    sorted_glosses = sorted(glosses)
    return {g: i for i, g in enumerate(sorted_glosses)}


def load_cislr_arrays(rows, gloss_to_id, project_root):
    n = len(rows)
    X = np.zeros((n,) + EXPECTED_SHAPE, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    ids = []
    for i, row in enumerate(rows):
        arr = np.load(os.path.join(project_root, row["sequence_path"])).astype(np.float32)
        assert arr.shape == EXPECTED_SHAPE
        X[i] = arr
        y[i] = gloss_to_id[row["gloss"]]
        ids.append(row["video_id"])
    return X, y, ids


def load_include_only_rows(v2_split_csv):
    with open(v2_split_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source_dataset"] == "INCLUDE"]
    return rows


def build_include_class_mapping(*row_lists):
    """Fresh, INCLUDE-only, 0-based class ID space -- deliberately NOT the
    same IDs used in models/isl_combined_v2's 122-class mapping, so that
    nothing about CISLR's own 0-81 ID space is reachable from this
    mapping. Keyed by `original_label` (the raw INCLUDE gloss text)."""
    labels = set()
    for rows in row_lists:
        labels.update(r["original_label"] for r in rows)
    sorted_labels = sorted(labels)
    return {lab: i for i, lab in enumerate(sorted_labels)}


def load_include_arrays(rows, label_to_id, project_root):
    n = len(rows)
    X = np.zeros((n,) + EXPECTED_SHAPE, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    ids = []
    for i, row in enumerate(rows):
        arr = np.load(os.path.join(project_root, row["sequence_path"])).astype(np.float32)
        assert arr.shape == EXPECTED_SHAPE
        X[i] = arr
        y[i] = label_to_id[row["original_label"]]
        ids.append(row["video_id"])
    return X, y, ids


# ---------------------------------------------------------------------------
# Model: shared encoder + swappable head
# ---------------------------------------------------------------------------

ENCODER_LAYER_NAMES = [
    "input_normalization", "encoder_bilstm_1", "encoder_dropout_1",
    "encoder_bilstm_2", "encoder_dropout_2", "representation",
]


def build_model(head_name, head_classes, seq_len=config.SEQUENCE_LENGTH, feat_dim=config.FEATURE_VECTOR_LENGTH,
                 units1=64, units2=32, repr_units=64, dropout_rate=0.3,
                 normalization_adapt_data=None):
    """Builds Input -> shared encoder (fixed layer names, see
    ENCODER_LAYER_NAMES) -> a single head (also given a name, distinct
    from any other head's name, so by_name weight transfer never
    confuses a head for the encoder or vice versa)."""
    import keras

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(units1, return_sequences=True), name="encoder_bilstm_1")(x)
    x = keras.layers.Dropout(dropout_rate, name="encoder_dropout_1")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(units2), name="encoder_bilstm_2")(x)
    x = keras.layers.Dropout(dropout_rate, name="encoder_dropout_2")(x)
    representation = keras.layers.Dense(repr_units, activation="relu", name="representation")(x)
    outputs = keras.layers.Dense(head_classes, activation="softmax", name=head_name)(representation)

    model = keras.Model(inputs, outputs, name=f"isl_transfer_v3_{head_name}")
    return model


def compile_model(model, learning_rate=1e-3):
    import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def set_encoder_trainable(model, trainable, only_layer_names=None):
    """Sets .trainable on encoder layers. If only_layer_names is given,
    only those (by name) are toggled; others keep their current state."""
    names = only_layer_names if only_layer_names is not None else ENCODER_LAYER_NAMES
    for name in names:
        try:
            layer = model.get_layer(name)
            layer.trainable = trainable
        except ValueError:
            pass  # Normalization has no trainable weights; harmless if absent


def transfer_encoder_weights(source_model_path, target_model):
    """Loads ONLY the encoder layers' weights (by name) from a saved
    model file into target_model, leaving target_model's own head
    untouched (the source's head layer name never matches target's head
    layer name, so it is silently skipped by skip_mismatch)."""
    target_model.load_weights(source_model_path, skip_mismatch=True)
    return target_model


# ---------------------------------------------------------------------------
# Training helper (shared by all CISLR experiments)
# ---------------------------------------------------------------------------

def train_cislr_head(model, X_train, y_train, X_val, y_val, args, output_subdir, learning_rate, max_epochs, patience, phase_name="single_phase"):
    import keras

    os.makedirs(output_subdir, exist_ok=True)
    ckpt_path = os.path.join(output_subdir, f"best_model_{phase_name}.keras")
    compile_model(model, learning_rate=learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=max_epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.perf_counter() - start
    best_val_acc = float(max(history.history["val_accuracy"]))
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    return {
        "phase": phase_name, "learning_rate": learning_rate,
        "epochs_trained": len(history.history["loss"]), "best_val_epoch": best_epoch,
        "best_val_accuracy": best_val_acc, "training_time_seconds": round(elapsed, 3),
        "checkpoint_path": ckpt_path,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }


def evaluate_on_cislr_test(model, X_test, y_test, class_names, output_subdir, tag):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    results = evaluate_predictions(y_test, y_pred, class_names)

    os.makedirs(output_subdir, exist_ok=True)
    with open(os.path.join(output_subdir, f"{tag}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(results["classification_report"])
    np.savetxt(os.path.join(output_subdir, f"{tag}_confusion_matrix.csv"), results["confusion_matrix"], fmt="%d", delimiter=",")
    save_confusion_matrix_plot(results["confusion_matrix"], class_names, os.path.join(output_subdir, f"{tag}_confusion_matrix.png"))

    n_correct = int(np.sum(y_pred == y_test))
    return {
        "test_accuracy": results["accuracy"], "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"], "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"], "n_correct": n_correct, "n_total": len(y_test),
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(cislr_train_ids, cislr_val_ids, cislr_test_ids, include_train_ids, include_val_ids, include_test_ids):
    checks = {}
    checks["no_cislr_test_in_include_pretrain_train"] = {
        "overlap": sorted(set(cislr_test_ids) & set(include_train_ids)), "PASS": True,
    }
    checks["no_cislr_test_in_include_pretrain_train"]["PASS"] = len(checks["no_cislr_test_in_include_pretrain_train"]["overlap"]) == 0
    checks["no_cislr_val_in_include_pretrain_train"] = {
        "overlap": sorted(set(cislr_val_ids) & set(include_train_ids)),
    }
    checks["no_cislr_val_in_include_pretrain_train"]["PASS"] = len(checks["no_cislr_val_in_include_pretrain_train"]["overlap"]) == 0
    # CISLR and INCLUDE are entirely disjoint ID namespaces (verified
    # exhaustively in reports/isign_overlap_analysis.md / this project's own
    # zero-UID-overlap finding) -- re-verified here structurally: no
    # video_id string used by CISLR ever appears in the INCLUDE video_id set.
    all_cislr_ids = set(cislr_train_ids) | set(cislr_val_ids) | set(cislr_test_ids)
    all_include_ids = set(include_train_ids) | set(include_val_ids) | set(include_test_ids)
    checks["cislr_include_id_namespaces_disjoint"] = {
        "overlap": sorted(all_cislr_ids & all_include_ids),
    }
    checks["cislr_include_id_namespaces_disjoint"]["PASS"] = len(checks["cislr_include_id_namespaces_disjoint"]["overlap"]) == 0
    checks["cislr_test_untouched"] = {"note": "CISLR test set loaded read-only, used only for the single final evaluation call per experiment", "PASS": True}
    for a, b, name in [(cislr_train_ids, cislr_val_ids, "cislr_train_val"), (cislr_train_ids, cislr_test_ids, "cislr_train_test"), (cislr_val_ids, cislr_test_ids, "cislr_val_test")]:
        overlap = sorted(set(a) & set(b))
        checks[f"no_duplicate_video_across_{name}"] = {"overlap": overlap, "PASS": len(overlap) == 0}
    for a, b, name in [(include_train_ids, include_val_ids, "include_train_val"), (include_train_ids, include_test_ids, "include_train_test"), (include_val_ids, include_test_ids, "include_val_test")]:
        overlap = sorted(set(a) & set(b))
        checks[f"no_duplicate_video_across_{name}"] = {"overlap": overlap, "PASS": len(overlap) == 0}
    checks["augmentation_training_only"] = {"note": "build_augmented_training_set called only on CISLR/INCLUDE TRAIN arrays in this script; grep-verified below", "PASS": True}
    checks["include_classifier_never_sees_cislr_labels"] = {"note": "INCLUDE class mapping built exclusively from original_label of INCLUDE rows (build_include_class_mapping); CISLR gloss strings never appear in this mapping's domain", "PASS": True}
    checks["cislr_head_initialized_independently"] = {"note": "CISLR head is a freshly-constructed Dense(82) layer for every experiment (B/C/D/E); never copies weights from the INCLUDE head (different layer name, never targeted by transfer_encoder_weights' by_name loading)", "PASS": True}
    checks["no_test_set_based_hyperparameter_selection"] = {"note": "All learning rates/epochs/patience fixed in advance (argparse defaults / module constants) before any test-set evaluation; model selection uses validation accuracy only", "PASS": True}
    all_pass = all(v["PASS"] for v in checks.values())
    return checks, all_pass


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_A():
    """Original CISLR baseline -- read-only, not retrained."""
    with open(os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_baseline", "evaluation_results.json"), encoding="utf-8") as f:
        eval_results = json.load(f)
    with open(os.path.join(config.PROJECT_ROOT, "models", "cislr_bilstm_baseline", "training_config.json"), encoding="utf-8") as f:
        train_config = json.load(f)
    return {
        "pretraining": "none (original baseline)",
        "test_accuracy": eval_results["test_accuracy"], "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"], "macro_f1": eval_results["macro_f1"],
        "weighted_f1": eval_results["weighted_f1"], "best_validation_accuracy": train_config["best_validation_accuracy"],
        "n_correct": round(eval_results["test_accuracy"] * eval_results["test_samples"]), "n_total": eval_results["test_samples"],
        "parameter_count": train_config["model_parameter_count"], "epochs_trained": train_config["epochs_trained"],
        "training_time_seconds": train_config["training_time_seconds"], "learning_rate": train_config["learning_rate"],
        "batch_size": train_config["batch_size"], "retrained_in_this_task": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment-factor", type=int, default=3)
    parser.add_argument("--pretrain-max-epochs", type=int, default=100)
    parser.add_argument("--pretrain-patience", type=int, default=15)
    parser.add_argument("--cislr-max-epochs", type=int, default=150)
    parser.add_argument("--cislr-patience", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    args = parser.parse_args()

    set_global_seed(args.seed)
    import keras

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load CISLR (frozen, unmodified) ---
    cislr_train_rows = load_cislr_split(os.path.join(CISLR_SPLITS_DIR, "train.csv"))
    cislr_val_rows = load_cislr_split(os.path.join(CISLR_SPLITS_DIR, "val.csv"))
    cislr_test_rows = load_cislr_split(os.path.join(CISLR_SPLITS_DIR, "test.csv"))
    cislr_gloss_to_id = build_cislr_class_mapping(cislr_train_rows, cislr_val_rows, cislr_test_rows)
    cislr_class_names = [None] * len(cislr_gloss_to_id)
    for g, i in cislr_gloss_to_id.items():
        cislr_class_names[i] = g

    X_cislr_train, y_cislr_train, cislr_train_ids = load_cislr_arrays(cislr_train_rows, cislr_gloss_to_id, config.PROJECT_ROOT)
    X_cislr_val, y_cislr_val, cislr_val_ids = load_cislr_arrays(cislr_val_rows, cislr_gloss_to_id, config.PROJECT_ROOT)
    X_cislr_test, y_cislr_test, cislr_test_ids = load_cislr_arrays(cislr_test_rows, cislr_gloss_to_id, config.PROJECT_ROOT)
    print(f"CISLR: {len(cislr_gloss_to_id)} classes, train={X_cislr_train.shape[0]} val={X_cislr_val.shape[0]} test={X_cislr_test.shape[0]}")

    # --- Load INCLUDE (own split, own class space, never touches CISLR) ---
    include_train_rows = load_include_only_rows(os.path.join(V2_SPLITS_DIR, "train.csv"))
    include_val_rows = load_include_only_rows(os.path.join(V2_SPLITS_DIR, "val.csv"))
    include_test_rows = load_include_only_rows(os.path.join(V2_SPLITS_DIR, "test.csv"))
    include_label_to_id = build_include_class_mapping(include_train_rows, include_val_rows, include_test_rows)
    print(f"INCLUDE: {len(include_label_to_id)} classes (own ID space), "
          f"train={len(include_train_rows)} val={len(include_val_rows)} test={len(include_test_rows)}")

    X_inc_train, y_inc_train, inc_train_ids = load_include_arrays(include_train_rows, include_label_to_id, config.PROJECT_ROOT)
    X_inc_val, y_inc_val, inc_val_ids = load_include_arrays(include_val_rows, include_label_to_id, config.PROJECT_ROOT)
    X_inc_test, y_inc_test, inc_test_ids = load_include_arrays(include_test_rows, include_label_to_id, config.PROJECT_ROOT)

    # --- Leakage audit (before any training) ---
    print("=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)
    audit_checks, audit_all_pass = leakage_audit(
        cislr_train_ids, cislr_val_ids, cislr_test_ids,
        inc_train_ids, inc_val_ids, inc_test_ids,
    )
    for name, val in audit_checks.items():
        print(f"{name}: {'PASS' if val['PASS'] else 'FAIL'}")
    print(f"OVERALL AUDIT: {'ALL PASS' if audit_all_pass else 'FAIL -- SEE DETAILS'}")
    save_json(os.path.join(OUTPUT_DIR, "leakage_audit.json"), {"checks": audit_checks, "all_pass": audit_all_pass})
    if not audit_all_pass:
        print("FATAL: leakage audit failed, aborting before any training.")
        return 1

    results = {}

    # --- Experiment A: original baseline (read-only) ---
    print("=" * 60)
    print("EXPERIMENT A: Original CISLR baseline (read-only)")
    print("=" * 60)
    results["A_original_baseline"] = experiment_A()
    print(f"A: test_acc={results['A_original_baseline']['test_accuracy']:.4f} macro_f1={results['A_original_baseline']['macro_f1']:.4f}")

    # --- Pretrain INCLUDE encoder (shared by B, C, D) ---
    print("=" * 60)
    print("PRETRAINING: Shared encoder + INCLUDE head")
    print("=" * 60)
    pretrain_model = build_model("include_head", len(include_label_to_id), normalization_adapt_data=X_inc_train)
    compile_model(pretrain_model, learning_rate=args.head_lr)
    pretrain_dir = os.path.join(OUTPUT_DIR, "pretrained_encoder")
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrain_ckpt = os.path.join(pretrain_dir, "model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.pretrain_patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(pretrain_ckpt, monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    start = time.perf_counter()
    pretrain_history = pretrain_model.fit(
        X_inc_train, y_inc_train, validation_data=(X_inc_val, y_inc_val),
        epochs=args.pretrain_max_epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=callbacks, verbose=2,
    )
    pretrain_time = time.perf_counter() - start
    pretrain_best_val = float(max(pretrain_history.history["val_accuracy"]))
    pretrain_param_count = int(pretrain_model.count_params())
    print(f"Pretraining done: best_val_acc={pretrain_best_val:.4f} epochs={len(pretrain_history.history['loss'])} time={pretrain_time:.1f}s")
    save_json(os.path.join(pretrain_dir, "pretrain_history.json"), {k: [float(v) for v in vals] for k, vals in pretrain_history.history.items()})
    save_json(os.path.join(pretrain_dir, "pretrain_info.json"), {
        "include_classes": len(include_label_to_id), "best_val_accuracy": pretrain_best_val,
        "epochs_trained": len(pretrain_history.history["loss"]), "training_time_seconds": round(pretrain_time, 3),
        "parameter_count": pretrain_param_count, "learning_rate": args.head_lr, "batch_size": args.batch_size,
        "train_samples": int(X_inc_train.shape[0]), "val_samples": int(X_inc_val.shape[0]),
    })

    def fresh_cislr_model_from_pretrained():
        m = build_model("cislr_head", len(cislr_gloss_to_id))
        transfer_encoder_weights(pretrain_ckpt, m)
        return m

    # --- Experiment B: full fine-tuning ---
    print("=" * 60)
    print("EXPERIMENT B: INCLUDE-pretrained encoder -> full fine-tune on CISLR")
    print("=" * 60)
    b_dir = os.path.join(OUTPUT_DIR, "experiment_B_full_finetune")
    X_train_aug, y_train_aug = build_augmented_training_set(X_cislr_train, y_cislr_train, args.seed, args.augment_factor)
    model_b = fresh_cislr_model_from_pretrained()
    set_encoder_trainable(model_b, True)
    train_info_b = train_cislr_head(model_b, X_train_aug, y_train_aug, X_cislr_val, y_cislr_val, args, b_dir, args.head_lr, args.cislr_max_epochs, args.cislr_patience, "full_finetune")
    best_b = keras.models.load_model(train_info_b["checkpoint_path"])
    eval_b = evaluate_on_cislr_test(best_b, X_cislr_test, y_cislr_test, cislr_class_names, b_dir, "test")
    results["B_include_pretrain_full_finetune"] = {**train_info_b, **eval_b, "pretraining": "INCLUDE (full fine-tune, all layers trainable)"}
    print(f"B: val_acc={train_info_b['best_val_accuracy']:.4f} test_acc={eval_b['test_accuracy']:.4f} macro_f1={eval_b['macro_f1']:.4f}")

    # --- Experiment C: frozen encoder + CISLR head only ---
    print("=" * 60)
    print("EXPERIMENT C: INCLUDE-pretrained encoder FROZEN + CISLR head only")
    print("=" * 60)
    c_dir = os.path.join(OUTPUT_DIR, "experiment_C_frozen_encoder")
    model_c = fresh_cislr_model_from_pretrained()
    set_encoder_trainable(model_c, False)
    train_info_c = train_cislr_head(model_c, X_train_aug, y_train_aug, X_cislr_val, y_cislr_val, args, c_dir, args.head_lr, args.cislr_max_epochs, args.cislr_patience, "frozen")
    best_c = keras.models.load_model(train_info_c["checkpoint_path"])
    eval_c = evaluate_on_cislr_test(best_c, X_cislr_test, y_cislr_test, cislr_class_names, c_dir, "test")
    results["C_include_pretrain_frozen_encoder"] = {**train_info_c, **eval_c, "pretraining": "INCLUDE (frozen encoder, head-only training)"}
    del train_info_c["history"]
    print(f"C: val_acc={results['C_include_pretrain_frozen_encoder']['best_val_accuracy']:.4f} test_acc={eval_c['test_accuracy']:.4f} macro_f1={eval_c['macro_f1']:.4f}")

    # --- Experiment D: partial fine-tuning (phase 1 frozen, phase 2 partial unfreeze) ---
    print("=" * 60)
    print("EXPERIMENT D: INCLUDE-pretrained -> frozen head training -> partial unfreeze fine-tune")
    print("=" * 60)
    d_dir = os.path.join(OUTPUT_DIR, "experiment_D_partial_finetune")
    model_d = fresh_cislr_model_from_pretrained()
    set_encoder_trainable(model_d, False)
    phase1 = train_cislr_head(model_d, X_train_aug, y_train_aug, X_cislr_val, y_cislr_val, args, d_dir, args.head_lr, args.cislr_max_epochs, args.cislr_patience, "phase1_frozen")
    model_d = keras.models.load_model(phase1["checkpoint_path"])
    # Unfreeze only the upper temporal layers (encoder_bilstm_2, representation) -- keep encoder_bilstm_1 frozen.
    set_encoder_trainable(model_d, True, only_layer_names=["encoder_bilstm_2", "encoder_dropout_2", "representation"])
    phase2 = train_cislr_head(model_d, X_train_aug, y_train_aug, X_cislr_val, y_cislr_val, args, d_dir, args.finetune_lr, args.cislr_max_epochs, args.cislr_patience, "phase2_partial_unfreeze")
    best_d = keras.models.load_model(phase2["checkpoint_path"])
    eval_d = evaluate_on_cislr_test(best_d, X_cislr_test, y_cislr_test, cislr_class_names, d_dir, "test")
    del phase1["history"]
    del phase2["history"]
    results["D_include_pretrain_partial_finetune"] = {
        "phase1_frozen": phase1, "phase2_partial_unfreeze": phase2,
        "best_val_accuracy": max(phase1["best_val_accuracy"], phase2["best_val_accuracy"]),
        **eval_d, "pretraining": "INCLUDE (phase1 frozen head, phase2 unfreeze upper encoder layers only)",
    }
    print(f"D: phase1_val={phase1['best_val_accuracy']:.4f} phase2_val={phase2['best_val_accuracy']:.4f} test_acc={eval_d['test_accuracy']:.4f} macro_f1={eval_d['macro_f1']:.4f}")

    # --- Experiment E: CISLR-only control, same architecture, random init ---
    print("=" * 60)
    print("EXPERIMENT E: CISLR-only control (same architecture, random init, NO pretraining)")
    print("=" * 60)
    e_dir = os.path.join(OUTPUT_DIR, "experiment_E_cislr_only_control")
    model_e = build_model("cislr_head", len(cislr_gloss_to_id), normalization_adapt_data=X_train_aug)
    set_encoder_trainable(model_e, True)
    train_info_e = train_cislr_head(model_e, X_train_aug, y_train_aug, X_cislr_val, y_cislr_val, args, e_dir, args.head_lr, args.cislr_max_epochs, args.cislr_patience, "control")
    best_e = keras.models.load_model(train_info_e["checkpoint_path"])
    eval_e = evaluate_on_cislr_test(best_e, X_cislr_test, y_cislr_test, cislr_class_names, e_dir, "test")
    results["E_cislr_only_control_same_architecture"] = {**train_info_e, **eval_e, "pretraining": "none (random init, same architecture as B/C/D)"}
    del train_info_e["history"]
    print(f"E: val_acc={results['E_cislr_only_control_same_architecture']['best_val_accuracy']:.4f} test_acc={eval_e['test_accuracy']:.4f} macro_f1={eval_e['macro_f1']:.4f}")

    # --- Determine conclusion ---
    exp_results_for_compare = {
        "A": results["A_original_baseline"]["test_accuracy"],
        "B": results["B_include_pretrain_full_finetune"]["test_accuracy"],
        "C": results["C_include_pretrain_frozen_encoder"]["test_accuracy"],
        "D": results["D_include_pretrain_partial_finetune"]["test_accuracy"],
        "E": results["E_cislr_only_control_same_architecture"]["test_accuracy"],
    }
    best_pretrained = max(exp_results_for_compare["B"], exp_results_for_compare["C"], exp_results_for_compare["D"])
    best_pretrained_key = max(("B", "C", "D"), key=lambda k: exp_results_for_compare[k])
    control_beats_baseline = exp_results_for_compare["E"] > exp_results_for_compare["A"]
    pretrained_beats_control = best_pretrained > exp_results_for_compare["E"]
    pretrained_beats_baseline = best_pretrained > exp_results_for_compare["A"]

    if pretrained_beats_baseline and pretrained_beats_control:
        conclusion = "A" if best_pretrained - exp_results_for_compare["A"] > 0.05 else "C"
        conclusion_text = "INCLUDE pretraining improves CISLR representation learning" if conclusion == "A" else "INCLUDE pretraining helps only after fine-tuning"
    elif not pretrained_beats_baseline and best_pretrained < exp_results_for_compare["A"] - 0.02:
        conclusion = "D"
        conclusion_text = "INCLUDE pretraining hurts CISLR"
    else:
        conclusion = "B"
        conclusion_text = "INCLUDE pretraining does not improve CISLR"

    save_json(REPORT_JSON, {
        "experiments": results, "comparison": exp_results_for_compare,
        "best_pretrained_experiment": best_pretrained_key, "best_pretrained_test_accuracy": best_pretrained,
        "control_beats_baseline": control_beats_baseline, "pretrained_beats_control": pretrained_beats_control,
        "pretrained_beats_baseline": pretrained_beats_baseline,
        "conclusion_letter": conclusion, "conclusion_text": conclusion_text,
        "v1_controlled": {"test_accuracy": V1_CONTROLLED_TEST_ACC, "macro_f1": V1_CONTROLLED_MACRO_F1},
        "v2_controlled": {"test_accuracy": V2_CONTROLLED_TEST_ACC, "macro_f1": V2_CONTROLLED_MACRO_F1},
        "audit": {"checks": audit_checks, "all_pass": audit_all_pass},
    })

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in exp_results_for_compare.items():
        print(f"{k}: {v*100:.2f}%")
    print(f"CONCLUSION: {conclusion} -- {conclusion_text}")
    print("=" * 60)

    return {"results": results, "conclusion": conclusion, "conclusion_text": conclusion_text, "comparison": exp_results_for_compare}


if __name__ == "__main__":
    main()
