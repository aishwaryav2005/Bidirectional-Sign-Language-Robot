"""
Trains the Sign->Speech DEMO classifier: a controlled, 20-class isolated-sign
BiLSTM (or GRU) classifier on the INCLUDE "People" category vocabulary
selected by src/build_sign_demo_dataset.py.

Completely isolated from CISLR/INCLUDE/iSign/fingerspelling experiments:
reads only data/sign_demo/splits/{train,val,test}.csv (built from
data/external_processed/include/, read-only) and writes only to
models/sign_demo/, reports/sign_demo/, results/sign_demo/.

Reuses the existing feature format (60,147) and this project's established
training conventions (src/train_bilstm_baseline.py): keras.Input ->
Normalization -> Bidirectional LSTM/GRU -> Dropout -> Dense(softmax),
Adam, EarlyStopping, seed 42. Adds, specifically for this demo (per the
task's explicit requirements): feature-level augmentation (Gaussian
coordinate noise, temporal jitter, temporal scaling, feature masking),
class-balanced training, and ReduceLROnPlateau.

Usage:
    python src/train_sign_demo.py --seed 42 --cell bilstm
    python src/train_sign_demo.py --seed 42 --cell gru
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SPLITS_DIR = os.path.join(config.DATA_DIR, "sign_demo", "splits")
MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models", "sign_demo")
REPORT_DIR = os.path.join(config.PROJECT_ROOT, "reports", "sign_demo")
RESULTS_DIR = os.path.join(config.PROJECT_ROOT, "results", "sign_demo")

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH)  # (60, 147)


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_class_mapping(*manifests):
    labels = set()
    for m in manifests:
        labels.update(row["label"] for row in m)
    sorted_labels = sorted(labels)
    label_to_id = {l: i for i, l in enumerate(sorted_labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_to_id, id_to_label


def load_split_arrays(rows, label_to_id):
    X_list, y_list, video_ids = [], [], []
    skipped = []
    for row in rows:
        path = os.path.join(config.PROJECT_ROOT, row["sequence_path"])
        try:
            arr = np.load(path)
        except OSError as e:
            # OneDrive cloud-placeholder files that fail to hydrate even after
            # repeated retries (seen for a small number of INCLUDE files during
            # this session) are skipped rather than crashing the whole run.
            skipped.append((row["video_id"], str(e)))
            continue
        if arr.shape != EXPECTED_SHAPE:
            raise ValueError(f"{row['video_id']}: expected {EXPECTED_SHAPE}, got {arr.shape}")
        X_list.append(arr)
        y_list.append(label_to_id[row["label"]])
        video_ids.append(row["video_id"])
    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable file(s):")
        for vid, err in skipped:
            print(f"  - {vid}: {err}")
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, video_ids


# ---------------------------------------------------------------------------
# Feature-level augmentation (applied to TRAIN only, never val/test).
# Each one preserves sign semantics: small coordinate jitter, mild temporal
# resampling, and masking a few frames -- none of these change *which* sign
# is being performed.
# ---------------------------------------------------------------------------

def augment_gaussian_noise(seq, sigma=0.01, rng=np.random):
    noise = rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)
    return seq + noise


def augment_temporal_scaling(seq, rng=np.random, scale_range=(0.85, 1.15)):
    """Resample along time by a random scale factor, then re-fit to 60
    frames (same uniform-subsampling approach as temporal_sampling.py)."""
    from temporal_sampling import sample_frame_indices
    T = seq.shape[0]
    scale = rng.uniform(*scale_range)
    virtual_len = max(2, int(round(T * scale)))
    stretch_idx = sample_frame_indices(T, virtual_len)
    stretched = seq[stretch_idx]
    back_idx = sample_frame_indices(virtual_len, T)
    return stretched[back_idx]


def augment_temporal_jitter(seq, rng=np.random, max_shift=2):
    """Small random per-frame index jitter (within +-max_shift, clipped),
    simulating minor frame-sampling timing differences."""
    T = seq.shape[0]
    shifts = rng.randint(-max_shift, max_shift + 1, size=T)
    idx = np.clip(np.arange(T) + shifts, 0, T - 1)
    idx = np.sort(idx)  # preserve temporal order
    return seq[idx]


def augment_feature_masking(seq, rng=np.random, max_masked_frames=3):
    """Zero out a few randomly chosen frames (simulates brief detection
    dropout) -- never masks more than max_masked_frames of 60."""
    seq = seq.copy()
    n_mask = rng.randint(0, max_masked_frames + 1)
    if n_mask == 0:
        return seq
    frame_idx = rng.choice(seq.shape[0], size=n_mask, replace=False)
    seq[frame_idx] = 0.0
    return seq


def augment_one(seq, rng):
    """Applies a random subset of the four augmentations to one (60,147)
    sequence. At least noise is always applied (cheapest, safest)."""
    out = augment_gaussian_noise(seq, sigma=0.01, rng=rng)
    if rng.random() < 0.6:
        out = augment_temporal_scaling(out, rng=rng)
    if rng.random() < 0.4:
        out = augment_temporal_jitter(out, rng=rng)
    if rng.random() < 0.3:
        out = augment_feature_masking(out, rng=rng)
    return out.astype(np.float32)


def build_augmented_training_set(X_train, y_train, augment_factor, seed):
    """Returns (X_aug, y_aug) = original training set + augment_factor
    additional augmented copies of EVERY training sample. TRAIN ONLY."""
    rng = np.random.RandomState(seed)
    X_parts, y_parts = [X_train], [y_train]
    for _ in range(augment_factor):
        X_aug = np.stack([augment_one(X_train[i], rng) for i in range(X_train.shape[0])])
        X_parts.append(X_aug)
        y_parts.append(y_train)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes, cell="bilstm", seq_len=config.SEQUENCE_LENGTH, feat_dim=config.FEATURE_VECTOR_LENGTH,
                units1=64, units2=32, dropout_rate=0.3, learning_rate=1e-3, normalization_adapt_data=None):
    import keras

    inputs = keras.Input(shape=(seq_len, feat_dim), name="landmark_sequence")
    norm_layer = keras.layers.Normalization(axis=-1, name="input_normalization")
    if normalization_adapt_data is not None:
        norm_layer.adapt(normalization_adapt_data)
    x = norm_layer(inputs)

    if cell == "bilstm":
        x = keras.layers.Bidirectional(keras.layers.LSTM(units1, return_sequences=True), name="rnn_1")(x)
        x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(units2), name="rnn_2")(x)
    elif cell == "gru":
        x = keras.layers.GRU(units1, return_sequences=True, name="rnn_1")(x)
        x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = keras.layers.GRU(units2, name="rnn_2")(x)
    else:
        raise ValueError(f"unknown cell type: {cell}")

    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="sign_softmax")(x)

    model = keras.Model(inputs, outputs, name=f"sign_demo_{cell}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, class_names):
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, f1_score,
        classification_report, confusion_matrix,
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report_str = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        per_class_accuracy[name] = float((y_pred[mask] == i).mean()) if mask.sum() > 0 else None

    return {
        "accuracy": accuracy, "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1), "weighted_f1": weighted_f1,
        "classification_report": report_str, "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy,
    }


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell", choices=["bilstm", "gru"], default="bilstm")
    parser.add_argument("--units1", type=int, default=64)
    parser.add_argument("--units2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--augment-factor", type=int, default=4, help="extra augmented copies of the train set, 0=disabled")
    parser.add_argument("--use-class-weight", action="store_true", default=True)
    parser.add_argument("--tag", default="", help="suffix for output filenames, to compare runs without overwriting")
    args = parser.parse_args()

    set_global_seed(args.seed)
    import keras

    suffix = f"_{args.tag}" if args.tag else f"_{args.cell}"
    best_model_path = os.path.join(MODEL_DIR, f"best_model{suffix}.keras")
    if os.path.exists(best_model_path):
        raise FileExistsError(f"Refusing to overwrite existing {best_model_path} -- use a different --tag.")

    train_rows = load_manifest(os.path.join(SPLITS_DIR, "train.csv"))
    val_rows = load_manifest(os.path.join(SPLITS_DIR, "val.csv"))
    test_rows = load_manifest(os.path.join(SPLITS_DIR, "test.csv"))

    label_to_id, id_to_label = build_class_mapping(train_rows, val_rows, test_rows)
    num_classes = len(label_to_id)
    class_names = [id_to_label[i] for i in range(num_classes)]

    print("=" * 60)
    print(f"Sign Demo Training -- cell={args.cell}  seed={args.seed}  augment_factor={args.augment_factor}")
    print("=" * 60)
    print(f"Classes: {num_classes}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    X_train_raw, y_train_raw, train_ids = load_split_arrays(train_rows, label_to_id)
    X_val, y_val, val_ids = load_split_arrays(val_rows, label_to_id)
    X_test, y_test, test_ids = load_split_arrays(test_rows, label_to_id)

    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    if args.augment_factor > 0:
        X_train, y_train = build_augmented_training_set(X_train_raw, y_train_raw, args.augment_factor, args.seed)
        print(f"Augmented train set: {X_train_raw.shape[0]} -> {X_train.shape[0]} samples "
              f"(factor={args.augment_factor}: Gaussian noise + temporal scaling/jitter + feature masking)")
    else:
        X_train, y_train = X_train_raw, y_train_raw

    class_weight = None
    if args.use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes_present, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes_present, weights)}
        print("Class weights (balanced):", {class_names[k]: round(v, 3) for k, v in class_weight.items()})

    model = build_model(
        num_classes, cell=args.cell, units1=args.units1, units2=args.units2,
        dropout_rate=args.dropout, learning_rate=args.learning_rate,
        normalization_adapt_data=X_train_raw,  # adapt on RAW train only, never augmented/val/test
    )
    try:
        model.summary(print_fn=print)
    except Exception as e:
        print(f"(model.summary() display failed non-fatally: {e})")
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=max(5, args.patience // 3), min_lr=1e-6, verbose=1),
    ]

    print("-" * 60)
    print("Training...")
    start_time = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.max_epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=callbacks, class_weight=class_weight, verbose=2,
    )
    train_elapsed = time.perf_counter() - start_time
    epochs_trained = len(history.history["loss"])
    best_val_accuracy = float(max(history.history["val_accuracy"]))

    final_model_path = os.path.join(MODEL_DIR, f"final_model{suffix}.keras")
    model.save(final_model_path)
    print(f"Training finished in {train_elapsed:.2f}s ({epochs_trained} epochs). Best val accuracy: {best_val_accuracy:.4f}")

    # ---- train-set accuracy (on RAW, unaugmented train set, for honest reporting) ----
    best_model = keras.models.load_model(best_model_path)
    train_pred = np.argmax(best_model.predict(X_train_raw, verbose=0), axis=1)
    train_accuracy = float((train_pred == y_train_raw).mean())

    val_pred = np.argmax(best_model.predict(X_val, verbose=0), axis=1)
    val_results = evaluate_predictions(y_val, val_pred, class_names)

    test_pred_probs = best_model.predict(X_test, verbose=0)
    test_pred = np.argmax(test_pred_probs, axis=1)
    test_results = evaluate_predictions(y_test, test_pred, class_names)

    print("=" * 60)
    print("FINAL RESULTS")
    print(f"Train accuracy (raw, unaugmented): {train_accuracy:.4f}")
    print(f"Val accuracy:   {val_results['accuracy']:.4f}")
    print(f"Test accuracy:  {test_results['accuracy']:.4f}")
    print(f"Test macro F1:  {test_results['macro_f1']:.4f}")
    print("=" * 60)

    # ---- save artifacts ----
    save_json(os.path.join(MODEL_DIR, f"class_mapping{suffix}.json"), {
        "label_to_id": label_to_id, "id_to_label": {str(k): v for k, v in id_to_label.items()}, "num_classes": num_classes,
    })
    save_json(os.path.join(MODEL_DIR, f"model_config{suffix}.json"), {
        "cell": args.cell, "input_shape": [config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH],
        "units1": args.units1, "units2": args.units2, "dropout": args.dropout, "num_classes": num_classes,
        "parameter_count": int(param_count),
    })
    import tensorflow as tf
    training_config = {
        "seed": args.seed, "cell": args.cell, "optimizer": "Adam", "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
        "reduce_lr_on_plateau": True, "class_weight_balanced": args.use_class_weight,
        "augment_factor": args.augment_factor,
        "augmentations": ["gaussian_coordinate_noise(sigma=0.01)", "temporal_scaling(0.85-1.15x)",
                           "temporal_jitter(+-2 frames)", "feature_masking(<=3 frames)"],
        "train_samples_raw": int(X_train_raw.shape[0]), "train_samples_augmented": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]), "test_samples": int(X_test.shape[0]),
        "epochs_trained": epochs_trained, "best_val_accuracy": best_val_accuracy,
        "training_time_seconds": round(train_elapsed, 3),
        "model_parameter_count": int(param_count),
        "framework_versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "test_set_usage": "Test set was not used for training, validation, early stopping, or model selection -- evaluated once after training.",
        "signer_independent_evaluation": False,
        "signer_independent_note": "INCLUDE provides no signer/participant identifier; this is a seeded "
                                    "stratified random split by label, not signer-independent.",
    }
    save_json(os.path.join(MODEL_DIR, f"training_config{suffix}.json"), training_config)
    save_json(os.path.join(MODEL_DIR, f"training_history{suffix}.json"),
              {k: [float(v) for v in vals] for k, vals in history.history.items()})

    save_json(os.path.join(REPORT_DIR, f"metrics{suffix}.json"), {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_results["accuracy"], "val_macro_f1": val_results["macro_f1"],
        "test_accuracy": test_results["accuracy"], "test_macro_precision": test_results["macro_precision"],
        "test_macro_recall": test_results["macro_recall"], "test_macro_f1": test_results["macro_f1"],
        "test_weighted_f1": test_results["weighted_f1"],
        "per_class_accuracy": test_results["per_class_accuracy"],
        "num_classes": num_classes, "train_samples": len(train_rows), "val_samples": len(val_rows), "test_samples": len(test_rows),
        "target_85pct_achieved": test_results["accuracy"] >= 0.85,
    })

    with open(os.path.join(REPORT_DIR, f"classification_report{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(test_results["classification_report"])
    np.savetxt(os.path.join(REPORT_DIR, f"confusion_matrix{suffix}.csv"), test_results["confusion_matrix"], fmt="%d", delimiter=",")

    with open(os.path.join(REPORT_DIR, f"test_predictions{suffix}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "true_label", "predicted_label", "confidence", "correct"])
        for vid, true_i, pred_i, probs in zip(test_ids, y_test, test_pred, test_pred_probs):
            writer.writerow([vid, class_names[true_i], class_names[pred_i], round(float(probs[pred_i]), 4), true_i == pred_i])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(test_results["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Sign Demo Confusion Matrix (test, {args.cell})")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["accuracy"], label="train_acc")
        ax.plot(history.history["val_accuracy"], label="val_acc")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
        ax.set_title(f"Sign Demo Training History ({args.cell})")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"training_history{suffix}.png"), dpi=150)
        plt.close(fig)
        plots_ok = True
    except Exception as e:
        print(f"(plot generation failed non-fatally, skipping PNG outputs: {e})")
        plots_ok = False

    print(f"\nArtifacts saved under {MODEL_DIR}, {REPORT_DIR}, {RESULTS_DIR} (plots: {'ok' if plots_ok else 'skipped'})")
    return {"test_accuracy": test_results["accuracy"], "test_macro_f1": test_results["macro_f1"],
            "val_accuracy": val_results["accuracy"], "train_accuracy": train_accuracy}


if __name__ == "__main__":
    main()
