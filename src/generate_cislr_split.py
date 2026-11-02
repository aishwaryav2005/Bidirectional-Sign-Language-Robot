"""
Generates a deterministic, seeded, stratified-by-gloss train/val/test
split manifest for the processed CISLR dataset, restricted to glosses
with >= MIN_SAMPLES_PER_CLASS samples (default 5 -- see
reports/cislr_class_distribution_report.md and
reports/cislr_split_strategy_report.md for why).

IMPORTANT -- what this script does NOT do:
  - It does not copy or move any .npy sequence file. Splits are recorded
    as CSV/JSON manifests (video_id, gloss, category, sequence_path,
    split) that point at the existing files under
    data/public_processed/sequences/, which are left exactly as they are.
  - It does not claim signer-independent splitting. The local CISLR
    metadata (data/public_raw/dataset.csv: uid, gloss, duration,
    category) contains no signer/session/source-video identifier (see
    reports/cislr_split_strategy_report.md), so this is a seeded
    stratified RANDOM split by gloss -- the honest, documented fallback,
    not a signer-independent split. split_metadata.json records this
    limitation explicitly.

Split rule per eligible gloss (exactly, no exceptions):
    1 sample  -> validation
    1 sample  -> test
    remainder -> train   (>= 3 samples, since the minimum class size is 5)

Usage:
    python src/generate_cislr_split.py
    python src/generate_cislr_split.py --seed 42 --min-samples 5
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DEFAULT_SEED = 42
DEFAULT_MIN_SAMPLES = 5
DEFAULT_OUTPUT_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_82class")

_MANIFEST_FIELDS = ["video_id", "gloss", "category", "sequence_path", "split"]


def load_success_rows(metadata_path):
    """Reads processing_metadata.csv and returns only successfully
    processed rows, each augmented with a parsed 'category' field."""
    rows = []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["processing_status"] != "success":
                continue
            try:
                extra = json.loads(row["source_metadata"]) if row["source_metadata"] else {}
            except json.JSONDecodeError:
                extra = {}
            row["category"] = extra.get("category", "")
            rows.append(row)
    return rows


def group_by_gloss(rows):
    """{gloss: [row, ...]} with each gloss's rows sorted by video_id for
    deterministic input order to the random sampler."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    for gloss in groups:
        groups[gloss].sort(key=lambda r: r["video_id"])
    return groups


def select_eligible_glosses(groups, min_samples):
    """Returns {gloss: [rows]} restricted to glosses with >= min_samples
    rows, iterated/returned in sorted-gloss order for determinism."""
    return {gloss: rows for gloss, rows in sorted(groups.items()) if len(rows) >= min_samples}


def generate_split(metadata_path, seed=DEFAULT_SEED, min_samples=DEFAULT_MIN_SAMPLES):
    """Returns {"train": [...], "val": [...], "test": [...]} of manifest
    row dicts (video_id, gloss, category, sequence_path, split), fully
    deterministic for a given (metadata_path contents, seed, min_samples).
    Exactly 1 val + 1 test sample per eligible gloss; everything else
    (>= min_samples - 2, always >= 1 for min_samples >= 3) goes to train.
    """
    rows = load_success_rows(metadata_path)
    groups = group_by_gloss(rows)
    eligible = select_eligible_glosses(groups, min_samples)

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}

    for gloss in sorted(eligible.keys()):
        gloss_rows = eligible[gloss]  # already sorted by video_id
        val_row, test_row = rng.sample(gloss_rows, 2)
        val_id, test_id = val_row["video_id"], test_row["video_id"]

        for row in gloss_rows:
            if row["video_id"] == val_id:
                target = "val"
            elif row["video_id"] == test_id:
                target = "test"
            else:
                target = "train"
            split[target].append({
                "video_id": row["video_id"],
                "gloss": gloss,
                "category": row["category"],
                "sequence_path": row["sequence_path"],
                "split": target,
            })

    return split, eligible


def write_manifests(split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for name in ("train", "val", "test"):
        path = os.path.join(output_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
            writer.writeheader()
            for row in split[name]:
                writer.writerow(row)
        paths[name] = path
    return paths


def write_split_metadata(output_dir, split, eligible, seed, min_samples, metadata_path):
    n_classes = len(eligible)
    total = sum(len(v) for v in split.values())
    info = {
        "dataset_name": "CISLR (processed via this project's Phase 2 pipeline)",
        "vocabulary_rule": f">= {min_samples} samples per gloss",
        "num_classes": n_classes,
        "total_samples": total,
        "train_count": len(split["train"]),
        "val_count": len(split["val"]),
        "test_count": len(split["test"]),
        "random_seed": seed,
        "split_strategy": (
            "Deterministic seeded stratified split by gloss: for each "
            "eligible gloss, exactly 1 sample is randomly assigned to "
            "validation and exactly 1 to test (via random.Random(seed)."
            "sample on the gloss's video_ids sorted ascending, iterated "
            "in sorted-gloss order); all remaining samples for that "
            "gloss go to training. Every eligible gloss therefore "
            "appears at least once in train, exactly once in "
            "validation, and exactly once in test."
        ),
        "signer_independent_splitting": False,
        "signer_independent_splitting_note": (
            "Signer-independent splitting is NOT available: the local "
            "CISLR metadata (data/public_raw/dataset.csv) contains only "
            "uid, gloss, duration, category -- no signer, session, or "
            "verified source-video grouping identifier. video_id is not "
            "used as a signer proxy. See "
            "reports/cislr_split_strategy_report.md for the full "
            "investigation. This is a seeded stratified RANDOM split by "
            "gloss, not a signer-independent split; treat reported "
            "model accuracy accordingly."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata_files": [
            os.path.relpath(metadata_path, config.PROJECT_ROOT),
            os.path.relpath(os.path.join(config.PUBLIC_RAW_DIR, "dataset.csv"), config.PROJECT_ROOT),
        ],
    }
    path = os.path.join(output_dir, "split_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path, info


def validate_manifests(output_dir, project_root=None):
    """Re-reads the written manifests from disk and checks: video_id
    uniqueness within and across splits, every class present in all
    three splits, every referenced sequence_path exists on disk. Returns
    a report dict; never modifies anything. `project_root` is the base
    directory `sequence_path` values are relative to (default:
    config.PROJECT_ROOT; overridable for tests using synthetic fixtures)."""
    if project_root is None:
        project_root = config.PROJECT_ROOT
    loaded = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(output_dir, f"{name}.csv"), newline="", encoding="utf-8") as f:
            loaded[name] = list(csv.DictReader(f))

    report = {
        "counts": {name: len(rows) for name, rows in loaded.items()},
        "duplicate_ids_within_split": {},
        "cross_split_overlap": [],
        "missing_sequence_files": [],
        "classes_missing_from_a_split": [],
    }

    all_ids_by_split = {}
    for name, rows in loaded.items():
        ids = [r["video_id"] for r in rows]
        all_ids_by_split[name] = set(ids)
        dupes = {vid for vid in ids if ids.count(vid) > 1}
        if dupes:
            report["duplicate_ids_within_split"][name] = sorted(dupes)

    overlap_train_val = all_ids_by_split["train"] & all_ids_by_split["val"]
    overlap_train_test = all_ids_by_split["train"] & all_ids_by_split["test"]
    overlap_val_test = all_ids_by_split["val"] & all_ids_by_split["test"]
    for pair, s in [("train/val", overlap_train_val), ("train/test", overlap_train_test), ("val/test", overlap_val_test)]:
        if s:
            report["cross_split_overlap"].append({"pair": pair, "video_ids": sorted(s)})

    for name, rows in loaded.items():
        for row in rows:
            full_path = os.path.join(project_root, row["sequence_path"])
            if not os.path.exists(full_path):
                report["missing_sequence_files"].append({"split": name, "video_id": row["video_id"], "path": row["sequence_path"]})

    classes_by_split = {name: {r["gloss"] for r in rows} for name, rows in loaded.items()}
    all_classes = classes_by_split["train"] | classes_by_split["val"] | classes_by_split["test"]
    for gloss in sorted(all_classes):
        missing_from = [name for name in ("train", "val", "test") if gloss not in classes_by_split[name]]
        if missing_from:
            report["classes_missing_from_a_split"].append({"gloss": gloss, "missing_from": missing_from})

    report["passed"] = (
        not report["duplicate_ids_within_split"]
        and not report["cross_split_overlap"]
        and not report["missing_sequence_files"]
        and not report["classes_missing_from_a_split"]
    )
    return report, loaded


def build_per_class_report(loaded):
    per_class = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for name, rows in loaded.items():
        for row in rows:
            per_class[row["gloss"]][name] += 1
    return dict(sorted(per_class.items()))


def main():
    parser = argparse.ArgumentParser(description="Generate a deterministic, seeded, stratified train/val/test split manifest for the processed CISLR dataset.")
    parser.add_argument("--metadata-path", default=config.PUBLIC_METADATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = parser.parse_args()

    print("=" * 60)
    print("CISLR Split Generation")
    print("=" * 60)
    print(f"Metadata:    {args.metadata_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Seed:        {args.seed}")
    print(f"Min samples/class: {args.min_samples}")
    print("=" * 60)

    split, eligible = generate_split(args.metadata_path, seed=args.seed, min_samples=args.min_samples)
    manifest_paths = write_manifests(split, args.output_dir)
    metadata_json_path, split_info = write_split_metadata(
        args.output_dir, split, eligible, args.seed, args.min_samples, args.metadata_path
    )

    print(f"Wrote: {manifest_paths['train']}")
    print(f"Wrote: {manifest_paths['val']}")
    print(f"Wrote: {manifest_paths['test']}")
    print(f"Wrote: {metadata_json_path}")
    print("=" * 60)

    validation_report, loaded = validate_manifests(args.output_dir)
    per_class = build_per_class_report(loaded)

    print("FINAL REPORT")
    print("=" * 60)
    print(f"Classes:            {split_info['num_classes']}")
    print(f"Total samples:      {split_info['total_samples']}")
    print(f"Train count:        {split_info['train_count']}")
    print(f"Validation count:   {split_info['val_count']}")
    print(f"Test count:         {split_info['test_count']}")
    print(f"Random seed:        {split_info['random_seed']}")
    print("-" * 60)
    print("Per-class train/val/test counts:")
    for gloss, counts in per_class.items():
        print(f"  {gloss:35s} train={counts['train']:2d}  val={counts['val']:2d}  test={counts['test']:2d}")
    print("-" * 60)
    print(f"Duplicate IDs within a split:   {validation_report['duplicate_ids_within_split'] or 'none'}")
    print(f"Cross-split video_id overlap:   {validation_report['cross_split_overlap'] or 'none'}")
    print(f"Missing sequence files:         {validation_report['missing_sequence_files'] or 'none'}")
    print(f"Classes missing from a split:   {validation_report['classes_missing_from_a_split'] or 'none'}")
    print("=" * 60)
    print("RESULT: " + ("PASS" if validation_report["passed"] else "FAIL"))
    print("=" * 60)

    return 0 if validation_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
