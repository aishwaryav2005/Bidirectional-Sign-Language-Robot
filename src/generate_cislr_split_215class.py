"""
Generates the expanded (>= 4 samples/class) CISLR split, recommended in
reports/cislr_dataset_expansion_analysis.md. This is a NEW, isolated
script -- it does not modify src/generate_cislr_split.py, and the
existing 82-class split (data/splits/cislr_82class/) is never touched by
running this. It reuses (via import, unmodified) that module's
`load_success_rows`, `group_by_gloss`, `select_eligible_glosses`,
`write_manifests`, `validate_manifests`, and `build_per_class_report`.

Why a new script rather than just calling generate_split(min_samples=4):
a duplicate-content investigation (reports/cislr_split_strategy_report.md,
re-verified fresh here) found that the gloss "green" has 4 samples, two
of which (D9ihUabRKIQ_2, wwQV7dGvw0s_2) are byte-identical RAW SOURCE
VIDEOS. The naive per-gloss "pick 2 random samples for val+test" rule
could put one duplicate-pair member in validation and the other in test
(or train), leaking near-identical content across splits. This script
adds duplicate-awareness: within each eligible gloss, validation and
test are chosen only from samples that are NOT part of a same-gloss
duplicate group (falling back, with an explicitly logged conflict, to
one representative per duplicate group only if too few non-duplicated
samples exist -- see `generate_duplicate_aware_split`). Duplicate-group
members that aren't chosen for val/test always land in the SAME split
(train), never split across train/val/test.

Note: the two "green" videos are byte-identical as raw video files, but
their MediaPipe-derived .npy sequences turned out NOT to be byte-
identical (re-verified here) -- our pipeline assigns a running,
cross-video timestamp counter to MediaPipe's VIDEO-mode detector for
batch efficiency (see src/preprocess_public_dataset.py), and MediaPipe's
temporal smoothing is sensitive to that timestamp context even for
pixel-identical input. Despite the derived features not being literally
byte-identical, this script treats them as a duplicate pair anyway (the
scientifically safer choice) because they are the same real-world
recording/signer/take, which is still a meaningful near-duplicate risk
even where the exact feature vectors differ slightly.

Usage:
    python src/generate_cislr_split_215class.py
    python src/generate_cislr_split_215class.py --seed 42 --min-samples 4
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from generate_cislr_split import (
    load_success_rows, group_by_gloss, select_eligible_glosses,
    write_manifests, validate_manifests, build_per_class_report,
)

DEFAULT_SEED = 42
DEFAULT_MIN_SAMPLES = 4
DEFAULT_OUTPUT_DIR = os.path.join(config.DATA_DIR, "splits", "cislr_215class")
DEFAULT_VIDEOS_ROOT = os.path.join(config.PUBLIC_RAW_DIR, "CISLR_v1.5-a_videos", "extracted")


def _md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_groups(rows, videos_root):
    """Hashes each row's RAW SOURCE VIDEO (original_filename) and groups
    video_ids that share identical content. Returns
    (dup_group_of_video_id: {video_id: group_key or None},
     duplicate_groups: [[video_id, ...], ...] for groups with >1 member,
     each sorted; group order sorted by first member for determinism)."""
    hash_to_ids = defaultdict(list)
    for row in rows:
        video_path = os.path.join(videos_root, row["original_filename"])
        file_hash = _md5_of_file(video_path)
        hash_to_ids[file_hash].append(row["video_id"])

    dup_group_of_video_id = {}
    duplicate_groups = []
    for file_hash, ids in hash_to_ids.items():
        if len(ids) > 1:
            group = sorted(ids)
            duplicate_groups.append(group)
            for vid in group:
                dup_group_of_video_id[vid] = file_hash

    duplicate_groups.sort(key=lambda g: g[0])
    return dup_group_of_video_id, duplicate_groups


def generate_duplicate_aware_split(metadata_path, videos_root, seed=DEFAULT_SEED, min_samples=DEFAULT_MIN_SAMPLES):
    """Same contract as generate_cislr_split.generate_split, plus
    duplicate-content safety: validation and test are chosen only from
    samples with no same-gloss duplicate-content partner, wherever
    enough such samples exist. Returns
    (split, eligible, duplicate_groups, conflicts)."""
    rows = load_success_rows(metadata_path)
    groups = group_by_gloss(rows)
    eligible = select_eligible_glosses(groups, min_samples)
    all_eligible_rows = [r for rs in eligible.values() for r in rs]

    dup_group_of_video_id, duplicate_groups = find_duplicate_groups(all_eligible_rows, videos_root)

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    conflicts = []

    for gloss in sorted(eligible.keys()):
        gloss_rows = eligible[gloss]  # sorted by video_id

        dup_key_to_members = defaultdict(list)
        safe_rows = []
        for row in gloss_rows:
            key = dup_group_of_video_id.get(row["video_id"])
            if key is None:
                safe_rows.append(row)
            else:
                dup_key_to_members[key].append(row)

        if len(safe_rows) >= 2:
            candidates = safe_rows
        else:
            # Conflict: not enough duplicate-free samples in this gloss to
            # fill val+test without ever splitting a duplicate group.
            # Scientifically safer fallback: allow ONE representative per
            # duplicate group to be a val/test candidate (never more than
            # one, so two members of the same group can never land in
            # different splits), explicitly logged rather than silently
            # dropping data or silently accepting a leak.
            conflicts.append({
                "gloss": gloss,
                "safe_sample_count": len(safe_rows),
                "resolution": "used one representative per duplicate group as an additional val/test candidate",
            })
            candidates = list(safe_rows)
            for key, members in dup_key_to_members.items():
                candidates.append(sorted(members, key=lambda r: r["video_id"])[0])

        val_row, test_row = rng.sample(candidates, 2)
        val_id, test_id = val_row["video_id"], test_row["video_id"]

        for row in gloss_rows:
            vid = row["video_id"]
            if vid == val_id:
                target = "val"
            elif vid == test_id:
                target = "test"
            else:
                target = "train"
            split[target].append({
                "video_id": vid,
                "gloss": gloss,
                "category": row["category"],
                "sequence_path": row["sequence_path"],
                "split": target,
            })

    return split, eligible, duplicate_groups, conflicts


def write_split_metadata_215(output_dir, split, eligible, duplicate_groups, conflicts,
                              seed, min_samples, metadata_path, videos_root):
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
            "Deterministic seeded stratified split by gloss, duplicate-content "
            "aware: for each eligible gloss, validation and test are each "
            "assigned exactly 1 sample, chosen via random.Random(seed).sample "
            "restricted to samples with no same-gloss duplicate-content partner "
            "wherever >=2 such samples exist for that gloss (see "
            "generate_duplicate_aware_split docstring for the fallback used "
            "when fewer than 2 exist). All remaining samples for that gloss go "
            "to training. Every eligible gloss appears at least once in train, "
            "exactly once in validation, and exactly once in test."
        ),
        "duplicate_handling": {
            "method": "MD5 hash of each sample's RAW SOURCE VIDEO file (original_filename), restricted to the eligible (>= min_samples) rows",
            "duplicate_groups_found": duplicate_groups,
            "num_duplicate_groups": len(duplicate_groups),
            "conflicts_requiring_fallback": conflicts,
            "note": (
                "The known 'green' duplicate pair (D9ihUabRKIQ_2, wwQV7dGvw0s_2) "
                "is byte-identical at the RAW VIDEO level but the derived .npy "
                "sequences are NOT byte-identical (MediaPipe's VIDEO-mode temporal "
                "smoothing is sensitive to this pipeline's cross-video running "
                "timestamp counter, even for pixel-identical input). Both are "
                "treated as duplicates anyway (same real-world recording), the "
                "scientifically safer choice -- both members are kept in the same "
                "split (train), never split across train/val/test."
            ),
        },
        "signer_independent_splitting": False,
        "signer_independent_splitting_note": (
            "Signer-independent splitting is NOT available: the local CISLR "
            "metadata (data/public_raw/dataset.csv) contains only uid, gloss, "
            "duration, category -- no signer, session, or verified source-video "
            "grouping identifier beyond the exact-duplicate check performed "
            "here. video_id is not used as a signer proxy. See "
            "reports/cislr_split_strategy_report.md for the full investigation. "
            "This is a seeded stratified RANDOM split by gloss, not a "
            "signer-independent split; treat reported model accuracy accordingly."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata_files": [
            os.path.relpath(metadata_path, config.PROJECT_ROOT),
            os.path.relpath(os.path.join(config.PUBLIC_RAW_DIR, "dataset.csv"), config.PROJECT_ROOT),
            os.path.relpath(videos_root, config.PROJECT_ROOT) + " (raw videos, hashed for duplicate detection only)",
        ],
        "relationship_to_82class_split": (
            "Independent of data/splits/cislr_82class/, which is NOT modified, "
            "read, or regenerated by this script. This is a separate, larger "
            "(>= 4 samples/class) vocabulary."
        ),
    }
    path = os.path.join(output_dir, "split_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path, info


def validate_manifests_with_duplicates(output_dir, videos_root, metadata_path, project_root=None):
    """Extends generate_cislr_split.validate_manifests with a cross-split
    duplicate-content check (by raw-video hash). Never modifies anything.

    The written manifest CSVs only carry (video_id, gloss, category,
    sequence_path, split) -- not original_filename -- so the raw video
    filename needed for hashing is looked up from `metadata_path`
    (the source processing_metadata.csv) by video_id."""
    base_report, loaded = validate_manifests(output_dir, project_root=project_root)

    meta_rows = load_success_rows(metadata_path)
    video_id_to_filename = {r["video_id"]: r["original_filename"] for r in meta_rows}

    all_rows = [(name, row) for name, rows in loaded.items() for row in rows]
    rows_with_filename = [
        {"video_id": row["video_id"], "original_filename": video_id_to_filename[row["video_id"]]}
        for _, row in all_rows
    ]
    dup_group_of_video_id, _ = find_duplicate_groups(rows_with_filename, videos_root)

    split_of_video_id = {row["video_id"]: name for name, row in all_rows}
    group_members_by_key = defaultdict(list)
    for vid, key in dup_group_of_video_id.items():
        group_members_by_key[key].append(vid)

    cross_split_duplicate_content = []
    for key, members in group_members_by_key.items():
        splits_seen = {split_of_video_id[vid] for vid in members if vid in split_of_video_id}
        if len(splits_seen) > 1:
            cross_split_duplicate_content.append({
                "video_ids": sorted(members),
                "splits": sorted(splits_seen),
            })

    base_report["cross_split_duplicate_content"] = cross_split_duplicate_content
    base_report["passed"] = base_report["passed"] and not cross_split_duplicate_content
    return base_report, loaded


def main():
    parser = argparse.ArgumentParser(description="Generate the duplicate-aware, expanded (>=4 samples/class) CISLR split.")
    parser.add_argument("--metadata-path", default=config.PUBLIC_METADATA_PATH)
    parser.add_argument("--videos-root", default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = parser.parse_args()

    print("=" * 60)
    print("CISLR Expanded Split Generation (duplicate-aware)")
    print("=" * 60)
    print(f"Metadata:    {args.metadata_path}")
    print(f"Videos root: {args.videos_root}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Seed:        {args.seed}")
    print(f"Min samples/class: {args.min_samples}")
    print("=" * 60)

    split, eligible, duplicate_groups, conflicts = generate_duplicate_aware_split(
        args.metadata_path, args.videos_root, seed=args.seed, min_samples=args.min_samples
    )
    manifest_paths = write_manifests(split, args.output_dir)
    metadata_json_path, split_info = write_split_metadata_215(
        args.output_dir, split, eligible, duplicate_groups, conflicts,
        args.seed, args.min_samples, args.metadata_path, args.videos_root,
    )

    print(f"Wrote: {manifest_paths['train']}")
    print(f"Wrote: {manifest_paths['val']}")
    print(f"Wrote: {manifest_paths['test']}")
    print(f"Wrote: {metadata_json_path}")
    print(f"Duplicate groups found: {len(duplicate_groups)} -- {duplicate_groups}")
    if conflicts:
        print(f"Conflicts requiring fallback handling: {conflicts}")
    else:
        print("No conflicts: every duplicate group had enough safe samples to fill val+test normally.")
    print("=" * 60)

    validation_report, loaded = validate_manifests_with_duplicates(args.output_dir, args.videos_root, args.metadata_path)
    per_class = build_per_class_report(loaded)

    min_train_ok = all(counts["train"] >= 2 for counts in per_class.values())

    print("FINAL REPORT")
    print("=" * 60)
    print(f"Classes:            {split_info['num_classes']}")
    print(f"Total samples:      {split_info['total_samples']}")
    print(f"Train count:        {split_info['train_count']}")
    print(f"Validation count:   {split_info['val_count']}")
    print(f"Test count:         {split_info['test_count']}")
    print(f"Random seed:        {split_info['random_seed']}")
    print(f"Every class has >= 2 train samples: {min_train_ok}")
    print("-" * 60)
    print(f"Duplicate IDs within a split:     {validation_report['duplicate_ids_within_split'] or 'none'}")
    print(f"Cross-split video_id overlap:     {validation_report['cross_split_overlap'] or 'none'}")
    print(f"Cross-split duplicate CONTENT:    {validation_report['cross_split_duplicate_content'] or 'none'}")
    print(f"Missing sequence files:           {validation_report['missing_sequence_files'] or 'none'}")
    print(f"Classes missing from a split:     {validation_report['classes_missing_from_a_split'] or 'none'}")
    print("=" * 60)
    print("RESULT: " + ("PASS" if (validation_report["passed"] and min_train_ok) else "FAIL"))
    print("=" * 60)

    return 0 if (validation_report["passed"] and min_train_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
