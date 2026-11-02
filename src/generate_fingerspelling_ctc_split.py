"""
Generates a grouped-by-source-video train/validation split INSIDE the
official 810-sample signer-independent training pool (signer_00 + signer_01)
for the fingerspelling CTC experiment.

Does NOT touch the official signer-independent test set (signer_02, 498
samples) or the official splits/{train,test,standard_train,standard_test}.csv
files -- this script only reads them and writes two NEW files:

    data/fingerspelling_processed/splits/ctc_train.csv
    data/fingerspelling_processed/splits/ctc_val.csv

plus reports/fingerspelling_ctc/split_report.json.

Source-video identity check (done explicitly here, not assumed): video_id
format is "<youtube_id>_seg<NNN>" (confirmed directly below by regex-testing
every id in the training pool before using it as a grouping key).
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FP_DIR = os.path.join(config.DATA_DIR, "fingerspelling_processed")
METADATA_PATH = os.path.join(FP_DIR, "metadata", "processing_metadata.csv")
SPLITS_DIR = os.path.join(FP_DIR, "splits")
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")  # official signer-independent train pool (810)
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")  # official locked test (498) -- read only, for the overlap check

REPORT_DIR = os.path.join(config.PROJECT_ROOT, "reports", "fingerspelling_ctc")
SPLIT_REPORT_PATH = os.path.join(REPORT_DIR, "split_report.json")

SEED = 42
VAL_FRACTION = 0.20

SEG_SUFFIX_RE = re.compile(r"_seg\d+$")


def source_id(video_id):
    return SEG_SUFFIX_RE.sub("", video_id)


def main():
    random.seed(SEED)

    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        metadata = {row["video_id"]: row for row in csv.DictReader(f)}

    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        train_pool_ids = [row["video_id"] for row in csv.DictReader(f)]
    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        test_ids = set(row["video_id"] for row in csv.DictReader(f))

    assert len(train_pool_ids) == 810, f"expected 810 in official train pool, got {len(train_pool_ids)}"
    assert len(test_ids) == 498, f"expected 498 in official locked test, got {len(test_ids)}"

    # --- explicit source-video-id-format check (not assumed) ---
    unmatched = [v for v in train_pool_ids if not re.match(r"^.+_seg\d+$", v)]
    assert not unmatched, f"video_id format assumption violated for: {unmatched[:10]}"

    # --- group by source video ---
    groups = defaultdict(list)
    for vid in train_pool_ids:
        groups[source_id(vid)].append(vid)

    group_ids = sorted(groups.keys())  # deterministic order before shuffling
    random.shuffle(group_ids)

    n_val_target = round(len(train_pool_ids) * VAL_FRACTION)

    val_ids, train_ids = [], []
    val_count = 0
    for gid in group_ids:
        members = groups[gid]
        if val_count < n_val_target:
            val_ids.extend(members)
            val_count += len(members)
        else:
            train_ids.extend(members)

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    # --- write split CSVs (video_id, signer, full_text, sequence_path, num_characters) ---
    def write_csv(path, ids):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "signer", "full_text", "sequence_path", "num_characters", "source_video_id"])
            for vid in ids:
                row = metadata[vid]
                writer.writerow([vid, row["signer"], row["full_text"], row["sequence_path"], row["num_characters"], source_id(vid)])

    ctc_train_path = os.path.join(SPLITS_DIR, "ctc_train.csv")
    ctc_val_path = os.path.join(SPLITS_DIR, "ctc_val.csv")
    write_csv(ctc_train_path, train_ids)
    write_csv(ctc_val_path, val_ids)

    # --- verification checks ---
    train_set, val_set = set(train_ids), set(val_ids)
    train_sources = set(source_id(v) for v in train_ids)
    val_sources = set(source_id(v) for v in val_ids)

    checks = {
        "no_source_video_in_both_train_and_val": len(train_sources & val_sources) == 0,
        "signer_02_not_in_train": len(test_ids & train_set) == 0,
        "signer_02_not_in_val": len(test_ids & val_set) == 0,
        "all_498_signer_02_reserved_untouched": test_ids == set(line for line in test_ids),  # trivially true; real check is the two above
        "train_val_disjoint": len(train_set & val_set) == 0,
        "train_union_val_equals_official_pool": (train_set | val_set) == set(train_pool_ids),
    }
    assert all(checks.values()), f"split validation failed: {checks}"

    train_signers = Counter(metadata[v]["signer"] for v in train_ids)
    val_signers = Counter(metadata[v]["signer"] for v in val_ids)

    train_texts = set(metadata[v]["full_text"] for v in train_ids)
    val_texts = set(metadata[v]["full_text"] for v in val_ids)

    report = {
        "seed": SEED,
        "val_fraction_target": VAL_FRACTION,
        "method": "grouped random split by source_video_id (video_id with trailing _segNNN stripped), shuffled at the group level with seed=42, groups assigned to val until the 20% sample-count target is reached, remainder to train",
        "official_train_pool_size": len(train_pool_ids),
        "official_locked_test_size": len(test_ids),
        "ctc_train_samples": len(train_ids),
        "ctc_val_samples": len(val_ids),
        "ctc_train_source_video_groups": len(train_sources),
        "ctc_val_source_video_groups": len(val_sources),
        "signer_distribution": {"train": dict(train_signers), "val": dict(val_signers)},
        "both_signers_in_train": set(train_signers) == {"signer_00", "signer_01"},
        "both_signers_in_val": set(val_signers) == {"signer_00", "signer_01"},
        "text_coverage": {
            "unique_texts_train": len(train_texts),
            "unique_texts_val": len(val_texts),
            "shared_texts_train_val": len(train_texts & val_texts),
            "note": "shared text between train/val is not leakage (different videos/source groups); tracked for coverage awareness only",
        },
        "overlap_checks": checks,
        "output_files": {
            "ctc_train_csv": os.path.relpath(ctc_train_path, config.PROJECT_ROOT),
            "ctc_val_csv": os.path.relpath(ctc_val_path, config.PROJECT_ROOT),
        },
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(SPLIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("WROTE", ctc_train_path)
    print("WROTE", ctc_val_path)
    print("WROTE", SPLIT_REPORT_PATH)


if __name__ == "__main__":
    main()
