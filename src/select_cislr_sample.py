"""
Selects the first N labeled rows from a CISLR-style dataset.csv, robustly
matches each row's `uid` to an actual video file (searching recursively,
since dataset archives commonly nest an extra directory level and/or
include junk like macOS `__MACOSX/` metadata folders), verifies every
selected row has exactly one matching file BEFORE any processing happens,
and writes a `video_filename,label,category,duration` CSV consumable by
preprocess_public_dataset.py.

This does NOT run MediaPipe or extract any landmarks -- it only performs
selection + matching + verification, per the explicit instruction to
report a mismatch rather than silently substituting a different video.

Usage:
    python src/select_cislr_sample.py --limit 10
    python src/select_cislr_sample.py --dataset-csv data\\public_raw\\dataset.csv --videos-root data\\public_raw\\CISLR_v1.5-a_videos\\extracted --limit 10 --output data\\public_raw\\cislr_sample_10_labels.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_JUNK_DIR_MARKERS = ("__MACOSX",)


def find_video_files(videos_root):
    """Recursively finds real .mp4 files under videos_root, skipping known
    zip-artifact junk (`__MACOSX/` folders, AppleDouble `._name.mp4` files).

    Returns {uid_stem: absolute_path}. Raises ValueError if two different
    files would map to the same stem (ambiguous match) rather than
    silently picking one.
    """
    stem_to_path = {}
    conflicts = {}
    for root, dirnames, filenames in os.walk(videos_root):
        dirnames[:] = [d for d in dirnames if d not in _JUNK_DIR_MARKERS]
        for filename in filenames:
            if not filename.lower().endswith(".mp4"):
                continue
            if filename.startswith("._") or filename.startswith("."):
                continue  # AppleDouble / hidden metadata artifact, not a real video
            stem = os.path.splitext(filename)[0]
            full_path = os.path.join(root, filename)
            if stem in stem_to_path and stem_to_path[stem] != full_path:
                conflicts.setdefault(stem, [stem_to_path[stem]]).append(full_path)
            else:
                stem_to_path[stem] = full_path

    if conflicts:
        details = "; ".join(f"{stem}: {paths}" for stem, paths in conflicts.items())
        raise ValueError(f"Ambiguous video files -- multiple files share the same UID stem: {details}")

    return stem_to_path


def select_and_match(dataset_csv_path, videos_root, limit, offset=0):
    """Returns (matched_rows, mismatched_uids) where matched_rows is a list
    of dicts {uid, gloss, category, duration, video_path} for rows
    [offset : offset+limit] of dataset_csv_path (offset=0, the default,
    selects from the start), and mismatched_uids lists any of those rows
    whose video file could not be found.

    `offset` lets a later call select the *next* batch (e.g. offset=10,
    limit=100 selects rows 11-110) without re-selecting rows already
    chosen by an earlier call -- dataset.csv row order is stable across
    calls since the file is read fresh and never reordered."""
    with open(dataset_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"uid", "gloss", "duration", "category"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{dataset_csv_path} must have columns {required}; found {reader.fieldnames}")
        all_rows = list(reader)

    end = offset + limit if limit is not None else None
    selected = all_rows[offset:end]

    stem_to_path = find_video_files(videos_root)

    matched, mismatched = [], []
    for row in selected:
        uid = row["uid"].strip()
        video_path = stem_to_path.get(uid)
        if video_path is None:
            mismatched.append(uid)
        else:
            matched.append({
                "uid": uid,
                "gloss": row["gloss"].strip(),
                "category": row["category"].strip(),
                "duration": row["duration"].strip(),
                "video_path": video_path,
            })

    return matched, mismatched


def write_labels_csv(matched_rows, videos_root, output_csv_path):
    """Writes video_filename (relative to videos_root),label,category,duration."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_filename", "label", "category", "duration"])
        for row in matched_rows:
            rel_path = os.path.relpath(row["video_path"], videos_root)
            writer.writerow([rel_path, row["gloss"], row["category"], row["duration"]])


def main():
    parser = argparse.ArgumentParser(description="Select + verify a sample of CISLR videos and build a labels CSV for preprocess_public_dataset.py.")
    parser.add_argument("--dataset-csv", default=os.path.join(config.PUBLIC_RAW_DIR, "dataset.csv"))
    parser.add_argument("--videos-root", default=os.path.join(config.PUBLIC_RAW_DIR, "CISLR_v1.5-a_videos", "extracted"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows from the start of dataset.csv before selecting --limit rows (use to select the NEXT batch without re-selecting earlier ones)")
    parser.add_argument("--output", default=None, help="Where to write the labels CSV (default: <videos-root>/../cislr_sample_<offset>_<limit>_labels.csv)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        config.PUBLIC_RAW_DIR, f"cislr_sample_{args.offset}_{args.limit}_labels.csv"
    )

    print("=" * 60)
    print("CISLR Sample Selection & UID<->Video Matching")
    print("=" * 60)
    print(f"Dataset CSV:  {args.dataset_csv}")
    print(f"Videos root:  {args.videos_root}")
    print(f"Selecting rows [{args.offset}:{args.offset + args.limit}]")
    print("=" * 60)

    matched, mismatched = select_and_match(args.dataset_csv, args.videos_root, args.limit, offset=args.offset)

    print(f"Selected: {args.limit}")
    print(f"Matched to a real video file: {len(matched)}")
    print(f"MISMATCHED (no video file found): {len(mismatched)}")
    if mismatched:
        print("The following UIDs have NO corresponding video file:")
        for uid in mismatched:
            print(f"  - {uid}")

    for row in matched:
        print(f"  OK  {row['uid']}  -> {row['gloss']!r}  ({os.path.basename(row['video_path'])})")

    write_labels_csv(matched, args.videos_root, output_path)
    print("=" * 60)
    print(f"Labels CSV written: {output_path}")
    print(f"Videos root for preprocess_public_dataset.py --videos-dir: {args.videos_root}")
    print("=" * 60)

    return 0 if not mismatched else 2


if __name__ == "__main__":
    raise SystemExit(main())
