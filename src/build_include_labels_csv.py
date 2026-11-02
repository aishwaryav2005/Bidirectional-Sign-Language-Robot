"""
Walks an extracted INCLUDE dataset directory tree (Category/N. Gloss/*.MOV
or *.mp4) and builds a `video_filename,label,category` CSV compatible with
src/preprocess_public_dataset.py's expected input format (any extra
columns, here `category`, are preserved verbatim into each row's
`source_metadata`).

INCLUDE folder convention (per the official Zenodo distribution, record
4010759): <Category>/<N>. <Gloss>/<video files>. The leading "<N>. " index
prefix is stripped from the gloss folder name to produce a clean label
(e.g. "1. Dog" -> "Dog"); nothing else about the label is altered (no
lowercasing/merging here -- that is done later, explicitly, in the label
harmonization step against CISLR).

Does not touch data/public_raw/, data/public_processed/, or any frozen
CISLR path. Only reads under --videos-dir and writes one CSV there.

Usage:
    python src/build_include_labels_csv.py --videos-dir data/external/INCLUDE/raw_videos
"""

import argparse
import csv
import os
import re

VIDEO_EXTENSIONS = {".mov", ".mp4", ".MOV", ".MP4"}
# INCLUDE gloss folder names are prefixed either with a numeric index
# ("64. Fall") or, for a handful of category-level example entries, the
# literal "Ex." prefix ("Ex. Monsoon") -- confirmed by direct inspection
# of the official train/val/test metadata's `label` column (not assumed).
_LEADING_INDEX_RE = re.compile(r"^\s*(?:\d+|Ex)\.\s*", re.IGNORECASE)


def clean_gloss_label(folder_name):
    return _LEADING_INDEX_RE.sub("", folder_name).strip()


def build_rows(videos_dir):
    rows = []
    for category in sorted(os.listdir(videos_dir)):
        category_path = os.path.join(videos_dir, category)
        if not os.path.isdir(category_path):
            continue
        for gloss_folder in sorted(os.listdir(category_path)):
            gloss_path = os.path.join(category_path, gloss_folder)
            if not os.path.isdir(gloss_path):
                continue
            label = clean_gloss_label(gloss_folder)
            for fname in sorted(os.listdir(gloss_path)):
                ext = os.path.splitext(fname)[1]
                if ext not in VIDEO_EXTENSIONS:
                    continue
                rel_path = os.path.join(category, gloss_folder, fname)
                rows.append({
                    "video_filename": rel_path,
                    "label": label,
                    "category": category,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a video_filename,label,category CSV from an extracted INCLUDE directory tree.")
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-csv", default=None, help="default: <videos-dir>/labels.csv")
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(args.videos_dir, "labels.csv")
    rows = build_rows(args.videos_dir)

    if not rows:
        print(f"[build_include_labels_csv.py] WARNING: no video files found under {args.videos_dir}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_filename", "label", "category"])
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted(set(r["label"] for r in rows))
    categories = sorted(set(r["category"] for r in rows))
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Unique labels (glosses): {len(labels)}")
    print(f"Categories: {categories}")
    for lab in labels:
        count = sum(1 for r in rows if r["label"] == lab)
        print(f"  {lab}: {count} videos")


if __name__ == "__main__":
    main()
