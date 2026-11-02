# Public Dataset Investigation: iSign (Indian Sign Language)

This document records what was found by inspecting the **official** iSign
documentation (project page, ACL Findings 2024 paper, arXiv preprint, and
Hugging Face dataset cards) before writing any downloader/preprocessing
code, per the Phase 2 requirements. It also documents the CISLR subset,
which iSign itself is built partly on top of, because CISLR turns out to
be the structurally correct choice for our recognition task (see §8).

Sources consulted (all fetched directly, not from memory):
- Project page: https://exploration-lab.github.io/iSign/
- Paper (ACL Findings 2024): https://aclanthology.org/2024.findings-acl.643/
- Paper (arXiv, same content): https://arxiv.org/abs/2407.05404
- Dataset card: https://huggingface.co/datasets/Exploration-Lab/iSign
- CISLR dataset card: https://huggingface.co/datasets/IIT-K/CISLR
- CISLR paper (EMNLP 2022): https://aclanthology.org/2022.emnlp-main.707.pdf

## 1. Exact Dataset Structure

**iSign** (the full benchmark) is distributed as three parts:

| Part | Contents | Format |
|---|---|---|
| Videos | Raw ISL video clips | `.zip`, split into `iSign-videos_v1.1_part_aa`/`_ab` (`cat` together, then unzip) |
| Poses | Precomputed pose/skeleton data per video | `.pose` files (via the `pose-format` library), split into 4 parts `_aa`..`_ad` |
| CSVs | Text annotations / task labels | `iSign_v1.1.csv`, `word-presence-dataset_v1.1.csv`, `word-description-dataset_v1.1.csv` |

Each row/entry is keyed by a UID of the form `[video_id]-[sequence_number]`
(e.g. `1782bea75c7d-7`) — one source video can contribute multiple rows
(one per sentence spoken within it), which matters for §6/§9 below.

The official guidance for splitting: **split by `video_id`**, not by row,
to avoid leaking the same source video across train/val/test.

## 2. Available Subsets (5 benchmark tasks)

1. **SignVideo2Text** — ISL video → English sentence (translation)
2. **SignPose2Text** — ISL pose sequence → English sentence (translation)
3. **Text2Pose** — English text → generated ISL pose sequence
4. **Word/Gloss Recognition** — isolated single-sign recognition, built on
   the **CISLR** subset (see §8 — this is the one relevant to us)
5. **Word Presence / Sign Semantics** — auxiliary NLP tasks (word-in-sentence
   detection, semantic similarity) — not relevant to a vision pipeline

## 3. Video Data

- Source: scraped from ISLRTC (Indian Sign Language Research & Training
  Centre), ISH-News (YouTube ISL news channel), and DEF (Deaf Enabled
  Foundation) — 114 + 3,373 + 651 = **4,138 unique source videos**.
- Total duration: **252 hours**.
- These are largely **continuous, multi-word sentence videos** (a signer
  interpreting a news segment, etc.), not isolated single-sign clips.

## 4. Pose Data

- Extracted with **MediaPipe Holistic** (the legacy combined
  hand+pose+face pipeline — not the Tasks API we use, see §11 in the
  README for why our own installed MediaPipe no longer has this API).
- **75 landmarks**, face mesh excluded, 3D (x, y, z) coordinates.
- Normalization used by iSign: centered on the shoulder midpoint, scaled
  by the nose-to-shoulder distance — conceptually similar to our own
  wrist-relative / shoulder-relative normalization (see
  `src/feature_extractor.py`), though not numerically identical.
- Stored per-video in `.pose` format (`pose.body.data` / `.confidence`),
  which is a **different landmark indexing/layout** than our 21-point
  hand + 6-point pose selection. Reusing iSign's precomputed poses
  directly would require writing a format-conversion layer; running our
  own `landmark_detector.py` on the raw video frames instead is simpler,
  guarantees representation compatibility, and is what the project
  brief asked for ("use the SAME MediaPipe configuration... wherever
  possible"). **This is why our preprocessing pipeline (§ below)
  re-extracts landmarks from raw video rather than parsing `.pose`
  files.**

## 5. Text/Translation Annotations

`iSign_v1.1.csv` provides an English sentence/phrase for each UID —
sentence-level translation labels, not per-sign gloss labels. Not used by
our recognition pipeline (which needs single-word/sign labels).

## 6. Number of Samples/Records — read carefully (see §12 for full breakdown)

- **118,228** video-**sentence/phrase pairs** (the number iSign reports as
  its headline size) — each pair is one *row*, not one *video* and not one
  isolated *sign*.
- **~4,138** unique underlying source videos.
- **CISLR subset** (isolated words, embedded in iSign as Task 4):
  **4,765 words**, **7,050 videos** (~1.5 videos/word on average — low
  enough that the iSign paper itself frames CISLR word recognition as a
  **one-shot learning** task, not a conventional multi-example-per-class
  classification task).

## 7. License — flag: sources disagree, verify before large-scale use

- The **arXiv/ACL paper** states the license as **CC BY-NC-ND 4.0**
  (Attribution-NonCommercial-**NoDerivatives**).
- The **Hugging Face dataset card** for `Exploration-Lab/iSign` tags the
  license as **CC-BY-NC-SA-4.0** (Attribution-NonCommercial-**ShareAlike**,
  which — unlike ND — *does* permit derivative works).
- Both agree on **non-commercial, research-use only, attribution
  required**. They disagree on whether derivative works (like our
  extracted landmark sequences) may be shared/redistributed.
- **The separate CISLR dataset** (`IIT-K/CISLR` on Hugging Face) has its
  **own, more permissive license: AFL-3.0** (Academic Free License),
  which is an OSI-approved permissive license.
- **Recommendation:** since CISLR is the subset we actually want (§8) and
  carries the clearer, more permissive AFL-3.0 license, use CISLR as the
  license basis for this project. For academic use (training a model,
  writing a thesis/report, non-commercial demonstration) this is safe
  under any of the licenses found. **Do not publicly redistribute the
  raw videos or a bulk copy of the processed dataset** without first
  reading the actual license text on the Hugging Face page yourself —
  automated summarization tools (including the ones used to write this
  document) can miscategorize legal text, and this is exactly the kind
  of decision that shouldn't rest on a paraphrase.

## 8. Which Subset Is Appropriate for Our Sign-Language *Recognition* Task

**Recommended: CISLR** (`IIT-K/CISLR` on Hugging Face), **not** the full
118K continuous-sentence iSign corpus. Reasoning:

- Our target representation is **one `(60, 147)` sequence = one sign**,
  matching CISLR's structure directly: **one video = one word/sign**.
- iSign's 118K "pairs" are continuous multi-word sentences. Uniformly
  sampling 60 frames from a 10-second sentence containing 4 different
  signs would blend fragments of multiple unrelated signs into a single
  labeled sequence — that's not a scientifically valid training example
  for a per-sign classifier. Turning continuous sentences into per-sign
  clips requires **temporal sign segmentation**, a separate and much
  harder problem (akin to action segmentation) that is explicitly out of
  scope right now.
- CISLR provides **raw video only** (no precomputed pose), which is
  actually convenient: it means we run our *own* `landmark_detector.py`
  and `feature_extractor.py` unchanged, exactly as instructed.

**Caveat carried forward honestly:** CISLR's ~1.5 videos/word average
means most classes will have very few (often just 1) examples — the
original authors treat it as one-shot recognition, not a dataset you can
directly train a conventional LSTM/GRU classifier on with many examples
per class. See §12 for what this means for the "100,000 data points"
target and §13 for a concrete recommendation.

## 9. Can We Legally Use It for an Academic Project?

**Yes, for CISLR specifically** — AFL-3.0 permits academic (and broader)
use, with attribution, and access simply requires creating a free
Hugging Face account and accepting CISLR's access-request agreement (it
is a *gated* dataset — see §14, this is also why this project does not
auto-download anything). **For the broader iSign corpus**, yes for
non-commercial research use under either license reading found (§7), but
avoid public redistribution of derivatives until the ND/SA discrepancy is
resolved by reading the primary license file yourself.

## 10. Pose Extraction Method Comparison (informational)

| | iSign's own pose data | Our pipeline |
|---|---|---|
| Tool | MediaPipe Holistic (legacy solutions API) | MediaPipe Tasks API (`HandLandmarker` + `PoseLandmarker`) — the legacy API is gone from the MediaPipe version we have installed |
| Landmarks | 75 (hands + body, no face) | 21×2 hand + 6 pose = 48 raw points, expressed as a 147-value normalized vector (63+63+18+3) |
| Normalization | shoulder-midpoint-centered, nose-to-shoulder scaled | wrist-relative per hand, shoulder-midpoint-centered for pose (see `src/feature_extractor.py`) |

We do not attempt to reuse iSign's `.pose` files because of this layout
mismatch (§4); we re-run our own detector on raw video instead, so every
sample — ours or public — goes through the identical extraction and
normalization code path.

## 11. Access — Why This Project Does Not Auto-Download Anything

`IIT-K/CISLR` is a **gated** Hugging Face dataset: downloading requires a
free Hugging Face account, logging in, and accepting the dataset's access
agreement (a manual, one-time step tied to *your* identity/agreement, not
something that should happen invisibly inside a script). Combined with
the explicit instruction not to bulk-download automatically, our
`src/preprocess_public_dataset.py` **never fetches anything from the
network** — it only processes video files that already exist locally
under `data/public_raw/`, which you place there yourself after manually
downloading a (small, to start) batch from Hugging Face.

## 12. Counting "Data Points" Honestly

Different units answer different questions — conflating them is how a
dataset "grows" from 7K to 27 million without adding any real data:

| Unit | Count | Is this a usable training example for our LSTM/GRU/Transformer? |
|---|---|---|
| Unique source videos (all of iSign) | ~4,138 | No — too coarse, mixes many sentences |
| Video-sentence pairs (iSign's headline number) | 118,228 | No, not directly — each is a multi-sign continuous sentence, needs segmentation first (not built) |
| CISLR isolated-word videos | 7,050 | **Yes, structurally** — one video = one sign, matches our format |
| CISLR unique words/classes | 4,765 | N/A — this is the number of classes, not samples |
| Raw video frames (252 hours @ ~25–30 fps) | ~22–27 million | No — a single frame isn't a sign; our unit is a 60-frame sequence |
| Processed `(60, 147)` sequences we can actually produce **today** from CISLR | up to 7,050 | **Yes — this is the real, honest number** |

**Bottom line: with CISLR alone, this pipeline can produce at most
~7,050 real `(60, 147)` training sequences (across 4,765 classes,
averaging ~1.5 samples/class), not 100,000.** No interpretation of the
official numbers legitimately supports a >100K claim for *usable,
per-sign, non-duplicated* training sequences today.

## 13. Recommended Next Step for Reaching the >100K Target

This is a documentation/recommendation only — **not implemented in this
phase**:

1. **Combine multiple public isolated-sign ISL datasets** (CISLR +
   others such as the INCLUDE dataset from IIT Bombay, ~4,287 clips /
   263 signs — still modest, but additive) through the *same*
   preprocessing pipeline built here (it's dataset-agnostic: any
   videos-folder + labels-CSV works).
2. **Combine with your own webcam recordings** (Phase 1) — same `(60,
   147)` format, so they merge into one training set with zero
   conversion work.
3. **Data augmentation** (temporal jitter/speed variation, left-right
   mirroring, small coordinate noise) can multiply the *number of
   training examples* well past 100K, but be explicit in any report to
   faculty that augmented copies are not independent real data points —
   report both the real-sample count and the augmented-training-set size
   separately.
4. **Sign segmentation of the full continuous iSign corpus** (118K
   sentence pairs → many more isolated-sign clips) is the only path to a
   genuinely large *real* isolated-sign dataset from iSign itself, but is
   a nontrivial CV/NLP alignment problem and should be scoped as its own
   phase, not bolted onto Phase 2.

## 14. How to Actually Get Data Onto This Machine (manual, one-time)

1. Create a free account at https://huggingface.co
2. Visit https://huggingface.co/datasets/IIT-K/CISLR and accept the
   access agreement.
3. Download `dataset.csv` (or `prototype.csv` for the recognition split)
   plus a handful of videos from `CISLR_v1.5-a_videos/` — start small,
   e.g. 10 videos, to match the Phase 2 `--limit 10` test.
4. Build a two-column CSV (`video_filename,label`) from CISLR's own CSV
   and place both the videos and the label CSV under `data/public_raw/`
   (see `README.md` "Phase 2" section for the exact expected layout).
5. Run `src/preprocess_public_dataset.py --limit 10` against it.

No credentials or download automation are handled by this project's
code — that step is intentionally left to you.

## 15. Left/Right Hand Convention — Resolved

The real 10-video CISLR test surfaced a genuine cross-dataset
consistency question: `main.py` mirrored webcam frames
(`cv2.flip(frame, 1)`) before running MediaPipe, while
`preprocess_public_dataset.py` never mirrored CISLR frames.
MediaPipe's own docs state handedness assumes mirrored/selfie input —
which would suggest CISLR's *unmirrored* processing was the one at risk.

We tested this empirically rather than trusting that reading of the
docs, using `src/diagnose_handedness.py`:

- **CISLR** (`-0XujLX2_9I_2`, a video where one hand is used almost the
  entire time): the signer faces the camera in a normal, non-selfie
  recording. Their visibly raised hand sits on **screen-left** throughout
  — which is anatomically their own **right** hand for anyone facing a
  camera dead-on. MediaPipe's *unmirrored* pass (the pipeline's real,
  current behavior) reports **"Right"** — correct. Mirroring that same
  footage relocates the identical hand to screen-right and flips
  MediaPipe's label to "Left" — wrong. Verified visually
  (`outputs/handedness_diagnostic/cislr_unmirrored.png` /
  `cislr_mirrored.png`) as well as numerically.
- **Webcam**: the user confirmed live, on their own known physical right
  hand: unmirrored → MediaPipe reports "Right" (0.95–0.98 confidence);
  mirrored → reports "Left" (0.94–0.97). Same pattern as CISLR.

**Conclusion: unmirrored input is correct for this camera + model in
both pipelines.** `main.py`'s `cv2.flip` was removed (see README.md §7)
so both the webcam and CISLR pipelines now share one convention:
`category_name` from MediaPipe is used as-is, with **no swap** anywhere
in `feature_extractor.py`, and `left_hand`/`right_hand` in the 147-vector
mean the same physical hand regardless of data source.

## 16. Raw and Processed Datasets Are NOT Hosted on GitHub — Regeneration Guide

The `data/` directory (~21-22GB locally) is excluded from this repository
(`.gitignore`) for two reasons: GitHub is not designed to host multi-GB
datasets, and uploading it would not even recover local disk space unless
deleted afterward. Everything under `data/` is either a raw source
download or a byproduct of running a script already in `src/` against
that raw source — **fully reproducible, nothing unique is lost** by not
versioning it. This section is the authoritative map of what each
subdirectory is, where its raw source comes from, and the exact command
that regenerates it.

### 16.1 Raw sources (not produced by this project's code)

| Path | Size (local) | What it is | How it was obtained |
|---|---|---|---|
| `data/external/INCLUDE/raw_zips/*.zip` (`People_1of5.zip`...`5of5.zip`, `Seasons_1of1.zip`, `Electronics_1of2.zip`/`2of2.zip`) | ~18.8GB | Raw INCLUDE (IIT Bombay) category video archives | Manually downloaded from the INCLUDE dataset's official distribution (gated/manual download, same access-control pattern as CISLR — see §14) |
| `data/public_raw/CISLR_v1.5-a_videos/` + `labels.csv` | ~2.3GB | Raw CISLR isolated-word video clips | Manually downloaded from Hugging Face (`IIT-K/CISLR`) per the exact steps in §14 above |
| `C:\Users\<user>\OneDrive\ISL-Fingerspelling\videos\*.mp4` (**outside this project folder entirely**) | not measured as part of this repo | Raw ISL-Fingerspelling character-level video clips | Manually placed at this fixed path (see `src/stream_process_fingerspelling.py` header) — not under `data/` at all, so it isn't part of this project directory's size, but it IS a required raw source for regenerating `data/fingerspelling_processed/` |

### 16.2 Processed features (the actual training/eval inputs — small, reproducible)

| Path | Size | Produced by | Regeneration command |
|---|---|---|---|
| `data/external_processed/include/` | ~0.02GB (738 sequences) | INCLUDE preprocessing (the version that produced the currently-used `People` category features) | `python src/stream_process_include_v6.py` (reads `raw_zips/`, writes `sequences/` + `metadata/processing_metadata.csv`) |
| `data/include_streamed_processed/` | ~0.03GB | Same pipeline, different category batch (Greetings/Home/Jobs) | `python src/stream_process_include_v6.py --categories Greetings Home Jobs` |
| `data/public_processed/` | ~0.24GB (7,051 sequences) | CISLR preprocessing | `python src/preprocess_public_dataset.py` (reads `data/public_raw/`, writes `data/public_processed/`) |
| `data/cislr_feature_v4_processed/` | ~0.03GB | CISLR feature-extraction v4 experiment | `python src/extract_cislr_feature_v4.py` |
| `data/fingerspelling_processed/` | ~0.05GB | ISL-Fingerspelling preprocessing | `python src/stream_process_fingerspelling.py` (reads the external OneDrive path above — **not** under this project folder) |
| `data/sign_demo/splits/*.csv` | negligible | Sign→Speech demo vocabulary/split selection | `python src/build_sign_demo_dataset.py` (reads `data/external_processed/include/metadata/processing_metadata.csv`, which must already exist) |

### 16.3 Full regeneration order (raw → processed, from scratch)

```bash
# 1. Place raw sources (manual, one-time — see §14 and the table above):
#      data/external/INCLUDE/raw_zips/*.zip
#      data/public_raw/CISLR_v1.5-a_videos/ + labels.csv
#      <OneDrive>/ISL-Fingerspelling/videos/*.mp4   (outside this project folder)

# 2. Run each preprocessing script (independent, any order):
python src/stream_process_include_v6.py
python src/preprocess_public_dataset.py
python src/stream_process_fingerspelling.py
python src/extract_cislr_feature_v4.py          # optional, only for the v4 CISLR experiment branch

# 3. Build the sign-demo vocabulary/split (depends on step 2's INCLUDE output):
python src/build_sign_demo_dataset.py
```

Every script above is idempotent and resumable (each writes a
`processing_metadata.csv` and skips videos already marked
`processing_status == "success"` on re-run), matching the behavior
already documented throughout this file for the CISLR pipeline.

### 16.4 Raw archives proposed for eventual local deletion (NOT yet deleted)

Per explicit instruction, the ~21.1GB of raw archives in §16.1
(`data/external/INCLUDE/raw_zips/` + `data/public_raw/CISLR_v1.5-a_videos/`)
are **candidates** for local deletion once (a) the processed feature sets
in §16.2 are verified complete and usable by the existing training/eval
workflows, and (b) a backup of the raw archives exists. **Nothing has
been deleted as of this writing** — see the project's working notes for
the current verification status before deleting anything.
