# Sign Language Robot — Perception & Dataset Pipeline

## 1. Project Purpose

This is the foundation of **"AI-Based Bidirectional Sign Language Translation
Using Vision-Based Robot Perception and a Dual-Arm Humanoid Robot."**

The full project (across all phases) will let a deaf user sign to an RGB
camera, have the sign recognized and translated by an AI model, and
eventually have a dual-arm humanoid robot sign back. **This repository
currently implements Phase 1 and Phase 2.**

## 2. Current Phase

**Phase 1 — Sign → AI Recognition Pipeline (live webcam perception + data acquisition).**

```
DEAF USER → RGB CAMERA → HAND + POSE DETECTION → LANDMARK EXTRACTION
          → SAVE TEMPORAL LANDMARK SEQUENCES
```

**Phase 2 — Public Dataset → Standardized Landmark Dataset** (see §14 below
and `DATASET_INFO.md`): converts public isolated-sign ISL videos into the
same `(SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH)` representation Phase 1
produces, so public and custom-recorded data can be combined for training.

Explicitly **not** implemented yet (future phases): LSTM/Transformer
training, sign→English translation, speech-to-text, text-to-speech, NLP,
ROS2, MoveIt2, inverse kinematics, servo/robot-arm control, or robot
simulation. This repo only gets the data pipeline solid first.

## 3. Folder Structure

```
Sign Language Robot/
│
├── README.md
├── DATASET_INFO.md            # Phase 2: public dataset investigation & recommendation
├── requirements.txt
├── config.py                  # all tunable parameters live here
│
├── models/                    # downloaded MediaPipe Tasks model files
│   ├── hand_landmarker.task
│   └── pose_landmarker_lite.task
│
├── src/
│   ├── camera.py                      # webcam capture (Step 1)
│   ├── landmark_detector.py           # MediaPipe hand + pose detection (Steps 2-3)
│   ├── feature_extractor.py           # landmarks -> fixed-size feature vector (Step 4)
│   ├── sequence_collector.py          # buffers frames -> saved sequences (Step 5)
│   ├── main.py                        # live app: camera + HUD + recording controls (Steps 6-7)
│   ├── validate_dataset.py            # Phase 1 dataset inspection/statistics (Step 8)
│   ├── temporal_sampling.py           # Phase 2: variable-length video -> fixed frame count
│   ├── preprocess_public_dataset.py   # Phase 2: public videos -> (60,147) sequences
│   └── validate_public_dataset.py     # Phase 2 dataset inspection/statistics
│
├── data/
│   ├── raw/                   # reserved for raw video, if ever captured
│   ├── landmarks/             # reserved for intermediate per-frame dumps
│   ├── sequences/             # Phase 1: <label>/sample_NNN.npy + metadata.csv
│   ├── public_raw/            # Phase 2: your local copy of public videos + labels.csv (not auto-downloaded)
│   ├── public_processed/
│   │   ├── sequences/         # Phase 2: <label>/<video_id>.npy
│   │   └── metadata/          # Phase 2: processing_metadata.csv
│   └── mappings/
│       └── label_mapping.json # Phase 2: original label -> numeric class ID
│
├── tests/                     # unittest suite (Step 9)
│
├── outputs/                   # reserved for exported reports/plots
│
└── solidworks files/          # pre-existing CAD files for the physical robot
                                # (unrelated to this pipeline, left untouched)
```

## 4. Required Python Version

**Python 3.12.** MediaPipe does not yet publish wheels for Python 3.14, and
the machine also has a separate MSYS2 Python without `pip` — neither works
for this project. A real Python 3.12 install was found at:

```
C:\Users\Elitebook 840 G9\AppData\Local\Programs\Python\Python312\python.exe
```

A dedicated virtual environment (`venv/`) built from that interpreter
already exists in this project folder.

## 5. Installation

From the project root (PowerShell):

```powershell
# (venv/ already created for you from Python 3.12; recreate if needed with:)
# & "C:\Users\Elitebook 840 G9\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv

.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Model files

The current MediaPipe version (1.0.1) uses the newer **Tasks API**, which
loads pretrained models from standalone `.task` files rather than bundling
them in the package (see [§13 Known Limitations](#13-known-limitations) for
why this differs from older MediaPipe tutorials). These are already
downloaded into `models/`. If you ever need to re-download them:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "models\pose_landmarker_lite.task"
```

## 6. Connecting / Using the Webcam

`src/camera.py` auto-detects the first working camera index (tries 0..4) if
you don't pass one explicitly, so no camera index is hardcoded. It uses
OpenCV's DirectShow backend on Windows with a fallback to the default
backend. If no camera can be opened, it raises a clear `CameraError`
instead of crashing with a cryptic OpenCV error.

## 7. Running the Application

```powershell
.\venv\Scripts\python.exe src\main.py
```

This opens a live window showing your webcam feed with hand/pose landmarks
drawn on top, plus an on-screen HUD (see §8).

**Note: the live view is intentionally displayed unmirrored** (not the
"selfie view" you may expect from a typical video-call app) — so what you
see on screen is a normal, non-flipped camera image, matching how your
own hands actually move. This was a deliberate fix, not an oversight: an
earlier version mirrored the frame before running MediaPipe, which was
empirically found (see `DATASET_INFO.md` and `src/diagnose_handedness.py`)
to make MediaPipe report your physical right hand as "Left" and vice
versa. Leaving the frame unmirrored gives anatomically correct left/right
labels, and — just as importantly — matches the convention used by the
CISLR preprocessing pipeline (§14), so `left_hand`/`right_hand` mean the
same physical hand across both custom webcam recordings and processed
public-dataset samples.

**Controls:**

| Key | Action |
|-----|--------|
| `L` | Set/change the current sign label (typed in the terminal) |
| `R` | Start a `COUNTDOWN_SECONDS` countdown, then record a `SEQUENCE_LENGTH`-frame sample for the current label |
| `C` | Cancel an in-progress countdown or recording |
| `Q` | Quit |

## 8. Recording a Sign

`SEQUENCE_LENGTH` defaults to **60 frames** (about 2 seconds at ~30 FPS) —
increased from an initial 30-frame default because 30 frames proved too
short to comfortably perform a complete sign. Both `SEQUENCE_LENGTH` and
`COUNTDOWN_SECONDS` are configurable in `config.py`.

Recording flow:

```
Press R
   ↓
 3
 2
 1
   ↓
RECORDING
Frame 1/60
   ...
Frame 60/60
   ↓
Sample saved
```

Steps:

1. Run `src\main.py`.
2. Press `L`, then type a label in the terminal (e.g. `hello`) and press Enter.
3. Position yourself in frame and press `R`.
4. A large on-screen countdown (`3`, `2`, `1`) gives you time to get into
   position before capture starts — no frames are recorded during the
   countdown. Press `C` any time during the countdown (or the recording
   itself) to cancel.
5. Once the countdown ends, recording starts automatically. Perform the
   sign for the next `SEQUENCE_LENGTH` (default 60) frames. The HUD shows
   `RECORDING  Frame: 17/60` while capturing.
6. On completion, the terminal prints `Sample saved successfully: <path>`.
7. Repeat for as many samples/labels as needed — press `L` again to switch
   labels, or press `R` again to record another sample of the same label.

The on-screen HUD (Step 7) always shows: FPS, current label, left/right
hand detection status, pose detection status, and countdown/recording
status (including live frame count while recording) — useful for
demoing the perception pipeline live.

## 9. Where Data Is Stored

```
data/sequences/<LABEL>/sample_001.npy
data/sequences/<LABEL>/sample_002.npy
...
data/sequences/metadata.csv
```

Each `.npy` file is one recorded sample. `metadata.csv` has one row per
saved sample with: `label, sample_path, sequence_length, feature_dimension,
timestamp, signer_id, camera_width, camera_height,
frames_left_hand_detected, frames_right_hand_detected,
frames_pose_detected`.

## 10. Dataset Format

Each sample is a NumPy array of shape:

```
(SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH) = (60, 147)
```

i.e. 60 frames × a 147-value feature vector per frame (see §11).
`SEQUENCE_LENGTH` is configurable in `config.py` — it defaults to 60 (was
30 initially; increased because 30 frames was too short to comfortably
perform a complete sign). This format is intentionally **generic and not
tied to any specific dataset** — the same `(60, 147)` layout can hold
samples from your own webcam
recordings or from a public ISL dataset once its videos are run through
the same `landmark_detector.py` + `feature_extractor.py` pipeline. The
intended future workflow:

```
Public ISL Dataset  +  Custom ISL Dataset
                ↓
          Preprocessing
                ↓
        Landmark Extraction
                ↓
       Temporal Sequences (this repo's format)
                ↓
      AI Sign Recognition Model  (Phase 2, not yet built)
```

Use `src/validate_dataset.py` to inspect a collected dataset:

```powershell
.\venv\Scripts\python.exe src\validate_dataset.py
```

This reports: samples per label, total sample count, sequence-length /
feature-dimension consistency, missing-hand/pose statistics per label, and
any corrupted `.npy` files.

## 11. Feature-Vector Structure

Every frame produces a **fixed-length 147-value** `float32` NumPy vector,
regardless of what was actually detected:

```
[ left_hand (63) | right_hand (63) | pose (18) | detection_flags (3) ]
  index 0-62        63-125           126-143      144-146
```

- **Left/right hand** (63 values each): 21 MediaPipe hand landmarks × (x, y, z).
- **Pose** (18 values): 6 upper-body landmarks × (x, y, z) — specifically
  `left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
  right_wrist` (MediaPipe Pose indices 11–16). These were chosen because
  they capture arm position/reach during signing without the cost or
  noise of the full 33-point body skeleton or face mesh.
- **Detection flags** (3 values): `[left_hand_detected, right_hand_detected,
  pose_detected]` as 1.0/0.0, so a model can distinguish "hand really is at
  the origin" from "hand was not detected." Also recorded in
  `metadata.csv` (aggregated per sample).

**If a hand or the pose is not detected in a frame, its slice of the
vector is filled with zeros** — the vector length never changes, which is
required for feeding fixed-size sequences into an LSTM/GRU/Transformer
later.

### Normalization (kept intentionally simple)

- **Each hand** is translated so its wrist (landmark 0) sits at the
  origin, then scaled by the distance from the wrist to the middle-finger
  MCP joint (landmark 9). This makes hand shape roughly invariant to the
  signer's distance from the camera and position in frame.
- **Pose** is translated so the shoulder midpoint sits at the origin, then
  scaled by the shoulder-to-shoulder distance, for the same reason.
- If the scale reference distance is ~0 (a degenerate detection), a scale
  of 1.0 is used instead of dividing by zero.

No rotation-invariance, temporal smoothing, or other more complex
normalization is applied at this stage — kept deliberately simple so it's
easy to reason about and swap out later. See `src/feature_extractor.py`.

## 12. How This Connects to an LSTM/GRU/Transformer (Phase 2, not yet built)

Each saved `sample_NNN.npy` is already shaped `(SEQUENCE_LENGTH,
FEATURE_VECTOR_LENGTH)` — i.e. `(60, 147)` — which is exactly the
`(timesteps, features)` input shape expected by a Keras/PyTorch
`LSTM`/`GRU` layer, or the per-frame token embedding a Transformer encoder
would consume. `metadata.csv` supplies the label for supervised training.
Phase 2 will load these `.npy` files (grouped by label from
`metadata.csv`), stack them into a `(num_samples, 60, 147)` training
tensor, and train a sequence classifier. **That model is intentionally not
built in this phase.**

## 13. Known Limitations

- **MediaPipe API change:** the installed `mediapipe==1.0.1` removed the
  classic `mp.solutions.hands` / `mp.solutions.pose` API that most
  tutorials use. This project uses the current **Tasks API**
  (`HandLandmarker`, `PoseLandmarker`) instead, which requires separate
  `.task` model files (see §5). If you follow older MediaPipe tutorials
  elsewhere, expect the API surface to differ from what's implemented here.
- **Single signer assumed:** pose extraction uses `pose_landmarks[0]`
  (the first detected person). Multiple people in frame are not handled.
- **No rotation/viewpoint invariance:** normalization handles translation
  and scale only, not camera angle or hand rotation.
- **`pose_landmarker_lite` model:** the "lite" variant was used for speed;
  it's less accurate than the "full"/"heavy" variants under fast motion or
  partial occlusion.
- **No temporal smoothing/interpolation:** a frame with a missed detection
  is zero-filled rather than interpolated from neighboring frames.
- **Manual label typing:** labels are typed into the terminal (Step 6 asked
  for a simple, reliable interface rather than a GUI), so there's no
  autocomplete or validation against a fixed sign vocabulary yet.
- **Windows-only camera backend tuning:** `camera.py` prefers the
  DirectShow (`CAP_DSHOW`) backend, which is Windows-specific; it falls
  back to the default backend, which should still work cross-platform.

## 14. Phase 2 — Public Dataset Preprocessing

Converts a public isolated-sign ISL video dataset into the same
`(SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH)` = `(60, 147)` representation
used by Phase 1, so public and custom-recorded samples can later be
combined for training. Full investigation of the candidate dataset
(iSign / CISLR — structure, license, sample counts, and why CISLR rather
than the full continuous iSign corpus) is in **`DATASET_INFO.md`** —
read that first.

```
PUBLIC ISL VIDEOS -> FRAME SAMPLING -> MEDIAPIPE -> LANDMARKS
                   -> 147 FEATURES -> 60-FRAME SEQUENCE -> PROCESSED DATA
```

**This project never downloads anything automatically.** `IIT-K/CISLR` is
a gated Hugging Face dataset requiring a manual, authenticated
access-request — see `DATASET_INFO.md` §14 for the exact steps.

### Expected input layout

Place videos plus a two-column label CSV under `data/public_raw/`:

```
data/public_raw/
    labels.csv              # columns: video_filename,label
    <video files...>
```

`labels.csv` example:

```csv
video_filename,label
hello_001.mp4,hello
hello_002.mp4,hello
thank_you_001.mp4,thank_you
```

### Running the 10-video test

```powershell
.\venv\Scripts\python.exe src\preprocess_public_dataset.py --limit 10
.\venv\Scripts\python.exe src\validate_public_dataset.py
```

`--limit N` processes only the first N rows of `labels.csv` — always test
on a small number before processing a full dataset. Reruns are
idempotent: already-successfully-processed videos are skipped, so
processing a larger `--limit` later resumes rather than redoing work.

### How variable-length videos become exactly 60 frames

Public videos are **not** assumed to be 60 frames long. Frame indices are
chosen via **uniform temporal subsampling** (`src/temporal_sampling.py`):
`round(linspace(0, source_length-1, 60))`, spanning the entire video from
first frame to last frame. Longer videos are downsampled (evenly-spaced
frames skipped); shorter videos are upsampled by *repeating* frames
(nearest-neighbor), never blended or invented — chosen deliberately over
pixel/landmark interpolation because synthesizing an in-between hand
shape that never occurred would be scientifically indefensible for this
domain. Every sampled frame goes through the *same*
`landmark_detector.py` + `feature_extractor.py` code Phase 1 uses (same
MediaPipe models, same normalization), including zero-filling any
undetected hand/pose exactly as Phase 1 does.

### Output

```
data/public_processed/sequences/<label>/<video_id>.npy   # shape (60, 147)
data/public_processed/metadata/processing_metadata.csv
data/mappings/label_mapping.json                          # {label: {class_id, display_name}}
```

`processing_metadata.csv` has one row per attempted video: `video_id,
original_filename, label, class_id, original_video_length_frames,
sampled_frame_indices, processing_status, failure_reason,
sequence_length, feature_dimension, left_hand_detection_rate,
right_hand_detection_rate, pose_detection_rate, sequence_path`. Failed
videos (missing file, unreadable/corrupt file) are recorded with a
reason and skipped — one bad file never aborts the batch.

### Validating results

`src/validate_public_dataset.py` reports: total videos processed,
successful/failed counts, number of labels, samples per label, sequence
length / feature dimension consistency, average left-hand/right-hand/pose
detection rates, and any corrupted `.npy` files.

### On dataset scale — read `DATASET_INFO.md` §12 before citing a number

CISLR (the recommended subset) provides **at most ~7,050 real isolated-sign
videos across 4,765 words** (~1.5 samples/word on average) — not 100,000.
The 118,228 figure iSign advertises counts continuous multi-word sentence
clips, which are not directly convertible to single-sign `(60, 147)`
sequences without a sign-segmentation step this project does not
implement. `DATASET_INFO.md` §12 lays out the honest accounting (videos
vs. frames vs. usable sequences) and §13 recommends concrete next steps
for reaching a larger real training set.

## Testing

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Covers: camera initialization (incl. graceful failure via mocked
`cv2.VideoCapture`), a real-webcam smoke test (skipped automatically if no
camera is present), fixed feature-vector dimensions, missing-hand/pose
zero-fill behavior, normalization correctness, sequence-length triggering,
save/load round-trips, metadata correctness, dataset validation (including
corrupted-file detection), a full real-hardware integration test
(camera → MediaPipe detection → feature extraction → sequence save,
skipped automatically if no camera/model files are present), Phase 2's
temporal-sampling logic (downsampling/upsampling/identity/edge cases),
and a full Phase 2 preprocessing run against synthetically generated test
videos (variable lengths, a missing file, and a corrupt file, to exercise
both the success and failure paths without depending on gated dataset
access). All tests currently pass on this machine.
