"""Tests for src/preprocess_public_dataset.py and validate_public_dataset.py,
using small synthetically generated video files (not real dataset content)
so the suite doesn't depend on any gated download."""

import csv
import json
import os
import shutil
import tempfile
import unittest
import unittest.mock

import cv2
import numpy as np
import test_helpers  # noqa: F401

import config
import preprocess_public_dataset as ppd
from validate_public_dataset import validate_public_dataset


def _write_synthetic_video(path, n_frames, width=160, height=120, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _models_available():
    return os.path.exists(config.HAND_LANDMARKER_MODEL_PATH) and os.path.exists(
        config.POSE_LANDMARKER_MODEL_PATH
    )


class TestLoadLabelCsv(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_loads_rows_in_order(self):
        csv_path = os.path.join(self.tmp_dir, "labels.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_filename", "label"])
            w.writerow(["a.mp4", "hello"])
            w.writerow(["b.mp4", "thanks"])
        rows = ppd.load_label_csv(csv_path)
        self.assertEqual(
            rows,
            [
                {"video_filename": "a.mp4", "label": "hello", "extra": {}},
                {"video_filename": "b.mp4", "label": "thanks", "extra": {}},
            ],
        )

    def test_extra_columns_preserved(self):
        csv_path = os.path.join(self.tmp_dir, "labels.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_filename", "label", "category", "duration"])
            w.writerow(["a.mp4", "hello", "greeting", "6"])
        rows = ppd.load_label_csv(csv_path)
        self.assertEqual(rows[0]["extra"], {"category": "greeting", "duration": "6"})

    def test_rejects_wrong_columns(self):
        csv_path = os.path.join(self.tmp_dir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "class"])
            w.writerow(["a.mp4", "hello"])
        with self.assertRaises(ValueError):
            ppd.load_label_csv(csv_path)


class TestAssignClassIds(unittest.TestCase):
    def test_preserves_ids_across_calls(self):
        mapping = {}
        mapping = ppd.assign_class_ids(mapping, ["hello", "thanks"])
        self.assertEqual(mapping["hello"]["class_id"], 0)
        self.assertEqual(mapping["thanks"]["class_id"], 1)

        # New label added later must not renumber existing ones.
        mapping = ppd.assign_class_ids(mapping, ["thanks", "goodbye"])
        self.assertEqual(mapping["hello"]["class_id"], 0)
        self.assertEqual(mapping["thanks"]["class_id"], 1)
        self.assertEqual(mapping["goodbye"]["class_id"], 2)

    def test_preserves_display_name(self):
        mapping = ppd.assign_class_ids({}, ["hello"])
        self.assertEqual(mapping["hello"]["display_name"], "hello")


@unittest.skipUnless(_models_available(), "MediaPipe Tasks model files not found under models/")
class TestPreprocessEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.tmp_dir, "videos")
        self.output_dir = os.path.join(self.tmp_dir, "output")
        self.mapping_path = os.path.join(self.tmp_dir, "label_mapping.json")
        os.makedirs(self.videos_dir)

        # Deliberately varied lengths: shorter than, equal to, and longer
        # than config.SEQUENCE_LENGTH, to exercise all sampling regimes.
        self.plan = [
            ("short.mp4", "alpha", 10),
            ("exact.mp4", "alpha", config.SEQUENCE_LENGTH),
            ("long.mp4", "beta", config.SEQUENCE_LENGTH * 3),
        ]
        for filename, _, n_frames in self.plan:
            _write_synthetic_video(os.path.join(self.videos_dir, filename), n_frames)

        self.labels_csv = os.path.join(self.videos_dir, "labels.csv")
        with open(self.labels_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_filename", "label"])
            for filename, label, _ in self.plan:
                w.writerow([filename, label])
            w.writerow(["does_not_exist.mp4", "gamma"])  # failure case

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, limit=None):
        import sys
        argv_backup = sys.argv
        try:
            sys.argv = [
                "preprocess_public_dataset.py",
                "--videos-dir", self.videos_dir,
                "--labels-csv", self.labels_csv,
                "--output-dir", self.output_dir,
                "--label-mapping", self.mapping_path,
            ]
            if limit is not None:
                sys.argv += ["--limit", str(limit)]
            return ppd.main()
        finally:
            sys.argv = argv_backup

    def test_produces_correct_shapes_and_metadata(self):
        exit_code = self._run()
        self.assertEqual(exit_code, 0)

        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        with open(metadata_path, newline="") as f:
            rows = {r["video_id"]: r for r in csv.DictReader(f)}

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["short"]["processing_status"], "success")
        self.assertEqual(rows["exact"]["processing_status"], "success")
        self.assertEqual(rows["long"]["processing_status"], "success")
        self.assertEqual(rows["does_not_exist"]["processing_status"], "failed")
        self.assertNotEqual(rows["does_not_exist"]["failure_reason"], "")

        # original_video_length_frames must reflect the REAL source length,
        # proving the pipeline doesn't assume every video is 60 frames.
        self.assertEqual(int(rows["short"]["original_video_length_frames"]), 10)
        self.assertEqual(int(rows["exact"]["original_video_length_frames"]), config.SEQUENCE_LENGTH)
        self.assertEqual(int(rows["long"]["original_video_length_frames"]), config.SEQUENCE_LENGTH * 3)

        for key in ("short", "exact", "long"):
            seq_path = os.path.join(config.PROJECT_ROOT, rows[key]["sequence_path"])
            arr = np.load(seq_path)
            self.assertEqual(arr.shape, (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH))
            self.assertEqual(arr.dtype, np.float32)
            self.assertFalse(np.any(np.isnan(arr)))

        with open(self.mapping_path) as f:
            mapping = json.load(f)
        self.assertIn("alpha", mapping)
        self.assertIn("beta", mapping)
        self.assertIn("gamma", mapping)  # label preserved even though its video failed

    def test_limit_processes_only_first_n(self):
        exit_code = self._run(limit=2)
        self.assertEqual(exit_code, 0)
        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        with open(metadata_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)  # only first 2 of 4 CSV rows processed

    def test_rerun_skips_already_processed(self):
        self._run()
        # Second run should skip all 3 successes and only re-attempt the
        # already-failed row (idempotent, cumulative behavior).
        exit_code = self._run()
        self.assertEqual(exit_code, 0)

    def test_validate_public_dataset_matches(self):
        self._run()
        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        report = validate_public_dataset(metadata_path=metadata_path)
        self.assertEqual(report["total_processed"], 4)
        self.assertEqual(report["successful"], 3)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["sequence_lengths"], {config.SEQUENCE_LENGTH})
        self.assertEqual(report["feature_dimensions"], {config.FEATURE_VECTOR_LENGTH})
        self.assertEqual(len(report["corrupted_files"]), 0)


@unittest.skipUnless(_models_available(), "MediaPipe Tasks model files not found under models/")
class TestCrashSafety(unittest.TestCase):
    """Verifies the checkpointing fix: metadata is flushed after every
    video (not just at the end), so a genuine interruption mid-run loses
    no progress and a resumed run never redundantly reprocesses videos
    that already completed."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.tmp_dir, "videos")
        self.output_dir = os.path.join(self.tmp_dir, "output")
        self.mapping_path = os.path.join(self.tmp_dir, "label_mapping.json")
        os.makedirs(self.videos_dir)

        self.plan = [(f"clip_{i}.mp4", f"label_{i}", 20 + i) for i in range(5)]
        for filename, _, n_frames in self.plan:
            _write_synthetic_video(os.path.join(self.videos_dir, filename), n_frames)

        self.labels_csv = os.path.join(self.videos_dir, "labels.csv")
        with open(self.labels_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_filename", "label"])
            for filename, label, _ in self.plan:
                w.writerow([filename, label])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        import sys
        argv_backup = sys.argv
        try:
            sys.argv = [
                "preprocess_public_dataset.py",
                "--videos-dir", self.videos_dir,
                "--labels-csv", self.labels_csv,
                "--output-dir", self.output_dir,
                "--label-mapping", self.mapping_path,
            ]
            return ppd.main()
        finally:
            sys.argv = argv_backup

    def test_interruption_preserves_already_completed_progress(self):
        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        real_process_video = ppd.process_video
        call_count = {"n": 0}

        def flaky_process_video(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 3:
                # KeyboardInterrupt (not Exception) simulates a genuine
                # interruption that the per-video try/except cannot and
                # must not swallow.
                raise KeyboardInterrupt("simulated interruption")
            return real_process_video(*args, **kwargs)

        with unittest.mock.patch.object(ppd, "process_video", side_effect=flaky_process_video):
            with self.assertRaises(KeyboardInterrupt):
                self._run()

        # Videos 1-2 completed and must already be checkpointed to disk,
        # even though the run never reached its normal end-of-main() point.
        with open(metadata_path, newline="") as f:
            rows_after_crash = {r["video_id"]: r for r in csv.DictReader(f)}
        self.assertIn("clip_0", rows_after_crash)
        self.assertIn("clip_1", rows_after_crash)
        self.assertEqual(rows_after_crash["clip_0"]["processing_status"], "success")
        self.assertEqual(rows_after_crash["clip_1"]["processing_status"], "success")
        self.assertNotIn("clip_2", rows_after_crash)
        self.assertNotIn("clip_3", rows_after_crash)
        self.assertNotIn("clip_4", rows_after_crash)

        first_run_time_clip0 = rows_after_crash["clip_0"]["processing_time_seconds"]

        # Resume with a normal (unpatched) run.
        call_log = []
        def counting_process_video(*args, **kwargs):
            call_log.append(args[0])
            return real_process_video(*args, **kwargs)

        with unittest.mock.patch.object(ppd, "process_video", side_effect=counting_process_video):
            exit_code = self._run()
        self.assertEqual(exit_code, 0)

        # Only the 3 not-yet-completed videos should have actually run --
        # clip_0 and clip_1 must NOT be reprocessed.
        processed_paths = [os.path.basename(p) for p in call_log]
        self.assertEqual(sorted(processed_paths), sorted(["clip_2.mp4", "clip_3.mp4", "clip_4.mp4"]))

        with open(metadata_path, newline="") as f:
            rows_final = {r["video_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(rows_final), 5)
        for vid in ["clip_0", "clip_1", "clip_2", "clip_3", "clip_4"]:
            self.assertEqual(rows_final[vid]["processing_status"], "success")

        # clip_0's original timing is untouched -- proof it wasn't rewritten.
        self.assertEqual(rows_final["clip_0"]["processing_time_seconds"], first_run_time_clip0)

    def test_dry_run_reports_without_side_effects(self):
        import io
        from contextlib import redirect_stdout

        # Process 2 of 5 for real first.
        with unittest.mock.patch.object(ppd.sys, "argv", [
            "preprocess_public_dataset.py",
            "--videos-dir", self.videos_dir,
            "--labels-csv", self.labels_csv,
            "--output-dir", self.output_dir,
            "--label-mapping", self.mapping_path,
            "--limit", "2",
        ]):
            self.assertEqual(ppd.main(), 0)

        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        mtime_before = os.path.getmtime(metadata_path)

        buf = io.StringIO()
        with unittest.mock.patch.object(ppd.sys, "argv", [
            "preprocess_public_dataset.py",
            "--videos-dir", self.videos_dir,
            "--labels-csv", self.labels_csv,
            "--output-dir", self.output_dir,
            "--label-mapping", self.mapping_path,
            "--dry-run",
        ]):
            with redirect_stdout(buf):
                exit_code = ppd.main()

        self.assertEqual(exit_code, 0)
        output = buf.getvalue()
        self.assertIn("Total videos in labels CSV:            5", output)
        self.assertIn("Already processed (success, will skip): 2", output)
        self.assertIn("Remaining to process this run:          3", output)

        # No side effects: metadata file untouched, no video processed.
        self.assertEqual(os.path.getmtime(metadata_path), mtime_before)

    def test_metadata_csv_never_left_truncated_by_atomic_write(self):
        """write_metadata_csv uses temp-file-then-replace; verify the
        helper itself produces a complete, valid file (no partial/stray
        .tmp file left behind)."""
        metadata_path = os.path.join(self.output_dir, "metadata", "processing_metadata.csv")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        fake_rows = {
            "a": {f: "" for f in ppd._METADATA_FIELDS} | {"video_id": "a", "processing_status": "success"},
            "b": {f: "" for f in ppd._METADATA_FIELDS} | {"video_id": "b", "processing_status": "success"},
        }
        ppd.write_metadata_csv(metadata_path, fake_rows)

        self.assertTrue(os.path.exists(metadata_path))
        self.assertFalse(os.path.exists(metadata_path + ".tmp"))
        with open(metadata_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["video_id"] for r in rows}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
