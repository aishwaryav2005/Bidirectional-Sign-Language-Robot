"""End-to-end integration test: real webcam + real MediaPipe inference,
running the exact sequence main.py executes when a user presses R.

Skipped automatically if no webcam or no MediaPipe model files are
available, so it still passes in headless/CI environments. Writes to a
temp directory, never touches data/sequences/.
"""

import csv
import os
import shutil
import tempfile
import time
import unittest

import numpy as np
import test_helpers  # noqa: F401

import config
from camera import Camera, CameraError
from feature_extractor import extract_features
from landmark_detector import LandmarkDetector
from sequence_collector import SequenceCollector


def _camera_available():
    try:
        cam = Camera()
    except CameraError:
        return False
    cam.release()
    return True


def _models_available():
    return os.path.exists(config.HAND_LANDMARKER_MODEL_PATH) and os.path.exists(
        config.POSE_LANDMARKER_MODEL_PATH
    )


@unittest.skipUnless(_camera_available(), "No physical webcam available in this environment.")
@unittest.skipUnless(_models_available(), "MediaPipe Tasks model files not found under models/")
class TestFullPipelineWithRealHardware(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cam = Camera()
        self.detector = LandmarkDetector()
        self.collector = SequenceCollector(
            sequence_length=config.SEQUENCE_LENGTH,
            feature_dim=config.FEATURE_VECTOR_LENGTH,
            sequences_dir=self.tmp_dir,
            metadata_path=os.path.join(self.tmp_dir, "metadata.csv"),
        )

    def tearDown(self):
        self.detector.close()
        self.cam.release()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_record_real_sequence_end_to_end(self):
        """Mirrors main.py's frame loop for exactly SEQUENCE_LENGTH frames:
        read a real frame, run real detection, extract real features, buffer,
        then save -- verifying shape/dtype/metadata on genuine hardware
        output rather than synthetic fakes."""
        self.collector.start_recording()
        resolution = self.cam.get_resolution()

        frames_processed = 0
        completed = False
        for _ in range(config.SEQUENCE_LENGTH):
            ok, frame = self.cam.read()
            self.assertTrue(ok, "camera failed to deliver a frame during recording")

            timestamp_ms = int(time.time() * 1000) + frames_processed
            result = self.detector.detect(frame, timestamp_ms)
            fv, info = extract_features(result["hand_result"], result["pose_result"])

            self.assertEqual(fv.shape, (config.FEATURE_VECTOR_LENGTH,))
            self.assertEqual(fv.dtype, np.float32)
            self.assertIn("left_hand", info)
            self.assertIn("right_hand", info)
            self.assertIn("pose", info)

            completed = self.collector.add_frame(fv, info)
            frames_processed += 1

        self.assertEqual(frames_processed, config.SEQUENCE_LENGTH)
        self.assertTrue(completed, "collector should report completion at exactly SEQUENCE_LENGTH frames")
        self.assertFalse(self.collector.is_recording)

        saved_path = self.collector.save_sequence(
            "integration_test", camera_resolution=resolution, signer_id="test_harness"
        )
        self.assertTrue(os.path.exists(saved_path))

        loaded = np.load(saved_path)
        self.assertEqual(loaded.shape, (config.SEQUENCE_LENGTH, config.FEATURE_VECTOR_LENGTH))
        self.assertEqual(loaded.dtype, np.float32)
        self.assertFalse(np.any(np.isnan(loaded)))
        self.assertFalse(np.any(np.isinf(loaded)))

        with open(os.path.join(self.tmp_dir, "metadata.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "integration_test")
        self.assertEqual(int(rows[0]["sequence_length"]), config.SEQUENCE_LENGTH)
        self.assertEqual(int(rows[0]["feature_dimension"]), config.FEATURE_VECTOR_LENGTH)


if __name__ == "__main__":
    unittest.main()
