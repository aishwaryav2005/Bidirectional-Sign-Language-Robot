"""
Buffers per-frame feature vectors into fixed-length temporal sequences and
saves them to disk as labeled samples, alongside a metadata record.

Sample layout on disk:
    data/sequences/<label>/sample_XXX.npy   -> float32 array,
                                                shape (SEQUENCE_LENGTH, FEATURE_VECTOR_LENGTH)
    data/sequences/metadata.csv             -> one row per saved sample
"""

import csv
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_METADATA_FIELDS = [
    "label",
    "sample_path",
    "sequence_length",
    "feature_dimension",
    "timestamp",
    "signer_id",
    "camera_width",
    "camera_height",
    "frames_left_hand_detected",
    "frames_right_hand_detected",
    "frames_pose_detected",
]


class SequenceCollector:
    """Tracks recording state and accumulates frames for one sample at a time."""

    def __init__(self, sequence_length=config.SEQUENCE_LENGTH,
                 feature_dim=config.FEATURE_VECTOR_LENGTH,
                 sequences_dir=config.SEQUENCES_DIR,
                 metadata_path=config.METADATA_PATH):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.sequences_dir = sequences_dir
        self.metadata_path = metadata_path

        self.is_recording = False
        self._buffer = []
        self._detection_buffer = []

        os.makedirs(self.sequences_dir, exist_ok=True)
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
                writer.writeheader()

    def start_recording(self):
        self.is_recording = True
        self._buffer = []
        self._detection_buffer = []

    def add_frame(self, feature_vector, detection_info):
        """Add one frame's feature vector while recording.

        Returns True if the sequence just became complete (caller should
        then call save_sequence()).
        """
        if not self.is_recording:
            return False
        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature vector has length {feature_vector.shape[0]}, "
                f"expected {self.feature_dim}"
            )
        self._buffer.append(feature_vector)
        self._detection_buffer.append(detection_info)
        if len(self._buffer) >= self.sequence_length:
            self.is_recording = False
            return True
        return False

    @property
    def current_frame_count(self):
        return len(self._buffer)

    def save_sequence(self, label, camera_resolution=(None, None), signer_id=""):
        """Persist the buffered sequence to disk and append a metadata row.

        Returns the path to the saved .npy file.
        """
        if len(self._buffer) != self.sequence_length:
            raise ValueError(
                f"Cannot save: buffer has {len(self._buffer)} frames, "
                f"expected exactly {self.sequence_length}"
            )

        label_dir = os.path.join(self.sequences_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        existing = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        next_index = len(existing) + 1
        filename = f"sample_{next_index:03d}.npy"
        sample_path = os.path.join(label_dir, filename)

        sequence_array = np.stack(self._buffer, axis=0).astype(np.float32)
        np.save(sample_path, sequence_array)

        frames_left = sum(1 for d in self._detection_buffer if d["left_hand"])
        frames_right = sum(1 for d in self._detection_buffer if d["right_hand"])
        frames_pose = sum(1 for d in self._detection_buffer if d["pose"])

        row = {
            "label": label,
            "sample_path": os.path.relpath(sample_path, config.PROJECT_ROOT),
            "sequence_length": self.sequence_length,
            "feature_dimension": self.feature_dim,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signer_id": signer_id,
            "camera_width": camera_resolution[0],
            "camera_height": camera_resolution[1],
            "frames_left_hand_detected": frames_left,
            "frames_right_hand_detected": frames_right,
            "frames_pose_detected": frames_pose,
        }
        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
            writer.writerow(row)

        self._buffer = []
        self._detection_buffer = []
        return sample_path

    def cancel_recording(self):
        self.is_recording = False
        self._buffer = []
        self._detection_buffer = []


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    try:
        collector = SequenceCollector(
            sequence_length=5,
            feature_dim=config.FEATURE_VECTOR_LENGTH,
            sequences_dir=tmp_dir,
            metadata_path=os.path.join(tmp_dir, "metadata.csv"),
        )
        collector.start_recording()
        assert collector.is_recording

        complete = False
        for i in range(5):
            fv = np.random.rand(config.FEATURE_VECTOR_LENGTH).astype(np.float32)
            info = {"left_hand": True, "right_hand": False, "pose": True}
            complete = collector.add_frame(fv, info)
        assert complete, "sequence should be complete after 5 frames"
        assert not collector.is_recording

        path = collector.save_sequence("test_label", camera_resolution=(640, 480))
        loaded = np.load(path)
        assert loaded.shape == (5, config.FEATURE_VECTOR_LENGTH), loaded.shape

        with open(os.path.join(tmp_dir, "metadata.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["label"] == "test_label"

        print("Saved sequence shape:", loaded.shape)
        print("Metadata row:", rows[0])
        print("PASS")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
