"""Tests for src/camera.py: initialization and graceful failure."""

import unittest
from unittest.mock import patch, MagicMock

import test_helpers  # noqa: F401  (sets up sys.path)

import camera


class TestCameraGracefulFailure(unittest.TestCase):
    def test_raises_camera_error_when_no_device_opens(self):
        """If cv2.VideoCapture never reports isOpened()==True, Camera()
        must raise the project's own CameraError, not a generic exception,
        and must not hang."""
        with patch("camera.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_capture.return_value = mock_instance

            with self.assertRaises(camera.CameraError):
                camera.Camera(index=0)

    def test_read_returns_false_when_capture_not_opened(self):
        with patch("camera.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            # camera.py's __init__ checks isOpened() twice on the success
            # path (once after the DSHOW attempt, once before the final
            # raise/continue decision); both must be True for init to
            # succeed. read() then checks it a third time.
            mock_instance.isOpened.side_effect = [True, True, False]
            mock_capture.return_value = mock_instance

            cam = camera.Camera(index=0)
            ok, frame = cam.read()
            self.assertFalse(ok)
            self.assertIsNone(frame)


class TestCameraRealDevice(unittest.TestCase):
    """These exercise the actual webcam. Skipped automatically if none is
    available, so the suite still passes in headless/CI environments."""

    def test_open_real_camera_and_read_frame(self):
        try:
            cam = camera.Camera()
        except camera.CameraError:
            self.skipTest("No physical webcam available in this environment.")
            return

        try:
            ok, frame = cam.read()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.ndim, 3)
        finally:
            cam.release()


if __name__ == "__main__":
    unittest.main()
