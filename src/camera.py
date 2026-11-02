"""
Webcam capture wrapper.

Responsible only for finding/opening a camera and yielding frames.
Knows nothing about MediaPipe, features, or sequences.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CameraError(RuntimeError):
    """Raised when no working camera can be opened."""


class Camera:
    def __init__(self, index=None, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.index = index if index is not None else self._auto_detect_index()
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # Retry without the DirectShow backend hint (some cameras/drivers
            # only work with the default backend).
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open camera at index {self.index}. "
                "Check that a webcam is connected and not in use by another application."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    @staticmethod
    def _auto_detect_index():
        """Probe device indices 0..N and return the first one that opens and
        successfully returns a frame."""
        for idx in range(config.MAX_CAMERA_INDEX_TO_PROBE):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    return idx
            cap.release()
        raise CameraError(
            "No working webcam found. Tried camera indices "
            f"0..{config.MAX_CAMERA_INDEX_TO_PROBE - 1}. "
            "Connect a webcam and try again, or pass an explicit index."
        )

    def read(self):
        """Return (success, frame). Frame is BGR (OpenCV default)."""
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def get_resolution(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if __name__ == "__main__":
    # Minimal standalone smoke test: open the camera, show live feed, quit on 'q'.
    try:
        cam = Camera()
    except CameraError as e:
        print(f"[camera.py] ERROR: {e}")
        raise SystemExit(1)

    print(f"[camera.py] Opened camera index {cam.index} at resolution {cam.get_resolution()}")
    print("[camera.py] Press Q to quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[camera.py] WARNING: failed to read frame, stopping.")
                break
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
