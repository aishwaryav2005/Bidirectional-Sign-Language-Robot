"""
Converts a variable-length video (N frames) into a fixed number of frame
indices (config.SEQUENCE_LENGTH) to sample.

Method: uniform temporal subsampling (a.k.a. linspace / nearest-neighbor
resampling), the same simple, well-established approach used as the
standard baseline in video-action-recognition literature (e.g. Temporal
Segment Networks). We deliberately do NOT use pixel- or landmark-space
interpolation (blending two frames' hand shapes together) because that
would synthesize hand poses that never actually occurred -- scientifically
indefensible for a sign-language dataset where exact hand shape matters.

How it works:
    target_indices = round(linspace(0, source_length - 1, target_length))

- If source_length > target_length (long video): this picks
  `target_length` evenly-spaced source frames, i.e. uniform downsampling.
- If source_length < target_length (short video): some source indices
  necessarily repeat, i.e. nearest-neighbor upsampling by frame
  repetition -- simple, avoids inventing new hand shapes, and keeps the
  same function correct for both directions with no branching.
- If source_length == target_length: this returns 0..target_length-1
  unchanged (identity).

This is intentionally the simplest method that satisfies "do not blindly
crop the first N frames": it always spans the *entire* video from first
frame to last frame, regardless of length.
"""

import numpy as np


def sample_frame_indices(source_length, target_length):
    """Return a sorted, length-`target_length` NumPy int array of frame
    indices into a video with `source_length` frames, spanning the full
    video (first frame to last frame) via uniform temporal subsampling.

    Raises ValueError if source_length < 1 (nothing to sample from).
    """
    if source_length < 1:
        raise ValueError(f"source_length must be >= 1, got {source_length}")
    if target_length < 1:
        raise ValueError(f"target_length must be >= 1, got {target_length}")

    if source_length == 1:
        return np.zeros(target_length, dtype=np.int64)

    positions = np.linspace(0, source_length - 1, num=target_length)
    indices = np.round(positions).astype(np.int64)
    indices = np.clip(indices, 0, source_length - 1)
    return indices


if __name__ == "__main__":
    # Smoke tests covering the three regimes: downsample, upsample, identity.
    long_video = sample_frame_indices(source_length=180, target_length=60)
    assert long_video.shape == (60,)
    assert long_video[0] == 0 and long_video[-1] == 179
    assert np.all(np.diff(long_video) >= 0), "indices must be non-decreasing"

    short_video = sample_frame_indices(source_length=20, target_length=60)
    assert short_video.shape == (60,)
    assert short_video[0] == 0 and short_video[-1] == 19
    assert len(set(short_video.tolist())) <= 20, "upsampling must only repeat existing indices"

    exact_video = sample_frame_indices(source_length=60, target_length=60)
    assert exact_video.shape == (60,)
    np.testing.assert_array_equal(exact_video, np.arange(60))

    single_frame = sample_frame_indices(source_length=1, target_length=60)
    assert single_frame.shape == (60,)
    assert np.all(single_frame == 0)

    print("Downsample (180->60) sample indices:", long_video[:10], "...")
    print("Upsample (20->60) sample indices:", short_video[:10], "...")
    print("PASS")
