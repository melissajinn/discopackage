"""Sharpness utilities for selecting and saving frames.

All functions operate on OpenCV-style BGR numpy arrays unless otherwise noted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import cv2


def laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance sharpness for a BGR frame.

    Args:
        frame: BGR image as a numpy array (H, W, 3).

    Returns:
        Variance of the Laplacian of the grayscale image.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def is_frame_sharp(frame: np.ndarray, threshold: float = 50) -> bool:
    """Check whether a frame is sharp based on Laplacian variance.

    Args:
        frame: BGR image as a numpy array (H, W, 3).
        threshold: Minimum Laplacian variance to consider the frame sharp.

    Returns:
        True if sharpness >= threshold, otherwise False.
    """
    return laplacian_variance(frame) >= float(threshold)


def pick_sharpest_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    """Pick the sharpest frame (highest Laplacian variance) from a list.

    Args:
        frames: List of BGR numpy arrays.

    Returns:
        The sharpest frame, or None if the list is empty or all frames invalid.
    """
    if not frames:
        return None

    best_frame: np.ndarray | None = None
    best_score = -1.0
    for fr in frames:
        score = laplacian_variance(fr)
        if score > best_score:
            best_score = score
            best_frame = fr
    return best_frame


def save_frame(frame: np.ndarray, path: Path, fixed_height: int = 512) -> None:
    """Save a BGR frame to disk, resized to a fixed height (aspect preserved).

    Uses PIL with LANCZOS resampling.

    Args:
        frame: BGR image as a numpy array (H, W, 3).
        path: Output file path.
        fixed_height: Desired output height in pixels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Cannot save an empty frame.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions.")

    new_h = int(fixed_height)
    new_w = max(1, int(round((w / h) * new_h)))
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    img.save(path, format="JPEG", quality=95, optimize=True)

