"""Scene detection and frame extraction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from utils.sharpness import is_frame_sharp, pick_sharpest_frame, save_frame


def detect_scenes(movie_path: Path) -> list[tuple[float, float]]:
    """Detect scenes using PySceneDetect ContentDetector.

    Args:
        movie_path: Path to the movie file.

    Returns:
        List of (start_seconds, end_seconds) tuples.
    """
    movie_path = Path(movie_path)
    video = open_video(str(movie_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()
    out: list[tuple[float, float]] = []
    for start, end in scenes:
        out.append((float(start.get_seconds()), float(end.get_seconds())))
    return out


def extract_frames(
    movie_path: Path,
    scenes: list[tuple[float, float]],
    output_dir: Path,
    skip_start: float,
    skip_end: float,
) -> int:
    """Extract one representative frame per scene (midpoint, sharpness-selected).

    For each scene, sample 5 candidate frames in a 0.6s window around the midpoint
    (uniformly spaced), then pick the sharpest via Laplacian variance.

    Scenes with midpoint within the first `skip_start` seconds or within the last
    `skip_end` seconds of the movie are skipped.

    Args:
        movie_path: Path to the movie file.
        scenes: List of (start_seconds, end_seconds) tuples.
        output_dir: Directory to write frames.
        skip_start: Seconds to skip at the start of the movie.
        skip_end: Seconds to skip at the end of the movie.

    Returns:
        Number of frames saved.
    """
    movie_path = Path(movie_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(movie_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {movie_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    saved = 0
    window = 0.6
    n_candidates = 5

    for i, (start_s, end_s) in enumerate(scenes):
        mid_s = (float(start_s) + float(end_s)) / 2.0
        if mid_s <= float(skip_start):
            continue
        if duration > 0 and mid_s >= (duration - float(skip_end)):
            continue

        times = [
            mid_s + (-window / 2.0) + (j * (window / max(1, n_candidates - 1)))
            for j in range(n_candidates)
        ]

        frames = []
        for t in times:
            if t < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        chosen = pick_sharpest_frame(frames)
        if chosen is None:
            continue

        out_path = output_dir / f"scene_{i:04d}.jpg"
        save_frame(chosen, out_path)
        saved += 1

    cap.release()
    return saved


def capture_movie_screenshots(
    video_path: Path,
    output_dir: Path,
    interval_sec: float,
    sharpness_threshold: float = 50.0,
    fixed_height: int = 512,
) -> int:
    """Capture sharp screenshots from a movie at a fixed time interval.

    This mirrors the workflow in your snippet: step through a video at `interval_sec`,
    keep only frames that pass `is_frame_sharp`, and save resized images.

    Args:
        video_path: Path to movie file.
        output_dir: Directory to save screenshots.
        interval_sec: Interval between candidate frames in seconds (e.g. 0.2).
        sharpness_threshold: Laplacian variance threshold for `is_frame_sharp`.
        fixed_height: Output image height in pixels (aspect ratio preserved).

    Returns:
        Number of screenshots saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine FPS for video.")

    frame_interval = max(1, int(round(float(interval_sec) * fps)))
    frame_number = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_number % frame_interval == 0:
            if is_frame_sharp(frame, threshold=float(sharpness_threshold)):
                out_name = f"shot_{saved:06d}.jpg"
                save_frame(frame, output_dir / out_name, fixed_height=int(fixed_height))
                saved += 1

        frame_number += 1

    cap.release()
    return saved

