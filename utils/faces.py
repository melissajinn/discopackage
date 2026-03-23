"""Face detection and frame classification utilities using insightface."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import cv2
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

import config
from utils.dedup import farthest_point_sampling, load_image
from utils.sharpness import laplacian_variance


def load_face_model() -> FaceAnalysis:
    """Initialize insightface FaceAnalysis (buffalo_l) on CPU.

    Returns:
        Prepared FaceAnalysis app.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def frame_pixel_std_bgr(bgr: np.ndarray) -> float:
    """Compute pixel standard deviation for a BGR frame (blank/black frame heuristic).

    Args:
        bgr: BGR image as numpy array.

    Returns:
        Standard deviation of pixel values as float.
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
        return 0.0
    return float(np.std(bgr))


def has_any_face(img: Image.Image, face_app: FaceAnalysis, min_det_score: float = 0.3) -> bool:
    """Return True if insightface detects any face above a confidence threshold.

    This is intentionally *less strict* than `detect_quality_faces` and is used for
    neutral frame filtering (we want neutrals to be faceless).

    Args:
        img: PIL image (RGB).
        face_app: Prepared insightface FaceAnalysis app.
        min_det_score: Minimum detection score to count as a face.

    Returns:
        True if any face is detected with det_score >= min_det_score.
    """
    if img is None:
        return False
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)
    if not faces:
        return False
    for f in faces:
        if float(getattr(f, "det_score", 0.0)) >= float(min_det_score):
            return True
    return False


def detect_quality_faces(img: Image.Image, face_app: FaceAnalysis) -> list[dict[str, Any]]:
    """Detect faces and apply quality filters.

    Filters:
    - face size >= config.FACE_MIN_SIZE (both width and height)
    - |yaw| < config.FACE_MAX_YAW
    - det_score > config.FACE_MIN_CONFIDENCE
    - face sharpness > config.FACE_MIN_SHARPNESS (Laplacian variance on crop)

    Args:
        img: PIL image (RGB).
        face_app: Prepared insightface FaceAnalysis app.

    Returns:
        List of dicts with keys: bbox, embedding, pose, sharpness, det_score.
    """
    if img is None:
        return []

    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)
    if not faces:
        return []

    h, w = bgr.shape[:2]
    out: list[dict[str, Any]] = []

    for face in faces:
        bbox = np.array(face.bbox, dtype=np.float32)
        x1, y1, x2, y2 = bbox.astype(int).tolist()

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        fw, fh = x2 - x1, y2 - y1
        if fw < config.FACE_MIN_SIZE or fh < config.FACE_MIN_SIZE:
            continue

        pose = getattr(face, "pose", None)
        if pose is None or len(pose) < 2:
            continue
        yaw = float(pose[1])
        if abs(yaw) >= float(config.FACE_MAX_YAW):
            continue

        det_score = float(getattr(face, "det_score", 0.0))
        if det_score <= float(config.FACE_MIN_CONFIDENCE):
            continue

        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        sharp = laplacian_variance(crop)
        if sharp <= float(config.FACE_MIN_SHARPNESS):
            continue

        emb = np.asarray(getattr(face, "embedding", None), dtype=np.float32)
        if emb is None or emb.size == 0:
            continue
        emb = emb.reshape(-1)

        out.append(
            {
                "bbox": (x1, y1, x2, y2),
                "embedding": emb,
                "pose": np.asarray(pose, dtype=np.float32),
                "sharpness": float(sharp),
                "det_score": det_score,
            }
        )

    return out


def score_frame_quality(img_size: tuple[int, int], bbox: tuple[int, int, int, int], sharpness: float) -> float:
    """Compute aesthetic quality score for a face within an image.

    quality = size_score * 0.3 + center_score * 0.3 + sharp_score * 0.4

    Args:
        img_size: (width, height)
        bbox: (x1, y1, x2, y2)
        sharpness: Laplacian variance for face crop.

    Returns:
        Quality score in [0, 1] (approximately).
    """
    w, h = img_size
    x1, y1, x2, y2 = bbox

    frame_area = max(1.0, float(w) * float(h))
    face_w = max(1.0, float(x2 - x1))
    face_h = max(1.0, float(y2 - y1))
    face_area = face_w * face_h

    size_score = min(1.0, face_area / frame_area)

    face_cx = (float(x1) + float(x2)) / 2.0
    face_cy = (float(y1) + float(y2)) / 2.0
    frame_cx, frame_cy = float(w) / 2.0, float(h) / 2.0
    dx = abs(face_cx - frame_cx) / max(1.0, float(w) / 2.0)
    dy = abs(face_cy - frame_cy) / max(1.0, float(h) / 2.0)
    center_score = max(0.0, 1.0 - (dx + dy) / 2.0)

    sharp_score = min(1.0, float(sharpness) / 200.0)

    return float(size_score * 0.3 + center_score * 0.3 + sharp_score * 0.4)


def classify_frames(
    pool_frames: list[str],
    frame_dir: Path,
    face_app: FaceAnalysis,
    kept_frames: list[str],
    emb_matrix: np.ndarray,
    n_main: int,
    n_neutral: int,
) -> tuple[list[str], list[str]]:
    """Full classification pipeline to produce main and neutral frame sets.

    Steps:
    - Run quality face detection on all pool frames.
    - DBSCAN clustering on face embeddings (eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES).
    - Prioritize frames by cluster frequency (largest first, then smaller, then noise).
    - Score face frames by aesthetic quality, keep top config.QUALITY_KEEP_RATIO.
    - Run farthest_point_sampling on CLIP embeddings to select diverse main frames.
    - Neutral selection (ported from the working heuristic in `discoframes.ipynb`):
      choose frames **not in main** that are (a) not blank/black (pixel std > 20)
      and (b) **faceless** (any detected face with det_score >= 0.3 rejects).
      Then diversify with farthest_point_sampling.

    Args:
        pool_frames: Filenames to consider (subset of kept_frames).
        frame_dir: Directory containing images.
        face_app: Prepared insightface FaceAnalysis app.
        kept_frames: Filenames corresponding to rows of emb_matrix.
        emb_matrix: CLIP embedding matrix of shape (N, D).
        n_main: Number of main frames to select.
        n_neutral: Number of neutral frames to select.

    Returns:
        (main_frames, neutral_frames) as lists of filenames.
    """
    frame_dir = Path(frame_dir)

    clip_index = {name: i for i, name in enumerate(kept_frames)}
    pool = [f for f in pool_frames if f in clip_index]
    if not pool:
        return [], []

    face_rows: list[dict[str, Any]] = []
    face_embs: list[np.ndarray] = []

    for fname in pool:
        img = load_image(frame_dir / fname)
        if img is None:
            continue
        faces = detect_quality_faces(img, face_app)
        if not faces:
            continue
        for fd in faces:
            q = score_frame_quality(img.size, fd["bbox"], fd["sharpness"])
            face_rows.append({"frame": fname, "quality": q, **fd})
            face_embs.append(np.asarray(fd["embedding"], dtype=np.float32))

    if not face_embs:
        main_frames: list[str] = []

        neutral_candidates: list[str] = []
        for fname in pool:
            img = load_image(frame_dir / fname)
            if img is None:
                continue
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if frame_pixel_std_bgr(bgr) <= 20.0:
                continue
            if has_any_face(img, face_app, min_det_score=0.3):
                continue
            neutral_candidates.append(fname)

        idxs_n = [clip_index[f] for f in neutral_candidates]
        embs_n = emb_matrix[idxs_n] if idxs_n else np.zeros((0, emb_matrix.shape[1]), dtype=np.float32)
        sel_n = farthest_point_sampling(embs_n, n_neutral)
        neutral_frames = [neutral_candidates[i] for i in sel_n]
        return main_frames, neutral_frames

    X = np.stack(face_embs, axis=0)
    labels = DBSCAN(
        eps=float(config.DBSCAN_EPS),
        min_samples=int(config.DBSCAN_MIN_SAMPLES),
        metric="euclidean",
    ).fit_predict(X)

    frame_best_quality: dict[str, float] = {}
    frame_best_label: dict[str, int] = {}

    for row, lab in zip(face_rows, labels, strict=False):
        fname = str(row["frame"])
        q = float(row["quality"])
        prev_q = frame_best_quality.get(fname, -1.0)
        if q > prev_q:
            frame_best_quality[fname] = q
            frame_best_label[fname] = int(lab)

    non_noise = [int(l) for l in labels.tolist() if int(l) >= 0]
    counts = Counter(non_noise)

    def cluster_priority(label: int) -> tuple[int, int]:
        if label < 0:
            return (1, 0)
        return (0, -counts.get(label, 0))

    face_frames = list(frame_best_quality.keys())
    face_frames.sort(
        key=lambda f: (
            cluster_priority(frame_best_label.get(f, -1)),
            -frame_best_quality.get(f, 0.0),
        )
    )

    face_frames_by_q = sorted(face_frames, key=lambda f: frame_best_quality.get(f, 0.0), reverse=True)
    k_keep = max(1, int(np.ceil(float(config.QUALITY_KEEP_RATIO) * len(face_frames_by_q))))
    face_kept = set(face_frames_by_q[:k_keep])

    main_candidates = [f for f in face_frames if f in face_kept]
    idxs = [clip_index[f] for f in main_candidates]
    embs = emb_matrix[idxs] if idxs else np.zeros((0, emb_matrix.shape[1]), dtype=np.float32)
    sel = farthest_point_sampling(embs, n_main)
    main_frames = [main_candidates[i] for i in sel]
    main_set = set(main_frames)

    neutral_candidates: list[str] = []
    for fname in pool:
        if fname in main_set:
            continue
        img = load_image(frame_dir / fname)
        if img is None:
            continue
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if frame_pixel_std_bgr(bgr) <= 20.0:
            continue
        if has_any_face(img, face_app, min_det_score=0.3):
            continue
        neutral_candidates.append(fname)

    if len(neutral_candidates) < int(n_neutral):
        neutral_candidates = [f for f in pool if f not in main_set]

    idxs_r = [clip_index[f] for f in neutral_candidates]
    embs_r = emb_matrix[idxs_r] if idxs_r else np.zeros((0, emb_matrix.shape[1]), dtype=np.float32)
    sel_r = farthest_point_sampling(embs_r, n_neutral)
    neutral_frames = [neutral_candidates[i] for i in sel_r]

    return main_frames, neutral_frames

