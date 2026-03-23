"""CLIP deduplication and diversity sampling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import chromadb
from chromadb.config import Settings

import torch
import clip  # type: ignore


def load_image(path: Path) -> Image.Image | None:
    """Load an image from disk as RGB.

    Args:
        path: Path to an image file.

    Returns:
        PIL Image in RGB, or None if loading fails.
    """
    try:
        return Image.open(Path(path)).convert("RGB")
    except Exception:
        return None


def load_clip_model() -> tuple[Any, Any]:
    """Load CLIP ViT-B/32 on CPU.

    Returns:
        (model, preprocess) as returned by `clip.load`.
    """
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()
    return model, preprocess


def embed_image(img: Image.Image, model: Any, preprocess: Any) -> np.ndarray:
    """Embed an image with CLIP and return a normalized numpy vector.

    Args:
        img: PIL image (RGB).
        model: CLIP model.
        preprocess: CLIP preprocess callable.

    Returns:
        L2-normalized embedding as a numpy array of shape (D,).
    """
    with torch.no_grad():
        image_tensor = preprocess(img).unsqueeze(0)
        feats = model.encode_image(image_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype(np.float32, copy=False)


def deduplicate_frames(
    frame_dir: Path,
    model: Any,
    preprocess: Any,
    threshold: float = 0.90,
) -> tuple[list[str], np.ndarray]:
    """Deduplicate frames using CLIP cosine similarity backed by ChromaDB.

    Frames are embedded with CLIP, inserted into an in-memory Chroma collection,
    and rejected if their nearest neighbor has cosine similarity >= threshold.

    Args:
        frame_dir: Directory containing extracted frames.
        model: CLIP model.
        preprocess: CLIP preprocess callable.
        threshold: Cosine similarity threshold for duplicates.

    Returns:
        (kept_filenames, kept_embeddings_matrix)
    """
    frame_dir = Path(frame_dir)
    files = sorted(
        p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection_name = "frames_clip"
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    kept: list[str] = []
    kept_embs: list[np.ndarray] = []

    for path in files:
        img = load_image(path)
        if img is None:
            continue

        emb = embed_image(img, model, preprocess)

        if not kept:
            kept.append(path.name)
            kept_embs.append(emb)
            collection.add(
                ids=[path.name],
                embeddings=[emb.tolist()],
                metadatas=[{"filename": path.name}],
            )
            continue

        res = collection.query(
            query_embeddings=[emb.tolist()],
            n_results=min(5, len(kept)),
            include=["distances"],
        )

        is_dup = False
        dists = (res or {}).get("distances") or []
        if dists and dists[0]:
            for d in dists[0]:
                sim = 1.0 - float(d)  # cosine distance -> cosine similarity
                if sim >= float(threshold):
                    is_dup = True
                    break

        if is_dup:
            continue

        kept.append(path.name)
        kept_embs.append(emb)
        collection.add(
            ids=[path.name],
            embeddings=[emb.tolist()],
            metadatas=[{"filename": path.name}],
        )

    emb_matrix = np.stack(kept_embs, axis=0) if kept_embs else np.zeros((0, 0), dtype=np.float32)
    return kept, emb_matrix


def farthest_point_sampling(embs: np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point sampling for diversity.

    Uses cosine distance (assumes embeddings are L2-normalized).

    Args:
        embs: Embedding matrix of shape (N, D).
        k: Number of points to select.

    Returns:
        List of selected indices into `embs`.
    """
    if embs.ndim != 2 or embs.shape[0] == 0 or k <= 0:
        return []

    n = int(embs.shape[0])
    k = min(int(k), n)

    selected = [0]
    # cosine distance = 1 - dot (for normalized vectors)
    min_dists = 1.0 - (embs @ embs[selected[0]].T)

    for _ in range(1, k):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        d = 1.0 - (embs @ embs[next_idx].T)
        min_dists = np.minimum(min_dists, d)

    return selected


def build_diverse_pool(
    kept_frames: list[str],
    kept_embs: np.ndarray,
    pool_size: int,
) -> list[str]:
    """Build a diverse pool of frames using farthest-point sampling.

    Args:
        kept_frames: Filenames corresponding to rows of `kept_embs`.
        kept_embs: Embedding matrix of shape (N, D).
        pool_size: Desired pool size (will be capped by N).

    Returns:
        List of selected filenames (subset of kept_frames).
    """
    if not kept_frames or kept_embs.ndim != 2 or kept_embs.shape[0] == 0:
        return []

    idxs = farthest_point_sampling(kept_embs, int(pool_size))
    return [kept_frames[i] for i in idxs]

