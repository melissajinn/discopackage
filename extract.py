"""CLI entry point for extracting and classifying movie frames."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config
from utils.dedup import build_diverse_pool, deduplicate_frames, load_clip_model
from utils.faces import classify_frames, load_face_model
from utils.scene import detect_scenes, extract_frames


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the extraction pipeline."""
    p = argparse.ArgumentParser(description="Extract and classify representative movie frames.")
    p.add_argument("--path", required=True, type=Path, help="Path to movie file (e.g., .mp4).")
    p.add_argument("--output", required=True, type=Path, help="Output directory.")
    p.add_argument("--skip-start", type=float, default=float(config.SKIP_START), help="Seconds to skip at start.")
    p.add_argument("--skip-end", type=float, default=float(config.SKIP_END), help="Seconds to skip at end.")
    p.add_argument("--clip-threshold", type=float, default=float(config.CLIP_DUP_THRESHOLD), help="CLIP dup threshold.")
    p.add_argument("--n-main", type=int, default=int(config.N_MAIN), help="Number of main frames.")
    p.add_argument("--n-neutral", type=int, default=int(config.N_NEUTRAL), help="Number of neutral frames.")
    p.add_argument(
        "--pool-multiplier",
        type=int,
        default=int(config.POOL_MULTIPLIER),
        help="Pool size multiplier relative to (n_main + n_neutral).",
    )
    return p.parse_args()


def run_pipeline(
    movie_path: Path,
    output_dir: Path,
    skip_start: float = float(config.SKIP_START),
    skip_end: float = float(config.SKIP_END),
    clip_threshold: float = float(config.CLIP_DUP_THRESHOLD),
    n_main: int = int(config.N_MAIN),
    n_neutral: int = int(config.N_NEUTRAL),
    pool_multiplier: int = int(config.POOL_MULTIPLIER),
) -> None:
    """Run the full extraction + classification pipeline and write outputs."""
    movie_path = Path(movie_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "frames_raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    scenes = detect_scenes(movie_path)
    saved = extract_frames(
        movie_path=movie_path,
        scenes=scenes,
        output_dir=raw_dir,
        skip_start=float(skip_start),
        skip_end=float(skip_end),
    )

    clip_model, clip_preprocess = load_clip_model()
    kept_frames, kept_embs = deduplicate_frames(
        frame_dir=raw_dir,
        model=clip_model,
        preprocess=clip_preprocess,
        threshold=float(clip_threshold),
    )

    pool_size = int(pool_multiplier) * (int(n_main) + int(n_neutral))
    pool_frames = build_diverse_pool(kept_frames, kept_embs, pool_size=pool_size)

    face_app = load_face_model()
    main_frames, neutral_frames = classify_frames(
        pool_frames=pool_frames,
        frame_dir=raw_dir,
        face_app=face_app,
        kept_frames=kept_frames,
        emb_matrix=kept_embs,
        n_main=int(n_main),
        n_neutral=int(n_neutral),
    )

    main_dir = output_dir / "main"
    neutral_dir = output_dir / "neutral"
    shutil.rmtree(main_dir, ignore_errors=True)
    shutil.rmtree(neutral_dir, ignore_errors=True)
    main_dir.mkdir(parents=True, exist_ok=True)
    neutral_dir.mkdir(parents=True, exist_ok=True)

    for fname in main_frames:
        src = raw_dir / fname
        if src.exists():
            shutil.copy2(src, main_dir / fname)

    for fname in neutral_frames:
        src = raw_dir / fname
        if src.exists():
            shutil.copy2(src, neutral_dir / fname)

    print("DISCO frames extraction summary")
    print(f"- Movie: {movie_path}")
    print(f"- Scenes detected: {len(scenes)}")
    print(f"- Frames saved (pre-dedup): {saved}")
    print(f"- Frames kept (post-CLIP-dedup): {len(kept_frames)}")
    print(f"- Diverse pool size: {len(pool_frames)}")
    print(f"- Main frames exported: {len(main_frames)} -> {main_dir}")
    print(f"- Neutral frames exported: {len(neutral_frames)} -> {neutral_dir}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_pipeline(
        movie_path=args.path,
        output_dir=args.output,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        clip_threshold=args.clip_threshold,
        n_main=args.n_main,
        n_neutral=args.n_neutral,
        pool_multiplier=args.pool_multiplier,
    )


if __name__ == "__main__":
    main()

