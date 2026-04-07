"""Compare DIS-CO experiment results from CSV files (unified naming convention)."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path


def slug_movie(name: str) -> str:
    """Match run.py: lowercase, strip non-alphanumeric (e.g. 'Forrest Gump' -> 'forrestgump')."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "unknown"


def llm_filename_token(llm: str) -> str:
    """Filename segment for LLM: HF-style ids get slashes -> dashes, lowercased (same as run.py)."""
    if "/" in llm:
        return llm.replace("/", "-").lower()
    return llm


@dataclass
class Metrics:
    overall_pct: float
    main_pct: float
    neutral_pct: float
    frames: int
    errors: int


def compute_metrics(path: Path) -> Metrics:
    """Compute accuracy metrics from a DIS-CO results CSV."""
    main_total = main_correct = 0
    neutral_total = neutral_correct = 0
    total = 0
    correct_count = 0
    errors = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            llm_resp = row.get("llm_response") or ""
            if llm_resp.startswith("ERROR:"):
                errors += 1

            is_correct = (row.get("correct") or "").strip() == "True"
            if is_correct:
                correct_count += 1

            ft = (row.get("frame_type") or "").strip()
            if ft == "Main":
                main_total += 1
                if is_correct:
                    main_correct += 1
            elif ft == "Neutral":
                neutral_total += 1
                if is_correct:
                    neutral_correct += 1

    overall_pct = 100.0 * correct_count / total if total else 0.0
    main_pct = 100.0 * main_correct / main_total if main_total else 0.0
    neutral_pct = 100.0 * neutral_correct / neutral_total if neutral_total else 0.0

    return Metrics(
        overall_pct=overall_pct,
        main_pct=main_pct,
        neutral_pct=neutral_pct,
        frames=total,
        errors=errors,
    )


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def parse_llm_from_stem(stem: str, movie_slug: str, source: str, test_type: str) -> str | None:
    """Extract LLM token from filename stem: {movie}_{source}_{test_type}_{llm}."""
    prefix = f"{movie_slug}_{source}_{test_type}_"
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix) :] or None


def parse_result_csv_stem(stem: str) -> tuple[str, str, str, str] | None:
    """Parse `{movie}_{source}_{test_type}_{llm}` (movie slug has no underscores)."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    source = parts[1]
    test_type = parts[2]
    if source not in ("local", "dataset"):
        return None
    if test_type not in ("image", "caption"):
        return None
    movie_slug = parts[0]
    llm = "_".join(parts[3:])
    return movie_slug, source, test_type, llm


def aggregate_metrics_from_paths(paths: list[Path]) -> Metrics:
    """Pool rows from multiple CSVs into one Metrics (same weighting as one combined file)."""
    main_total = main_correct = 0
    neutral_total = neutral_correct = 0
    total = 0
    correct_count = 0
    errors = 0

    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                llm_resp = row.get("llm_response") or ""
                if llm_resp.startswith("ERROR:"):
                    errors += 1

                is_correct = (row.get("correct") or "").strip() == "True"
                if is_correct:
                    correct_count += 1

                ft = (row.get("frame_type") or "").strip()
                if ft == "Main":
                    main_total += 1
                    if is_correct:
                        main_correct += 1
                elif ft == "Neutral":
                    neutral_total += 1
                    if is_correct:
                        neutral_correct += 1

    overall_pct = 100.0 * correct_count / total if total else 0.0
    main_pct = 100.0 * main_correct / main_total if main_total else 0.0
    neutral_pct = 100.0 * neutral_correct / neutral_total if neutral_total else 0.0

    return Metrics(
        overall_pct=overall_pct,
        main_pct=main_pct,
        neutral_pct=neutral_pct,
        frames=total,
        errors=errors,
    )


def read_movie_display_name(path: Path) -> str:
    """First non-empty ``movie`` cell in CSV, else movie slug from filename."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = (row.get("movie") or "").strip()
                if m:
                    return m
    except OSError:
        pass
    parsed = parse_result_csv_stem(path.stem)
    return parsed[0] if parsed else path.stem


def compare_llm(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    movie_slug = slug_movie(args.movie)
    source = args.source
    test_type = args.test_type

    pattern = str(out_dir / f"{movie_slug}_{source}_{test_type}_*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(
            f"No results found for {movie_slug}_{source}_{test_type}_*.csv in {out_dir}. "
            "Run experiments first with run.py."
        )
        return

    title = f"Comparing LLMs on: {movie_slug} | {source} | {test_type}"
    print(title)
    print("=" * len(title))

    rows: list[tuple[str, Metrics]] = []
    for p_str in paths:
        p = Path(p_str)
        stem = p.stem
        llm_name = parse_llm_from_stem(stem, movie_slug, source, test_type)
        if llm_name is None:
            continue
        rows.append((llm_name, compute_metrics(p)))

    if not rows:
        print(
            f"No valid result files matched {movie_slug}_{source}_{test_type}_*.csv in {out_dir}. "
            "Run experiments first with run.py."
        )
        return

    # Sort by LLM name for stable output
    rows.sort(key=lambda x: x[0])

    col_w = 12
    print(
        f"{'LLM':<{col_w}}| {'Overall':>8} | {'Main':>8} | {'Neutral':>8} | {'Frames':>6} | {'Errors':>6}"
    )
    print("-" * col_w + "|----------|----------|----------|--------|--------")
    for llm_name, m in rows:
        print(
            f"{llm_name:<{col_w}}| {fmt_pct(m.overall_pct):>8} | {fmt_pct(m.main_pct):>8} | "
            f"{fmt_pct(m.neutral_pct):>8} | {m.frames:>6} | {m.errors:>6}"
        )


def compare_test(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    movie_slug = slug_movie(args.movie)
    source = args.source
    llm_tok = llm_filename_token(args.llm)

    image_path = out_dir / f"{movie_slug}_{source}_image_{llm_tok}.csv"
    caption_path = out_dir / f"{movie_slug}_{source}_caption_{llm_tok}.csv"

    missing: list[str] = []
    if not image_path.is_file():
        missing.append(str(image_path.name))
    if not caption_path.is_file():
        missing.append(str(caption_path.name))

    if missing:
        print(
            f"Missing expected file(s): {', '.join(missing)} in {out_dir}. "
            f"Expected {movie_slug}_{source}_image_{llm_tok}.csv and "
            f"{movie_slug}_{source}_caption_{llm_tok}.csv. Run experiments first with run.py."
        )
        return

    m_image = compute_metrics(image_path)
    m_caption = compute_metrics(caption_path)

    title = f"Comparing test types on: {movie_slug} | {source} | {llm_tok}"
    print(title)
    print("=" * len(title))

    col_w = 12
    print(
        f"{'Test Type':<{col_w}}| {'Overall':>8} | {'Main':>8} | {'Neutral':>8} | {'Frames':>6} | {'Errors':>6}"
    )
    print("-" * col_w + "|----------|----------|----------|--------|--------")

    def print_row(label: str, m: Metrics) -> None:
        print(
            f"{label:<{col_w}}| {fmt_pct(m.overall_pct):>8} | {fmt_pct(m.main_pct):>8} | "
            f"{fmt_pct(m.neutral_pct):>8} | {m.frames:>6} | {m.errors:>6}"
        )

    print_row("image", m_image)
    print_row("caption", m_caption)

    d_overall = m_caption.overall_pct - m_image.overall_pct
    d_main = m_caption.main_pct - m_image.main_pct
    d_neutral = m_caption.neutral_pct - m_image.neutral_pct

    print("-" * col_w + "|----------|----------|----------|--------|--------")
    print(
        f"{'delta':<{col_w}}| {d_overall:>+7.1f}% | {d_main:>+7.1f}% | {d_neutral:>+7.1f}% | {'':>6} | {'':>6}"
    )


def compare_leaderboard(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    source = args.source
    test_type = args.test_type
    assert source is not None and test_type is not None

    pattern = f"*_{source}_{test_type}_*.csv"
    paths = sorted(p for p in out_dir.glob(pattern) if p.is_file())
    paths = [p for p in paths if parse_result_csv_stem(p.stem) is not None]

    if not paths:
        print(
            f"No result CSVs matching *_{source}_{test_type}_*.csv in {out_dir}. "
            "Run experiments first with run.py."
        )
        return

    if args.by == "model":
        llm_to_paths: dict[str, list[Path]] = {}
        for p in paths:
            parsed = parse_result_csv_stem(p.stem)
            if not parsed:
                continue
            _movie, src, tt, llm = parsed
            if src != source or tt != test_type:
                continue
            llm_to_paths.setdefault(llm, []).append(p)

        rows: list[tuple[str, Metrics, int]] = []
        for llm, ps in sorted(llm_to_paths.items()):
            m = aggregate_metrics_from_paths(sorted(ps))
            n_movies = len(ps)
            rows.append((llm, m, n_movies))

        rows.sort(key=lambda x: (-x[1].overall_pct, x[0]))

        title = f"Model Leaderboard (all movies, {test_type} test)"
        print(title)
        print("=" * len(title))
        w_rank, w_ov, w_main, w_neu, w_mov, w_fr = 4, 7, 6, 7, 6, 6
        w_llm = max(18, max(len(llm) for llm, _, _ in rows))
        print(
            f"{'Rank':<{w_rank}}| {'LLM':<{w_llm}}| {'Overall':>{w_ov}} | "
            f"{'Main':>{w_main}} | {'Neutral':>{w_neu}} | {'Movies':>{w_mov}} | {'Frames':>{w_fr}}"
        )
        sep = (
            f"{'-' * w_rank}|{'-' * w_llm}|{'-' * (w_ov + 2)}|"
            f"{'-' * (w_main + 2)}|{'-' * (w_neu + 2)}|{'-' * (w_mov + 2)}|{'-' * (w_fr + 2)}"
        )
        print(sep)
        for rank, (llm, m, n_movies) in enumerate(rows, start=1):
            print(
                f"{rank:<{w_rank}}| {llm:<{w_llm}}| {fmt_pct(m.overall_pct):>{w_ov}} | "
                f"{fmt_pct(m.main_pct):>{w_main}} | {fmt_pct(m.neutral_pct):>{w_neu}} | "
                f"{n_movies:>{w_mov}} | {m.frames:>{w_fr}}"
            )
        return

    # --by movie
    llm_tok = llm_filename_token(args.llm)
    movie_rows: list[tuple[str, Metrics]] = []
    for p in paths:
        parsed = parse_result_csv_stem(p.stem)
        if not parsed:
            continue
        _movie_slug, src, tt, llm = parsed
        if src != source or tt != test_type or llm != llm_tok:
            continue
        m = compute_metrics(p)
        display = read_movie_display_name(p)
        movie_rows.append((display, m))

    if not movie_rows:
        print(
            f"No result CSVs for LLM '{llm_tok}' matching *_{source}_{test_type}_{llm_tok}.csv "
            f"in {out_dir}. Run experiments first with run.py."
        )
        return

    movie_rows.sort(key=lambda x: (-x[1].overall_pct, x[0].lower()))

    title = f"Movie Leaderboard ({llm_tok}, {test_type} test)"
    print(title)
    print("=" * len(title))
    w_rank, w_ov, w_main, w_neu, w_fr = 4, 7, 6, 7, 6
    w_mov = max(18, max(len(name) for name, _ in movie_rows))
    print(
        f"{'Rank':<{w_rank}}| {'Movie':<{w_mov}}| {'Overall':>{w_ov}} | "
        f"{'Main':>{w_main}} | {'Neutral':>{w_neu}} | {'Frames':>{w_fr}}"
    )
    sep = (
        f"{'-' * w_rank}|{'-' * w_mov}|{'-' * (w_ov + 2)}|"
        f"{'-' * (w_main + 2)}|{'-' * (w_neu + 2)}|{'-' * (w_fr + 2)}"
    )
    print(sep)
    for rank, (name, m) in enumerate(movie_rows, start=1):
        print(
            f"{rank:<{w_rank}}| {name:<{w_mov}}| {fmt_pct(m.overall_pct):>{w_ov}} | "
            f"{fmt_pct(m.main_pct):>{w_main}} | {fmt_pct(m.neutral_pct):>{w_neu}} | {m.frames:>{w_fr}}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare DIS-CO CSV results (unified {movie}_{source}_{test_type}_{llm}.csv naming)."
    )
    p.add_argument(
        "--compare",
        required=True,
        choices=["llm", "test", "leaderboard"],
        help="llm: compare LLMs for same movie/source/test-type; test: image vs caption for same LLM; "
        "leaderboard: ranked summary across all result CSVs in --output-dir.",
    )
    p.add_argument(
        "--by",
        choices=["model", "movie"],
        default=None,
        help='With --compare leaderboard: "model" ranks LLMs overall; "movie" ranks movies for one LLM (needs --llm).',
    )
    p.add_argument(
        "--movie",
        required=False,
        default=None,
        type=str,
        help='Human-readable movie name (e.g. "Frozen", "Forrest Gump"). Required for --compare llm and test.',
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory containing result CSVs (default: current directory).",
    )
    p.add_argument(
        "--test-type",
        choices=["image", "caption"],
        default=None,
        help='Required for --compare llm. For --compare leaderboard defaults to "image" if omitted.',
    )
    p.add_argument(
        "--source",
        choices=["local", "dataset"],
        default=None,
        help='Required for --compare llm and test. For --compare leaderboard defaults to "dataset" if omitted.',
    )
    p.add_argument(
        "--llm",
        type=str,
        default=None,
        help="For --compare test: LLM token as in filename. For --compare leaderboard --by movie: required.",
    )

    args = p.parse_args()

    if args.by is not None and args.compare != "leaderboard":
        p.error("--by is only valid with --compare leaderboard.")

    if args.compare == "llm":
        if args.movie is None:
            p.error("--movie is required when --compare llm.")
        if args.test_type is None:
            p.error("--test-type is required when --compare llm.")
        if args.source is None:
            p.error("--source is required when --compare llm.")
    elif args.compare == "test":
        if args.movie is None:
            p.error("--movie is required when --compare test.")
        if args.llm is None:
            p.error("--llm is required when --compare test.")
        if args.source is None:
            p.error("--source is required when --compare test.")
    elif args.compare == "leaderboard":
        if args.by is None:
            p.error('--by is required when --compare leaderboard ("model" or "movie").')
        if args.by == "movie" and args.llm is None:
            p.error("--llm is required when --compare leaderboard --by movie.")
        if args.test_type is None:
            args.test_type = "image"
        if args.source is None:
            args.source = "dataset"

    return args


def main() -> None:
    args = parse_args()
    if args.compare == "llm":
        compare_llm(args)
    elif args.compare == "test":
        compare_test(args)
    else:
        compare_leaderboard(args)


if __name__ == "__main__":
    main()
