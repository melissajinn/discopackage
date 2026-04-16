"""Run DIS-CO movie identification experiments with VLMs.

This CLI supports two modes:
- ``local``: run on locally extracted frames (output of ``extract.py``).
- ``dataset``: run on the HuggingFace DIS-CO datasets.
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import string
import time
import multiprocessing as mp
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Literal, Tuple
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the DIS-CO runner."""
    p = argparse.ArgumentParser(description="Run DIS-CO VLM movie identification experiments.")

    p.add_argument(
        "--mode",
        required=True,
        choices=["local", "dataset"],
        help="Run on locally extracted frames ('local') or the DIS-CO dataset ('dataset').",
    )

    # Local mode arguments
    p.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory containing 'main/' and 'neutral/' subfolders with extracted frames (for --mode local).",
    )
    p.add_argument(
        "--movie-name",
        type=str,
        help="Ground truth movie name for local frames (for --mode local).",
    )

    # Dataset mode arguments
    p.add_argument(
        "--model-size",
        choices=["main", "mini"],
        help="DIS-CO dataset size to use for --mode dataset: 'main' or 'mini'.",
    )
    p.add_argument(
        "--movie",
        type=str,
        help="Optional movie name filter for --mode dataset (e.g., 'Forrest Gump').",
    )

    # LLM + run configuration
    p.add_argument(
        "--llm",
        required=True,
        choices=["gemini", "claude", "chatgpt", "vllm"],
        help="Which vision-language backend: cloud APIs or local vLLM.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the auto-generated results CSV (default: current directory).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls to avoid rate limiting (default: 1.0). Ignored for --llm vllm.",
    )
    p.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help="HuggingFace model id for vLLM offline inference (required when --llm vllm).",
    )
    p.add_argument(
        "--caption-mode",
        action="store_true",
        help=(
            "Caption-based test: guess the movie from text only. "
            "Local: generate a caption from each image, then guess from that text. "
            "Dataset: use the dataset Caption column only (no image download)."
        ),
    )

    args = p.parse_args()

    if args.mode == "local":
        if args.frames_dir is None:
            p.error("--frames-dir is required when --mode local.")
        if args.movie_name is None:
            p.error("--movie-name is required when --mode local.")
    elif args.mode == "dataset":
        if args.model_size is None:
            p.error("--model-size is required when --mode dataset.")

    if args.llm == "vllm":
        if not args.vllm_model:
            p.error("--vllm-model is required when --llm vllm.")
    elif args.vllm_model:
        p.error("--vllm-model is only valid when --llm vllm.")

    return args

PromptType = Literal["gemini", "claude", "chatgpt", "vllm"]

CAPTION_IMAGE_PROMPT = (
    "Describe this movie frame in detail. Include information about the characters, setting, "
    "lighting, actions, and any notable visual elements. Do not mention the movie title."
)


def guess_from_caption_prompt(caption: str) -> str:
    return (
        "Based on this description of a movie frame, what movie is it from? "
        "Reply with only the movie title, nothing else.\n\n"
        f"Description: {caption}"
    )


def _slug_movie_for_filename(name: str) -> str:
    """Lowercase; remove spaces and non-alphanumeric characters (e.g. 'Forrest Gump' -> 'forrestgump')."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "unknown"


def _llm_token_for_filename(llm: str, vllm_model: str | None) -> str:
    """LLM segment: cloud names as-is; vLLM uses HF id with slashes -> dashes, lowercased."""
    if llm == "vllm" and vllm_model:
        return vllm_model.replace("/", "-").lower()
    return llm


def generate_output_filename(
    movie_name: str,
    mode: str,
    caption_mode: bool,
    llm: str,
    vllm_model: str | None = None,
) -> Path:
    """Build `{movie}_{source}_{test_type}_{llm}.csv` as a Path (filename only)."""
    movie = _slug_movie_for_filename(movie_name)
    source = "dataset" if mode == "dataset" else "local"
    test_type = "caption" if caption_mode else "image"
    llm_part = _llm_token_for_filename(llm, vllm_model)
    return Path(f"{movie}_{source}_{test_type}_{llm_part}.csv")


def read_completed_frame_files(csv_path: Path) -> set[str]:
    """Load frame_file values that already have a successful result (last row per frame wins).

    Rows with empty ``llm_response`` or responses starting with ``ERROR:`` are not completed.
    """
    if not csv_path.is_file():
        return set()
    last_by_frame: dict[str, dict[str, str]] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set()
            for row in reader:
                ff = (row.get("frame_file") or "").strip()
                if ff:
                    last_by_frame[ff] = row
    except OSError:
        return set()
    completed: set[str] = set()
    for ff, row in last_by_frame.items():
        llm_resp = (row.get("llm_response") or "").strip()
        if llm_resp and not llm_resp.startswith("ERROR:"):
            completed.add(ff)
    return completed


def dataset_row_frame_file(row: Any) -> str:
    """Frame id string for a dataset row (matches process_and_write_row)."""
    scene_number = row.get("Scene_Number")
    shot_number = row.get("Shot_Number")
    frame_type = row.get("Frame_Type", "Unknown")
    return f"{frame_type}_scene_{scene_number}_shot_{shot_number}.jpg"


def estimate_gpus_needed(model_name: str) -> int:
    """Estimate how many GPUs a model needs based on its name."""
    name = (model_name or "").lower()
    match = re.search(r"(\d+)[bB]", name)
    if match:
        billions = int(match.group(1))
        if billions <= 8:
            return 1  # 7B/8B fits on 1 GPU
        if billions <= 15:
            return 2  # 13B/15B needs 2 GPUs
        if billions <= 34:
            return 4  # 34B needs 4 GPUs
        return 8  # 70B+ needs 8 GPUs
    return 1  # Default: assume small model


def _chunk_round_robin(items: list[Any], num_chunks: int) -> list[list[Any]]:
    chunks: list[list[Any]] = [[] for _ in range(max(1, num_chunks))]
    for i, item in enumerate(items):
        chunks[i % len(chunks)].append(item)
    return chunks


def _vllm_parallel_plan(vllm_model: str) -> tuple[int, int, int, list[int]]:
    """Return (num_workers, gpus_needed, num_gpus, free_gpus) for vLLM auto parallelism.

    - num_gpus: total visible GPUs (per torch).
    - free_gpus: subset of visible GPUs with enough free memory to safely schedule work.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0 or not torch.cuda.is_available():
        # CPU-only fallback so the rest of the script can still run (vLLM will fail later).
        return 1, 1, 1, [0]

    # Filter to only GPUs with enough free memory to avoid grabbing GPUs in use by others.
    # Note: This is a best-effort heuristic; CUDA memory reporting can be noisy.
    MIN_FREE_GB = 20.0
    free_gpus: list[int] = []
    try:
        for i in range(num_gpus):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024**3)
            if free_gb >= MIN_FREE_GB:
                free_gpus.append(i)
    except Exception:  # noqa: BLE001
        # If querying memory fails, fall back to "all visible GPUs".
        free_gpus = list(range(num_gpus))

    if not free_gpus:
        free_gpus = [0]

    gpus_needed = estimate_gpus_needed(vllm_model)
    gpus_needed = max(1, min(gpus_needed, len(free_gpus)))

    if gpus_needed >= len(free_gpus):
        # Tensor parallel across all eligible GPUs
        return 1, gpus_needed, num_gpus, free_gpus

    # Data parallel: multiple workers, each a TP group of size gpus_needed
    num_workers = max(1, len(free_gpus) // gpus_needed)
    return num_workers, gpus_needed, num_gpus, free_gpus


def _merge_tmp_csvs(final_path: Path, tmp_paths: list[Path], *, resume: bool) -> None:
    """Merge worker CSVs into final output, writing header once."""
    if not tmp_paths:
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and final_path.exists() else "w"
    with final_path.open(open_mode, newline="", encoding="utf-8") as out_f:
        out_wrote_header = resume and final_path.exists()
        w = csv.writer(out_f)
        for p in tmp_paths:
            with p.open(newline="", encoding="utf-8") as in_f:
                r = csv.reader(in_f)
                try:
                    header = next(r)
                except StopIteration:
                    continue
                if not out_wrote_header:
                    w.writerow(header)
                    out_wrote_header = True
                for row in r:
                    w.writerow(row)


def _run_local_vllm_worker(
    gpu_ids: list[int],
    items: list[tuple[str, str, str]],
    vllm_model_name: str,
    tp_size: int,
    tmp_csv: str,
    caption_mode_flag: bool,
) -> None:
    """Worker for vLLM local-mode data parallelism."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    from vllm import LLM

    llm_instance = LLM(
        model=vllm_model_name,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )

    image_prompt = "What movie is this frame from? Reply with only the movie title, nothing else."
    with Path(tmp_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if caption_mode_flag:
            writer.writerow(
                ["movie", "frame_type", "frame_file", "llm", "caption", "llm_response", "correct"]
            )
        else:
            writer.writerow(["movie", "frame_type", "frame_file", "llm", "llm_response", "correct"])

        for idx, (m_name, f_type, f_path_str) in enumerate(items, start=1):
            frame_path = Path(f_path_str)
            print(
                f"[worker {os.environ.get('CUDA_VISIBLE_DEVICES','?')}] {idx}/{len(items)} {frame_path.name}"
            )
            caption = ""
            try:
                img = Image.open(frame_path)
            except Exception as e:  # noqa: BLE001
                if caption_mode_flag:
                    caption = f"ERROR: Failed to open image {frame_path}: {e}"
                    llm_response = "ERROR: caption step failed"
                else:
                    llm_response = f"ERROR: Failed to open image {frame_path}: {e}"
                correct = False
            else:
                if caption_mode_flag:
                    caption, cap_err = call_llm_with_status(
                        "vllm",
                        CAPTION_IMAGE_PROMPT,
                        img,
                        vllm_instance=llm_instance,
                        vllm_model=vllm_model_name,
                        vllm_max_tokens=1024,
                    )
                    if cap_err:
                        llm_response = "ERROR: caption step failed"
                        correct = False
                    else:
                        guess_prompt = guess_from_caption_prompt(caption)
                        llm_response, guess_err = call_llm_text_with_status(
                            "vllm",
                            guess_prompt,
                            vllm_instance=llm_instance,
                        )
                        correct = False if guess_err else is_correct_response(llm_response, m_name)
                else:
                    llm_response, had_error = call_llm_with_status(
                        "vllm",
                        image_prompt,
                        img,
                        vllm_instance=llm_instance,
                        vllm_model=vllm_model_name,
                    )
                    correct = False if had_error else is_correct_response(llm_response, m_name)

            if caption_mode_flag:
                writer.writerow([m_name, f_type, frame_path.name, "vllm", caption, llm_response, str(correct)])
            else:
                writer.writerow([m_name, f_type, frame_path.name, "vllm", llm_response, str(correct)])


def _run_dataset_vllm_worker(
    gpu_ids: list[int],
    row_indices: list[int],
    dataset_name: str,
    vllm_model_name: str,
    tp_size: int,
    tmp_csv: str,
    caption_mode_flag: bool,
) -> None:
    """Worker for vLLM dataset-mode data parallelism (single-movie path)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    from datasets import load_dataset
    from vllm import LLM

    ds = load_dataset(dataset_name, split="train")
    if caption_mode_flag:
        ds = ds.remove_columns(["Image_File"])

    llm_instance = LLM(
        model=vllm_model_name,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )

    image_prompt = "What movie is this frame from? Reply with only the movie title, nothing else."
    with Path(tmp_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if caption_mode_flag:
            writer.writerow(["movie", "frame_type", "frame_file", "llm", "caption", "llm_response", "correct"])
        else:
            writer.writerow(["movie", "frame_type", "frame_file", "llm", "llm_response", "correct"])

        for idx, row_idx in enumerate(row_indices, start=1):
            row = ds[row_idx]
            movie = row.get("Movie", "")
            frame_type = row.get("Frame_Type", "")
            frame_file = dataset_row_frame_file(row)
            print(f"[worker gpu={gpu_ids}] {idx}/{len(row_indices)} {frame_file}")

            caption = ""
            if caption_mode_flag:
                raw_cap = row.get("Caption")
                if not raw_cap or not str(raw_cap).strip():
                    caption = "ERROR: Missing or empty Caption."
                    llm_response = "ERROR: caption step failed"
                    correct = False
                else:
                    caption = str(raw_cap).strip()
                    guess_prompt = guess_from_caption_prompt(caption)
                    llm_response, guess_err = call_vllm_text(guess_prompt, llm_instance)
                    correct = False if guess_err else is_correct_response(llm_response, movie)
            else:
                image = row.get("Image_File")
                if image is None:
                    llm_response = "ERROR: Missing image."
                    correct = False
                else:
                    try:
                        img = _row_image_to_pil(image)
                        llm_response, had_error = call_vllm(
                            image_prompt,
                            img,
                            vllm_model_name,
                            llm_instance,
                        )
                        correct = False if had_error else is_correct_response(llm_response, movie)
                    except Exception as e:  # noqa: BLE001
                        llm_response = f"ERROR: {e}"
                        correct = False

            if caption_mode_flag:
                writer.writerow([movie, frame_type, frame_file, "vllm", caption, llm_response, str(correct)])
            else:
                writer.writerow([movie, frame_type, frame_file, "vllm", llm_response, str(correct)])


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image to a base64 JPEG string."""
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_gemini(prompt: str, base64_image: str) -> Tuple[str, bool]:
    """Call Google Gemini with the given prompt and base64-encoded image."""
    try:
        import google.generativeai as genai

        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            [
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(base64_image),
                },
            ]
        )
        text = getattr(response, "text", None) or ""
        if not text and hasattr(response, "candidates") and response.candidates:
            # Fallback if .text is not populated
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        return text.strip(), False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_claude(prompt: str, base64_image: str) -> Tuple[str, bool]:
    """Call Anthropic Claude with the given prompt and base64-encoded image."""
    try:
        import anthropic

        api_key = os.environ["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text_parts = []
        for block in message.content or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        text = "".join(text_parts).strip()
        return text, False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_chatgpt(prompt: str, base64_image: str) -> Tuple[str, bool]:
    """Call OpenAI GPT-4o with the given prompt and base64-encoded image."""
    try:
        from openai import OpenAI

        api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        choice = response.choices[0].message
        text = choice.content or ""
        return text.strip(), False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_gemini_text(prompt: str) -> Tuple[str, bool]:
    """Call Gemini with text-only prompt."""
    try:
        import google.generativeai as genai

        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or ""
        if not text and hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        return text.strip(), False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_claude_text(prompt: str) -> Tuple[str, bool]:
    """Call Claude with text-only prompt."""
    try:
        import anthropic

        api_key = os.environ["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        text_parts = []
        for block in message.content or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return "".join(text_parts).strip(), False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_chatgpt_text(prompt: str) -> Tuple[str, bool]:
    """Call OpenAI GPT-4o with text-only prompt."""
    try:
        from openai import OpenAI

        api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return text.strip(), False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_vllm_text(prompt: str, llm_instance: Any) -> Tuple[str, bool]:
    """Run vLLM text-only inference (no image)."""
    try:
        from vllm import SamplingParams

        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
        outputs = llm_instance.chat([conversation], sampling_params=sampling_params)
        text = outputs[0].outputs[0].text.strip()
        return text, False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_vllm(
    prompt: str,
    img: Image.Image,
    model_name: str,
    llm_instance: Any = None,
    *,
    max_tokens: int = 64,
) -> Tuple[str, bool]:
    """Run offline vLLM inference with PIL image via LLM.chat."""
    try:
        from vllm import LLM, SamplingParams

        if llm_instance is None:
            llm_instance = LLM(
                model=model_name,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 1},
                trust_remote_code=True,
            )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = llm_instance.chat([conversation], sampling_params=sampling_params)
        text = outputs[0].outputs[0].text.strip()
        return text, False
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}", True


def call_llm(
    llm: PromptType,
    prompt: str,
    img: Image.Image,
    *,
    vllm_instance: Any = None,
    vllm_model: str | None = None,
    vllm_max_tokens: int = 64,
) -> str:
    """Dispatch to the requested LLM with error handling."""
    if llm == "vllm":
        text, _ = call_vllm(prompt, img, vllm_model or "", vllm_instance, max_tokens=vllm_max_tokens)
        return text
    base64_image = encode_image_to_base64(img)
    if llm == "gemini":
        text, _ = call_gemini(prompt, base64_image)
    elif llm == "claude":
        text, _ = call_claude(prompt, base64_image)
    else:
        text, _ = call_chatgpt(prompt, base64_image)
    return text


def call_llm_with_status(
    llm: PromptType,
    prompt: str,
    img: Image.Image,
    *,
    vllm_instance: Any = None,
    vllm_model: str | None = None,
    vllm_max_tokens: int = 64,
) -> Tuple[str, bool]:
    """Call the LLM and return (response_text, had_error_flag)."""
    if llm == "vllm":
        return call_vllm(prompt, img, vllm_model or "", vllm_instance, max_tokens=vllm_max_tokens)
    base64_image = encode_image_to_base64(img)
    if llm == "gemini":
        return call_gemini(prompt, base64_image)
    if llm == "claude":
        return call_claude(prompt, base64_image)
    return call_chatgpt(prompt, base64_image)


def call_llm_text_with_status(
    llm: PromptType,
    prompt: str,
    *,
    vllm_instance: Any = None,
) -> Tuple[str, bool]:
    """Call the LLM with text only (no image). vllm_instance must be set when llm is vllm."""
    if llm == "gemini":
        return call_gemini_text(prompt)
    if llm == "claude":
        return call_claude_text(prompt)
    if llm == "chatgpt":
        return call_chatgpt_text(prompt)
    if llm == "vllm":
        if vllm_instance is None:
            return "ERROR: vLLM instance is required for text-only calls.", True
        return call_vllm_text(prompt, vllm_instance)
    return "ERROR: unknown llm backend.", True


def is_correct_response(llm_response: str, movie_name: str) -> bool:
    """Return True iff LLM response exactly matches the ground truth title.

    Matching is done case-insensitively after stripping whitespace and removing
    quotes/punctuation from both strings.
    """
    if not llm_response or llm_response.startswith("ERROR:"):
        return False

    def normalize(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        # Replace punctuation/quotes with spaces so hyphens/quotes don't glue words together.
        bad_chars = string.punctuation + '“”‘’`–—…'
        tbl = {ord(ch): " " for ch in bad_chars}
        s = s.translate(tbl)
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()

    return normalize(llm_response) == normalize(movie_name)


def iter_local_frames(frames_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (movie, frame_type, frame_path) for local mode.

    The movie name is not known here and is expected to be injected separately.
    This iterator yields empty string for movie, to be replaced by the caller.
    """
    for frame_type_label, sub in (("Main", "main"), ("Neutral", "neutral")):
        subdir = frames_dir / sub
        if not subdir.exists():
            continue
        files = sorted(p for p in subdir.iterdir() if p.is_file())
        for f in files:
            yield "", frame_type_label, f


def run_local(
    frames_dir: Path,
    movie_name: str,
    llm: PromptType,
    output: Path,
    delay: float,
    vllm_model: str | None = None,
    caption_mode: bool = False,
) -> None:
    """Run DIS-CO evaluation on locally extracted frames."""
    frames = list(iter_local_frames(frames_dir))
    total = len(frames)
    if total == 0:
        print(f"No frames found under {frames_dir} (expected 'main/' and/or 'neutral/').")
        return

    completed = read_completed_frame_files(output)
    remaining_frames = [(a, b, c) for (a, b, c) in frames if c.name not in completed]
    n_done = total - len(remaining_frames)
    if not remaining_frames:
        print(f"All {total} frames already processed in {output}. Nothing to do.")
        return
    if n_done > 0:
        print(f"Found existing results: {n_done}/{total} frames already processed. Resuming...")

    num_workers = 1
    gpus_needed = 1
    num_gpus = 1

    # vLLM: auto decide tensor parallelism vs data parallelism
    if llm == "vllm":
        assert vllm_model is not None
        num_workers, gpus_needed, num_gpus, free_gpus = _vllm_parallel_plan(vllm_model)
        if gpus_needed >= len(free_gpus):
            print(
                f"vLLM: model needs {gpus_needed} GPU(s), using tensor parallelism across {len(free_gpus)} eligible GPU(s)"
            )
        else:
            print(
                f"vLLM: model fits on {gpus_needed} GPU(s), launching {num_workers} data-parallel workers across {len(free_gpus)} eligible GPU(s)"
            )

        if num_workers > 1:
            work_items: list[tuple[str, str, str]] = [
                (movie_name, frame_type, str(frame_path)) for (_, frame_type, frame_path) in remaining_frames
            ]
            chunks = _chunk_round_robin(work_items, num_workers)

            gpu_groups: list[list[int]] = []
            for w in range(num_workers):
                start = w * gpus_needed
                gpu_groups.append(free_gpus[start : start + gpus_needed])

            with tempfile.TemporaryDirectory(prefix="disco_vllm_local_", dir=str(output.parent)) as td:
                tmp_paths: list[Path] = []
                procs: list[mp.Process] = []
                ctx = mp.get_context("spawn")
                for wi, (gpu_ids, items) in enumerate(zip(gpu_groups, chunks, strict=False)):
                    tmp = Path(td) / f"worker_{wi}.csv"
                    tmp_paths.append(tmp)
                    p = ctx.Process(
                        target=_run_local_vllm_worker,
                        args=(gpu_ids, items, vllm_model, gpus_needed, str(tmp), caption_mode),
                    )
                    p.start()
                    procs.append(p)

                for p in procs:
                    p.join()

                bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
                if bad:
                    raise RuntimeError(f"One or more vLLM worker processes failed: exit codes={bad}")

                resume = n_done > 0
                _merge_tmp_csvs(output, tmp_paths, resume=resume)
            return

    vllm_instance: Any = None
    if llm == "vllm":
        from vllm import LLM

        assert vllm_model is not None
        # Use tensor parallelism across eligible GPUs in the current process.
        _, gpus_needed, _num_gpus, free_gpus = _vllm_parallel_plan(vllm_model)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpus[:gpus_needed])
        vllm_instance = LLM(
            model=vllm_model,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            tensor_parallel_size=gpus_needed,
        )

    image_prompt = "What movie is this frame from? Reply with only the movie title, nothing else."

    output.parent.mkdir(parents=True, exist_ok=True)
    resume = n_done > 0
    open_mode = "a" if resume else "w"
    with output.open(open_mode, newline="") as f:
        writer = csv.writer(f)
        if not resume:
            if caption_mode:
                writer.writerow(
                    ["movie", "frame_type", "frame_file", "llm", "caption", "llm_response", "correct"]
                )
            else:
                writer.writerow(["movie", "frame_type", "frame_file", "llm", "llm_response", "correct"])

        n_remain = len(remaining_frames)
        for idx, (_, frame_type, frame_path) in enumerate(remaining_frames, start=1):
            print(f"Processing frame {idx}/{n_remain}: {frame_path.name}...")
            caption = ""
            try:
                img = Image.open(frame_path)
            except Exception as e:  # noqa: BLE001
                if caption_mode:
                    caption = f"ERROR: Failed to open image {frame_path}: {e}"
                    llm_response = "ERROR: caption step failed"
                else:
                    llm_response = f"ERROR: Failed to open image {frame_path}: {e}"
                correct = False
            else:
                if caption_mode:
                    caption, cap_err = call_llm_with_status(
                        llm,
                        CAPTION_IMAGE_PROMPT,
                        img,
                        vllm_instance=vllm_instance,
                        vllm_model=vllm_model,
                        vllm_max_tokens=1024,
                    )
                    if llm != "vllm":
                        time.sleep(float(delay))
                    if cap_err:
                        llm_response = "ERROR: caption step failed"
                        correct = False
                    else:
                        guess_prompt = guess_from_caption_prompt(caption)
                        llm_response, guess_err = call_llm_text_with_status(
                            llm,
                            guess_prompt,
                            vllm_instance=vllm_instance,
                        )
                        correct = False if guess_err else is_correct_response(llm_response, movie_name)
                    if llm != "vllm":
                        time.sleep(float(delay))
                else:
                    llm_response, had_error = call_llm_with_status(
                        llm,
                        image_prompt,
                        img,
                        vllm_instance=vllm_instance,
                        vllm_model=vllm_model,
                    )
                    correct = False if had_error else is_correct_response(llm_response, movie_name)
                    if llm != "vllm":
                        time.sleep(float(delay))

            if caption_mode:
                writer.writerow(
                    [
                        movie_name,
                        frame_type,
                        frame_path.name,
                        llm,
                        caption,
                        llm_response,
                        str(correct),
                    ]
                )
            else:
                writer.writerow(
                    [
                        movie_name,
                        frame_type,
                        frame_path.name,
                        llm,
                        llm_response,
                        str(correct),
                    ]
                )
            f.flush()


def _row_image_to_pil(image: Any) -> Image.Image:
    """Normalize dataset image field to PIL Image."""
    img = image
    if not isinstance(img, Image.Image):
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(BytesIO(img["bytes"]))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
    return img


def run_dataset(
    model_size: Literal["main", "mini"],
    movie_filter: str | None,
    llm: PromptType,
    output_dir: Path,
    output_file: Path | None,
    delay: float,
    vllm_model: str | None = None,
    caption_mode: bool = False,
) -> None:
    """Run DIS-CO evaluation on the HuggingFace DIS-CO dataset.

    When ``movie_filter`` is set, writes a single CSV to ``output_file``.
    When ``movie_filter`` is None (all movies), writes one CSV per movie under ``output_dir``.
    In caption mode, each row uses the dataset ``Caption`` column (images are not loaded).
    """
    from datasets import load_dataset

    if model_size == "main":
        dataset_name = "DIS-CO/MovieTection"
    else:
        dataset_name = "DIS-CO/MovieTection_Mini"

    print(f"Loading dataset '{dataset_name}'...")
    ds = load_dataset(dataset_name, split="train")

    if caption_mode:
        ds = ds.remove_columns(["Image_File"])

    # Precompute indices to process so we know total for progress printing.
    indices = []
    movie_filter_norm = movie_filter.lower() if movie_filter else None
    for i, row in enumerate(ds):
        movie = row.get("Movie", "")
        if movie_filter_norm is not None and movie.lower() != movie_filter_norm:
            continue
        indices.append(i)

    total = len(indices)
    if total == 0:
        if movie_filter:
            print(f"No rows found for movie '{movie_filter}' in dataset '{dataset_name}'.")
        else:
            print(f"No rows found in dataset '{dataset_name}'.")
        return

    vllm_instance: Any = None

    image_prompt = "What movie is this frame from? Reply with only the movie title, nothing else."

    def write_header(w: csv.writer) -> None:
        if caption_mode:
            w.writerow(
                ["movie", "frame_type", "frame_file", "llm", "caption", "llm_response", "correct"]
            )
        else:
            w.writerow(["movie", "frame_type", "frame_file", "llm", "llm_response", "correct"])

    def process_and_write_row(writer: csv.writer, row_idx: int, frame_num: int, frame_total: int, out_fp) -> None:
        row = ds[row_idx]
        movie = row.get("Movie", "")
        frame_type = row.get("Frame_Type", "")
        image = row.get("Image_File")
        frame_file = dataset_row_frame_file(row)

        print(f"Processing frame {frame_num}/{frame_total}: {frame_file}...")

        caption = ""

        if caption_mode:
            raw_cap = row.get("Caption")
            if raw_cap is None or not str(raw_cap).strip():
                caption = "ERROR: Missing or empty Caption in dataset row."
                llm_response = "ERROR: caption step failed"
                correct = False
            else:
                caption = str(raw_cap).strip()
                guess_prompt = guess_from_caption_prompt(caption)
                llm_response, guess_err = call_llm_text_with_status(
                    llm,
                    guess_prompt,
                    vllm_instance=vllm_instance,
                )
                correct = False if guess_err else is_correct_response(llm_response, movie)
            if llm != "vllm":
                time.sleep(float(delay))
        else:
            if image is None:
                llm_response = "ERROR: Missing image in dataset row."
                correct = False
            else:
                try:
                    img = _row_image_to_pil(image)
                    llm_response, had_error = call_llm_with_status(
                        llm,
                        image_prompt,
                        img,
                        vllm_instance=vllm_instance,
                        vllm_model=vllm_model,
                    )
                    correct = False if had_error else is_correct_response(llm_response, movie)
                except Exception as e:  # noqa: BLE001
                    llm_response = f"ERROR: Failed to process image from dataset: {e}"
                    correct = False
            if llm != "vllm":
                time.sleep(float(delay))

        if caption_mode:
            writer.writerow(
                [
                    movie,
                    frame_type,
                    frame_file,
                    llm,
                    caption,
                    llm_response,
                    str(correct),
                ]
            )
        else:
            writer.writerow(
                [
                    movie,
                    frame_type,
                    frame_file,
                    llm,
                    llm_response,
                    str(correct),
                ]
            )
        out_fp.flush()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if movie_filter is not None:
        if output_file is None:
            raise ValueError("output_file is required when movie_filter is set.")
        completed = read_completed_frame_files(output_file)
        remaining_indices = [
            i for i in indices if dataset_row_frame_file(ds[i]) not in completed
        ]
        n_done = total - len(remaining_indices)
        if not remaining_indices:
            print(f"All {total} frames already processed in {output_file}. Nothing to do.")
            return
        if n_done > 0:
            print(f"Found existing results: {n_done}/{total} frames already processed. Resuming...")

        if llm == "vllm":
            assert vllm_model is not None
            num_workers, gpus_needed, num_gpus, free_gpus = _vllm_parallel_plan(vllm_model)
            if num_workers > 1:
                chunks = _chunk_round_robin(remaining_indices, num_workers)
                gpu_groups: list[list[int]] = []
                for w in range(num_workers):
                    start = w * gpus_needed
                    gpu_groups.append(free_gpus[start : start + gpus_needed])

                dataset_name = "DIS-CO/MovieTection_Mini" if model_size == "mini" else "DIS-CO/MovieTection"
                with tempfile.TemporaryDirectory(prefix="disco_vllm_ds_", dir=str(output_dir)) as td:
                    tmp_paths: list[Path] = []
                    procs: list[mp.Process] = []
                    ctx = mp.get_context("spawn")
                    for wi, (gpu_ids, chunk) in enumerate(zip(gpu_groups, chunks, strict=False)):
                        tmp = Path(td) / f"worker_{wi}.csv"
                        tmp_paths.append(tmp)
                        p = ctx.Process(
                            target=_run_dataset_vllm_worker,
                            args=(
                                gpu_ids,
                                chunk,
                                dataset_name,
                                vllm_model,
                                gpus_needed,
                                str(tmp),
                                caption_mode,
                            ),
                        )
                        p.start()
                        procs.append(p)

                    for p in procs:
                        p.join()
                    bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
                    if bad:
                        raise RuntimeError(f"vLLM worker(s) failed: exit codes={bad}")
                    _merge_tmp_csvs(output_file, tmp_paths, resume=n_done > 0)
                return

            from vllm import LLM

            print(
                f"vLLM: using {gpus_needed} GPU(s) (tensor parallel) out of {num_gpus} visible; eligible={len(free_gpus)}"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpus[:gpus_needed])
            vllm_instance = LLM(
                model=vllm_model,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 1},
                trust_remote_code=True,
                tensor_parallel_size=gpus_needed,
            )

        resume = n_done > 0
        open_mode = "a" if resume else "w"
        n_remain = len(remaining_indices)
        with output_file.open(open_mode, newline="") as f:
            writer = csv.writer(f)
            if not resume:
                write_header(writer)
            for idx, row_idx in enumerate(remaining_indices, start=1):
                process_and_write_row(writer, row_idx, idx, n_remain, f)
    else:
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for row_idx in indices:
            m = ds[row_idx].get("Movie", "") or "Unknown"
            groups[m].append(row_idx)
        sorted_movies = sorted(groups.items(), key=lambda x: x[0])
        n_movies = len(sorted_movies)

        movie_jobs: list[tuple[str, list[int], Path, list[int]]] = []
        for movie_name, movie_indices in sorted_movies:
            out_path = output_dir / generate_output_filename(
                movie_name, "dataset", caption_mode, llm, vllm_model
            )
            completed_m = read_completed_frame_files(out_path)
            remaining_m = [
                i for i in movie_indices if dataset_row_frame_file(ds[i]) not in completed_m
            ]
            movie_jobs.append((movie_name, movie_indices, out_path, remaining_m))

        if all(len(rem) == 0 for *_, rem in movie_jobs):
            for movie_name, movie_indices, out_path, _ in movie_jobs:
                print(
                    f"All {len(movie_indices)} frames already processed in {out_path}. Nothing to do."
                )
            return

        if llm == "vllm":
            from vllm import LLM

            assert vllm_model is not None
            _, gpus_needed, num_gpus, free_gpus = _vllm_parallel_plan(vllm_model)
            print(
                f"vLLM: using {gpus_needed} GPU(s) (tensor parallel) out of {num_gpus} visible; eligible={len(free_gpus)}"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpus[:gpus_needed])
            vllm_instance = LLM(
                model=vllm_model,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 1},
                trust_remote_code=True,
                tensor_parallel_size=gpus_needed,
            )

        for mi, (movie_name, movie_indices, out_path, remaining) in enumerate(movie_jobs, start=1):
            n_frames = len(movie_indices)
            if not remaining:
                print(
                    f"All {n_frames} frames already processed in {out_path}. Nothing to do."
                )
                continue
            n_done_m = n_frames - len(remaining)
            print(f"Processing movie {mi}/{n_movies}: {movie_name} ({n_frames} frames)...")
            if n_done_m > 0:
                print(
                    f"Found existing results: {n_done_m}/{n_frames} frames already processed. Resuming..."
                )
            resume_m = n_done_m > 0
            open_mode = "a" if resume_m else "w"
            n_remain = len(remaining)
            with out_path.open(open_mode, newline="") as f:
                writer = csv.writer(f)
                if not resume_m:
                    write_header(writer)
                for fi, row_idx in enumerate(remaining, start=1):
                    process_and_write_row(writer, row_idx, fi, n_remain, f)
        print(f"Wrote results for {n_movies} movies to {output_dir}/")


def main() -> None:
    """CLI entrypoint for DIS-CO run script."""
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.mode == "local":
        out_path = output_dir / generate_output_filename(
            movie_name=args.movie_name,
            mode="local",
            caption_mode=args.caption_mode,
            llm=args.llm,
            vllm_model=args.vllm_model,
        )
        print(f"Writing results to: {out_path}")
        run_local(
            frames_dir=Path(args.frames_dir),
            movie_name=args.movie_name,
            llm=args.llm,
            output=out_path,
            delay=float(args.delay),
            vllm_model=args.vllm_model,
            caption_mode=args.caption_mode,
        )
    elif args.movie:
        out_path = output_dir / generate_output_filename(
            movie_name=args.movie,
            mode="dataset",
            caption_mode=args.caption_mode,
            llm=args.llm,
            vllm_model=args.vllm_model,
        )
        print(f"Writing results to: {out_path}")
        run_dataset(
            model_size=args.model_size,
            movie_filter=args.movie,
            llm=args.llm,
            output_dir=output_dir,
            output_file=out_path,
            delay=float(args.delay),
            vllm_model=args.vllm_model,
            caption_mode=args.caption_mode,
        )
    else:
        print(f"Writing one CSV per movie under: {output_dir}/")
        run_dataset(
            model_size=args.model_size,
            movie_filter=None,
            llm=args.llm,
            output_dir=output_dir,
            output_file=None,
            delay=float(args.delay),
            vllm_model=args.vllm_model,
            caption_mode=args.caption_mode,
        )


if __name__ == "__main__":
    main()

