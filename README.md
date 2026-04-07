## discopackage (movie frame extraction + classification)

This project extracts representative frames from a movie, deduplicates them with CLIP embeddings, then classifies frames into **main** (character-focused) vs **neutral** (faceless / background-y) selections. The **`run.py`** script runs the **DIS-CO** step: it asks a VLM to name the movie from each frame and writes results to CSV (cloud APIs or local **vLLM**). The **`compare.py`** script aggregates those CSVs and prints accuracy tables (compare LLMs, image vs caption, or **leaderboard** rankings across all result files).

### What it does

- **Scene detection**: detects scenes with PySceneDetect.
- **Frame extraction**: saves one sharp frame per scene (midpoint, choose sharpest from a small time window).
- **CLIP deduplication**: embeds frames with OpenAI CLIP and removes near-duplicates using ChromaDB cosine similarity.
- **Diverse pool**: builds a diverse frame pool using farthest-point sampling.
- **Face-based selection**:
  - Main frames: uses insightface + DBSCAN clustering + quality scoring + diversity sampling.
  - Neutral frames: filters to **faceless + non-blank** frames, then diversity sampling.

### Setup (on your laptop)

#### 1) Create a virtual environment

Requirements:
- Use **Python 3.10+**.

From the `discopackage/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

#### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Notes:
- `git+https://github.com/openai/CLIP.git` installs CLIP from source.
- `insightface` will download model weights on first use (into your user cache).
- `onnxruntime` is CPU by default here.
- Video decoding relies on OpenCV. If a video fails to decode, install **ffmpeg** on your system (e.g. `brew install ffmpeg` on macOS).
- **`vllm`** is listed in `requirements.txt` for local VLMs. It needs a **CUDA-capable GPU** and matching PyTorch/CUDA setup; if you only use cloud APIs (`gemini` / `claude` / `chatgpt`), you can skip installing vLLM or install dependencies in a separate environment.

### Run the extractor (CLI)

From `discopackage/`:

```bash
python3 extract.py --path "/path/to/movie.mp4" --output "./output"
```

Optional flags:

```bash
python3 extract.py \
  --path "/path/to/movie.mp4" \
  --output "./output" \
  --skip-start 180 \
  --skip-end 420 \
  --clip-threshold 0.90 \
  --n-main 100 \
  --n-neutral 40 \
  --pool-multiplier 4
```

What the main flags mean:
- `--skip-start`: seconds to skip from the start of the video (titles/intro).
- `--skip-end`: seconds to skip from the end of the video (credits/outro).
- `--pool-multiplier`: controls how large the “diverse CLIP pool” is before face-based selection.
  - `pool_size = pool_multiplier * (n_main + n_neutral)`
  - Higher = more candidate frames (often more diverse, but slower).
- `--clip-threshold`: CLIP cosine similarity threshold used for deduplication.
  - Frames are treated as duplicates if their nearest-neighbor similarity is `>= threshold`.
  - Lower threshold = more frames count as duplicates (more aggressive dedup).
  - Higher threshold = fewer frames count as duplicates (less aggressive dedup).

### Output layout

Inside your `--output` directory you’ll get:

- `frames_raw/`: extracted frames (pre-classification)
- `main/`: final selected main frames
- `neutral/`: final selected neutral frames

The script prints a short summary at the end (scenes detected, frames saved, frames kept post-dedup, pool size, and export counts).

### Common issues

- **Movie won’t open / OpenCV errors**: try a different `.mp4` encode or re-export the video; OpenCV can fail on some containers/codecs.
- **Insightface downloads**: the first run may take longer while models download.
- **Performance**: everything runs on CPU by default; large movies can take time.

### Run DIS-CO VLM identification (`run.py`)

`run.py` is the **DIS-CO** evaluation step: it shows each frame to a vision-language model (VLM) with a fixed prompt and records whether the model’s answer matches the **ground-truth movie title**.

**Image mode (default):** for each frame, the model sees the image and answers:  
`What movie is this frame from? Reply with only the movie title, nothing else.`

**Caption mode (`--caption-mode`):** tests whether the model can identify the movie from **text alone** (copyright signal via description, not only pixels).

- **`--mode local`:** (1) **Image → caption** —  
  `Describe this movie frame in detail. Include information about the characters, setting, lighting, actions, and any notable visual elements. Do not mention the movie title.`  
  (2) **Caption → title (text only):**  
  `Based on this description of a movie frame, what movie is it from? Reply with only the movie title, nothing else.`  
  followed by `Description: {caption}`

- **`--mode dataset`:** only step (2) — the **`Caption`** column from each row is used; **images are not loaded** (the HuggingFace loader omits **`Image_File`** so frames are not downloaded). Same text-only movie-guess prompt as in local step 2.

**Output CSV filename (auto-generated):** The file is named `{movie}_{source}_{test_type}_{llm}.csv` in **`--output-dir`** (default: current directory). Example: `forrestgump_local_image_chatgpt.csv`, `frozen_dataset_caption_gemini.csv`. For **`--mode dataset` without `--movie`** (full dataset), `run.py` writes **one CSV per movie** (e.g. `frozen_dataset_image_chatgpt.csv`, `forrestgump_dataset_image_chatgpt.csv`). For `--llm vllm`, the last segment is the HuggingFace model id with slashes replaced by dashes and lowercased (e.g. `qwen-qwen2-vl-7b-instruct`).

#### Modes

| Mode | What it reads | Required flags |
|------|----------------|----------------|
| **`local`** | Frames on disk (e.g. output of `extract.py`) | `--frames-dir`, `--movie-name` |
| **`dataset`** | HuggingFace DIS-CO datasets | `--model-size` |

- **`--mode local`**: `--frames-dir` must contain `main/` and `neutral/` subfolders. Every file in those folders is treated as one frame (same layout as `extract.py` output).
- **`--mode dataset`**: loads `DIS-CO/MovieTection` or `DIS-CO/MovieTection_Mini` via `datasets`. Use `--movie "Title"` to filter rows by the `Movie` column (case-insensitive). In **image** mode, `Frame_Type` and `Image_File` are used. In **caption** mode, `Frame_Type` and **`Caption`** are used (no images).

#### Vision backends (`--llm`)

| `--llm` | How it runs | Extra setup |
|---------|-------------|-------------|
| `gemini` | Google Gemini API | `GEMINI_API_KEY` |
| `claude` | Anthropic API | `ANTHROPIC_API_KEY` |
| `chatgpt` | OpenAI GPT-4o | `OPENAI_API_KEY` |
| `vllm` | Local [vLLM](https://github.com/vllm-project/vllm) offline inference | **`--vllm-model`** (HuggingFace id). GPU + CUDA stack recommended. No API keys. |

**API keys (cloud only):** set only the key for the backend you use.

```bash
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

**`--delay`:** seconds to sleep between cloud API calls (after each image call; after each text call in **local** caption mode; once per row in **dataset** caption mode). Default `1.0`. **Not used** when `--llm vllm` (local inference has no API quota).

**`--output-dir`:** directory for the auto-generated CSV (default `.`).

**`--vllm-model`:** required if and only if `--llm vllm`. The model weights are loaded **once** per `run.py` invocation (after any resume checks) and reused for all frames processed in that run.

**Resume & checkpointing:** If the target CSV already exists, `run.py` reads it and **skips frames that already finished successfully**. A frame counts as done when the latest row for that `frame_file` has a non-empty `llm_response` that does **not** start with `ERROR:` (so failed/API rows are **retried**). New rows are **appended**; the header is not written again. You’ll see messages like `Found existing results: 85/140 frames already processed. Resuming...` or `All 140 frames already processed in {path}. Nothing to do.` For **dataset mode without `--movie`**, each per-movie file resumes independently—if a run stops on movie 30, rerunning picks up from movie 30’s file without redoing movies 1–29.

**Dataset mode `frame_file` ids:** In **`--mode dataset`**, each row’s `frame_file` column is a stable id derived from **`Frame_Type`**, **`Scene_Number`**, and **`Shot_Number`**: `{Frame_Type}_scene_{Scene_Number}_shot_{Shot_Number}.jpg` (missing **`Frame_Type`** defaults to `Unknown`). The same scene/shot can appear twice (e.g. **Main** and **Neutral**); including **`Frame_Type`** keeps resume keys unique. **Resuming against an older CSV** that used only `scene_X_shot_Y.jpg` will not match those rows—finish the old run or use a new output file if you change formats mid-project.

**Recommended vLLM models** (for movie-frame identification; strongest first):

| Model | HuggingFace id | VRAM | Notes |
|-------|----------------|------|-------|
| Qwen2-VL 7B | `Qwen/Qwen2-VL-7B-Instruct` | ~16 GB | Strong visual understanding, good at scene/title recognition |
| LLaVA 1.6 Mistral 7B | `llava-hf/llava-v1.6-mistral-7b-hf` | ~16 GB | Improved over LLaVA 1.5 |
| LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` | ~14 GB | Reliable baseline, widely tested |
| Qwen2-VL 2B | `Qwen/Qwen2-VL-2B-Instruct` | ~6 GB | Smaller / faster; lower accuracy |
| LLaVA-NeXT 34B | `llava-hf/llava-v1.6-34b-hf` | ~24 GB+ | Best quality; needs larger GPU |

#### End-to-end: local frames (extract → evaluate)

1. Extract frames from a video:

```bash
python3 extract.py --path "/path/to/movie.mp4" --output ./my_movie_frames
```

2. Run evaluation on `./my_movie_frames` (must contain `main/` and `neutral/`):

```bash
python3 run.py --mode local \
  --frames-dir ./my_movie_frames \
  --movie-name "Your Movie Title" \
  --llm gemini
```

#### Examples: cloud APIs

**Image mode** (default — model sees each frame directly):

Gemini on local frames:

```bash
python3 run.py --mode local \
  --frames-dir ./forrestgump_frames \
  --movie-name "Forrest Gump" \
  --llm gemini
```

Claude on full DIS-CO main dataset:

```bash
python3 run.py --mode dataset \
  --model-size main \
  --llm claude
```

ChatGPT on one movie (mini dataset):

```bash
python3 run.py --mode dataset \
  --model-size mini \
  --movie "The Truman Show" \
  --llm chatgpt
```

**Caption mode** (`--caption-mode` — local: describe frame then guess from text; dataset: guess from dataset **`Caption`** only; filenames include `caption` as the test type, e.g. `forrestgump_local_caption_gemini.csv`):

Gemini on local frames (caption pipeline):

```bash
python3 run.py --mode local \
  --frames-dir ./forrestgump_frames \
  --movie-name "Forrest Gump" \
  --llm gemini \
  --caption-mode
```

ChatGPT on one movie, mini dataset, caption mode (uses dataset **`Caption`** column only):

```bash
python3 run.py --mode dataset \
  --model-size mini \
  --movie "Frozen" \
  --llm chatgpt \
  --caption-mode
```

Claude on one movie, mini dataset, caption mode:

```bash
python3 run.py --mode dataset \
  --model-size mini \
  --movie "The Truman Show" \
  --llm claude \
  --caption-mode
```

#### Examples: local vLLM (`--llm vllm`)

LLaVA 1.5 on local frames:

```bash
python3 run.py --mode local \
  --frames-dir ./forrestgump_frames \
  --movie-name "Forrest Gump" \
  --llm vllm \
  --vllm-model "llava-hf/llava-1.5-7b-hf"
```

Qwen2-VL on the HuggingFace dataset (mini + movie filter):

```bash
python3 run.py --mode dataset \
  --model-size mini \
  --movie "Frozen" \
  --llm vllm \
  --vllm-model "Qwen/Qwen2-VL-7B-Instruct"
```

#### Output CSV

**Image mode (default):** columns are `movie`, `frame_type`, `frame_file`, `llm`, `llm_response`, `correct`.

**Caption mode (`--caption-mode`):** adds `caption` (generated description or dataset caption); `llm_response` is the movie guess from the text-only step. The filename uses `test_type` = `caption` (e.g. `frozen_dataset_caption_chatgpt.csv`).

**Dataset mode:** `frame_file` is written as `{Frame_Type}_scene_{Scene_Number}_shot_{Shot_Number}.jpg` (see **Resume & checkpointing** above).

**Normalization for `correct`:** strip whitespace; replace quotes and punctuation (including hyphens) with spaces; collapse whitespace; compare with Unicode case-folding. Leading words like “The” are **not** removed: `The Matrix` and `Matrix` are not treated as the same title.

### Compare experiment results (`compare.py`)

`compare.py` reads DIS-CO result CSVs (same auto-generated names as `run.py`: `{movie}_{source}_{test_type}_{llm}.csv`) and prints summary tables to the terminal: per-movie LLM comparison, image vs caption for one LLM, or **leaderboard** rankings across all CSVs in a directory. It does **not** write files.

**`--output-dir`** (default: current directory) is where it looks for CSVs.

**Metrics:** overall / Main / Neutral accuracy (% of rows with `correct` = `True`), total row count (**Frames**), and rows where `llm_response` starts with `ERROR:` (**Errors**).

#### Mode 1: compare LLMs (`--compare llm`)

For one movie, one source, and one test type, find every CSV matching `{movie_slug}_{source}_{test_type}_*.csv` and list each LLM file side by side.

Required: `--movie`, `--test-type` (`image` or `caption`), `--source` (`local` or `dataset`).

```bash
python3 compare.py --compare llm --movie "Frozen" --test-type image --source dataset
```

If nothing matches, you’ll get a hint to run `run.py` first.

#### Mode 2: compare image vs caption (`--compare test`)

For one movie, one source, and one LLM, load the two files `{movie_slug}_{source}_image_{llm}.csv` and `{movie_slug}_{source}_caption_{llm}.csv`, print both tables, and a **delta** row (caption accuracy minus image accuracy, in percentage points) for overall, Main, and Neutral.

Required: `--movie`, `--llm`, `--source`.

For cloud backends, `--llm` is the short name (`gemini`, `claude`, `chatgpt`). For vLLM runs, use the same token as in the filename (e.g. `qwen-qwen2-vl-7b-instruct`) or pass a HuggingFace id with slashes (they are normalized like `run.py`).

```bash
python3 compare.py --compare test --movie "Frozen" --llm chatgpt --source dataset
```

#### Mode 3: leaderboards (`--compare leaderboard`)

Scans **all** result CSVs in **`--output-dir`** whose names match `{movie}_{source}_{test_type}_{llm}.csv` (movie slug has no underscores; the LLM segment is everything after the third `_`). Optional filters:

- **`--test-type`** `image` or `caption` — **default `image`** for leaderboard if omitted.
- **`--source`** `local` or `dataset` — **default `dataset`** for leaderboard if omitted.

**`--by model`:** ranks **LLMs** by **overall** accuracy pooled across every matching movie file (also **Main** / **Neutral** breakdown, **Movies** count, **Frames**). Sorted by overall accuracy, highest first.

**`--by movie`:** for a single LLM (**`--llm` required,** as in the filename token), ranks **movies** by overall accuracy (Main / Neutral / Frames). The first non-empty **`movie`** cell in each CSV is used for the display name.

```bash
# Which LLM is best overall?
python3 compare.py --compare leaderboard --by model --output-dir /path/to/results

# Which LLM is best at caption testing?
python3 compare.py --compare leaderboard --by model --test-type caption --output-dir /path/to/results

# Which movies are easiest/hardest for ChatGPT?
python3 compare.py --compare leaderboard --by movie --llm chatgpt --output-dir /path/to/results

# Which movies for Qwen2-VL (filename token)?
python3 compare.py --compare leaderboard --by movie --llm qwen-qwen2-vl-7b-instruct --output-dir /path/to/results
```

