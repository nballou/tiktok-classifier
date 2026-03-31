# TikTok Mental Health Classifier

Classifies TikTok videos as mental-health-related or not, based on their text
metadata (description, transcript, suggested words). Each video is labelled
**TRUE** (meaningfully engages with mental health experience) or **FALSE**
(does not) by a large language model running locally via
[vLLM](https://docs.vllm.ai/).

The classifier uses guided decoding to constrain the model's output to the
allowed labels, and supports resume, batched checkpointing, and HPC
array-job parallelism.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running the LLM server

Start a [vLLM](https://docs.vllm.ai/) server with an instruct-tuned model
before classifying. The model must support chat completions and guided
decoding.

```bash
vllm serve <model-name>

# On Mac (vllm-metal), enable paged attention for better memory efficiency:
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve <model-name>
```

The classifier connects to `http://localhost:8000/v1` by default (override
with `--endpoint`). Pass `--model` to match whatever model the server is
running.

> **Current local testing setup (Mac, vllm-metal):**
> ```bash
> VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve Qwen/Qwen3-8B
> ```
> Qwen3.5 would be preferable but requires a linear attention kernel that
> vllm-metal doesn't support yet
> ([tracking issue](https://github.com/vllm-project/vllm-metal/issues/194)).
> Qwen3-8B is the closest supported alternative.

## Usage

### Stage 1: Mental health classification

The default mode classifies each video as mental-health-related or not,
writing results to the `is_mental_health` column:

```bash
uv run python classify.py <file>.parquet --model <model-name>
```

Results are written back to the same parquet file. Rows that already have a
classification are skipped, so you can safely interrupt and resume. Rows with
`scrape_status='error'` (no metadata available) are automatically skipped.

Add `-v` / `--verbose` to print each row's label and URL as it is classified.

### Stage 2: Subclassification

After stage 1, mental-health-related videos can be further classified using a
custom prompt and label set. For example, to categorise videos as self-help
vs commiseration:

```bash
uv run python classify.py <file>.parquet \
  --prompt prompts/mh_subtype.txt \
  --labels SELF_HELP COMMISERATION OTHER \
  --output-col mh_subtype \
  --filter is_mental_health=TRUE \
  --model <model-name>
```

This only processes rows already labelled TRUE in stage 1, and writes results
to a new `mh_subtype` column. The same resume, checkpointing, and HPC
features apply.

The `--labels`, `--output-col`, and `--filter` flags are general-purpose, so
additional classification stages can be added in the same way with different
prompts and label sets.

### Extract and classify a test set

Extract specific videos by URL for validation against ground-truth labels, or
pull a random sample for spot-checks:

```bash
# Extract by URL (one per line in a text file)
uv run python classify.py <file>.parquet --urls <urls-file>.txt -o <output>.parquet

# Or extract a random sample
uv run python classify.py <file>.parquet --sample <N>

# Then classify
uv run python classify.py <output>.parquet -v --model <model-name>
```

> **Current local testing commands:**
> ```bash
> uv run python classify.py tiktok_metadata.parquet --urls ground_truth_urls.txt -o test_set.parquet
> uv run python classify.py test_set.parquet -v --model Qwen/Qwen3-8B
> ```

### HPC parallel workflow

Split the data into chunks, run one job per chunk, then merge:

```bash
# 1. Partition
uv run python classify.py <file>.parquet --partition <N>

# 2. Classify each chunk (e.g. in a PBS array job)
uv run python classify.py chunks/chunk_000.parquet --model <model-name>

# 3. Merge results
uv run python classify.py chunks/ --merge -o classified.parquet
```

## Options

| Flag | Description |
|------|-------------|
| `--model MODEL` | HuggingFace model ID (must match the model the vLLM server is running) |
| `--endpoint URL` | vLLM server URL (default: `http://localhost:8000/v1`) |
| `--labels LABEL ...` | Allowed output labels for guided decoding (default: `TRUE FALSE`) |
| `--output-col NAME` | Column to write results to (default: `is_mental_health`) |
| `--filter COL=VAL` | Only classify rows where an existing column matches a value |
| `--concurrency N` | Parallel requests to the server (default: 4) |
| `--batch_size N` | Rows per checkpoint batch (default: 200) |
| `--cols COL ...` | Metadata columns to include in the prompt (default: `description transcript suggested_words`) |
| `--prompt FILE` | Path to a custom prompt template (must contain `{text}`) |
| `-o, --output PATH` | Output path for `--merge`, `--sample`, or `--urls` |
| `-v, --verbose` | Print each classification result with URL |

## Output columns

The classifier adds columns to the parquet file depending on the stage:

- **is_mental_health** (stage 1): `TRUE`, `FALSE`, `SKIPPED`, or `UNKNOWN`
- **mh_subtype** (stage 2, example): whichever labels are specified via `--labels`
- **prob_true**: Probability of `TRUE` from logprobs (currently `NA` due to a
  vLLM compatibility issue; the column is reserved for when this is resolved)
