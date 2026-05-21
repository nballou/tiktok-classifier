# TikTok Mental Health Classifier

Two-pass LLM pipeline that classifies TikTok videos as mental-health-related
based on text metadata (description, transcript, suggested words).

- **Pass 1 — screen** (`--stage screen`): liberal first pass over all videos.
  Casts a wide net; false positives are acceptable. Writes to
  `is_mental_health_broad`.
- **Pass 2 — classify** (`--stage classify`): fine-grained second pass over
  screen-passing rows only. Requires genuine engagement with a recognised
  mental health condition. Writes to `is_mental_health`.

Supports two backends:
- **Ollama** — local development on Mac
- **vLLM** — HPC with guided decoding

Both stages support resume (already-classified rows are skipped), atomic
checkpointing, and HPC array-job parallelism.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running the LLM server

### Local (Ollama)

```bash
ollama pull gemma4:e4b    # or another instruct model
ollama serve              # starts automatically on Mac after install
```

### HPC (vLLM)

```bash
vllm serve google/gemma-3-4b-it   # or Qwen/Qwen3-8B, meta-llama/..., etc.
```

The vLLM backend uses guided decoding to constrain output to TRUE/FALSE.
When `--rationale` is used, guided decoding is disabled and the model is
prompted for a two-line response (label on line 1, explanation on line 2).

## Full pipeline

```bash
# Pass 1: screen all videos (~8% expected to pass)
python 02_classify.py tiktok_metadata.parquet \
  --stage screen \
  --model gemma4:e4b \
  --concurrency 4 \
  -v

# Pass 2: classify screen-positives only
python 02_classify.py tiktok_metadata.parquet \
  --stage classify \
  --filter is_mental_health_broad=TRUE \
  --model gemma4:e4b \
  --concurrency 4 \
  -v
```

Results are written back to the same parquet. Safe to interrupt and resume.
Rows with `scrape_status='error'` are automatically skipped.

## Evaluation

The ground-truth CSV (`handcoded_examples.csv`) contains hand-labelled videos
with these columns beyond the standard metadata:

| Column | Description |
|--------|-------------|
| `is_mental_health_handcoded` | Ground truth for classify stage (`TRUE`/`FALSE`) |
| `is_superficial_mental_health_handcoded` | Ground truth for screen stage |
| `use` | `train` (few-shot example) or `test` (evaluation) |
| `rationale` | Optional explanation, included in few-shot prompts for classify stage |

```bash
# Evaluate classify stage (filter to screen-passing rows)
python 02_classify.py tiktok_metadata.parquet \
  --stage classify \
  --evaluate handcoded_examples.csv \
  --filter is_superficial_mental_health_handcoded=TRUE \
  --rationale \
  --model gemma4:e4b \
  --concurrency 2 \
  -v
```

Evaluation prints accuracy, a confusion matrix, and mismatches with rationales.
Full results (URL, prediction, ground truth, rationale) are saved to
`eval_results_{stage}.csv` for inspection.

Experiment results are appended to `experiment_log.csv` after each eval run.

### --rationale flag

Requests a one-sentence explanation alongside each label. Stored in
`{output_col}_rationale` in the parquet. Useful for debugging and spot-checks;
disables guided decoding on vLLM.

```bash
python 02_classify.py sample.parquet \
  --stage classify \
  --filter is_mental_health_broad=TRUE \
  --rationale \
  --model gemma4:e4b \
  -v
```

## HPC workflow (Imperial CX3, PBS)

The cluster uses PBS with GPU nodes (`gpu72` queue). The workflow runs vLLM
as a background process alongside the classifier in a single job.

### One-time setup (run on login node)

```bash
scp -r tiktok-classifier/ your_username@login.cx3.hpc.ic.ac.uk:~/
ssh your_username@login.cx3.hpc.ic.ac.uk
cd ~/tiktok-classifier
bash setup_hpc_env.sh   # creates conda env + pre-downloads model weights
```

The setup script installs vLLM and classifier dependencies into a `tiktok`
conda environment and caches model weights to `$EPHEMERAL/huggingface`
(10 TB, avoids filling `$HOME`).

### Submit a job

Edit the `MODEL` and `INPUT` variables at the top of `run_classify.pbs`,
then:

```bash
mkdir -p ~/tiktok-classifier/logs
qsub run_classify.pbs
qstat -u $USER   # monitor status
```

The job starts a vLLM server, waits for it to be ready, runs the screen pass,
then the classify pass with `--rationale`. Results and rationales are written
back to the input parquet. Server logs go to `logs/vllm.log`.

### Full dataset with array jobs

For 1M rows, partition first and run a screen pass per chunk:

```bash
# 1. Partition into chunks
python 01_partition.py tiktok_metadata.parquet --partition 100

# 2. One PBS job per chunk — set #PBS -J 1-100 in the job script, then:
python 02_classify.py chunks/chunk_$(printf "%03d" $((PBS_ARRAY_INDEX-1))).parquet \
  --stage screen --backend vllm --concurrency 16

# 3. Merge
python 03_merge.py chunks/ -o screened.parquet

# 4. Classify (single job; screen reduces volume ~10x)
python 02_classify.py screened.parquet \
  --stage classify --filter is_mental_health_broad=TRUE \
  --rationale --backend vllm --concurrency 8
```

## CLI reference

### 02_classify.py

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | required | `screen` or `classify` |
| `--backend` | `ollama` | `ollama` or `vllm` |
| `--model` | backend-specific | Model name |
| `--endpoint` | backend-specific | Server URL |
| `--evaluate CSV` | — | Evaluate against ground-truth CSV; reports accuracy + confusion matrix |
| `--filter COL=VAL` | — | Only classify rows where column matches value |
| `--rationale` | off | Request one-sentence explanation alongside each label |
| `--examples CSV` | — | Inject `use=train` rows as few-shot examples |
| `--concurrency N` | `4` | Parallel requests |
| `--batch_size N` | `200` | Rows per checkpoint |
| `--timeout N` | `120` | Request timeout in seconds (increase for large models) |
| `--output-col NAME` | stage-specific | Override output column name |
| `--cols COL ...` | `description transcript suggested_words` | Metadata fields to include |
| `-v` | off | Verbose: print each result as it arrives |

### 01_partition.py

| Flag | Description |
|------|-------------|
| `--partition N` | Split into N equal chunks (for array jobs) |
| `--sample N` | Extract N random rows |
| `--urls FILE` | Extract rows matching URLs in FILE (one per line) |
| `-o PATH` | Output path |

### 03_merge.py

| Flag | Description |
|------|-------------|
| `-o PATH` | Output parquet path (default: `classified.parquet`) |

## Output columns

| Column | Stage | Values |
|--------|-------|--------|
| `is_mental_health_broad` | screen | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN` |
| `is_mental_health` | classify | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN` |
| `is_mental_health_broad_rationale` | screen + `--rationale` | one-sentence explanation |
| `is_mental_health_rationale` | classify + `--rationale` | one-sentence explanation |
| `prob_true` | both | reserved; currently `NA` |
