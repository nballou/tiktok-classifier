# TikTok Mental Health Classifier

Two-pass LLM pipeline that classifies TikTok videos as mental-health-related
based on text metadata (description, transcript, suggested words).

- **Pass 1 — screen** (`--stage screen`): liberal first pass over all videos.
  Casts a wide net; false positives are acceptable.
  Writes `is_mental_health_broad_<model>` to the parquet.
- **Pass 2 — classify** (`--stage classify`): fine-grained second pass over
  screen-passing rows only. Requires genuine engagement with a recognised
  mental health condition.
  Writes `is_mental_health_<model>` and optionally `is_mental_health_<model>_rationale`.

Both stages support atomic checkpointing: safe to interrupt and resubmit —
already-classified rows are skipped on resume.

## Setup

### Local (requires Python 3.11+ and [uv](https://docs.astral.sh/uv/))

```bash
uv sync
ollama pull gemma4:e4b   # or any other instruct model
```

### HPC (Imperial CX3)

Run once on the login node (internet access required, no GPU needed):

```bash
bash hpc/setup_hpc_env.sh
```

This creates the `tiktok` conda environment, installs dependencies from
`pyproject.toml`, and downloads the model weights specified in `model.env`
to `$HOME/huggingface`. To switch models, update `model.env` and rerun
the setup script to download the new weights.

## Running the classifier

### Local

```bash
python classify.py metadata/data.parquet --stage screen --concurrency 4 -v
python classify.py metadata/data.parquet --stage classify --rationale --concurrency 4 -v
```

`--model`, `--model-label`, and the classify-pass `--filter` are all derived
from `model.env` automatically. Pass any of them explicitly on the CLI to
override.

### HPC (PBS)

```bash
qsub hpc/run_smoke_test.pbs               # 20-row sanity check
qsub -v ROWS=1000 hpc/run_smoke_test.pbs  # scale test
qsub hpc/run_classify.pbs                 # full dataset; resubmit if walltime hit
```

Model configuration (name, label, quantization, thinking mode) lives in
`model.env` at the repo root — edit that file to switch models. The PBS
scripts source it automatically.

## Evaluation

`handcoded_examples.csv` contains hand-labelled videos:

| Column | Description |
|--------|-------------|
| `is_mental_health_handcoded` | Ground truth for classify stage (`TRUE`/`FALSE`) |
| `is_superficial_mental_health_handcoded` | Ground truth for screen stage |
| `use` | `train` (few-shot example) or `test` (held-out evaluation) |
| `rationale` | Explanation included in few-shot prompts for the classify stage |

```bash
python classify.py metadata/data.parquet \
  --stage classify \
  --evaluate handcoded_examples.csv \
  --filter is_superficial_mental_health_handcoded=TRUE \
  --rationale --model gemma4:e4b --concurrency 2 -v
```

Prints accuracy, confusion matrix, and mismatches. Full results are saved to
`eval_results_classify.csv`. Metrics are appended to `experiment_log.csv`.

### IMHI benchmark

`prepare_imhi_eval.py` converts the [IMHI dataset](https://huggingface.co/datasets/Tianlin668/IMHI)
test split into the evaluation format for cross-dataset comparison:

```bash
pip install datasets
python prepare_imhi_eval.py   # writes imhi_eval.csv

python classify.py metadata/data.parquet \
  --stage classify --evaluate imhi_eval.csv \
  --model-label <your_model_label> --backend vllm
```

## CLI reference

### classify.py

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | required | `screen` or `classify` |
| `--backend` | `ollama` | `ollama` or `vllm` |
| `--model` | backend-specific | Model name |
| `--model-label LABEL` | derived from model | Suffix for output columns (e.g. `gemma4_26b`) |
| `--endpoint` | backend-specific | Server URL |
| `--evaluate CSV` | — | Evaluate against ground-truth CSV; prints accuracy + confusion matrix |
| `--filter COL=VAL` | — | Only classify rows where column matches value |
| `--rationale` | off | Request a one-sentence explanation alongside each label |
| `--examples CSV` | — | Inject `use=train` rows from CSV as few-shot examples |
| `--concurrency N` | `4` | Parallel requests |
| `--batch_size N` | `200` | Rows per checkpoint |
| `--timeout N` | `120` | Per-request timeout in seconds |
| `--output-col NAME` | stage-specific | Override output column name |
| `--cols COL ...` | `description transcript suggested_words` | Metadata fields to include |
| `-v` | off | Verbose: print each result as it arrives |

## Output columns

Column names are suffixed with the model label to support multi-model comparison:

| Column | Stage | Values |
|--------|-------|--------|
| `is_mental_health_broad_<model>` | screen | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN` |
| `is_mental_health_<model>` | classify | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN` |
| `is_mental_health_<model>_rationale` | classify + `--rationale` | one-sentence explanation |
