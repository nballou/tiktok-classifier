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

## For collaborators

### Where to start

The classification logic lives in two files:

- **`prompts.py`** — the methodological core. Contains both system prompts with their
  full TRUE/FALSE criteria, how video metadata is formatted for the model, and how
  few-shot examples are constructed. Start here if you want to review or improve the
  classification criteria.
- **`classify.py`** — the implementation. `classify_one_vllm()` shows how structured
  decoding is used to constrain outputs; `parse_response()` handles output parsing.
  `TEMPERATURE`, `RATIONALE_MAX_TOKENS`, `LABELS`, and `TEXT_COLS` at the top of the
  file are the key parameters.

Everything else (`hpc/`, `prepare_imhi_eval.py`, the batch loop in `classify()`) is
infrastructure and can be ignored on a first read.

### Testing a change locally

The fastest feedback loop is Ollama + the held-out test set:

```bash
uv sync
ollama pull gemma4:27b-it-qat   # or any instruct model

# Optionally edit prompts.py, then:
python classify.py metadata/data.parquet \
  --stage classify --evaluate handcoded_examples.csv \
  --filter is_superficial_mental_health_handcoded=TRUE \
  --rationale --backend ollama -v
```

This classifies the ~50 held-out rows in `handcoded_examples.csv` (use=test),
prints accuracy and a confusion matrix, and lists mismatches with rationales so
you can see exactly where the model disagrees with the ground truth. Results are
appended to `experiment_log.csv` for comparison across runs.

### Key design decisions

- **Two-pass rather than one**: the screen pass uses a high-recall prompt to avoid
  losing edge cases; the classify pass applies stricter criteria only to the ~8% of
  videos that pass. This keeps the expensive classify prompt away from the bulk of data.
- **Model label suffix on output columns** (e.g. `is_mental_health_gemma4_26b`): allows
  results from multiple models to coexist in the same parquet for direct comparison.
- **Structured/guided decoding**: the model is constrained at the token level to
  produce only valid output — `TRUE`/`FALSE` for label-only mode, or a JSON object
  with a hard-constrained `label` field and a free `rationale` string. See the
  [Structured and guided decoding](#structured-and-guided-decoding) section for details.

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

The model name, model label, and the classify-pass `--filter` are all derived
from `model.env` automatically — no extra flags needed.

### HPC (PBS)

```bash
qsub hpc/run_smoke_test.pbs               # 20-row sanity check
qsub -v ROWS=1000 hpc/run_smoke_test.pbs  # scale test
qsub hpc/run_classify.pbs                 # full dataset; resubmit if walltime hit
```

Model configuration (name, label, quantization, thinking mode) lives in
`model.env` at the repo root — edit that file to switch models. The PBS
scripts source it automatically.

## Structured and guided decoding

This is the most finicky part of the pipeline.

### Why structured decoding?

Without constraints, LLMs can produce verbose output ("The answer is TRUE because…")
or refuse to answer with a valid label at all. Post-hoc string matching is fragile.
Structured/guided decoding solves this at the token level: the model physically cannot
emit tokens that would violate the constraint. A `TRUE`/`FALSE` label is always returned.

### The four cases

The approach to enforcing structured output varies depending on the backend, and whether we are prompting for a rationale. The `--rationale` flag is used exclusively for debugging - by having the model report its reasoning for a certain label, we were able to iterate on the prompts. Rationale is not used in production. 

| Backend | Mode | Mechanism | Where in code |
|---------|------|-----------|---------------|
| vLLM | label only | `structured_outputs: {choice: ["TRUE","FALSE"]}` (XGrammar) | `classify_one_vllm` |
| vLLM | `--rationale` | `guided_json: RATIONALE_SCHEMA` | `classify_one_vllm` |
| Ollama | label only | free generation + `parse_label` fallback | `classify_one_ollama` |
| Ollama | `--rationale` | `format: RATIONALE_SCHEMA` (native Ollama) | `classify_one_ollama` |

Ollama's label-only mode does not use structured decoding (the Ollama OpenAI-compatible
endpoint does not expose `guided_choice`), but in practice the models we use follow
the "Reply with a single token" instruction reliably on label-only prompts.

### RATIONALE_SCHEMA

For rationale mode on both backends, the model is constrained to produce a JSON object
matching this schema (defined as `RATIONALE_SCHEMA` in `classify.py`):

```json
{
  "type": "object",
  "properties": {
    "label":     {"type": "string", "enum": ["TRUE", "FALSE"]},
    "rationale": {"type": "string"}
  },
  "required": ["label", "rationale"]
}
```

`label` is hard-constrained to `TRUE`/`FALSE`; `rationale` is a free string.
`parse_response()` JSON-decodes the output and extracts both fields, falling back to
line-based parsing only if JSON decoding fails (which it shouldn't when constraints
are active).

### vLLM legacy API fallback

vLLM has two guided-decoding backends:
- **XGrammar** (newer): `extra_body: {structured_outputs: {choice: [...]}}` — used by default
- **Outlines** (older): `extra_body: {guided_choice: [...]}` — used as fallback

The first request attempts XGrammar; if it raises an error mentioning `structured_outputs`,
`use_legacy[0]` is set to `True` and all subsequent requests use `guided_choice`.
This means there's no manual configuration needed when switching vLLM versions.

### Disabling thinking mode (Qwen3)

Qwen3 models include a chain-of-thought "thinking" mode that emits a `<think>…</think>`
block before the answer. This contaminates structured-output parsing — the grammar tries
to match the thinking block against the schema and fails.

Thinking mode must be disabled at **two** levels:
1. **Server level**: `--default-chat-template-kwargs '{"enable_thinking": false}'` in the
   `vllm serve` command (see `hpc/config.sh`)
2. **Per-request level**: `extra_body: {chat_template_kwargs: {enable_thinking: false}}`
   sent with every request (the `no_think` dict in `classify_one_vllm`)

Both are required. The server-level flag alone is unreliable on some vLLM versions
(tracked in [vLLM issue #35574](https://github.com/vllm-project/vllm/issues/35574));
the per-request flag alone has no effect without the server-level flag. The double-layer
approach is the only reliable solution we found.

Non-Qwen models (Gemma, Llama, etc.) are unaffected — the `enable_thinking` key is
ignored if the model's chat template doesn't recognise it.

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
  --rationale --concurrency 2 -v
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
  --stage classify --evaluate imhi_eval.csv --backend vllm
```

## CLI reference

### classify.py

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | required | `screen` or `classify` |
| `--backend` | `ollama` | `ollama` or `vllm` |
| `--endpoint` | backend-specific | Server URL (model is set via `model.env`) |
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
