"""
classify.py — Two-pass LLM classifier for TikTok video metadata.

Pass 1 (--stage screen): broad liberal screen — labels TRUE anything that
    might relate to mental health. Writes to is_mental_health_broad.
Pass 2 (--stage classify): fine-grained classification applied to rows that
    passed the screen. Writes to is_mental_health.

Supports Ollama (local Mac) and vLLM (HPC) backends. Atomic checkpointing
and full resume support.

Prompts and input formatting live in prompts.py — start there to understand
or modify the classification criteria.

Model defaults (--model, --model-label) are read from model.env in the repo
root if present; CLI flags override them.

Local (Mac, Ollama):
    ollama serve
    python classify.py data.parquet --stage screen -v
    python classify.py data.parquet --stage classify -v

Evaluation:
    python classify.py data.parquet --stage classify --evaluate ground_truth.csv --rationale
"""

import argparse
import asyncio
import csv
import hashlib
import httpx
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI, BadRequestError

from prompts import STAGE_PROMPTS, STAGE_TRUTH_COLS, USER_TEMPLATE, format_examples, row_text


def _load_model_env() -> dict:
    """Parse model.env (repo root) and return its key/value pairs.

    Exits with a clear error if the file is missing or lacks required keys.
    """
    path = Path(__file__).parent / "model.env"
    env = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            if val.startswith('"'):
                end = val.find('"', 1)
                val = val[1:end] if end != -1 else val.strip('"')
            elif val.startswith("'"):
                end = val.find("'", 1)
                val = val[1:end] if end != -1 else val.strip("'")
            else:
                val = val.split('#')[0].strip()
            env[key.strip()] = val
    except FileNotFoundError:
        sys.exit(f"model.env not found at {path} — see reference configs in hpc/config.sh")
    for key in ("MODEL", "MODEL_LABEL"):
        if key not in env:
            sys.exit(f"model.env is missing required key: {key}")
    return env


# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------

LABELS = ["TRUE", "FALSE"]
TEXT_COLS = ["description", "transcript", "suggested_words"]

STAGE_OUTPUT_COLS = {
    "screen":   "is_mental_health_broad",
    "classify": "is_mental_health",
}

# ---------------------------------------------------------------------------
# Decode parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 0
RATIONALE_MAX_TOKENS = 100  # max new tokens when --rationale is set; label-only uses ~5

# JSON schema used for --rationale mode on both backends.
# Constrains `label` to TRUE/FALSE while leaving `rationale` as a free string.
RATIONALE_SCHEMA = {
    "type": "object",
    "properties": {
        "label":     {"type": "string", "enum": ["TRUE", "FALSE"]},
        "rationale": {"type": "string"},
    },
    "required": ["label", "rationale"],
}

# ---------------------------------------------------------------------------
# Backend defaults (overridden by model.env / --model / --endpoint)
# ---------------------------------------------------------------------------

BACKENDS = {
    "ollama": {"endpoint": "http://localhost:11434/v1"},
    "vllm":   {"endpoint": "http://localhost:8000/v1"},
}

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_label(raw: str, labels: set[str]) -> str:
    """Extract a valid label from model output, handling verbose responses."""
    text = raw.strip().upper()
    if text in labels:
        return text
    for label in labels:
        if text.startswith(label) or f" {label}" in f" {text} ":
            return label
    return "UNKNOWN"


def parse_response(raw: str, labels: set[str], with_rationale: bool) -> tuple[str, str | None]:
    """Parse model output into (label, rationale).

    Without rationale: raw output is the label (or close to it).
    With rationale: output is JSON matching RATIONALE_SCHEMA; falls back to
    line-based parsing if JSON decoding fails.
    """
    if not with_rationale:
        return parse_label(raw, labels), None
    try:
        data = json.loads(raw)
        label = parse_label(str(data.get("label", "")), labels)
        rationale = str(data.get("rationale", "")).strip() or None
        return label, rationale
    except (json.JSONDecodeError, AttributeError):
        lines = raw.strip().splitlines()
        return parse_label(lines[0] if lines else "", labels), (lines[1].strip() if len(lines) > 1 else None)


# ---------------------------------------------------------------------------
# Single-row classification
# ---------------------------------------------------------------------------


async def classify_one_ollama(
    http_client: httpx.AsyncClient,
    text: str,
    system_prompt: str,
    model: str,
    labels: list[str],
    with_rationale: bool,
    sem: asyncio.Semaphore,
) -> tuple[str, float | None, str | None]:
    """Classify a single row using the native Ollama API (supports think=false).

    Without rationale: free generation, output parsed with parse_label.
    With rationale: JSON schema via `format` field, label hard-constrained.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
        ],
        "think": False,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    if with_rationale:
        body["format"] = RATIONALE_SCHEMA
    async with sem:
        resp = await http_client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()

    raw = data.get("message", {}).get("content", "")
    label, rationale = parse_response(raw, set(labels), with_rationale)
    return label, None, rationale  # Ollama native API doesn't provide logprobs


async def classify_one_vllm(
    client: AsyncOpenAI,
    text: str,
    system_prompt: str,
    model: str,
    labels: list[str],
    with_rationale: bool,
    use_legacy: list[bool],
    sem: asyncio.Semaphore,
) -> tuple[str, float | None, str | None]:
    """Classify a single row using vLLM with guided decoding.

    Without rationale: structured_outputs (XGrammar) with fallback to
    guided_choice (outlines) — label is hard-constrained to TRUE/FALSE.

    With rationale: guided_json with RATIONALE_SCHEMA — label is still
    hard-constrained to TRUE/FALSE, rationale is a free string.

    Thinking mode is disabled per-request via chat_template_kwargs; the
    server-level flag alone is unreliable on some vLLM versions (issue #35574).
    """
    max_tok = RATIONALE_MAX_TOKENS if with_rationale else max(len(l.split()) for l in labels) + 10
    body = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tok,
    )

    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    if with_rationale:
        body["extra_body"] = {"guided_json": RATIONALE_SCHEMA, **no_think}
    elif not use_legacy[0]:
        body["extra_body"] = {"structured_outputs": {"choice": labels}, **no_think}
    else:
        body["extra_body"] = {"guided_choice": labels, **no_think}

    async with sem:
        try:
            resp = await client.chat.completions.create(**body)
        except BadRequestError as exc:
            if not use_legacy[0] and not with_rationale and "structured_outputs" in str(exc):
                use_legacy[0] = True
                body["extra_body"] = {"guided_choice": labels, **no_think}
                resp = await client.chat.completions.create(**body)
            elif "maximum context length" in str(exc):
                print(f"[WARN] Skipping row: prompt exceeds context limit ({exc})", flush=True)
                return "TOO_LONG", None, None
            else:
                raise

    raw = resp.choices[0].message.content.strip()
    label, rationale = parse_response(raw, set(labels), with_rationale)
    p = _prob_true(resp)
    return label, p, rationale


def _prob_true(response) -> float | None:
    """Extract p(TRUE) from the first output token's top logprobs."""
    try:
        top = response.choices[0].logprobs.content[0].top_logprobs
        probs = {e.token.strip().upper(): math.exp(e.logprob) for e in top}
        return probs.get("TRUE")
    except (AttributeError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Infrastructure: data loading, checkpointing, batch loop
# ---------------------------------------------------------------------------


def _load_pending(
    path: Path,
    output_col: str,
    row_filter: str | None,
    prob_col: str,
) -> tuple[pd.DataFrame, list]:
    """Read a parquet file and return (df, pending_indices).

    Ensures output columns exist, marks failed-scrape rows as SKIPPED,
    and applies any --filter.
    """
    df = pd.read_parquet(path)

    for col in (output_col, prob_col):
        if col not in df.columns:
            df[col] = pd.NA

    if "scrape_status" in df.columns:
        skippable = df[output_col].isna() & (df["scrape_status"] == "error")
        n_skipped = skippable.sum()
        if n_skipped:
            df.loc[skippable, output_col] = "SKIPPED"
            print(f"Skipped {n_skipped} rows with scrape_status='error'")

    candidates = df[df[output_col].isna()]
    if row_filter:
        filter_col, filter_val = row_filter.split("=", 1)
        candidates = candidates[df.loc[candidates.index, filter_col] == filter_val]

    pending = candidates.index.tolist()
    total = len(df)
    print(f"{path.name}: {len(pending)} pending, {total - len(pending)} already done, {total} total")
    return df, pending


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet via tmp-then-rename (crash-safe)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.parquet")
    try:
        os.close(fd)
        df.to_parquet(tmp, index=True)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _build_prompt(args: argparse.Namespace, cols: list[str], sample_text: str) -> str:
    """Assemble and return the system prompt, logging estimated token counts."""
    label_col = STAGE_TRUTH_COLS[args.stage]
    examples_block = ""
    if args.examples:
        examples_df, _ = _load_ground_truth(Path(args.examples), label_col)
        examples_block = format_examples(
            examples_df, cols, label_col,
            include_rationale=(args.stage == "classify"),
        )
        if examples_block:
            print(f"Loaded {len(examples_df)} few-shot examples from {args.examples}")

    if args.prompt:
        system_prompt = Path(args.prompt).read_text()
        if "{examples}" in system_prompt:
            system_prompt = system_prompt.replace("{examples}", examples_block)
        elif examples_block:
            print("Warning: prompt file has no {examples} placeholder; examples not included")
    else:
        system_prompt = STAGE_PROMPTS[args.stage].replace("{examples}", examples_block)

    if args.rationale:
        system_prompt += (
            "\n\nAlso provide a one-sentence rationale explaining your decision."
        )

    sys_chars = len(system_prompt)
    user_chars = len(USER_TEMPLATE) - len("{text}") + len(sample_text)
    est_tokens = (sys_chars + user_chars) // 4
    print(f"Prompt size: ~{sys_chars // 4:,} system + ~{user_chars // 4:,} user "
          f"≈ {est_tokens:,} tokens/request")
    if est_tokens > 6000:
        print(f"Warning: large prompts (~{est_tokens:,} tokens) may cause GPU "
              f"memory pressure with concurrency={args.concurrency}")

    return system_prompt


async def classify(args: argparse.Namespace) -> None:
    """Orchestrate classification: load data, build prompt, run batch loop."""
    path = Path(args.file)
    output_col = args.output_col
    rationale_col = f"{output_col}_rationale"
    labels = [l.upper() for l in args.labels]
    cols = args.cols
    with_rationale = args.rationale

    df, pending = _load_pending(path, output_col, args.filter, args.prob_col)
    if not pending:
        return

    if with_rationale and rationale_col not in df.columns:
        df[rationale_col] = pd.NA

    system_prompt = _build_prompt(args, cols, row_text(df.loc[pending[0]], cols))

    sem = asyncio.Semaphore(args.concurrency)
    use_legacy = [False]  # mutable flag: True after first structured_outputs failure
    has_url = "url" in df.columns
    n_pending = len(pending)

    if args.backend == "ollama":
        base_url = args.endpoint.replace("/v1", "")
        http_client: httpx.AsyncClient | None = httpx.AsyncClient(base_url=base_url, timeout=args.timeout)
        openai_client = None
    else:
        http_client = None
        openai_client = AsyncOpenAI(base_url=args.endpoint, api_key="not-needed")

    n_batches = math.ceil(n_pending / args.batch_size)
    print(f"Classifying {n_pending} rows in {n_batches} batches "
          f"(batch_size={args.batch_size}, concurrency={args.concurrency})")
    print("Sending first batch to server...")
    t0 = time.perf_counter()
    done = 0

    try:
        for i in range(0, n_pending, args.batch_size):
            batch = pending[i : i + args.batch_size]
            tasks = []
            for idx in batch:
                text = row_text(df.loc[idx], cols)
                if args.backend == "ollama":
                    coro = classify_one_ollama(
                        http_client, text, system_prompt,
                        args.model, labels, with_rationale, sem,
                    )
                else:
                    coro = classify_one_vllm(
                        openai_client, text, system_prompt,
                        args.model, labels, with_rationale, use_legacy, sem,
                    )
                tasks.append((idx, text, coro))

            results = await asyncio.gather(*[coro for _, _, coro in tasks])

            for (idx, text, _), (label, p, rationale) in zip(tasks, results):
                df.at[idx, output_col] = label
                df.at[idx, args.prob_col] = p
                if with_rationale:
                    df.at[idx, rationale_col] = rationale
                if args.verbose:
                    url = df.loc[idx, "url"] if has_url else ""
                    print(f"    [{idx}] {label}  {url}")
                    print(f"           {text[:80].replace(chr(10), ' ')}…")
                    if rationale:
                        print(f"           rationale: {rationale}")

            done += len(batch)
            _atomic_write(df, path)

            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            eta = (n_pending - done) / rate if rate > 0 else 0
            print(f"  {done}/{n_pending} ({done/n_pending*100:.0f}%)  "
                  f"{rate:.1f} rows/s  ETA {eta:.0f}s")
    finally:
        if http_client:
            await http_client.aclose()

    elapsed = time.perf_counter() - t0
    counts = df[output_col].value_counts()
    summary = "  ".join(f"{k}: {v}" for k, v in counts.items() if k != "SKIPPED")
    print(f"\nDone in {elapsed:.1f}s ({n_pending/elapsed:.1f} rows/s)")
    print(f"  {summary}")


# ---------------------------------------------------------------------------
# Ground truth, evaluation, experiment logging
# ---------------------------------------------------------------------------


def _load_ground_truth(
    csv_path: Path,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a ground-truth CSV and return (examples_df, test_df)."""
    gt = pd.read_csv(csv_path, dtype=str, quoting=csv.QUOTE_ALL)
    gt.columns = gt.columns.str.strip()

    required = {"url", label_col, "use"}
    missing = required - set(gt.columns)
    if missing:
        sys.exit(f"Ground-truth CSV missing columns: {', '.join(sorted(missing))}")

    gt[label_col] = gt[label_col].str.strip().str.upper()
    gt["use"] = gt["use"].str.strip().str.lower()

    n_missing = gt[label_col].isna().sum()
    if n_missing:
        print(f"Warning: dropping {n_missing} rows with missing {label_col}")
        gt = gt.dropna(subset=[label_col])

    invalid_labels = set(gt[label_col]) - {"TRUE", "FALSE"}
    if invalid_labels:
        sys.exit(f"Invalid {label_col} values: {invalid_labels}")

    invalid_use = set(gt["use"]) - {"train", "test"}
    if invalid_use:
        sys.exit(f"Invalid 'use' values: {invalid_use} (expected 'train' or 'test')")

    return gt[gt["use"] == "train"].copy(), gt[gt["use"] == "test"].copy()


_EXPERIMENT_LOG = Path("experiment_log.csv")


def _log_experiment(args, accuracy, total, tp, tn, fp, fn, elapsed, prompt_hash):
    file_exists = _EXPERIMENT_LOG.exists()
    row = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "stage":        args.stage,
        "model":        args.model,
        "backend":      args.backend,
        "accuracy":     f"{accuracy:.3f}",
        "total":        total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision":    f"{tp/(tp+fp):.3f}" if (tp+fp) > 0 else "N/A",
        "recall":       f"{tp/(tp+fn):.3f}" if (tp+fn) > 0 else "N/A",
        "rows_per_sec": f"{total/elapsed:.2f}",
        "elapsed_sec":  f"{elapsed:.1f}",
        "n_examples":   len(pd.read_csv(args.examples)) if args.examples else 0,
        "prompt_hash":  prompt_hash[:8],
        "concurrency":  args.concurrency,
    }
    with open(_EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nLogged to {_EXPERIMENT_LOG}")


async def evaluate(args: argparse.Namespace) -> None:
    """Classify ground-truth test rows and compare predictions to labels."""
    eval_start = time.perf_counter()
    csv_path = Path(args.evaluate)
    output_col = args.output_col
    label_col = STAGE_TRUTH_COLS[args.stage]

    _, test_df = _load_ground_truth(csv_path, label_col)
    if test_df.empty:
        sys.exit("No rows with use=test in the CSV.")

    subset = test_df.copy()
    for col in ([output_col, "prob_true"] + ([f"{output_col}_rationale"] if args.rationale else [])):
        subset[col] = pd.NA

    print(f"Evaluating {len(subset)} test rows from {csv_path.name}")

    tmp_path = Path(args.file).parent / ".eval_tmp.parquet"
    try:
        subset.to_parquet(tmp_path, index=False)
        eval_args = argparse.Namespace(**vars(args))
        eval_args.file = str(tmp_path)
        await classify(eval_args)
        result = pd.read_parquet(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    truth = dict(zip(test_df["url"], test_df[label_col]))
    result["ground_truth"] = result["url"].map(truth)
    classified = result[result[output_col].isin({"TRUE", "FALSE"})].copy()

    if classified.empty:
        print("No rows were classified (all UNKNOWN or SKIPPED).")
        return

    total = len(classified)
    tp = ((classified[output_col] == "TRUE")  & (classified["ground_truth"] == "TRUE")).sum()
    tn = ((classified[output_col] == "FALSE") & (classified["ground_truth"] == "FALSE")).sum()
    fp = ((classified[output_col] == "TRUE")  & (classified["ground_truth"] == "FALSE")).sum()
    fn = ((classified[output_col] == "FALSE") & (classified["ground_truth"] == "TRUE")).sum()
    accuracy = (tp + tn) / total

    print(f"\n{'='*60}")
    print(f"Evaluation results ({total} test rows)")
    print(f"{'='*60}")
    print(f"Accuracy: {tp+tn}/{total} ({accuracy:.1%})\n")
    print("Confusion matrix (rows = predicted, cols = ground truth):")
    ct = pd.crosstab(classified[output_col], classified["ground_truth"],
                     rownames=["predicted"], colnames=["actual"])
    print(ct.to_string())

    rationale_col = f"{output_col}_rationale"
    has_rationale = rationale_col in result.columns
    mismatches = classified[classified[output_col] != classified["ground_truth"]]
    if not mismatches.empty:
        print(f"\nMismatches ({len(mismatches)}):")
        for _, row in mismatches.iterrows():
            print(f"  {row['url']}")
            print(f"    predicted={row[output_col]}  truth={row['ground_truth']}")
            if has_rationale and pd.notna(row.get(rationale_col)):
                print(f"    rationale: {row[rationale_col]}")
    else:
        print("\nNo mismatches — perfect accuracy.")

    eval_csv = Path(args.file).parent / f"eval_results_{args.stage}.csv"
    save_cols = ["url", output_col, "ground_truth"] + ([rationale_col] if has_rationale else [])
    classified[save_cols].to_csv(eval_csv, index=False)
    print(f"\nFull results saved to {eval_csv.name}")

    elapsed = time.perf_counter() - eval_start
    prompt_hash = hashlib.md5(STAGE_PROMPTS[args.stage].encode()).hexdigest()
    _log_experiment(args, accuracy, total, tp, tn, fp, fn, elapsed, prompt_hash)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Two-pass LLM classifier for TikTok mental-health content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("file", help="Parquet file to classify")
    p.add_argument("--stage", choices=["screen", "classify"], required=True,
                   help="'screen': broad first pass; 'classify': fine-grained second pass")
    p.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                   help="LLM server backend (default: ollama)")
    p.add_argument("--endpoint", default=None,
                   help="Server URL (default: localhost:11434/v1 for ollama, :8000/v1 for vllm)")
    p.add_argument("--evaluate", metavar="CSV",
                   help="Evaluate against ground-truth CSV; prints accuracy + confusion matrix")
    p.add_argument("--filter", default=None, metavar="COL=VAL",
                   help="Only classify rows where COL equals VAL")
    p.add_argument("--rationale", action="store_true",
                   help="Request a one-sentence explanation alongside each label")
    p.add_argument("--examples", metavar="CSV", default=None,
                   help="Inject use=train rows as few-shot examples")
    p.add_argument("--prompt", default=None,
                   help="Path to a .txt prompt file (overrides --stage prompt)")
    p.add_argument("--labels", nargs="+", default=LABELS,
                   help="Allowed output labels (default: TRUE FALSE)")
    p.add_argument("--output-col", default=None,
                   help="Override output column name")
    p.add_argument("--cols", nargs="+", default=TEXT_COLS,
                   help="Metadata fields to include (default: description transcript suggested_words)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-request timeout in seconds")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print each result as it arrives")

    args = p.parse_args()

    env = _load_model_env()
    args.model = env["MODEL"]
    args.model_label = env["MODEL_LABEL"]

    if args.endpoint is None:
        args.endpoint = BACKENDS[args.backend]["endpoint"]
    if args.output_col is None:
        args.output_col = STAGE_OUTPUT_COLS[args.stage]
    args.output_col = f"{args.output_col}_{args.model_label}"
    args.prob_col = f"prob_true_{args.model_label}"
    if args.filter is None and args.stage == "classify":
        args.filter = f"is_mental_health_broad_{args.model_label}=TRUE"

    if args.evaluate:
        asyncio.run(evaluate(args))
    else:
        asyncio.run(classify(args))


if __name__ == "__main__":
    main()
