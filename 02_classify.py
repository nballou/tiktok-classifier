"""
02_classify.py — Two-pass LLM classifier for TikTok video metadata.

Pass 1 (--stage screen): broad liberal screen — labels TRUE anything that
    might relate to mental health. Writes to is_mental_health_broad.
Pass 2 (--stage classify): fine-grained classification applied to rows that
    passed the screen (--filter is_mental_health_broad=TRUE). Writes to
    is_mental_health.

Supports Ollama (local Mac) and vLLM (HPC) backends. Atomic checkpointing
and full resume support.

Requirements:
    pip install pandas pyarrow openai httpx

Local (Mac, Ollama):
    ollama serve
    python 02_classify.py data.parquet --stage screen -v
    python 02_classify.py data.parquet --stage classify --filter is_mental_health_broad=TRUE -v

Evaluation (with rationale):
    python 02_classify.py data.parquet --stage classify --evaluate ground_truth.csv --rationale

Design:
    - Rows with a non-null output column are skipped on resume.
    - Batches are checkpointed atomically (write-tmp-then-rename).
    - vLLM uses guided decoding; Ollama uses prompt + parsing.
    - --rationale requests JSON {label, rationale} output for inspection.
"""

import argparse
import asyncio
import csv
import hashlib
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Pass 1: cast a wide net — anything that could plausibly relate to mental
# health should be labelled TRUE. False positives are acceptable here; they
# will be filtered out in the second pass.
SCREEN_SYSTEM_PROMPT = """\
You are a first-pass filter for social media video content.

You will receive up to three metadata fields per video:
- description: text written by the creator
- transcript: auto-generated speech-to-text of the audio
- suggested_words: TikTok-generated search terms derived from the full video content


Label TRUE if the video contains at least one clear signal of mental health
relevance:
- A named mental health condition (depression, anxiety disorder, ADHD, PTSD, OCD, bipolar disorder, eating disorders, self-harm, psychosis, autism, BPD, substance use disorder, etc.) — whether as a hashtag, in a description, or in a transcript
- Reference to psychological or psychiatric treatment (seeing a therapist,
  psychologist, or psychiatrist; psychiatric medication; mental health diagnosis)
- First-person description of psychological symptoms or emotional struggles
  explicitly tied to a condition
- Community or peer support content explicitly framed around mental health

Label FALSE if mental health terms appear only as:
- Clear jokes or colloquial hyperbole ("that's so depressing")
- Song or audio lyrics — identifiable by ♪/♫ symbols in the transcript,
  "(singing)", "(music)", or similar ASR markers — unless the creator is
  clearly speaking about their own experience outside the lyrics
- Generic wellness, self-care, fitness, or relaxation content — including
  colloquial "therapy" (e.g. "ocean therapy", "retail therapy", "colouring therapy")
- Physical health treatments (e.g. chemotherapy) with no psychological framing

When uncertain, label TRUE. Missing a relevant video is worse than passing
an irrelevant one through to the next stage.

Reply with a single token: TRUE or FALSE.

{examples}"""

# Pass 2: applied only to rows that passed the screen. Distinguishes genuine
# mental-health content from spurious matches.
CLASSIFY_SYSTEM_PROMPT = """\
You are a second-pass classifier for social media video content. Every video
you see has already passed a broad screen confirming it contains some mental
health signal. Your task is to determine whether that signal reflects genuine
engagement with mental health.

You will receive up to three metadata fields per video:
- description: text written by the creator
- transcript: auto-generated speech-to-text of the audio
- suggested_words: TikTok-generated search terms derived from the full video content

The content must engage with mental health specifically — not just physical
health, disability, or everyday emotional experience. Recognised mental health
conditions include: depression, anxiety disorders, PTSD, OCD, ADHD, bipolar
disorder, eating disorders, self-harm, psychosis, BPD, phobias, substance use
disorder, autism spectrum, dissociation, and similar. Physical health conditions
(sickle cell, PCOS, hypermobility, chronic pain, wheelchair use, food allergies,
cancer, etc.) do NOT qualify unless the content explicitly discusses a
co-occurring mental health condition by name or clear description.

Read all fields together: a hashtag is not isolated — if the description or
transcript provides MH context, a condition hashtag confirms the topic. Weight
the description heavily; creators often use it to frame what the video is about.

Label TRUE if the content meaningfully engages with mental health through any of:
- First-person description of symptoms, diagnosis, or lived experience of a
  recognised MH condition — including condition-specific language or slang
  (e.g. "crash out" for emotional dysregulation, "being sectioned" for
  psychiatric hospitalisation, "masking" for autism camouflaging)
- Discussing psychological or psychiatric treatment in a personal context:
  therapy sessions, therapeutic modalities (IFS, CBT, DBT, trauma therapy,
  psychotherapy), psychiatric medication, or psychiatric hospitalisation
- Offering or seeking coping strategies, support, or solidarity explicitly
  around a MH condition
- Expressing community membership, identity, or solidarity framed around a
  recognised MH condition — including awareness/educational content,
  participation in MH-coded communities, and meme or humour-format content
  whose central subject is a condition (e.g. "my ADHD be like", BPD community
  content, autism community content)
- Profound grief or trauma (e.g. bereavement, loss of a loved one, severe
  traumatic events) where the psychological impact is explicitly discussed —
  not incidental sadness or routine disappointment
- When the transcript contains only music or is absent: a preponderance of
  converging signals across description, hashtags, and suggested_words that
  make a specific MH condition clearly the central topic. Apply this when the
  condition appears to be the primary subject — not when a single MH hashtag
  sits alongside many unrelated lifestyle or entertainment tags.

Label FALSE if the mental health signal is spurious:
- A single MH-adjacent hashtag (e.g. #anxiety, #stress) embedded among hobby,
  lifestyle, or entertainment hashtags where the video is clearly about something
  else — such hashtags function as personality identifiers, not content
  descriptors. This is different from a condition being the primary subject: if
  the condition is the main or sole hashtag and the description or suggested_words
  are also centred on it, label TRUE.
- Colloquial expressions or jokes with no supporting content (e.g. "I have PTSD
  from this", "that's so OCD", "Teams ringtone PTSD")
- MH terms appear only in song/audio lyrics (identifiable by ♪/♫, "(singing)",
  "(music)" in the transcript) and the creator is not speaking about their own
  experience outside the lyrics
- The mention is incidental to content primarily about something else (e.g.
  therapy mentioned once in passing; ADHD as a throwaway excuse)
- The content discusses a third party's or fictional character's condition
  without the creator expressing personal connection, solidarity, or lived
  experience (e.g. analysing a TV character's diagnosis with no personal framing)
- Physical health conditions, chronic illness, or disability content (sickle
  cell, PCOS, hypermobility, EDS, chronic pain, food allergies, wheelchair use,
  cancer, chemotherapy, cosmetic surgery, etc.) unless a co-occurring MH
  condition is explicitly named or clearly described — "psychological impact"
  or "living with" a physical condition is not sufficient
- Wellness, self-care, or relaxation content using MH-adjacent language with
  no clinical or lived-experience framing (e.g. "ocean therapy", "retail
  therapy", "good for my mental health")
- Routine emotional experiences (unrequited love, heartbreak, relationship
  drama, career frustration, social conflict, general stress) not tied to a
  recognised MH condition — profound grief or trauma may qualify only if the
  psychological impact is explicitly discussed (see TRUE criteria above)
- Content about colloquial "narcissism" or "toxic" behaviour in relationships
  without clinical framing of a personality disorder

Reply with a single token: TRUE or FALSE.

{examples}"""

STAGE_PROMPTS = {
    "screen": SCREEN_SYSTEM_PROMPT,
    "classify": CLASSIFY_SYSTEM_PROMPT,
}

STAGE_OUTPUT_COLS = {
    "screen": "is_mental_health_broad",
    "classify": "is_mental_health",
}

# Ground-truth column in handcoded CSV used for evaluation and few-shot examples
STAGE_TRUTH_COLS = {
    "screen": "is_superficial_mental_health_handcoded",
    "classify": "is_mental_health_handcoded",
}

USER_TEMPLATE = "Video metadata:\n{text}"

LABELS = ["TRUE", "FALSE"]
TEXT_COLS = ["description", "transcript", "suggested_words"]

# Backend-specific defaults
BACKENDS = {
    "ollama": {"endpoint": "http://localhost:11434/v1", "model": "qwen3:8b"},
    "vllm": {"endpoint": "http://localhost:8000/v1", "model": "Qwen/Qwen3-8B"},
}

# ---------------------------------------------------------------------------
# Helpers
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

    When with_rationale=True, expects the label on the first line and a
    rationale sentence on the second line.
    """
    if not with_rationale:
        return parse_label(raw, labels), None
    lines = raw.strip().splitlines()
    label = parse_label(lines[0] if lines else "", labels)
    rationale = lines[1].strip() if len(lines) > 1 else None
    return label, rationale


def row_text(row: pd.Series, cols: list[str]) -> str:
    """Concatenate non-empty metadata columns into a labelled prompt input."""
    return "\n".join(
        f"{c}: {row[c]}" for c in cols
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip()
    )


def prob_true(response) -> float | None:
    """Extract p(TRUE) from the first output token's top logprobs."""
    try:
        top = response.choices[0].logprobs.content[0].top_logprobs
        probs = {e.token.strip().upper(): math.exp(e.logprob) for e in top}
        return probs.get("TRUE")
    except (AttributeError, IndexError, TypeError):
        return None


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet via tmp-then-rename (crash-safe)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.parquet")
    try:
        os.close(fd)
        df.to_parquet(tmp, index=True)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def load_ground_truth(
    csv_path: Path,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a ground-truth CSV and return (examples_df, test_df).

    label_col is the column used as the classification ground truth
    (varies by stage). The `rationale` column is optional and included
    in the prompt for train examples. `use` should be 'train' or 'test'.
    """
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

    examples_df = gt[gt["use"] == "train"].copy()
    test_df = gt[gt["use"] == "test"].copy()
    return examples_df, test_df


def format_examples(
    examples_df: pd.DataFrame,
    cols: list[str],
    label_col: str,
    include_rationale: bool = True,
) -> str:
    """Build the few-shot examples block for injection into the prompt."""
    if examples_df.empty:
        return ""

    parts = ["Examples:\n"]
    for _, ex_row in examples_df.iterrows():
        text = row_text(ex_row, cols)
        if not text.strip():
            print(f"Warning: no metadata for example {ex_row['url']}, skipping")
            continue

        block = f"Video metadata:\n{text}\nLabel: {ex_row[label_col]}"

        if include_rationale:
            rationale = ex_row.get("rationale")
            if pd.notna(rationale) and str(rationale).strip():
                block += f"\nRationale: {rationale}"

        parts.append(block)

    if len(parts) == 1:
        return ""

    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_pending(
    path: Path,
    output_col: str,
    row_filter: str | None,
) -> tuple[pd.DataFrame, list]:
    """Read a parquet file and return (df, pending_indices).

    Ensures output columns exist, marks failed-scrape rows as SKIPPED,
    and applies any --filter. Returns an empty list when nothing is pending.
    """
    df = pd.read_parquet(path)

    for col in (output_col, "prob_true"):
        if col not in df.columns:
            df[col] = pd.NA

    # Skip rows with no usable metadata (failed scrapes)
    if "scrape_status" in df.columns:
        skippable = df[output_col].isna() & (df["scrape_status"] == "error")
        n_skipped = skippable.sum()
        if n_skipped:
            df.loc[skippable, output_col] = "SKIPPED"
            print(f"Skipped {n_skipped} rows with scrape_status='error'")

    # Apply row filter (e.g. --filter is_mental_health=TRUE)
    candidates = df[df[output_col].isna()]
    if row_filter:
        filter_col, filter_val = row_filter.split("=", 1)
        candidates = candidates[df.loc[candidates.index, filter_col] == filter_val]

    pending = candidates.index.tolist()
    total = len(df)
    already = total - len(pending)
    print(f"{path.name}: {len(pending)} pending, {already} already done, {total} total")

    return df, pending


def build_prompt(
    args: argparse.Namespace,
    cols: list[str],
    sample_text: str,
) -> tuple[str, str]:
    """Assemble system prompt and user template; log estimated token counts."""
    label_col = STAGE_TRUTH_COLS[args.stage]
    examples_block = ""
    if args.examples:
        examples_df, _ = load_ground_truth(Path(args.examples), label_col)
        examples_block = format_examples(
            examples_df, cols, label_col,
            include_rationale=(args.stage == "classify"),
        )
        if examples_block:
            print(f"Loaded {len(examples_df)} few-shot examples from {args.examples}")

    if args.prompt:
        # Custom prompt file — everything in the system message
        system_prompt = Path(args.prompt).read_text()
        if "{examples}" in system_prompt:
            system_prompt = system_prompt.replace("{examples}", examples_block)
        elif examples_block:
            print("Warning: prompt file has no {examples} placeholder; examples not included")
    else:
        system_prompt = STAGE_PROMPTS[args.stage].replace("{examples}", examples_block)

    if args.rationale:
        system_prompt += (
            "\n\nFirst line: TRUE or FALSE only. "
            "Second line: one sentence explaining your decision."
        )

    user_template = USER_TEMPLATE

    # Estimate prompt size (~4 chars/token)
    sys_chars = len(system_prompt)
    user_chars = len(user_template) - len("{text}") + len(sample_text)
    est_tokens = (sys_chars + user_chars) // 4
    print(f"Prompt size: ~{sys_chars // 4:,} system + ~{user_chars // 4:,} user "
          f"≈ {est_tokens:,} tokens/request")
    if est_tokens > 6000:
        print(f"Warning: large prompts (~{est_tokens:,} tokens) may cause GPU "
              f"memory pressure with concurrency={args.concurrency}")

    return system_prompt, user_template


# ---------------------------------------------------------------------------
# Single-row classification
# ---------------------------------------------------------------------------

import httpx


async def classify_one_ollama(
    http_client: httpx.AsyncClient,
    text: str,
    system_prompt: str,
    user_template: str,
    model: str,
    labels: list[str],
    with_rationale: bool,
    sem: asyncio.Semaphore,
) -> tuple[str, float | None, str | None]:
    """Classify using native Ollama API (supports think=false)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {"temperature": 0},
    }

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
    user_template: str,
    model: str,
    labels: list[str],
    with_rationale: bool,
    use_legacy: list[bool],
    sem: asyncio.Semaphore,
) -> tuple[str, float | None, str | None]:
    """Classify using vLLM with guided decoding."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
    # With rationale we need free generation (guided_choice forces a single token)
    max_tok = 100 if with_rationale else max(len(l.split()) for l in labels) + 10
    body = dict(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tok,
    )

    if not with_rationale:
        if not use_legacy[0]:
            body["extra_body"] = {"structured_outputs": {"choice": labels}}
        else:
            body["extra_body"] = {"guided_choice": labels}

    async with sem:
        try:
            resp = await client.chat.completions.create(**body)
        except Exception as exc:
            if not use_legacy[0] and not with_rationale and "structured_outputs" in str(exc):
                use_legacy[0] = True
                body["extra_body"] = {"guided_choice": labels}
                resp = await client.chat.completions.create(**body)
            else:
                raise

    raw = resp.choices[0].message.content.strip()
    label, rationale = parse_response(raw, set(labels), with_rationale)
    p = prob_true(resp)
    return label, p, rationale


# ---------------------------------------------------------------------------
# Classification orchestrator
# ---------------------------------------------------------------------------


async def classify(args: argparse.Namespace) -> None:
    """Orchestrate classification: load data, build prompts, run batch loop."""
    path = Path(args.file)
    output_col = args.output_col
    rationale_col = f"{output_col}_rationale"
    labels = [l.upper() for l in args.labels]
    cols = args.cols
    with_rationale = args.rationale

    df, pending = load_pending(path, output_col, args.filter)
    if not pending:
        return

    if with_rationale and rationale_col not in df.columns:
        df[rationale_col] = pd.NA

    sample_text = row_text(df.loc[pending[0]], cols)
    system_prompt, user_template = build_prompt(args, cols, sample_text)

    sem = asyncio.Semaphore(args.concurrency)
    backend = args.backend
    use_legacy = [False]  # mutable flag for vLLM structured_outputs fallback
    verbose = args.verbose
    has_url = "url" in df.columns
    n_pending = len(pending)

    if backend == "ollama":
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
                if backend == "ollama":
                    coro = classify_one_ollama(
                        http_client, text, system_prompt, user_template,
                        args.model, labels, with_rationale, sem,
                    )
                else:
                    coro = classify_one_vllm(
                        openai_client, text, system_prompt, user_template,
                        args.model, labels, with_rationale, use_legacy, sem,
                    )
                tasks.append((idx, text, coro))

            results = await asyncio.gather(*[coro for _, _, coro in tasks])

            for (idx, text, _), (label, p, rationale) in zip(tasks, results):
                df.at[idx, output_col] = label
                df.at[idx, "prob_true"] = p
                if with_rationale:
                    df.at[idx, rationale_col] = rationale
                if verbose:
                    url = df.loc[idx, "url"] if has_url else ""
                    preview = text[:80].replace("\n", " ")
                    print(f"    [{idx}] {label}  {url}")
                    print(f"           {preview}…")
                    if rationale:
                        print(f"           rationale: {rationale}")

            done += len(batch)
            atomic_write(df, path)

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
# Experiment logging
# ---------------------------------------------------------------------------

EXPERIMENT_LOG = Path("experiment_log.csv")


def log_experiment(
    args: argparse.Namespace,
    accuracy: float,
    total: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    elapsed: float,
    prompt_hash: str,
) -> None:
    """Append experiment results to CSV log."""
    file_exists = EXPERIMENT_LOG.exists()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "model": args.model,
        "backend": args.backend,
        "accuracy": f"{accuracy:.3f}",
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": f"{tp/(tp+fp):.3f}" if (tp+fp) > 0 else "N/A",
        "recall": f"{tp/(tp+fn):.3f}" if (tp+fn) > 0 else "N/A",
        "rows_per_sec": f"{total/elapsed:.2f}",
        "elapsed_sec": f"{elapsed:.1f}",
        "n_examples": len(pd.read_csv(args.examples)) if args.examples else 0,
        "prompt_hash": prompt_hash[:8],
        "concurrency": args.concurrency,
    }

    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nLogged to {EXPERIMENT_LOG}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate(args: argparse.Namespace) -> None:
    """Classify ground-truth test rows and compare predictions to labels.

    The ground-truth CSV contains full video metadata, so test rows are
    written directly to a temp parquet for classification (no lookup against
    the source parquet is needed).
    """
    eval_start = time.perf_counter()
    csv_path = Path(args.evaluate)
    output_col = args.output_col

    label_col = STAGE_TRUTH_COLS[args.stage]
    _, test_df = load_ground_truth(csv_path, label_col)
    if test_df.empty:
        sys.exit("No rows with use=test in the CSV.")

    # Build a classification-ready DataFrame from the CSV rows
    subset = test_df.copy()
    init_cols = [output_col, "prob_true"]
    if args.rationale:
        init_cols.append(f"{output_col}_rationale")
    for col in init_cols:
        subset[col] = pd.NA

    print(f"Evaluating {len(subset)} test rows from {csv_path.name}")

    # Write to a temp parquet and classify via the normal pipeline
    tmp_path = Path(args.file).parent / ".eval_tmp.parquet"
    try:
        subset.to_parquet(tmp_path, index=False)
        eval_args = argparse.Namespace(**vars(args))
        eval_args.file = str(tmp_path)
        await classify(eval_args)

        # Read back classified results
        result = pd.read_parquet(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Compare predictions to ground truth
    truth = dict(zip(test_df["url"], test_df[label_col]))
    result["ground_truth"] = result["url"].map(truth)
    classified = result[result[output_col].isin({"TRUE", "FALSE"})].copy()

    if classified.empty:
        print("No rows were classified (all UNKNOWN or SKIPPED).")
        return

    # Calculate metrics
    total = len(classified)
    tp = ((classified[output_col] == "TRUE") & (classified["ground_truth"] == "TRUE")).sum()
    tn = ((classified[output_col] == "FALSE") & (classified["ground_truth"] == "FALSE")).sum()
    fp = ((classified[output_col] == "TRUE") & (classified["ground_truth"] == "FALSE")).sum()
    fn = ((classified[output_col] == "FALSE") & (classified["ground_truth"] == "TRUE")).sum()
    accuracy = (tp + tn) / total

    print(f"\n{'='*60}")
    print(f"Evaluation results ({total} test rows)")
    print(f"{'='*60}")
    print(f"Accuracy: {tp+tn}/{total} ({accuracy:.1%})\n")

    # Confusion matrix
    print("Confusion matrix (rows = predicted, cols = ground truth):")
    ct = pd.crosstab(classified[output_col], classified["ground_truth"],
                     rownames=["predicted"], colnames=["actual"])
    print(ct.to_string())

    # List mismatches for inspection
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

    # Save full results (with rationale) to CSV for debugging
    eval_csv = Path(args.file).parent / f"eval_results_{args.stage}.csv"
    save_cols = ["url", output_col, "ground_truth"]
    if has_rationale:
        save_cols.append(rationale_col)
    classified[save_cols].to_csv(eval_csv, index=False)
    print(f"\nFull results saved to {eval_csv.name}")

    # Log experiment results
    elapsed = time.perf_counter() - eval_start
    prompt_hash = hashlib.md5(STAGE_PROMPTS[args.stage].encode()).hexdigest()
    log_experiment(args, accuracy, total, tp, tn, fp, fn, elapsed, prompt_hash)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Two-pass LLM classifier for TikTok mental-health content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", help="Parquet file to classify")

    p.add_argument("--stage", choices=["screen", "classify"], required=True,
                   help="'screen': broad first pass; 'classify': fine-grained second pass")
    p.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                   help="LLM server backend (default: ollama)")
    p.add_argument("--evaluate", metavar="CSV",
                   help="Classify ground-truth test rows and report accuracy")
    p.add_argument("--rationale", action="store_true",
                   help="Request a rationale from the model (stored in {output_col}_rationale)")
    p.add_argument("--examples", metavar="CSV", default=None,
                   help="Ground-truth CSV; use=train rows are injected as few-shot examples")
    p.add_argument("--model", default=None,
                   help="Model name (default: qwen3:8b for ollama, Qwen/Qwen3-8B for vllm)")
    p.add_argument("--endpoint", default=None,
                   help="Server URL (default: localhost:11434/v1 for ollama, :8000/v1 for vllm)")
    p.add_argument("--cols", nargs="+", default=TEXT_COLS)
    p.add_argument("--prompt", default=None, help="Path to a .txt prompt file (overrides --stage prompt)")
    p.add_argument("--labels", nargs="+", default=LABELS,
                   help="Allowed output labels (default: TRUE FALSE)")
    p.add_argument("--output-col", default=None,
                   help="Column to write results to (default: is_mental_health_broad for screen, "
                        "is_mental_health for classify)")
    p.add_argument("--filter", default=None, metavar="COL=VAL",
                   help="Only classify rows where COL equals VAL")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-request timeout in seconds (default: 120; increase for large models)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print each row's classification result")

    args = p.parse_args()

    # Apply defaults
    defaults = BACKENDS[args.backend]
    if args.model is None:
        args.model = defaults["model"]
    if args.endpoint is None:
        args.endpoint = defaults["endpoint"]
    if args.output_col is None:
        args.output_col = STAGE_OUTPUT_COLS[args.stage]

    if args.evaluate:
        asyncio.run(evaluate(args))
    else:
        asyncio.run(classify(args))


if __name__ == "__main__":
    main()
