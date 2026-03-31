"""
classify.py — LLM-based binary classifier for TikTok video metadata.

Classifies rows as mental-health-related (TRUE/FALSE) via a vLLM server
with guided decoding. Works directly on the source parquet file with
atomic checkpointing and full resume support.

Requirements:
    pip install pandas pyarrow openai

Local (Mac, vllm-metal):
    VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve Qwen/Qwen3-8B
    python classify.py tiktok_metadata.parquet --verbose

HPC (single GPU job):
    python classify.py tiktok_metadata.parquet \\
        --model <model-name> \\
        --concurrency 32

HPC (array jobs — partition first, then one job per chunk):
    python classify.py data/tiktok_metadata.parquet --partition 20
    # produces chunks/chunk_000.parquet ... chunks/chunk_019.parquet
    # PBS array jobs each run:
    python classify.py chunks/chunk_${PBS_ARRAY_INDEX}.parquet
    # merge when all jobs finish:
    python classify.py chunks/ --merge -o classified.parquet

Design:
    - Rows with a non-null is_mental_health value are skipped (resume).
    - Rows are processed in fixed-size batches; each batch is checkpointed
      atomically (write-tmp-then-rename) so a crash loses at most one batch.
    - Guided decoding constrains output to exactly TRUE or FALSE; logprobs
      give you p(TRUE) for free.
"""

import argparse
import asyncio
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = """\
You are a binary classifier for social media video content.

Label a video TRUE if its content meaningfully engages with direct personal or
collective experience of a mental health condition. Label it FALSE otherwise.

Mental health conditions include, but are not limited to:
- Mood disorders (depression, bipolar), anxiety disorders, trauma/stressor-
  related disorders (PTSD, C-PTSD), OCD and related, eating disorders,
  dissociative disorders, psychotic disorders, self-harm and suicidality,
  neurodevelopmental conditions discussed in a mental-health context (ADHD,
  autism), and substance use disorders framed around psychological dependency.

TRUE requires one or more of: describing symptoms or lived experience; offering
coping strategies or support; discussing diagnosis, treatment, or medication;
reflecting on psychological impact on daily life; expressing solidarity or
shared experience around a condition.

FALSE if the content: uses a mental-health term only as a hashtag, metaphor,
or joke; describes ordinary situational emotions without reference to a
condition; or is primarily about another topic that incidentally mentions
a mental-health term.

Reply with a single token: TRUE or FALSE.

Video metadata:
{text}"""

LABELS = ["TRUE", "FALSE"]
TEXT_COLS = ["description", "transcript", "suggested_words"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def row_text(row: pd.Series, cols: list[str]) -> str:
    """Concatenate non-empty metadata columns into a single prompt input."""
    return "\n".join(
        str(row[c]) for c in cols
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
# Partition & merge (HPC utilities)
# ---------------------------------------------------------------------------


def partition(input_path: Path, n: int, output_dir: Path) -> None:
    """Split a parquet file into n chunks for parallel processing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)
    for col in ("is_mental_health", "prob_true"):
        if col not in df.columns:
            df[col] = pd.NA
    chunks = [df.iloc[idx] for idx in np.array_split(range(len(df)), n)]
    for i, chunk in enumerate(chunks):
        p = output_dir / f"chunk_{i:03d}.parquet"
        chunk.to_parquet(p, index=True)
        print(f"  {len(chunk):>6} rows → {p}")
    print(f"\n{len(df)} rows across {n} chunks in {output_dir}/")


def extract_sample(input_path: Path, n: int, output_path: Path) -> None:
    """Extract n random rows as a test set."""
    df = pd.read_parquet(input_path)
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=42)
    for col in ("is_mental_health", "prob_true"):
        if col not in sample.columns:
            sample[col] = pd.NA
    sample.to_parquet(output_path, index=True)
    print(f"Sampled {n} rows → {output_path}")


def extract_urls(input_path: Path, urls_file: Path, output_path: Path) -> None:
    """Extract rows whose URL matches entries in a file (one URL per line)."""
    urls = {line.strip() for line in urls_file.read_text().splitlines() if line.strip()}
    df = pd.read_parquet(input_path)
    if "url" not in df.columns:
        sys.exit("No 'url' column found in the parquet file.")
    subset = df[df["url"].isin(urls)]
    missing = urls - set(subset["url"])
    if missing:
        print(f"Warning: {len(missing)} URLs not found in data")
    for col in ("is_mental_health", "prob_true"):
        if col not in subset.columns:
            subset[col] = pd.NA
    subset.to_parquet(output_path, index=True)
    print(f"Extracted {len(subset)} rows from {len(urls)} URLs → {output_path}")


def merge(chunks_dir: Path, output_path: Path) -> None:
    """Merge classified chunks back into a single parquet file."""
    paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not paths:
        sys.exit(f"No chunk_*.parquet files found in {chunks_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    df.to_parquet(output_path, index=True)
    done = df["is_mental_health"].notna().sum()
    print(f"Merged {len(paths)} chunks → {output_path}  ({done}/{len(df)} classified)")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


async def classify(args: argparse.Namespace) -> None:
    path = Path(args.file)
    df = pd.read_parquet(path)
    output_col = args.output_col
    labels = [l.upper() for l in args.labels]

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
    if args.filter:
        filter_col, filter_val = args.filter.split("=", 1)
        candidates = candidates[df.loc[candidates.index, filter_col] == filter_val]

    pending = candidates.index.tolist()
    total, n_pending = len(df), len(pending)
    already = total - n_pending
    print(f"{path.name}: {n_pending} pending, {already} already done, {total} total")
    if not pending:
        return

    prompt_template = Path(args.prompt).read_text() if args.prompt else DEFAULT_PROMPT
    client = AsyncOpenAI(base_url=args.endpoint, api_key="not-needed")
    sem = asyncio.Semaphore(args.concurrency)
    cols = args.cols
    verbose = args.verbose
    label_set = set(labels)

    # Detect whether server supports the new structured_outputs API or the
    # legacy guided_choice parameter. Try new first, fall back once.
    use_legacy = False

    def extra_body():
        if use_legacy:
            return {"guided_choice": labels}
        return {"structured_outputs": {"choice": labels}}

    async def classify_one(idx: int) -> tuple[int, str, float | None]:
        nonlocal use_legacy
        text = row_text(df.loc[idx], cols)
        body = dict(
            model=args.model,
            messages=[{"role": "user", "content": prompt_template.format(text=text)}],
            temperature=0,
            max_tokens=max(len(l.split()) for l in labels) + 2,
            extra_body=extra_body(),
        )
        async with sem:
            try:
                resp = await client.chat.completions.create(**body)
            except Exception as exc:
                if not use_legacy and "structured_outputs" in str(exc):
                    use_legacy = True
                    body["extra_body"] = extra_body()
                    resp = await client.chat.completions.create(**body)
                else:
                    raise

        raw = resp.choices[0].message.content.strip().upper()
        label = raw if raw in label_set else "UNKNOWN"
        p = prob_true(resp)

        if verbose:
            url = df.loc[idx, "url"] if "url" in df.columns else ""
            preview = text[:80].replace("\n", " ")
            print(f"    [{idx}] {label}  {url}")
            print(f"           {preview}…")

        return idx, label, p

    # Process in batches — each batch is checkpointed on completion.
    n_batches = math.ceil(n_pending / args.batch_size)
    print(f"Classifying {n_pending} rows in {n_batches} batches "
          f"(batch_size={args.batch_size}, concurrency={args.concurrency})")
    print("Sending first batch to server...")
    t0 = time.perf_counter()
    done = 0
    for i in range(0, n_pending, args.batch_size):
        batch = pending[i : i + args.batch_size]
        results = await asyncio.gather(*[classify_one(j) for j in batch])

        for idx, label, p in results:
            df.at[idx, output_col] = label
            df.at[idx, "prob_true"] = p

        done += len(batch)
        atomic_write(df, path)

        elapsed = time.perf_counter() - t0
        rate = done / elapsed
        eta = (n_pending - done) / rate if rate > 0 else 0
        print(f"  {done}/{n_pending} ({done/n_pending*100:.0f}%)  "
              f"{rate:.1f} rows/s  ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    counts = df[output_col].value_counts()
    summary = "  ".join(f"{k}: {v}" for k, v in counts.items() if k != "SKIPPED")
    print(f"\nDone in {elapsed:.1f}s ({n_pending/elapsed:.1f} rows/s)")
    print(f"  {summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Classify TikTok metadata as mental-health-related.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", help="Parquet file (or chunks dir with --merge)")

    # Modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--partition", type=int, metavar="N",
                      help="Split input into N chunks (for HPC array jobs)")
    mode.add_argument("--merge", action="store_true",
                      help="Merge chunk_*.parquet files from a directory")
    mode.add_argument("--sample", type=int, metavar="N",
                      help="Extract N random rows as a test set")
    mode.add_argument("--urls", metavar="FILE",
                      help="Extract rows matching URLs listed in FILE (one per line)")

    # Classification options
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--cols", nargs="+", default=TEXT_COLS)
    p.add_argument("--prompt", default=None, help="Path to .txt prompt override")
    p.add_argument("--labels", nargs="+", default=LABELS,
                   help="Allowed output labels for guided decoding (default: TRUE FALSE)")
    p.add_argument("--output-col", default="is_mental_health",
                   help="Column to write results to (default: is_mental_health)")
    p.add_argument("--filter", default=None, metavar="COL=VAL",
                   help="Only classify rows where COL equals VAL (e.g. is_mental_health=TRUE)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("-o", "--output", default=None, help="Output path (for --merge)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print each row's classification result")

    args = p.parse_args()

    if args.partition:
        partition(Path(args.file), args.partition, Path("chunks"))
    elif args.merge:
        out = Path(args.output or "classified.parquet")
        merge(Path(args.file), out)
    elif args.sample:
        out = Path(args.output or "test_sample.parquet")
        extract_sample(Path(args.file), args.sample, out)
    elif args.urls:
        out = Path(args.output or "test_urls.parquet")
        extract_urls(Path(args.file), Path(args.urls), out)
    else:
        asyncio.run(classify(args))


if __name__ == "__main__":
    main()