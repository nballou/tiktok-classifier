"""
0_partition.py — Data preparation utilities for the TikTok classifier.

Partition a parquet file into chunks for HPC array jobs, extract a random
sample for spot-checks, or pull specific rows by URL.

Usage:
    python 0_partition.py <file>.parquet --partition 20
    python 0_partition.py <file>.parquet --sample 100
    python 0_partition.py <file>.parquet --urls <urls-file>.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Partition, sample, and URL extraction
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Data preparation utilities: partition, sample, or extract by URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", help="Input parquet file")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--partition", type=int, metavar="N",
                      help="Split input into N chunks (for HPC array jobs)")
    mode.add_argument("--sample", type=int, metavar="N",
                      help="Extract N random rows as a test set")
    mode.add_argument("--urls", metavar="FILE",
                      help="Extract rows matching URLs listed in FILE (one per line)")

    p.add_argument("-o", "--output", default=None,
                   help="Output path (default varies by mode)")

    args = p.parse_args()

    if args.partition:
        partition(Path(args.file), args.partition, Path("chunks"))
    elif args.sample:
        out = Path(args.output or "test_sample.parquet")
        extract_sample(Path(args.file), args.sample, out)
    elif args.urls:
        out = Path(args.output or "test_urls.parquet")
        extract_urls(Path(args.file), Path(args.urls), out)


if __name__ == "__main__":
    main()
