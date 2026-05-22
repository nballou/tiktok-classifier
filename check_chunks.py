"""
check_chunks.py — Classification progress across all chunk files.

Usage:
    python check_chunks.py [chunks_dir]       # full table (default: chunks/)
    python check_chunks.py --pending          # pending indices only (for scripting)
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DONE           = "done"
NEEDS_CLASSIFY = "needs_classify"
NEEDS_SCREEN   = "needs_screen"
EMPTY          = "empty"


def chunk_status(path: Path) -> tuple[str, dict]:
    pf    = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    avail = set(pf.schema_arrow.names)

    if total == 0:
        return EMPTY, {"total": 0, "screened": 0, "broad_true": 0, "classified": 0}

    if "is_mental_health_broad" not in avail:
        return NEEDS_SCREEN, {"total": total, "screened": 0, "broad_true": 0, "classified": 0}

    read_cols = [c for c in ["is_mental_health_broad", "is_mental_health"] if c in avail]
    df = pd.read_parquet(path, columns=read_cols)

    screened   = int(df["is_mental_health_broad"].notna().sum())
    broad_true = int((df["is_mental_health_broad"] == "TRUE").sum())
    classified = 0

    if "is_mental_health" in df.columns:
        classified = int(
            ((df["is_mental_health_broad"] == "TRUE") & df["is_mental_health"].notna()).sum()
        )

    needs_screen = screened < total
    needs_classify = (
        "is_mental_health" in df.columns and
        ((df["is_mental_health_broad"] == "TRUE") & df["is_mental_health"].isna()).any()
    )

    if needs_screen:
        status = NEEDS_SCREEN
    elif needs_classify:
        status = NEEDS_CLASSIFY
    else:
        status = DONE

    return status, {
        "total": total, "screened": screened,
        "broad_true": broad_true, "classified": classified,
    }


def chunk_index(path: Path) -> int:
    return int(re.search(r'(\d+)', path.stem).group(1))


def main():
    ap = argparse.ArgumentParser(description="Show classification progress for chunk files.")
    ap.add_argument("chunks_dir", nargs="?", default="chunks")
    ap.add_argument("--pending", action="store_true",
                    help="Print pending indices only (one per line)")
    args = ap.parse_args()

    paths = sorted(Path(args.chunks_dir).glob("chunk_*.parquet"))
    if not paths:
        print(f"No chunk_*.parquet files in {args.chunks_dir}/")
        return

    rows = [(p, *chunk_status(p)) for p in paths]

    if args.pending:
        for p, status, _ in rows:
            if status not in (DONE, EMPTY):
                print(chunk_index(p))
        return

    # Full table
    cw = max(len(str(p)) for p, _, _ in rows) + 2
    header = f"{'Chunk':<{cw}}  {'Status':<18}  {'Rows':>5}  {'Screened':>9}  {'MH-TRUE':>7}  {'Classified':>11}"
    print(f"\n{header}")
    print("-" * len(header))

    counts = {NEEDS_SCREEN: 0, NEEDS_CLASSIFY: 0, DONE: 0, EMPTY: 0}
    pending_indices = []

    for p, status, s in rows:
        idx = chunk_index(p)
        screened_str = f"{s['screened']}/{s['total']}"
        classify_str = f"{s['classified']}/{s['broad_true']}" if s["broad_true"] else "-"
        marker = {"done": "+", "needs_classify": "~", "needs_screen": " ", "empty": "?"}[status]
        print(f"{marker} {str(p):<{cw}}  {status:<18}  {s['total']:>5}  "
              f"{screened_str:>9}  {s['broad_true']:>7}  {classify_str:>11}")
        counts[status] += 1
        if status not in (DONE, EMPTY):
            pending_indices.append(idx)

    print()
    labels = {
        DONE: "+ done",
        NEEDS_CLASSIFY: "~ needs_classify",
        NEEDS_SCREEN:   "  needs_screen",
        EMPTY: "? empty",
    }
    for s in (NEEDS_SCREEN, NEEDS_CLASSIFY, DONE, EMPTY):
        if counts[s]:
            print(f"  {labels[s]:<24} {counts[s]} chunks")

    if not pending_indices:
        print("\nAll chunks classified.")
        return

    lo, hi = pending_indices[0], pending_indices[-1]
    is_contiguous = list(range(lo, hi + 1)) == pending_indices
    chunk_spec = f"{lo}-{hi}" if is_contiguous else " ".join(map(str, pending_indices))

    print(f"\n{len(pending_indices)} pending chunk(s): {chunk_spec}")
    print(f"\n  qsub -v CHUNKS=\"{chunk_spec}\" run_batch.pbs        # all {len(pending_indices)} pending")
    print(f"  qsub -v N=5 run_batch.pbs                           # next 5 pending")


if __name__ == "__main__":
    main()
