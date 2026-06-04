"""
2_merge.py — Merge classified chunks back into a single parquet file.

After HPC array jobs classify each chunk independently, this script
reassembles them into one file.

Usage:
    python 2_merge.py chunks/ -o classified.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def merge(chunks_dir: Path, output_path: Path, col: str) -> None:
    """Merge classified chunks back into a single parquet file."""
    paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not paths:
        sys.exit(f"No chunk_*.parquet files found in {chunks_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    df.to_parquet(output_path, index=True)
    done = df[col].notna().sum() if col in df.columns else "?"
    print(f"Merged {len(paths)} chunks → {output_path}  ({done}/{len(df)} classified in '{col}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Merge classified chunk_*.parquet files into one parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("chunks_dir", help="Directory containing chunk_*.parquet files")
    p.add_argument("-o", "--output", default="classified.parquet",
                   help="Output parquet path (default: classified.parquet)")
    p.add_argument("--col", default="is_mental_health",
                   help="Column to report completion count for (default: is_mental_health)")

    args = p.parse_args()
    merge(Path(args.chunks_dir), Path(args.output), args.col)


if __name__ == "__main__":
    main()
