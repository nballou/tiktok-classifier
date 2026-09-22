"""
prepare_imhi_eval.py — Convert IMHI test data to 02_classify.py --evaluate format.

Loads binary classification tasks from the Tianlin668/IMHI HuggingFace dataset
and writes a CSV compatible with:

    python classify.py <data.parquet> --stage classify --evaluate imhi_eval.csv

Only the four binary yes/no tasks are included by default (DR, CLP, dreaddit,
loneliness). The multi-class tasks (swmh, t-sid, SAD, CAMS, MultiWD, IRF) are
excluded because their label schemas don't map cleanly to TRUE/FALSE.

Caveats
-------
IMHI asks "does this poster have [condition]?" while our classify pass asks
"does this content genuinely engage with mental health?" The mapping
yes→TRUE / no→FALSE is directionally correct but not identical, so treat
resulting accuracy numbers as a lower-bound proxy rather than a direct
comparison.

The post text is mapped to the `description` field; transcript and
suggested_words are left empty, so the model sees less signal than it does
on real TikTok metadata.

Requirements
------------
    pip install datasets

Usage
-----
    python prepare_imhi_eval.py                        # all binary tasks
    python prepare_imhi_eval.py --tasks DR CLP         # depression only
    python prepare_imhi_eval.py --output my_eval.csv   # custom output path
    python prepare_imhi_eval.py --max-per-task 200     # subsample for speed
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

BINARY_TASKS = ["DR", "CLP", "dreaddit", "loneliness"]

FIELDNAMES = [
    "url", "description", "transcript", "suggested_words",
    "is_mental_health_handcoded", "rationale", "use",
]


def extract_post(query: str) -> str:
    """Pull the post body out of the IMHI instruction template."""
    m = re.search(r'Consider this post:\s*["\'](.+?)["\'](?:\s+Question:|\s*$)',
                  query, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'Consider this post:\s*(.+?)\s*Question:', query, re.DOTALL)
    if m:
        return m.group(1).strip()
    return query.strip()


def extract_label_and_rationale(response: str) -> tuple[str | None, str]:
    """Parse 'Yes/No[, the poster...]. Reasoning: ...' into (label, rationale)."""
    text = response.strip()
    if text.lower().startswith("yes"):
        label = "TRUE"
    elif text.lower().startswith("no"):
        label = "FALSE"
    else:
        return None, ""

    rationale = ""
    m = re.search(r"[Rr]easoning:\s*(.+)", text, re.DOTALL)
    if m:
        rationale = m.group(1).strip()

    return label, rationale


def make_url(task: str, idx: int, post: str) -> str:
    h = hashlib.md5(post.encode()).hexdigest()[:8]
    return f"imhi:{task}:{idx}:{h}"


def load_task(task: str, max_rows: int | None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets library not installed — run: pip install datasets")

    print(f"  {task}...", end=" ", flush=True)
    try:
        ds = load_dataset("Tianlin668/IMHI", task, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"FAILED ({e})")
        return []

    rows = []
    for i, rec in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        post = extract_post(rec.get("query", ""))
        response = rec.get("gpt-3.5-turbo", "")
        label, rationale = extract_label_and_rationale(response)
        if label is None:
            continue
        rows.append({
            "url": make_url(task, i, post),
            "description": post,
            "transcript": "",
            "suggested_words": "",
            "is_mental_health_handcoded": label,
            "rationale": rationale,
            "use": "test",
        })

    true_n = sum(1 for r in rows if r["is_mental_health_handcoded"] == "TRUE")
    print(f"{len(rows)} rows (TRUE={true_n}, FALSE={len(rows)-true_n})")
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Convert IMHI test data to 02_classify.py --evaluate format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", default="imhi_eval.csv",
                   help="Output CSV path (default: imhi_eval.csv)")
    p.add_argument("--tasks", nargs="+", default=BINARY_TASKS,
                   choices=BINARY_TASKS,
                   help=f"Tasks to include (default: all binary tasks: {BINARY_TASKS})")
    p.add_argument("--max-per-task", type=int, default=None, metavar="N",
                   help="Cap rows per task — useful for a quick smoke test")
    args = p.parse_args()

    print(f"Loading IMHI binary tasks from HuggingFace (Tianlin668/IMHI):")
    all_rows = []
    for task in args.tasks:
        all_rows.extend(load_task(task, args.max_per_task))

    if not all_rows:
        sys.exit("No rows loaded.")

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)

    true_n = sum(1 for r in all_rows if r["is_mental_health_handcoded"] == "TRUE")
    print(f"\nWrote {len(all_rows)} rows → {out}")
    print(f"Label balance: TRUE={true_n} ({true_n/len(all_rows):.0%}), "
          f"FALSE={len(all_rows)-true_n} ({(len(all_rows)-true_n)/len(all_rows):.0%})")
    print(f"\nTo evaluate:")
    print(f"  python classify.py <data.parquet> --stage classify \\")
    print(f"      --evaluate {out} --model-label <your_model_label> --backend vllm")