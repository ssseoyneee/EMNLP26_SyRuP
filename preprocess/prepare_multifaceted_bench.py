"""Download kaist-ai/Multifaceted-Bench and write one SyRuP benchmark file per group:

    data/benchmarks/mf-{AlpacaEval,FLASK,Koala,MT-Bench,Self-Instruct}.json

Each record's `id` is a hash of source + system + prompt, so it stays the same across runs.

    python preprocess/prepare_multifaceted_bench.py
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
GROUP_OF_SOURCE = {"alpaca": "mf-AlpacaEval", "koala": "mf-Koala", "mt-bench": "mf-MT-Bench",
                   "self-instruct": "mf-Self-Instruct"}   # every other source -> mf-FLASK


def item_key(source: str, system: str, prompt: str) -> str:
    return hashlib.sha1(f"{source}\n{system}\n{prompt}".encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "benchmarks"))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ds = load_dataset("kaist-ai/Multifaceted-Bench", split="train", token=token)

    groups = defaultdict(list)
    for ex in ds:
        groups[GROUP_OF_SOURCE.get(ex["source"], "mf-FLASK")].append({
            "id": item_key(ex["source"], ex["system"], ex["prompt"]),
            "source": ex["source"],
            "benchmark": GROUP_OF_SOURCE.get(ex["source"], "mf-FLASK"),
            "system": ex["system"],
            "prompt": ex["prompt"],
            "reference_answer": ex["reference_answer"],
            "rubrics": ex["rubrics"],
        })

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in sorted(groups.items()):
        path = Path(args.out_dir) / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"{path}: {len(rows)} items")


if __name__ == "__main__":
    main()
