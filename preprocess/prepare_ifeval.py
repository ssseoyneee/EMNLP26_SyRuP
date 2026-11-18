"""IFEval with the verifiable instructions moved into the system prompt.

The split file (not included in this repository) holds the 539 IFEval prompts used in the
paper: for each, `system` carries the formatting constraints, `prompt` the task without them,
and `key` / `original_prompt` / `instruction_id_list` / `kwargs` are the original IFEval
fields used by the official verifier. This script only adds the `id` field SyRuP expects.

    python preprocess/prepare_ifeval.py --split_file /path/to/ifeval_system_split.json
        -> data/benchmarks/ifeval.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True, help="JSON list of IFEval prompts with a system/prompt split")
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmarks" / "ifeval.json"))
    args = ap.parse_args()

    rows = json.load(open(args.split_file, encoding="utf-8"))
    out = [{"id": f"ifeval-{r['key']}", "benchmark": "ifeval", **r} for r in rows]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{path}: {len(out)} items")


if __name__ == "__main__":
    main()
