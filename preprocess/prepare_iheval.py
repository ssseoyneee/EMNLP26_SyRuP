"""IHEval rule-following (github.com/ytyz1307zzh/IHEval) -> data/benchmarks/iheval-<variant>.json

    git clone https://github.com/ytyz1307zzh/IHEval /path/to/IHEval
    python preprocess/prepare_iheval.py --repo /path/to/IHEval

Variants (default system prompt strength, as in the paper):
    single_aligned   rule-following/single-turn/aligned/default
    single_conflict  rule-following/single-turn/conflict/default
    multi_aligned    rule-following/multi-turn/aligned/default_system_prompt
    multi_conflict   rule-following/multi-turn/conflict/both-turn-conflict-default-system-prompt

Scored with the IFEval verifier (evaluation/ifeval.py) using each item's `answer`
(instruction_id_list, kwargs).

Paper protocol (reproduced by default):
  * the model sees only [system, instruction]. For the multi-turn variants the
    `conversation_history` turns are NOT given to the model. Pass --with_history to
    feed them as earlier user/assistant turns instead.
  * the reward head reads the plain system / instruction text, not chat-template
    rendered (record field reward_inputs="raw").
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {
    "single_aligned": "rule-following/single-turn/aligned/default",
    "single_conflict": "rule-following/single-turn/conflict/default",
    "multi_aligned": "rule-following/multi-turn/aligned/default_system_prompt",
    "multi_conflict": "rule-following/multi-turn/conflict/both-turn-conflict-default-system-prompt",
}


def _maybe_literal(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def convert(repo: Path, variant: str, with_history: bool):
    rows = json.load(open(repo / "benchmark" / VARIANTS[variant] / "input_data.json", encoding="utf-8"))
    out = []
    for r in rows:
        answer = _maybe_literal(r["answer"])
        rec = {
            "id": f"iheval-{variant}-{r['id']}",
            "key": r["id"],
            "benchmark": f"iheval-{variant}",
            "system": r["system"],
            "prompt": r["instruction"],
            # key the verifier matches responses by; the verifier does not read its content
            "original_prompt": r.get("original_prompt") or r["instruction"],
            "instruction_id_list": answer["instruction_id_list"],
            "kwargs": answer["kwargs"],
            "reward_inputs": "raw",
        }
        history = _maybe_literal(r.get("conversation_history")) or []
        if history:
            rec["conversation_history"] = history
            if with_history:
                msgs = [{"role": "system", "content": r["system"]}]
                for i, turn in enumerate(history):
                    msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": turn})
                msgs.append({"role": "user", "content": r["instruction"]})
                rec["messages"] = msgs
        out.append(rec)
    if len({r["original_prompt"] for r in out}) != len(out):
        for r in out:
            r["original_prompt"] = f"[{r['id']}] {r['original_prompt']}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="clone of github.com/ytyz1307zzh/IHEval")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--with_history", action="store_true",
                    help="give multi-turn variants their conversation history (not the paper protocol)")
    args = ap.parse_args()
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        rows = convert(Path(args.repo), v, args.with_history)
        name = f"iheval-{v}" + ("-hist" if args.with_history and v.startswith("multi") else "")
        path = out_dir / f"{name}.json"
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{path}: {len(rows)} items")


if __name__ == "__main__":
    main()
