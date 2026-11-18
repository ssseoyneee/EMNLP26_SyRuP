"""Rule-verifier benchmarks with the official IFEval verifier:

  ifeval      IFEval with the constraints moved into the system prompt
  iheval-*    IHEval rule-following (single/multi-turn, aligned/conflict)

Each response is checked against the item's instruction_id_list / kwargs, i.e. the
constraints the model only saw in the system prompt. Deterministic, so one run suffices.

    python evaluation/ifeval.py --ifeval_dir /path/to/google-research --pred "runs/*/rm/*/generate/ifeval/syrup_*.json"
    python evaluation/ifeval.py --ifeval_dir /path/to/google-research --pred "runs/*/rm/*/generate/iheval-*/syrup_*.json"

The verifier is not included in this repository. --ifeval_dir (or $IFEVAL_DIR) is a checkout
of github.com/google-research/google-research that contains instruction_following_eval/.

Writes <prediction dir>/eval/ifeval/<stem>/eval_results_{strict,loose}.jsonl and
*_summary.json, and prints prompt-/instruction-level accuracy (strict and loose).
"""
import argparse
import collections
import importlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from langdetect import DetectorFactory  # noqa: E402
from judge import expand  # noqa: E402

ROOT = HERE.parent
evaluation_lib = None


def load_verifier(ifeval_dir: str):
    global evaluation_lib
    root = Path(ifeval_dir).expanduser().resolve()
    if not (root / "instruction_following_eval" / "evaluation_lib.py").exists():
        raise FileNotFoundError(
            f"{root}/instruction_following_eval/evaluation_lib.py not found; pass --ifeval_dir pointing to a "
            "checkout of github.com/google-research/google-research (see README)")
    sys.path.insert(0, str(root))
    evaluation_lib = importlib.import_module("instruction_following_eval.evaluation_lib")


def build_report(outputs) -> dict:
    """Prompt-/instruction-level accuracy, plus per-instruction accuracy by group (tier0) and id (tier1)."""
    n_prompt = n_prompt_ok = n_inst = n_inst_ok = 0
    tiers = {"tier0": (collections.Counter(), collections.Counter()),
             "tier1": (collections.Counter(), collections.Counter())}
    for ex in outputs:
        n_prompt += 1
        n_prompt_ok += all(ex.follow_instruction_list)
        n_inst += len(ex.instruction_id_list)
        n_inst_ok += sum(ex.follow_instruction_list)
        for inst_id, ok in zip(ex.instruction_id_list, ex.follow_instruction_list):
            for tier, key in (("tier0", inst_id.split(":")[0]), ("tier1", inst_id)):
                tiers[tier][0][key] += 1
                tiers[tier][1][key] += bool(ok)
    report = {
        "prompt_level_accuracy": n_prompt_ok / n_prompt,
        "instruction_level_accuracy": n_inst_ok / n_inst,
        "prompt_total": n_prompt,
        "prompt_correct": n_prompt_ok,
        "instruction_total": n_inst,
        "instruction_correct": n_inst_ok,
    }
    for tier, (total, correct) in tiers.items():
        report[tier] = {k: correct[k] / total[k] for k in sorted(total)}
    return report


def clean_kwargs(kwargs_list):
    """The verifier calls build_description(**kwargs); drop the None placeholders."""
    out = []
    for kw in kwargs_list:
        d = {}
        for k, v in (kw or {}).items():
            if v is None:
                continue
            d[k] = int(v) if isinstance(v, float) and v.is_integer() else v
        out.append(d)
    return out


def evaluate(pred_path: Path, out_dir: Path):
    bench_file = ROOT / "data" / "benchmarks" / f"{pred_path.parent.name}.json"
    source = {}
    if bench_file.exists():
        source = {r["id"]: r for r in json.load(open(bench_file, encoding="utf-8"))}
    preds = json.load(open(pred_path, encoding="utf-8"))
    inputs, responses = [], {}
    for p in preds:
        src = source.get(p["id"], {})
        prompt = p.get("original_prompt") or src.get("original_prompt") or p.get("prompt")
        if prompt in responses:
            raise ValueError(f"{pred_path}: duplicate verifier key for id={p['id']}")
        inputs.append(evaluation_lib.InputExample(
            key=p.get("key", src.get("key")),
            instruction_id_list=p.get("instruction_id_list", src.get("instruction_id_list")),
            prompt=prompt,
            kwargs=clean_kwargs(p.get("kwargs", src.get("kwargs"))),
        ))
        responses[prompt] = p["response"]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, fn in [("strict", evaluation_lib.test_instruction_following_strict),
                     ("loose", evaluation_lib.test_instruction_following_loose)]:
        # the verifier uses `random` and langdetect internally
        random.seed(0)
        DetectorFactory.seed = 0
        outputs = [fn(inp, responses) for inp in inputs]
        evaluation_lib.write_outputs(str(out_dir / f"eval_results_{name}.jsonl"), outputs)
        report = build_report(outputs)
        with open(out_dir / f"eval_results_{name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        summary[name] = {k: report[k] for k in ("prompt_level_accuracy", "instruction_level_accuracy")}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--ifeval_dir", default=os.environ.get("IFEVAL_DIR", ""),
                    help="google-research checkout containing instruction_following_eval/ (default: $IFEVAL_DIR)")
    args = ap.parse_args()
    if not args.ifeval_dir:
        ap.error("--ifeval_dir (or $IFEVAL_DIR) is required; see README")
    load_verifier(args.ifeval_dir)
    print("file,strict_prompt,strict_instruction,loose_prompt,loose_instruction")
    for pred in expand(args.pred):
        s = evaluate(pred, pred.parent / "eval" / "ifeval" / pred.stem)
        print(f"{pred},{s['strict']['prompt_level_accuracy']:.4f},{s['strict']['instruction_level_accuracy']:.4f},"
              f"{s['loose']['prompt_level_accuracy']:.4f},{s['loose']['instruction_level_accuracy']:.4f}")


if __name__ == "__main__":
    main()
