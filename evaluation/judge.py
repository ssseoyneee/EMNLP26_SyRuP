"""Multifaceted-Bench: rubric-based absolute grading (1-5) by an LLM judge.

Every rubric of every item is graded separately. Decoding is greedy, so the only source of
variance is the judge (temperature 1.0, top_p 0.9). The paper therefore grades each
prediction file once per judge seed (42, 43, 44) and reports mean ± std over the three runs
(see aggregate.py).

    export OPENAI_API_KEY=...
    python evaluation/judge.py --pred "runs/*/rm/*/generate/mf-*/syrup_*.json"

Judgements go to <prediction dir>/eval/judge-<model>/seed<seed>/<prediction file name>;
files that already exist are skipped, so the command can simply be re-run after an
interruption. Judge models with a "/" in the name (e.g. deepseek/deepseek-chat-v3-0324) are
called through OpenRouter (OPENROUTER_API_KEY).
"""
import argparse
import asyncio
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLM  # noqa: E402

JUDGE_SYSTEM = (
    "You are a fair judge assistant tasked with providing clear, objective feedback based on specific "
    "criteria, ensuring each assessment reflects the absolute standards set for performance."
)
JUDGE_TEMPLATE = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback: """
SAMPLING = {"max_tokens": 1024, "temperature": 1.0, "top_p": 0.9}
DEFAULT_SEEDS = (42, 43, 44)


_PATTERNS = [
    r"^(?P<feedback>.*?)\s*\[RESULT\]\s*(?P<score>[1-5])\s*$",
    r"^(?P<feedback>.*?)\s*\[(?P<score>[1-5])\]\s*$",
    r"^(?P<feedback>.*?)\s*(?:Score|Result)\s*[:=]\s*(?P<score>[1-5])\s*$",
    r"^(?P<feedback>.*?)\s*\[\s*(?:Score|Result)\s*[:=]\s*(?P<score>[1-5])\s*\]\s*$",
    r"^(?P<feedback>.*?)\((?P<score>[1-5])\)\s*$",
]
_FALLBACK = r"(?:\[RESULT\]|Score|\[SCORE\]|\[RESULT\]:|Score:|score:|Result:|\[Result\]|score of)\s*(?:\(\s*|\[\s*|)\s*(\d+)"


def parse_judgement(output):
    """-> (feedback, score) or (None, None)."""
    if not isinstance(output, str):
        return None, None
    text = output.strip()
    for pat in _PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group("feedback").strip(), int(m.group("score"))
    m = re.search(_FALLBACK, text, re.IGNORECASE | re.DOTALL)
    if m:
        return text[: m.start()].rstrip(" (["), int(m.group(1))
    return None, None


def bench_lookup():
    table = {}
    for path in sorted((ROOT / "data" / "benchmarks").glob("mf-*.json")):
        for r in json.load(open(path, encoding="utf-8")):
            table[r["id"]] = r
    return table


def build_messages(pred: dict, bench: dict):
    """One judge request per rubric of the item."""
    src = bench.get(pred["id"], {})
    get = lambda k: pred.get(k, src.get(k))
    system, prompt, reference, rubrics = get("system"), get("prompt"), get("reference_answer"), get("rubrics")
    if system is None or prompt is None or rubrics is None:
        raise ValueError(f"id={pred['id']}: system/prompt/rubrics not in the prediction nor in data/benchmarks "
                         "(run preprocess/prepare_multifaceted_bench.py)")
    rubrics = rubrics if isinstance(rubrics, list) else [rubrics]
    out = []
    for rubric in rubrics:
        rubric_str = rubric if isinstance(rubric, str) else json.dumps(rubric, indent=4, ensure_ascii=False)
        content = JUDGE_TEMPLATE.format(
            instruction=f"{system}\n{prompt}", response=pred["response"],
            reference_answer=reference or "", score_rubric=rubric_str,
        ).strip()
        out.append([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}])
    return out


class Judge:
    def __init__(self, model: str):
        self.llm = LLM(model)
        self.model = model

    async def grade(self, requests, seed, batch_size=100, max_retries=5):
        outputs = await self.llm.many(requests, batch_size, desc=f"seed {seed}", seed=seed, **SAMPLING)
        for attempt in range(1, max_retries + 1):
            bad = [i for i, o in enumerate(outputs) if parse_judgement(o)[1] is None]
            if not bad:
                break
            print(f"[judge] retry {attempt}/{max_retries}: {len(bad)} unparsable outputs")
            redo = await self.llm.many([requests[i] for i in bad], batch_size, desc="retry", seed=seed, **SAMPLING)
            for i, o in zip(bad, redo):
                outputs[i] = o
        return [parse_judgement(o) for o in outputs]


async def judge_file(judge: Judge, pred_path: Path, out_path: Path, seed: int, bench: dict, batch_size: int):
    preds = json.load(open(pred_path, encoding="utf-8"))
    requests, spans = [], []
    for p in preds:
        msgs = build_messages(p, bench)
        spans.append((len(requests), len(requests) + len(msgs)))
        requests += msgs
    results = await judge.grade(requests, seed, batch_size)
    judgements = [{
        "id": p["id"],
        "feedback": [f for f, _ in results[a:b]],
        "score": [s for _, s in results[a:b]],
    } for p, (a, b) in zip(preds, spans)]
    n_fail = sum(s is None for j in judgements for s in j["score"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"judge_model": judge.model, "seed": seed, "prediction_file": str(pred_path),
                   "sampling": SAMPLING, "judgements": judgements}, f, ensure_ascii=False, indent=1)
    tmp.replace(out_path)
    print(f"[judge] {out_path} ({len(judgements)} items, {n_fail} unparsable scores)")


def judgement_path(pred_path: Path, judge_model: str, seed: int, out_root: str = "") -> Path:
    name = "judge-" + judge_model.replace("/", "_")
    base = Path(out_root) if out_root else pred_path.parent / "eval"
    return base / name / f"seed{seed}" / pred_path.name


def expand(patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        files += [Path(h) for h in hits if h.endswith(".json") and not h.endswith(".partial.json")]
    if not files:
        raise FileNotFoundError(f"no prediction files match {patterns}")
    return files


async def main(args):
    judge = Judge(args.judge_model)
    bench = bench_lookup()
    for pred in expand(args.pred):
        for seed in args.seeds:
            out = judgement_path(pred, args.judge_model, seed, args.out_root)
            if out.exists() and not args.overwrite:
                print(f"[judge] exists, skipping: {out}")
                continue
            await judge_file(judge, pred, out, seed, bench, args.batch_size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, help="prediction JSON files or glob patterns")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="one full grading run per seed (paper: 42 43 44)")
    ap.add_argument("--batch_size", type=int, default=100, help="concurrent requests per batch")
    ap.add_argument("--out_root", default="", help="write judgements here instead of next to the predictions")
    ap.add_argument("--overwrite", action="store_true")
    asyncio.run(main(ap.parse_args()))
