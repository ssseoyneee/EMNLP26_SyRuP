"""Mean ± std over judge runs for Multifaceted-Bench.

Score of one prediction file under one judge run
    = mean over items of (mean over that item's rubric scores); unparsable scores are skipped.
For each method (prediction file name) and benchmark we report mean ± sample std over the
judge seeds, and for "Average" we first average the benchmarks within each seed and then
take mean ± std over seeds.

    python evaluation/aggregate.py --pred "runs/qwen2-5-1-5b-instruct/rm/*/generate/mf-*/syrup_*.json" \
        --out results_mf.csv
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import DEFAULT_SEEDS, expand, judgement_path  # noqa: E402

BENCH_ORDER = ["mf-AlpacaEval", "mf-FLASK", "mf-Koala", "mf-MT-Bench", "mf-Self-Instruct"]


def file_score(path: Path):
    data = json.load(open(path, encoding="utf-8"))
    judgements = data["judgements"] if isinstance(data, dict) else data
    per_item = []
    for j in judgements:
        scores = j["score"] if isinstance(j["score"], list) else [j["score"]]
        scores = [s for s in scores if s is not None]
        if scores:
            per_item.append(statistics.fmean(scores))
    return statistics.fmean(per_item) if per_item else None


def mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main(args):
    # scores[method][bench][seed]
    scores = defaultdict(lambda: defaultdict(dict))
    missing = []
    for pred in expand(args.pred):
        method = f"{pred.parents[2].name}/{pred.stem}" if args.by_rm else pred.stem
        bench = pred.parent.name
        for seed in args.seeds:
            jp = judgement_path(pred, args.judge_model, seed, args.out_root)
            if not jp.exists():
                missing.append(str(jp))
                continue
            s = file_score(jp)
            if s is not None:
                scores[method][bench][seed] = s

    benches = [b for b in BENCH_ORDER if any(b in v for v in scores.values())]
    benches += sorted({b for v in scores.values() for b in v} - set(benches))
    header = ["method"] + benches + ["Average", "n_runs"]
    rows = []
    for method in sorted(scores):
        row = {"method": method}
        for b in benches:
            m, s = mean_std(list(scores[method][b].values()))
            row[b] = "" if m is None else f"{m:.4f} ± {s:.4f}"
        complete = [seed for seed in args.seeds if all(seed in scores[method][b] for b in benches)]
        per_seed_avg = [statistics.fmean(scores[method][b][seed] for b in benches) for seed in complete]
        m, s = mean_std(per_seed_avg)
        row["Average"] = "" if m is None else f"{m:.4f} ± {s:.4f}"
        row["n_runs"] = len(complete)
        rows.append(row)

    w = csv.DictWriter(sys.stdout, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {args.out}", file=sys.stderr)
    if missing:
        print(f"\n[warn] {len(missing)} judgement files missing, e.g. {missing[0]}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--out_root", default="")
    ap.add_argument("--by_rm", action="store_true", help="prefix each method with its reward-head run tag")
    ap.add_argument("--out", default="")
    main(ap.parse_args())
