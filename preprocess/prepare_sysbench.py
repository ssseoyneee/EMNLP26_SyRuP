"""SysBench (github.com/PKU-Baichuan-MLSystemLab/SysBench) -> data/benchmarks/sysbench.json

    git clone https://github.com/PKU-Baichuan-MLSystemLab/SysBench /path/to/SysBench
    python preprocess/prepare_sysbench.py --repo /path/to/SysBench

Ground-truth-history protocol: each of the 500 five-turn dialogues becomes 5 records (2500
in total, ordered dialogue by dialogue). Turn t is answered given the system prompt and the
dataset's own ground-truth turns 1..t-1 (the official default instead feeds back the model's
own earlier answers). The paper used the first 100 or 250 records (20 / 50 dialogues), i.e.
`--end_idx 100` / `--end_idx 250` at generation time. Scored by evaluation/sysbench.py.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="clone of github.com/PKU-Baichuan-MLSystemLab/SysBench")
    args = ap.parse_args()
    dialogues = json.load(open(Path(args.repo) / "datas" / "system_benchmark_eval_datas.json", encoding="utf-8"))

    rows = []
    for d in dialogues:
        msgs = d["messages"]                       # [system, u1, a1, ..., u5, a5]
        for r in range(5):
            u = 1 + 2 * r
            user, reference = msgs[u]["content"], msgs[u + 1]["content"]
            history = [{"role": m["role"], "content": m["content"]} for m in msgs[1:u]]
            # prompt_infos is keyed by the user text, as in the official evaluator
            info = d["prompt_infos"][user]
            rows.append({
                "id": f"sysbench_{d['system_id']}_t{r + 1}",
                "benchmark": "sysbench",
                "system": d["system_prompt"],
                "prompt": user,
                "history": history,
                "messages": [{"role": "system", "content": d["system_prompt"].strip()}] + history
                            + [{"role": "user", "content": user.strip()}],
                "reference": reference,
                "system_id": d["system_id"],
                "round_index": r + 1,
                "alignment": info["alignment"],
                "criteria": info["criteria"],
                "rounds_related": d["rounds_related"],
                "domain": d.get("领域", ""),
                "scenario": d.get("场景", ""),
            })
    out = ROOT / "data" / "benchmarks" / "sysbench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(rows)} turns from {len(dialogues)} dialogues")


if __name__ == "__main__":
    main()
