"""SysBench turn-level system-message following (LLM judge), ground-truth-history protocol.

For every turn the judge sees (system prompt, ground-truth earlier turns, current user turn,
the model's response) plus that turn's constraint checklist, and returns 是/否 per
constraint, using the official repository's Chinese prompt. Invalid replies (not JSON, wrong
constraint ids, values other than 是/否) are re-asked up to --max_retries times, then the
run fails.

Metrics, as in the official utils.py analysis:
  ISR  fraction of turns where every constraint passes (overall / by alignment / by
       multi-turn relation / by round)
  CSR  fraction of constraints satisfied (overall / by constraint type)
  SSR  mean number of consecutive passing turns from turn 1, per dialogue
       (only meaningful when all 5 turns of a dialogue are generated)

    python evaluation/sysbench.py --pred "runs/*/rm/*/generate/sysbench/syrup_*_i0-250.json"

Judge gpt-4o-mini at temperature 0 (the official verifier is gpt-4o-2024-05-13, so numbers are
not directly comparable with the SysBench paper). The judge is not deterministic even at
temperature 0, so each file is judged once per seed (default 42 43 44) and mean ± std is
reported. Judgements: <prediction dir>/eval/sysbench-<judge>/seed<seed>/<file name>.jsonl,
metrics alongside as <file name>.metrics.json.
"""
import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import expand  # noqa: E402
from llm import LLM  # noqa: E402

CRITERIA_TYPES = ["内容约束", "动作约束", "风格约束", "格式约束", "角色约束", "背景约束"]

JUDGE_PROMPT_TEMPLATE = """# 背景与目标
你现在是一个大模型结果评估专家。以下，你将会面对一个大模型system prompt遵循能力评估任务。
我将会给出你对应的system prompt、历史对话、当前轮次的问题、和当前轮次的回答。
需要你准确判断当前轮次的回答是否合格。
为了让你的评判足够准确，我还将会给出评判细则，需要你准确判断并告知评判细则当中每一条约束的遵循情况。

# 对话信息
## system prompt
<system prompt>
{system_prompt}
</system prompt>

## 历史对话轮次
{history}

## 当前待评估的对话轮次
<role:>user</role>
<content>{user_turn}</content>
<role:>assistant</role>
<content>{response}</content>

# 评判细则
<评判细则>
{checklist}
</评判细则>

请你认真阅读上述system prompt设定与历史对话轮次，并严格以评判细则为评判标准，针对评判细则当中的逐条要求，判断当前对话轮次的回答是否遵循。
请只输出合法JSON，包含两个字段：评判理由、评判结果。评判结果是一个对象，key为评判细则序号字符串，value只能为“是”或“否”。

{{
  "评判理由": "……",
  "评判结果": {{"1": "是", "2": "否"}}
}}"""
RETRY_SUFFIX = "\n\n重要：上次输出格式无效。评判结果必须且只能包含这些序号：{ids}。不得遗漏任何一项。"


def clean_criteria(criteria):
    return {str(k): v for k, v in (criteria or {}).items() if isinstance(v, dict)}


def format_history(history):
    if not history:
        return "（无历史对话）"
    lines = []
    for i in range(0, len(history), 2):
        lines.append(f"<round-{i // 2 + 1}>")
        for m in history[i:i + 2]:
            lines.append(f"<role:>{m.get('role', '')}</role>")
            lines.append(f"<content>{m.get('content', '')}</content>")
        lines.append(f"</round-{i // 2 + 1}>")
    return "\n".join(lines)


def judge_messages(rec, retry=False):
    criteria = rec["criteria"]
    checklist = "\n".join(f"{cid}. {criteria[cid]['criteria_content']} | {criteria[cid]['criteria_type']}"
                          for cid in sorted(criteria, key=int))
    content = JUDGE_PROMPT_TEMPLATE.format(
        system_prompt=rec.get("system", ""), history=format_history(rec.get("history") or []),
        user_turn=rec.get("prompt", ""), response=rec.get("response", ""), checklist=checklist)
    if retry:
        content += RETRY_SUFFIX.format(ids=json.dumps(sorted(criteria, key=int), ensure_ascii=False))
    return [{"role": "user", "content": content}]


def parse(rec, reply):
    if not isinstance(reply, dict) or not isinstance(reply.get("评判结果"), dict):
        raise ValueError(f"invalid reply: {reply!r}"[:300])
    got = {str(k): v for k, v in reply["评判结果"].items()}
    if set(got) != set(rec["criteria"]):
        raise ValueError(f"criterion ids {sorted(got)} != {sorted(rec['criteria'])}")
    if any(v not in ("是", "否") for v in got.values()):
        raise ValueError(f"value other than 是/否: {got}")
    return reply.get("评判理由", ""), {cid: got[cid] == "是" for cid in rec["criteria"]}


async def judge_file(llm, recs, seed, batch_size, max_retries):
    params = dict(json_mode=True, temperature=0, seed=seed)
    replies = await llm.many([judge_messages(r) for r in recs], batch_size, desc=f"seed {seed}", **params)
    parsed, pending, errors = [None] * len(recs), list(range(len(recs))), {}
    for attempt in range(max_retries + 1):
        still = []
        for i in pending:
            try:
                parsed[i] = parse(recs[i], replies[i])
            except ValueError as exc:
                errors[i] = str(exc)
                still.append(i)
        if not still:
            break
        if attempt == max_retries:
            raise RuntimeError(f"{len(still)} invalid judge replies after {max_retries} retries, e.g. "
                               f"{recs[still[0]]['id']}: {errors[still[0]]}")
        print(f"[sysbench] retry {attempt + 1}/{max_retries}: {len(still)} invalid replies")
        redo = await llm.many([judge_messages(recs[i], retry=True) for i in still], batch_size, desc="retry", **params)
        for i, reply in zip(still, redo):
            replies[i] = reply
        pending = still
    return [{"id": r["id"], "system_id": r["system_id"], "round_index": r["round_index"],
             "alignment": r["alignment"], "rounds_related": r["rounds_related"], "criteria": r["criteria"],
             "judge_reasoning": reason, "judge_results": res, "turn_pass": all(res.values()) if res else True}
            for r, (reason, res) in zip(recs, parsed)]


def metrics(judged):
    rate = lambda v: sum(v) / len(v) if v else float("nan")
    isr_all, by_align, by_rel, by_round = [], defaultdict(list), defaultdict(list), defaultdict(list)
    csr_n, csr_ok = defaultdict(int), defaultdict(int)
    dialogs, relation = defaultdict(dict), {}
    for r in judged:
        rel = "related" if r.get("rounds_related") else "parallel"
        isr_all.append(r["turn_pass"])
        by_align[r.get("alignment", "unknown")].append(r["turn_pass"])
        by_rel[rel].append(r["turn_pass"])
        by_round[str(r["round_index"])].append(r["turn_pass"])
        for cid, c in r["criteria"].items():
            csr_n[c["criteria_type"]] += 1
            csr_ok[c["criteria_type"]] += bool(r["judge_results"].get(cid))
        dialogs[r["system_id"]][r["round_index"]] = r["turn_pass"]
        relation[r["system_id"]] = rel
    ssr, ssr_rel = [], defaultdict(list)
    for sid, rounds in dialogs.items():
        streak = 0
        for k in sorted(rounds):
            if not rounds[k]:
                break
            streak += 1
        ssr.append(streak)
        ssr_rel[relation[sid]].append(streak)
    return {
        "n_turns": len(judged), "n_dialogues": len(dialogs),
        "complete_dialogues": all(set(r) == set(range(1, 6)) for r in dialogs.values()),
        "ISR_overall": rate(isr_all),
        "ISR_by_alignment": {k: rate(v) for k, v in by_align.items()},
        "ISR_by_relation": {k: rate(v) for k, v in by_rel.items()},
        "ISR_by_round": {k: rate(v) for k, v in sorted(by_round.items(), key=lambda x: int(x[0]))},
        "CSR_overall": sum(csr_ok.values()) / sum(csr_n.values()) if csr_n else float("nan"),
        "CSR_by_type": {t: csr_ok[t] / csr_n[t] for t in CRITERIA_TYPES if csr_n[t]},
        "SSR_avg_consecutive_turns": rate(ssr),
        "SSR_by_relation": {k: rate(v) for k, v in ssr_rel.items()},
    }


async def run(args):
    llm = None
    keys = ["ISR_overall", "CSR_overall", "SSR_avg_consecutive_turns"]
    print("file,n_turns,runs," + ",".join(f"{k}_mean,{k}_std" for k in keys))
    for pred in expand(args.pred):
        per_run = []
        for seed in args.seeds:
            base = pred.parent / "eval" / f"sysbench-{args.judge_model.replace('/', '_')}" / f"seed{seed}"
            out = base / f"{pred.stem}.jsonl"
            if out.exists() and not args.overwrite:
                judged = [json.loads(line) for line in open(out, encoding="utf-8") if line.strip()]
            else:
                recs = json.load(open(pred, encoding="utf-8"))
                for r in recs:
                    r["criteria"] = clean_criteria(r.get("criteria"))
                    if not r["criteria"]:
                        raise ValueError(f"{pred}: {r.get('id')} has no criteria (generate from data/benchmarks/sysbench.json)")
                llm = llm or LLM(args.judge_model)
                judged = await judge_file(llm, recs, seed, args.batch_size, args.max_retries)
                base.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    for j in judged:
                        f.write(json.dumps(j, ensure_ascii=False) + "\n")
            m = metrics(judged)
            json.dump(m, open(base / f"{pred.stem}.metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            per_run.append(m)
        if not per_run[0]["complete_dialogues"]:
            print(f"[warn] {pred.name}: some dialogues lack turns; SSR is only a partial-session diagnostic",
                  file=sys.stderr)
        cells = []
        for k in keys:
            vals = [m[k] for m in per_run]
            cells += [f"{statistics.fmean(vals):.4f}", f"{statistics.stdev(vals) if len(vals) > 1 else 0.0:.4f}"]
        print(f"{pred},{per_run[0]['n_turns']},{len(per_run)}," + ",".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="one judge run per seed")
    ap.add_argument("--batch_size", type=int, default=20, help="concurrent requests per batch")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    asyncio.run(run(ap.parse_args()))
