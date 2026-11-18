"""RealGuardrails (huggingface.co/datasets/normster/RealGuardrails) test suites
-> data/benchmarks/realguardrails-{handwritten,distractors}.json

    python preprocess/prepare_realguardrails.py

handwritten: 239 single-turn cases; distractors: 504 cases with in-context distractor turns
before the final user message. The model sees the full conversation (`messages`); the
reward head sees the system prompt and the final user turn. Scored by evaluation/realguardrails.py.
"""
import gzip
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ["handwritten", "distractors"]
NEUTRAL = "You are a helpful assistant."


def normalize(messages):
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("content", "") for x in content if isinstance(x, dict) and x.get("content"))
        content = "" if content is None else str(content)
        if role and content:
            out.append({"role": role, "content": content})
    if out and out[-1]["role"] == "assistant":
        out = out[:-1]
    return out


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        path = hf_hub_download("normster/RealGuardrails", f"{subset}.jsonl.gz", repo_type="dataset", token=token)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        rows = []
        for i, ex in enumerate(raw):
            msgs = normalize(ex.get("messages"))
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
            rec = {
                "id": f"rg-{subset}-{i}",
                "source_id": ex.get("id"),
                "benchmark": f"realguardrails-{subset}",
                "system": system,
                "prompt": user,
                "messages": msgs,
                "guardrails": ex.get("guardrails") or [],
                "banlist": ex.get("banlist") or [],
            }
            for k in ("num_turns", "distractor_task", "is_multiturn"):
                if k in ex:
                    rec[k] = ex[k]
            if not system.strip():
                rec["reward_system"] = NEUTRAL  # the reward head needs a non-empty memory
            rows.append(rec)
        out = out_dir / f"realguardrails-{subset}.json"
        json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{out}: {len(rows)} items")


if __name__ == "__main__":
    main()
