"""Training pairs (Multifaceted-Collection-DPO) and benchmark records.

Evaluation records are plain JSON lists produced by `preprocess/` with at least
    {"id", "system", "prompt"}
and optionally
    "messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]  (multi-turn input,
                ending with the user turn to answer; used instead of [system, prompt])
Every other field is passed through to the prediction file untouched.
"""
import json
from typing import List, Optional

import torch
from datasets import load_dataset

from .modeling import (
    apply_chat_template,
    normalize_messages,
    perturbed_prompt_text,
    prompt_text,
    system_only_text,
    tokenize,
    user_only_text,
    hf_token,
)

TRAIN_DATASET = "kaist-ai/Multifaceted-Collection-DPO"


def _assistant_text(dialogue) -> str:
    if isinstance(dialogue, list):
        for turn in reversed(dialogue):
            if turn.get("role") == "assistant":
                return turn.get("content", "").strip()
    if isinstance(dialogue, str):
        return dialogue.strip()
    return ""


def _pair_texts(tokenizer, ex):
    prefix = [{"role": "system", "content": ex["system"].strip()},
              {"role": "user", "content": ex["prompt"].strip()}]
    prompt = apply_chat_template(tokenizer, prefix, add_generation_prompt=True)
    chosen = apply_chat_template(tokenizer, prefix + [{"role": "assistant", "content": _assistant_text(ex["chosen"])}])
    rejected = apply_chat_template(tokenizer, prefix + [{"role": "assistant", "content": _assistant_text(ex["rejected"])}])
    return prompt, chosen, rejected


def _load_pairs(dataset_name: str, split: str):
    try:
        return load_dataset(dataset_name, split=split, token=hf_token())
    except Exception as exc:  # some `datasets` versions fail on this repo's list features
        if dataset_name != TRAIN_DATASET or split != "train":
            raise
        print(f"[data] load_dataset failed ({exc}); reading the parquet shards directly")
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        rows = []
        for i in range(2):
            path = hf_hub_download(TRAIN_DATASET, f"data/train-0000{i}-of-00002.parquet",
                                   repo_type="dataset", token=hf_token())
            rows.extend(pq.read_table(path).to_pylist())
        return rows


class PreferencePairs(torch.utils.data.Dataset):
    """chosen/rejected responses for the same (system, prompt). Pairs whose prompt already
    fills `max_length` (nothing left of a response after truncation) are dropped."""

    def __init__(self, tokenizer, dataset_name: str = TRAIN_DATASET, split: str = "train",
                 max_length: int = 1024, sources: Optional[set] = None, max_pairs: int = 0):
        raw = _load_pairs(dataset_name, split)
        if sources:
            keep = lambda ex: ex.get("original_source", ex.get("source", "")) in sources
            raw = raw.filter(keep) if hasattr(raw, "filter") else [ex for ex in raw if keep(ex)]
        if max_pairs:
            raw = raw.select(range(min(max_pairs, len(raw)))) if hasattr(raw, "select") else raw[:max_pairs]
        self.ds, self.tok, self.max_length = raw, tokenizer, max_length

        def n_tok(text):
            return len(tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"])

        self.index = []
        for i in range(len(raw)):
            p, c, r = _pair_texts(tokenizer, raw[i])
            n_p, n_c, n_r = n_tok(p), n_tok(c), n_tok(r)
            pl = min(n_p, n_c, n_r)
            if pl < n_c and pl < n_r:
                self.index.append(i)
        print(f"[data] {len(self.index):,}/{len(raw):,} pairs kept (prompt fits in {max_length} tokens)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ex = self.ds[self.index[idx]]
        p, c, r = _pair_texts(self.tok, ex)
        prompt_ids = tokenize(self.tok, p, self.max_length)
        chosen_ids = tokenize(self.tok, c, self.max_length)
        rejected_ids = tokenize(self.tok, r, self.max_length)
        return {
            "system_ids": tokenize(self.tok, system_only_text(self.tok, ex["system"]), self.max_length),
            "user_ids": tokenize(self.tok, user_only_text(self.tok, ex["prompt"]), self.max_length),
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "prefix_len": torch.tensor(min(len(prompt_ids), len(chosen_ids), len(rejected_ids))),
        }


def _pad(seqs: List[torch.Tensor], pad_id: int):
    n = max(x.size(0) for x in seqs)
    out = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, : x.size(0)] = x
        mask[i, : x.size(0)] = 1
    return out, mask


def collate_pairs(batch, pad_id: int):
    out = {}
    for key in ("system", "user", "chosen", "rejected"):
        out[f"{key}_ids"], out[f"{key}_mask"] = _pad([b[f"{key}_ids"] for b in batch], pad_id)
    out["prefix_lens"] = torch.stack([b["prefix_len"] for b in batch])
    return out


def load_records(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    for i, r in enumerate(records):
        if "id" not in r or "system" not in r or ("prompt" not in r and "messages" not in r):
            raise ValueError(f"{path}: record {i} needs 'id', 'system' and 'prompt' (or 'messages')")
    return records


def encode_record(tokenizer, rec: dict, perturb_mode: str, max_length: int) -> dict:
    """Token ids of the four inputs SyRuP needs for one record.

    Optional record fields set by the converters in preprocess/:
      reward_system   text given to the reward head as the system prompt (default: `system`)
      reward_inputs   "chat" (default): system/user texts for the reward head are rendered with
                      the chat template; "raw": plain stripped text (how the paper ran IHEval)
    """
    messages = normalize_messages(rec.get("messages"))
    if messages:
        user = rec.get("prompt") or next(m["content"] for m in reversed(messages) if m["role"] == "user")
        prefix = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    else:
        user = rec["prompt"]
        prefix = prompt_text(tokenizer, rec["system"], user)
    neutral = perturbed_prompt_text(tokenizer, user, perturb_mode, messages=messages or None)

    reward_system = rec.get("reward_system", rec["system"])
    if rec.get("reward_inputs", "chat") == "raw":
        system_text, user_text = reward_system.strip(), user.strip()
    else:
        system_text, user_text = system_only_text(tokenizer, reward_system), user_only_text(tokenizer, user)
    return {
        "prompt_ids": tokenize(tokenizer, prefix, max_length).unsqueeze(0),
        "neutral_ids": tokenize(tokenizer, neutral, max_length).unsqueeze(0),
        "system_ids": tokenize(tokenizer, system_text, max_length).unsqueeze(0),
        "user_ids": tokenize(tokenizer, user_text, max_length).unsqueeze(0),
    }
