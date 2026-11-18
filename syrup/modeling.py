"""Model loading, chat templates, and the SyRuP reward head.

Training and decoding must render prompts identically, so every chat-template string is
built here. The reward head reads the system prompt (keys/values) with the user turn and
the response so far as queries, and outputs a vector z. A candidate token v gets reward
r(v) = z · e_v, where e_v is the frozen input embedding of v.

Parameter names (q_proj, k_proj, v_proj, projector, layer_norm) match the checkpoints the
paper used, so `reward_model.pt` files from the original code load directly.
"""
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2-5-1-5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3-2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3-1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.strip().lower(), name.strip())


def model_tag(name: str) -> str:
    """'Qwen/Qwen2.5-1.5B-Instruct' -> 'qwen2-5-1-5b-instruct'."""
    return resolve_model_name(name).split("/")[-1].lower().replace(".", "-")


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(resolve_model_name(name), token=hf_token(), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name: str, torch_dtype=None):
    name = resolve_model_name(name)
    kwargs = {"torch_dtype": torch_dtype, "token": hf_token(), "trust_remote_code": True}
    config = AutoConfig.from_pretrained(name, token=hf_token(), trust_remote_code=True)
    if getattr(config, "model_type", "") == "gemma3" and "Gemma3ForConditionalGeneration" in (
        getattr(config, "architectures", None) or []
    ):
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration.from_pretrained(name, **kwargs)
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def hidden_size_of(model) -> int:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    return int(cfg.hidden_size)


def apply_chat_template(tokenizer, messages: List[dict], add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None) is not None:
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).lower():
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    chunks = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    if add_generation_prompt:
        chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def prompt_text(tokenizer, system: str, user: str) -> str:
    """Full generation prefix: [system, user] + generation prompt."""
    msgs = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
    return apply_chat_template(tokenizer, msgs, add_generation_prompt=True)


def system_only_text(tokenizer, system: str) -> str:
    """K/V memory of the reward head. Falls back to raw text for templates that drop
    a lone system message."""
    content = system.strip()
    out = apply_chat_template(tokenizer, [{"role": "system", "content": content}])
    if content and content not in out:
        return content
    return out


def user_only_text(tokenizer, user: str) -> str:
    return apply_chat_template(tokenizer, [{"role": "user", "content": user.strip()}])


def normalize_messages(messages) -> List[dict]:
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "") or "")
        if role and content:
            out.append({"role": role, "content": content})
    if out and out[-1]["role"] == "assistant":
        out = out[:-1]
    return out


def perturbed_prompt_text(tokenizer, user: str, perturb_mode: str, messages=None) -> str:
    """Prefix of the 'without system prompt' stream used for the contrastive term.

    perturb_mode: neutral_system (paper) | no_system | empty_system
    With `messages` (multi-turn benchmarks) only the system turn is replaced.
    """
    if perturb_mode not in ("neutral_system", "no_system", "empty_system"):
        raise ValueError(f"Unknown perturb_mode: {perturb_mode}")
    replacement = {"neutral_system": NEUTRAL_SYSTEM_PROMPT, "empty_system": ""}.get(perturb_mode)

    if not messages:
        msgs = [{"role": "user", "content": user.strip()}]
        if replacement is not None:
            msgs.insert(0, {"role": "system", "content": replacement})
        return apply_chat_template(tokenizer, msgs, add_generation_prompt=True)

    out, saw_system = [], False
    for m in normalize_messages(messages):
        if m["role"] == "system":
            saw_system = True
            if replacement is not None:
                out.append({"role": "system", "content": replacement})
        else:
            out.append(m)
    if not saw_system and replacement is not None:
        out.insert(0, {"role": "system", "content": replacement})
    return apply_chat_template(tokenizer, out, add_generation_prompt=True)


def tokenize(tokenizer, text: str, max_length: int) -> torch.Tensor:
    return tokenizer(
        text, truncation=True, max_length=max_length, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]


class RewardHead(nn.Module):
    """Single cross-attention head, query = [user tokens ; prefix tokens], pooled at the last
    valid position."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, user_h, user_mask, prefix_h, prefix_mask, system_h, system_mask):
        """All hidden states float32, shape (B, T, H). Returns z: (B, H)."""
        q_seq = self.layer_norm(torch.cat([user_h, prefix_h], dim=1))
        q_seq_mask = torch.cat([user_mask, prefix_mask], dim=1)

        q = self.q_proj(q_seq)
        k = self.k_proj(system_h)
        v = self.v_proj(system_h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill((system_mask == 0).unsqueeze(1), -1e9)
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)

        fused = self.projector(torch.cat([q_seq, ctx], dim=-1))
        last = q_seq_mask.long().sum(dim=1).clamp(min=1) - 1
        return fused[torch.arange(fused.size(0), device=fused.device), last, :]


class SyRuPModel(nn.Module):
    """Frozen base LM + trainable reward head."""

    def __init__(self, base_model_name: str, torch_dtype=None):
        super().__init__()
        self.base = load_base_model(base_model_name, torch_dtype=torch_dtype)
        self.base.config.output_hidden_states = True
        self.base.config.use_cache = True
        for p in self.base.parameters():
            p.requires_grad = False
        self.reward_head = RewardHead(hidden_size_of(self.base))
        self._logits_to_keep_ok = None

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    @torch.no_grad()
    def encode(self, input_ids, attention_mask, use_cache=False, past_key_values=None, last_logits_only=False):
        """Returns (last-layer hidden states, logits, past_key_values or None)."""
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if last_logits_only and self._logits_to_keep_ok is not False:
            try:
                out = self.base(**kwargs, logits_to_keep=1)
                self._logits_to_keep_ok = True
            except TypeError:
                self._logits_to_keep_ok = False
                out = self.base(**kwargs)
        else:
            out = self.base(**kwargs)
        return out.hidden_states[-1], out.logits, (out.past_key_values if use_cache else None)

    def sequence_reward(self, system_h, system_mask, user_h, user_mask, full_ids, full_mask, prefix_lens):
        """Mean token reward of each response in `full_ids` (= prompt + response).

        z is computed from the user turn and the *prompt* hidden states (positions
        [0, prefix_len)), then scored against the embedding of every response token.
        """
        full_h, _, _ = self.encode(full_ids, full_mask)
        with torch.no_grad():
            token_embeds = self.get_input_embeddings()(full_ids).float()

        rewards = []
        for b in range(full_ids.size(0)):
            pl = int(prefix_lens[b].item())
            valid_len = int(full_mask[b].sum().item())
            if valid_len <= 1 or pl >= valid_len:
                rewards.append(torch.tensor(0.0, device=full_ids.device))
                continue
            start = max(pl, 1)
            z = self.reward_head(
                user_h=user_h[b : b + 1].float(),
                user_mask=user_mask[b : b + 1],
                prefix_h=full_h[b : b + 1, :start, :].float(),
                prefix_mask=full_mask[b : b + 1, :start],
                system_h=system_h[b : b + 1].float(),
                system_mask=system_mask[b : b + 1],
            )
            token_reward = (z.unsqueeze(1) * token_embeds[b : b + 1, start:valid_len, :]).sum(-1)
            rewards.append(token_reward.mean())
        return torch.stack(rewards)

    def forward(self, system_ids, system_mask, user_ids, user_mask, chosen_ids, chosen_mask,
                rejected_ids, rejected_mask, prefix_lens, beta: float = 1.0):
        """Bradley-Terry loss  -log σ(β (r(chosen) − r(rejected)))."""
        system_h, _, _ = self.encode(system_ids, system_mask)
        user_h, _, _ = self.encode(user_ids, user_mask)
        r_c = self.sequence_reward(system_h, system_mask, user_h, user_mask, chosen_ids, chosen_mask, prefix_lens)
        r_r = self.sequence_reward(system_h, system_mask, user_h, user_mask, rejected_ids, rejected_mask, prefix_lens)
        loss = -torch.log(torch.sigmoid(beta * (r_c - r_r)) + 1e-12).mean()
        stats = {
            "loss": loss.detach().item(),
            "margin": (r_c - r_r).mean().detach().item(),
            "acc": (r_c > r_r).float().mean().item(),
        }
        return loss, stats


def load_checkpoint(path: str, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    case = str(ckpt.get("config", {}).get("case_kvq", "2_last"))
    if case not in ("2_last", "2.1", "case2_last"):
        raise ValueError(f"{path}: reward head type case_kvq={case}; this release implements 2_last only")
    return ckpt
