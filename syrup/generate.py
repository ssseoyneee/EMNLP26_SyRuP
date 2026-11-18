"""SyRuP decoding.

At every step, over the top-k candidates V_k of the system-conditioned LM:

    score(v) = l_sys(v) + λ_r · z_t·e_v + λ_d · (l_sys(v) − l_neutral(v))

    l_sys      logits with the real system prompt (after repetition penalty)
    l_neutral  logits of a second stream whose system prompt is replaced by
               "You are a helpful assistant." (perturb_mode=neutral_system)
    z_t        reward-head vector from (system prompt, user turn, response so far)
    e_v        frozen input embedding of v

and the next token is argmax_v score(v) (greedy).

The frozen LM runs as two streams, each with its own KV cache (TwoStreamCache):

    sys      prefix with the real system prompt          -> l_sys and the reward-head queries
    neutral  prefix with the system prompt replaced       -> l_neutral for the contrastive term

Both streams receive every chosen token. Only the sys stream's hidden states of the
generated tokens are kept: they are the reward head's prefix.
"""
import torch

from .modeling import SyRuPModel


class TwoStreamCache:
    def __init__(self, model, prompt_ids: torch.Tensor, neutral_ids: torch.Tensor):
        self.model = model
        self.sys_len, self.neutral_len = prompt_ids.size(1), neutral_ids.size(1)
        hidden, self.sys_logits, self.sys_past = model.encode(
            prompt_ids, torch.ones_like(prompt_ids), use_cache=True, last_logits_only=True
        )
        _, self.neutral_logits, self.neutral_past = model.encode(
            neutral_ids, torch.ones_like(neutral_ids), use_cache=True, last_logits_only=True
        )
        self.generated_h = hidden[:, :0, :]

    def next_logits(self):
        """(l_sys, l_neutral) for the next position, float32, shape (1, V)."""
        return self.sys_logits[:, -1, :].float(), self.neutral_logits[:, -1, :].float()

    def append(self, token: torch.Tensor):
        """Feed the chosen token (shape (1, 1)) to both streams."""
        device = token.device
        self.sys_len += 1
        self.neutral_len += 1
        hidden, self.sys_logits, self.sys_past = self.model.encode(
            token, torch.ones(1, self.sys_len, dtype=torch.long, device=device),
            use_cache=True, past_key_values=self.sys_past, last_logits_only=True,
        )
        self.generated_h = torch.cat([self.generated_h, hidden], dim=1)
        _, self.neutral_logits, self.neutral_past = self.model.encode(
            token, torch.ones(1, self.neutral_len, dtype=torch.long, device=device),
            use_cache=True, past_key_values=self.neutral_past, last_logits_only=True,
        )


def stop_token_ids(model: SyRuPModel, tokenizer) -> set:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    gen_eos = getattr(model.base.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, list):
        ids.update(gen_eos)
    elif gen_eos is not None:
        ids.add(int(gen_eos))
    return ids


def reward_vector(model, system_h, system_mask, user_h, user_mask, generated_h):
    """z_t from the hidden states of the tokens generated so far (none at step 0)."""
    gen_len = generated_h.size(1)
    if gen_len > 0:
        prefix_h = generated_h.float()
        prefix_mask = torch.ones(1, gen_len, dtype=torch.long, device=generated_h.device)
    else:
        prefix_h = torch.zeros(1, 1, user_h.size(-1), dtype=torch.float32, device=generated_h.device)
        prefix_mask = torch.zeros(1, 1, dtype=torch.long, device=generated_h.device)
    return model.reward_head(user_h, user_mask, prefix_h, prefix_mask, system_h.float(), system_mask)


@torch.no_grad()
def syrup_generate(
    model: SyRuPModel,
    tokenizer,
    prompt_ids: torch.Tensor,       # (1, L)  [system, user] prefix
    neutral_ids: torch.Tensor,      # (1, L') prefix with the system prompt perturbed
    system_ids: torch.Tensor,       # (1, S)  system-only text
    user_ids: torch.Tensor,         # (1, U)  user-only text
    lambda_reward: float = 0.05,
    lambda_delta: float = 2.0,
    top_k: int = 10,
    max_new_tokens: int = 1024,
    rep_penalty: float = 1.03,
):
    """Returns (generated token ids incl. prompt, stats)."""
    model.eval()
    device = next(model.reward_head.parameters()).device
    stops = stop_token_ids(model, tokenizer)
    pad = tokenizer.pad_token_id

    prompt_ids, neutral_ids = prompt_ids.to(device), neutral_ids.to(device)
    system_ids, user_ids = system_ids.to(device), user_ids.to(device)

    system_mask = (system_ids != pad).long()
    user_mask = (user_ids != pad).long()
    system_h, _, _ = model.encode(system_ids, system_mask)
    user_h, _, _ = model.encode(user_ids, user_mask, last_logits_only=True)
    user_h = user_h.float()

    cache = TwoStreamCache(model, prompt_ids, neutral_ids)
    generated = prompt_ids.clone()
    embeddings = model.get_input_embeddings()
    flips_by_reward = flips_by_delta = 0

    for _ in range(max_new_tokens):
        l_sys, l_neutral = cache.next_logits()
        if rep_penalty != 1.0 and generated.size(1) > 1:
            seen = generated[0].unique()
            l_sys[0, seen] = l_sys[0, seen] / rep_penalty

        cand_logits, cand_ids = torch.topk(l_sys, top_k, dim=-1)
        cand_logits, cand_ids_1d = cand_logits[0], cand_ids[0]

        z = reward_vector(model, system_h, system_mask, user_h, user_mask, cache.generated_h)
        reward = (z[0] * embeddings(cand_ids_1d).detach().float()).sum(-1)
        delta = cand_logits - l_neutral.gather(-1, cand_ids)[0]

        r_term = lambda_reward * reward
        d_term = lambda_delta * delta
        score = cand_logits + r_term + d_term
        best = int(torch.argmax(score).item())
        flips_by_reward += int(best != int(torch.argmax(score - r_term).item()))
        flips_by_delta += int(best != int(torch.argmax(score - d_term).item()))

        next_tok = cand_ids_1d[best].view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() in stops:
            break
        cache.append(next_tok)

    steps = generated.size(1) - prompt_ids.size(1)
    stats = {
        "num_new_tokens": steps,
        "reward_flip_ratio": flips_by_reward / steps if steps else 0.0,
        "delta_flip_ratio": flips_by_delta / steps if steps else 0.0,
    }
    return generated, stats
