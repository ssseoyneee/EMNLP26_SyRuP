"""Stage `train`: the SyRuP reward head on preference pairs (the base LM stays frozen).

    python main.py --stage train --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 4
    torchrun --standalone --nproc_per_node 4 main.py --stage train --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 4

Writes reward_model.pt and train_config.json to the run directory (see paths.py).
An existing checkpoint is not retrained unless --overwrite is given.
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import PreferencePairs, collate_pairs
from .modeling import SyRuPModel, load_tokenizer, resolve_model_name
from .paths import rm_dir


@dataclass
class Dist:
    on: bool = False
    rank: int = 0
    local_rank: int = 0
    world: int = 1

    @property
    def main(self):
        return self.rank == 0

    def barrier(self):
        if self.on:
            dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)


def init_dist() -> Dist:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return Dist()
    d = Dist(True, int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ["WORLD_SIZE"]))
    if torch.cuda.is_available():
        torch.cuda.set_device(d.local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_warmup(optimizer, warmup: int, total: int):
    def f(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return LambdaLR(optimizer, f)


def reduce_stats(stats, device, d: Dist):
    if not d.on:
        return stats
    v = torch.tensor([stats["loss"], stats["margin"], stats["acc"], 1.0], device=device)
    dist.all_reduce(v)
    return {"loss": (v[0] / v[3]).item(), "margin": (v[1] / v[3]).item(), "acc": (v[2] / v[3]).item()}


def train(args):
    d = init_dist()
    try:
        _train(args, d)
    finally:
        if d.on and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, d: Dist):
    out_dir = Path(args.output_dir) if args.output_dir else rm_dir(args, d.world)
    ckpt_path = out_dir / "reward_model.pt"
    if ckpt_path.exists() and not args.overwrite:
        if d.main:
            print(f"[train] {ckpt_path} exists; skipping (use --overwrite to retrain)")
        return
    if d.main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] run dir: {out_dir}")
    d.barrier()

    set_seed(args.seed + d.rank)
    device = torch.device(f"cuda:{d.local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = load_tokenizer(args.model_name)
    sources = {s.strip() for s in args.train_sources.split(",") if s.strip()} or None
    model = SyRuPModel(args.model_name, torch_dtype=dtype).to(device)
    if d.on:
        model = DDP(model, device_ids=[d.local_rank] if device.type == "cuda" else None,
                    output_device=d.local_rank if device.type == "cuda" else None)
    core = model.module if d.on else model

    ds = PreferencePairs(tokenizer, args.train_dataset, "train", args.max_length, sources, args.max_train_pairs)
    sampler = DistributedSampler(ds, num_replicas=d.world, rank=d.rank, shuffle=True, seed=args.seed) if d.on else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        collate_fn=lambda b: collate_pairs(b, tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(core.reward_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader) / args.grad_accum_steps) * args.num_epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = linear_warmup(optimizer, warmup, total_steps)
    if d.main:
        print(f"[train] pairs={len(ds):,} batches/epoch/GPU={len(loader):,} GPUs={d.world} "
              f"optimizer steps={total_steps:,} (warmup {warmup})")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, running, t0 = 0, {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}, time.time()
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, stats = model(
                system_ids=batch["system_ids"], system_mask=batch["system_mask"],
                user_ids=batch["user_ids"], user_mask=batch["user_mask"],
                chosen_ids=batch["chosen_ids"], chosen_mask=batch["chosen_mask"],
                rejected_ids=batch["rejected_ids"], rejected_mask=batch["rejected_mask"],
                prefix_lens=batch["prefix_lens"], beta=args.pair_beta,
            )
            stats = reduce_stats(stats, device, d)
            if loss.requires_grad:
                (loss / args.grad_accum_steps).backward()
            else:  # every item in the batch was degenerate; keep DDP ranks in sync
                (sum(p.sum() * 0.0 for p in core.reward_head.parameters()) / args.grad_accum_steps).backward()
            for k in ("loss", "margin", "acc"):
                running[k] += stats[k]
            running["n"] += 1

            last = (i + 1) == len(loader)
            if (i + 1) % args.grad_accum_steps == 0 or last:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if d.main and (step % args.log_every == 0 or last):
                    n = running["n"]
                    print(f"[epoch {epoch + 1}/{args.num_epochs}] step {step}/{total_steps} "
                          f"loss={running['loss'] / n:.4f} margin={running['margin'] / n:+.4f} "
                          f"acc={running['acc'] / n:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
                    running = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}

    if d.main:
        config = {
            "base_model_name": resolve_model_name(args.model_name),
            "case_kvq": "2_last",
            "max_length": args.max_length,
            "train_batch_size": args.batch_size,
            "world_size": d.world,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "pair_beta": args.pair_beta,
            "seed": args.seed,
            "dataset_name": args.train_dataset,
            "sources": args.train_sources,
            "max_train_pairs": args.max_train_pairs,
        }
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"reward_head": core.reward_head.state_dict(),
                    "base_model_name": config["base_model_name"], "config": config}, tmp)
        os.replace(tmp, ckpt_path)
        with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"[train] saved {ckpt_path}")
    d.barrier()

