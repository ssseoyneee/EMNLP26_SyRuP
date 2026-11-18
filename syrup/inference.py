"""Stage `generate`: SyRuP decoding over a benchmark file.

    python main.py --stage generate --rm_ckpt runs/qwen2-5-1-5b-instruct/rm/bs4x4_ga8_lr0.0001_ep2_seed42 \
        --benchmark mf-AlpacaEval

The base LM is the one the reward head was trained on (stored in the checkpoint).
Finished items are appended to <name>.partial.jsonl, so a killed job resumes where it
stopped. An existing prediction file is not regenerated unless --overwrite is given.
"""
import json
import time

import torch
from tqdm import tqdm

from .data import encode_record, load_records
from .generate import syrup_generate
from .modeling import SyRuPModel, load_checkpoint, load_tokenizer, resolve_model_name
from .paths import benchmark_file, checkpoint_file, prediction_dir, prediction_stem, save_config


def inference(args):
    ckpt_path = checkpoint_file(args.rm_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--rm_ckpt not found: {ckpt_path} (run --stage train first)")
    bench_path = benchmark_file(args.benchmark)
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark file not found: {bench_path} (see preprocess/)")

    records = load_records(str(bench_path))
    end = len(records) if args.end_idx < 0 else min(args.end_idx, len(records))
    todo = records[args.start_idx:end]

    out_dir = prediction_dir(args, ckpt_path, bench_path)
    stem = prediction_stem(args, len(records))
    out_path, partial = out_dir / f"{stem}.json", out_dir / f"{stem}.partial.jsonl"
    print(f"[generate] rm: {ckpt_path}\n[generate] output: {out_path}")
    if out_path.exists() and not args.overwrite:
        print("[generate] output exists; skipping (use --overwrite to regenerate)")
        return
    if args.overwrite and partial.exists():
        partial.unlink()

    done = {}
    if partial.exists():
        with open(partial, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
        print(f"[generate] resuming: {len(done)}/{len(todo)} already done")

    ckpt = load_checkpoint(str(ckpt_path))
    base_name = resolve_model_name(ckpt["base_model_name"])
    if args.model_name and resolve_model_name(args.model_name) != base_name:
        raise ValueError(f"--model_name {args.model_name} but the reward head was trained on {base_name}")
    save_config(out_dir / "configs" / f"{stem}.json", args,
                {"rm_ckpt": str(ckpt_path), "base_model_name": base_name, "rm_config": ckpt.get("config", {})})

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = load_tokenizer(base_name)
    model = SyRuPModel(base_name, torch_dtype=dtype).to(device)
    model.reward_head.load_state_dict(ckpt["reward_head"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as fout:
        for rec in tqdm(todo, desc=f"SyRuP {bench_path.stem}", dynamic_ncols=True):
            if rec["id"] in done:
                continue
            inputs = encode_record(tokenizer, rec, args.perturb_mode, args.max_length)
            t0 = time.time()
            generated, stats = syrup_generate(
                model, tokenizer, **inputs,
                lambda_reward=args.lambda_reward, lambda_delta=args.lambda_delta, top_k=args.top_k,
                max_new_tokens=args.max_new_tokens, rep_penalty=args.rep_penalty,
            )
            new_ids = generated[0, inputs["prompt_ids"].size(1):]
            out = dict(rec)
            out["response"] = tokenizer.decode(new_ids, skip_special_tokens=True)
            out["decode_stats"] = {**stats, "seconds": round(time.time() - t0, 3)}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done[rec["id"]] = out

    ordered = [done[r["id"]] for r in todo]
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    partial.unlink()
    print(f"[generate] saved {len(ordered)} predictions -> {out_path}")
