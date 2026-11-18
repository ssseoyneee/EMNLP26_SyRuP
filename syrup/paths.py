"""Every file SyRuP reads or writes.

Everything produced from one reward head lives under its run directory:

<run_root>/<model>/rm/<hparams>/
    reward_model.pt, train_config.json                 reward head                  (stage train)
    generate/<benchmark>/<decoding><items>.json        predictions                  (stage generate)
    generate/<benchmark>/<decoding><items>.partial.jsonl   finished items of an unfinished run
    generate/<benchmark>/configs/<decoding><items>.json    arguments of that run
    generate/<benchmark>/eval/...                      judgements and verifier outputs (evaluation/)

  <model>     qwen2-5-1-5b-instruct
  <hparams>   bs4x4_ga8_lr0.0001_ep2_seed42   per-GPU batch x GPUs (+ _src-<sources> / _n<N> when restricted)
  <decoding>  syrup_lr0.05_ld2_k10_neutral_rp1.03_mnt1024
  <items>     _i0-250 when --start_idx/--end_idx restrict the range, empty otherwise

Every setting that changes a file's content appears in its path, so two different
configurations never overwrite each other.
"""
import json
from pathlib import Path

from .modeling import model_tag

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
PERTURB_TAGS = {"neutral_system": "neutral", "no_system": "nosys", "empty_system": "emptysys"}


def fmt_num(x) -> str:
    """1.0 -> '1', 0.05 -> '0.05', 1e-4 -> '0.0001'."""
    s = f"{float(x):.10f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def rm_dir(args, world_size: int) -> Path:
    tag = (f"bs{args.batch_size}x{world_size}_ga{args.grad_accum_steps}_lr{fmt_num(args.lr)}"
           f"_ep{args.num_epochs}_seed{args.seed}")
    if args.train_sources:
        tag += "_src-" + "-".join(sorted(s.strip() for s in args.train_sources.split(",") if s.strip()))
    if args.max_train_pairs:
        tag += f"_n{args.max_train_pairs}"
    return Path(args.run_root) / model_tag(args.model_name) / "rm" / tag


def checkpoint_file(rm_ckpt: str) -> Path:
    path = Path(rm_ckpt)
    return path / "reward_model.pt" if path.is_dir() else path


def benchmark_file(benchmark: str) -> Path:
    """'mf-AlpacaEval' -> data/benchmarks/mf-AlpacaEval.json; a path to a .json file is used as is."""
    return Path(benchmark) if benchmark.endswith(".json") else BENCHMARK_DIR / f"{benchmark}.json"


def prediction_stem(args, n_records: int) -> str:
    stem = (f"syrup_lr{fmt_num(args.lambda_reward)}_ld{fmt_num(args.lambda_delta)}_k{args.top_k}"
            f"_{PERTURB_TAGS[args.perturb_mode]}_rp{fmt_num(args.rep_penalty)}_mnt{args.max_new_tokens}")
    end = n_records if args.end_idx < 0 else min(args.end_idx, n_records)
    if args.start_idx > 0 or end < n_records:
        stem += f"_i{args.start_idx}-{end}"
    return stem


def prediction_dir(args, ckpt_path: Path, bench_path: Path) -> Path:
    return (Path(args.output_dir) if args.output_dir else ckpt_path.parent / "generate") / bench_path.stem


def save_config(path: Path, args, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(vars(args))
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
