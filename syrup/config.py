import argparse

from .data import TRAIN_DATASET
from .paths import PERTURB_TAGS

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def get_args(argv=None):
    p = argparse.ArgumentParser(description="SyRuP. Defaults are the paper settings.")

    p.add_argument("--stage", required=True, choices=["train", "generate"],
                   help="train: reward-head training | generate: SyRuP decoding over a benchmark")

    # model / outputs
    p.add_argument("--model_name", type=str, default=None,
                   help=f"train: base LM (default {DEFAULT_MODEL}). generate: optional check against the checkpoint.")
    p.add_argument("--run_root", type=str, default="runs",
                   help="Everything is written under <run_root>/<model>/rm/... (see syrup/paths.py).")
    p.add_argument("--output_dir", type=str, default="",
                   help="train: reward-head directory. generate: prediction root (default <rm dir>/generate).")
    p.add_argument("--max_length", type=int, default=1024, help="token budget of each tokenized input")
    p.add_argument("--overwrite", action="store_true", help="redo even if the output exists")

    # reward-head training
    p.add_argument("--train_dataset", type=str, default=TRAIN_DATASET)
    p.add_argument("--train_sources", type=str, default="", help="comma-separated original_source filter; '' = all")
    p.add_argument("--max_train_pairs", type=int, default=0, help="use only the first N pairs (0 = all)")
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU micro batch")
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--pair_beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=1)

    # decoding
    p.add_argument("--rm_ckpt", type=str, default="", help="generate: reward_model.pt or the directory that contains it")
    p.add_argument("--benchmark", type=str, default="",
                   help="generate: name of data/benchmarks/<benchmark>.json, or a path to a JSON list")
    p.add_argument("--lambda_reward", type=float, default=0.05, help="λ_r, weight of the reward-head term")
    p.add_argument("--lambda_delta", type=float, default=2.0, help="λ_d, weight of the system-contrast term")
    p.add_argument("--top_k", type=int, default=10, help="candidate set = top-k of the system-conditioned LM")
    p.add_argument("--perturb_mode", type=str, default="neutral_system", choices=list(PERTURB_TAGS))
    p.add_argument("--rep_penalty", type=float, default=1.03)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1, help="exclusive; -1 = until the end")
    p.add_argument("--device", type=str, default="cuda:0")

    args = p.parse_args(argv)
    if args.stage == "train" and not args.model_name:
        args.model_name = DEFAULT_MODEL
    if args.stage == "generate" and (not args.rm_ckpt or not args.benchmark):
        p.error("--stage generate needs --rm_ckpt and --benchmark")
    return args
