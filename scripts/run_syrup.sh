#!/bin/bash
# Full SyRuP pipeline for one base model:
#   reward-head training -> decoding on every prepared benchmark -> evaluation.
#
#   bash scripts/run_syrup.sh Qwen/Qwen2.5-1.5B-Instruct 4      # model, per-GPU batch size
#
# Env: HF_TOKEN (Llama models are gated), OPENAI_API_KEY (judge), NGPU (training GPUs, default 1),
#      IFEVAL_SPLIT (IFEval split file, not included; IFEval is skipped without it),
#      IFEVAL_DIR (google-research checkout with instruction_following_eval/; needed to score IFEval/IHEval).
# Re-running skips everything that is already finished.
set -euo pipefail

MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
BATCH=${2:-4}
NGPU=${NGPU:-1}
SEED=${SEED:-42}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

TRAIN_ARGS=(--model_name "${MODEL}" --batch_size "${BATCH}" --grad_accum_steps 8 --lr 1e-4 --num_epochs 2 --seed "${SEED}")
RM_DIR=$(python - "${NGPU}" "${TRAIN_ARGS[@]}" <<'PY'
import sys
from syrup.config import get_args
from syrup.paths import rm_dir
print(rm_dir(get_args(["--stage", "train"] + sys.argv[2:]), int(sys.argv[1])))
PY
)

[ -f data/benchmarks/mf-AlpacaEval.json ] || python preprocess/prepare_multifaceted_bench.py
if [ ! -f data/benchmarks/ifeval.json ] && [ -n "${IFEVAL_SPLIT:-}" ]; then
  python preprocess/prepare_ifeval.py --split_file "${IFEVAL_SPLIT}"
fi

echo "[$(date)] 1/3 reward-head training (${NGPU} GPU) -> ${RM_DIR}"
if [ "${NGPU}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node "${NGPU}" main.py --stage train "${TRAIN_ARGS[@]}" --output_dir "${RM_DIR}"
else
  python main.py --stage train "${TRAIN_ARGS[@]}" --output_dir "${RM_DIR}"
fi

echo "[$(date)] 2/3 SyRuP decoding"
GEN_ARGS=(--stage generate --rm_ckpt "${RM_DIR}")
for BENCH in mf-AlpacaEval mf-FLASK mf-Koala mf-MT-Bench mf-Self-Instruct; do
  python main.py "${GEN_ARGS[@]}" --benchmark "${BENCH}"
done
# Optional benchmarks: run only if their converter was run (see README).
if [ -f data/benchmarks/ifeval.json ]; then
  python main.py "${GEN_ARGS[@]}" --benchmark ifeval
fi
for V in single_aligned single_conflict multi_aligned multi_conflict; do
  if [ -f "data/benchmarks/iheval-${V}.json" ]; then
    python main.py "${GEN_ARGS[@]}" --benchmark "iheval-${V}" --max_length 1024 --max_new_tokens 256
  fi
done
for V in handwritten distractors; do
  if [ -f "data/benchmarks/realguardrails-${V}.json" ]; then
    python main.py "${GEN_ARGS[@]}" --benchmark "realguardrails-${V}" --max_length 4096
  fi
done
if [ -f data/benchmarks/sysbench.json ]; then
  python main.py "${GEN_ARGS[@]}" --benchmark sysbench --max_length 4096 --end_idx "${SYSBENCH_TURNS:-250}"
fi

echo "[$(date)] 3/3 evaluation"
GEN="${RM_DIR}/generate"
python evaluation/judge.py --pred "${GEN}/mf-*/syrup_*.json" --seeds 42 43 44
python evaluation/aggregate.py --pred "${GEN}/mf-*/syrup_*.json" --seeds 42 43 44 --out "${RM_DIR}/multifaceted_bench.csv"
for B in ifeval "iheval-*"; do
  if compgen -G "${GEN}/${B}/syrup_*.json" >/dev/null; then
    if [ -n "${IFEVAL_DIR:-}" ]; then
      python evaluation/ifeval.py --pred "${GEN}/${B}/syrup_*.json"
    else
      echo "[skip] ${B} scoring: set IFEVAL_DIR to a google-research checkout (see README)"
    fi
  fi
done
if compgen -G "${GEN}/realguardrails-*/syrup_*.json" >/dev/null; then
  python evaluation/realguardrails.py --pred "${GEN}/realguardrails-*/syrup_*.json"
fi
if compgen -G "${GEN}/sysbench/syrup_*.json" >/dev/null; then
  python evaluation/sysbench.py --pred "${GEN}/sysbench/syrup_*.json"
fi
