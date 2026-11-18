# SyRuP: Enhancing System-Prompt Following via Reward-Guided Prediction in LLM Decoding

[![Paper](https://img.shields.io/badge/EMNLP_2026-Paper-b31b1b)](https://aclanthology.org/2026.emnlp-main.1537/)
[![PDF](https://img.shields.io/badge/PDF-ACL_Anthology-blue)](https://aclanthology.org/2026.emnlp-main.1537.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ssseoyneee.github.io/syrup_project_page/)

## Overview

System prompts are how developers set an LLM's role, rules and output format, but models often
lose track of them: as the user turn and the response grow, generation drifts back to the
model's default behaviour. Prompt engineering gives no guarantee, and fine-tuning the model for
every new system prompt is costly.

**SyRuP** is a decoding-time method that improves system-prompt following while the LLM stays
frozen. A lightweight cross-attention reward head, trained once on system-prompt-conditioned
preference pairs, gives token-level adherence scores. At every decoding step, the top-k
candidates of the base model are reranked by combining the base logits, the learned reward, and
a contrastive signal that measures how much the system prompt shifts each token's logit.

## Key Mechanism

1. **Reward head**: a cross-attention head reads the frozen LM's hidden states, with the user
   turn and the response so far as queries and the system prompt as keys/values, and outputs a
   vector z_t. The reward of a candidate token v is `z_t · e_v`, where e_v is its frozen input
   embedding. The head is trained with a Bradley-Terry loss on
   `kaist-ai/Multifaceted-Collection-DPO`; the LM is never updated.
2. **System-prompt contrast**: a second decoding stream replaces the system prompt with the
   neutral "You are a helpful assistant." The gap `l_sys(v) − l_neutral(v)` measures how much the
   real system prompt supports v.
3. **Reward-guided prediction**: at each step, the top-k (k = 10) candidates of the
   system-conditioned LM are rescored and the best one is taken greedily:

   `score(v) = l_sys(v) + λ_r · z_t·e_v + λ_d · (l_sys(v) − l_neutral(v))`, with λ_r = 0.05, λ_d = 2.0.

   Both streams keep their own KV cache and receive every chosen token, so a step costs two
   single-token forward passes and one reward-head call.

## Installation

```bash
conda create -n syrup python=3.10 -y
conda activate syrup
pip install -r requirements.txt
export HF_TOKEN=...   # the Llama models are gated
```

`requirements.txt` pins the versions this release was tested with (PyTorch 2.6.0, Transformers 4.55.4).

## Data

The reward head is trained on
[Multifaceted-Collection-DPO](https://huggingface.co/datasets/kaist-ai/Multifaceted-Collection-DPO),
which is downloaded from Hugging Face at training time.

Evaluation benchmarks are converted into `data/benchmarks/<name>.json`:

```bash
python preprocess/prepare_multifaceted_bench.py                      # kaist-ai/Multifaceted-Bench
python preprocess/prepare_ifeval.py --split_file /path/to/ifeval_system_split.json
python preprocess/prepare_iheval.py --repo /path/to/IHEval          # git clone https://github.com/ytyz1307zzh/IHEval
python preprocess/prepare_realguardrails.py                          # normster/RealGuardrails
python preprocess/prepare_sysbench.py --repo /path/to/SysBench      # git clone https://github.com/PKU-Baichuan-MLSystemLab/SysBench
```

| benchmark file | items | model input | scored by |
|---|---|---|---|
| `mf-{AlpacaEval,FLASK,Koala,MT-Bench,Self-Instruct}` | 134/349/147/150/141 | system + user | `judge.py` + `aggregate.py` |
| `ifeval` | 539 | system (constraints) + task | `ifeval.py` |
| `iheval-{single,multi}_{aligned,conflict}` | 541 each | system (rules) + instruction | `ifeval.py` |
| `realguardrails-{handwritten,distractors}` | 239 / 504 | full conversation incl. distractor turns | `realguardrails.py` |
| `sysbench` | 2500 turns (500 dialogues × 5) | system + ground-truth earlier turns + current turn | `sysbench.py` |

Each benchmark file is a list of records:

```json
{"id": "...", "system": "<system prompt>", "prompt": "<user turn>",
 "messages": [ ...optional full conversation ending with the user turn... ]}
```

Every other field is copied into the predictions, so the evaluators need only the prediction
file. For IFEval, the verifiable constraints of the 539 prompts are moved into the system prompt;
this split file is not included in the repository and is passed with `--split_file`.

## Usage

Run the full pipeline (reward-head training → SyRuP decoding → evaluation) for one base model:

```bash
NGPU=4 bash scripts/run_syrup.sh Qwen/Qwen2.5-1.5B-Instruct 4
```

or stage by stage:

```bash
torchrun --standalone --nproc_per_node 4 main.py --stage train --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 4
python main.py --stage generate --rm_ckpt runs/qwen2-5-1-5b-instruct/rm/bs4x4_ga8_lr0.0001_ep2_seed42 --benchmark mf-AlpacaEval
```

Use plain `python main.py --stage train ...` for one GPU. In the `generate` stage, the base LM is
read from the reward-head checkpoint.

Evaluate (Multifaceted-Bench: LLM judge with three seeds, then mean ± std; IFEval/IHEval: the
official IFEval verifier):

```bash
export OPENAI_API_KEY=...
python evaluation/judge.py     --pred "runs/qwen2-5-1-5b-instruct/rm/*/generate/mf-*/syrup_*.json"
python evaluation/aggregate.py --pred "runs/qwen2-5-1-5b-instruct/rm/*/generate/mf-*/syrup_*.json" --out mf.csv

git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git
git -C google-research sparse-checkout set instruction_following_eval
python evaluation/ifeval.py --ifeval_dir google-research --pred "runs/qwen2-5-1-5b-instruct/rm/*/generate/ifeval/syrup_*.json"
```

Everything a configuration produces (the reward head, predictions, the arguments of every run,
judgements) is written to `runs/<model>/rm/<hparams>/` (see `syrup/paths.py`). Finished work is skipped and decoding
resumes item by item, so an interrupted job can be resubmitted as is. Use `--start_idx/--end_idx`
to split a benchmark across GPUs.

To run SyRuP on your own prompt:

```python
import torch
from syrup import SyRuPModel, encode_record, load_checkpoint, load_tokenizer, syrup_generate

ckpt = load_checkpoint("runs/qwen2-5-1-5b-instruct/rm/bs4x4_ga8_lr0.0001_ep2_seed42/reward_model.pt")
tokenizer = load_tokenizer(ckpt["base_model_name"])
model = SyRuPModel(ckpt["base_model_name"], torch_dtype=torch.float16).to("cuda")
model.reward_head.load_state_dict(ckpt["reward_head"])

record = {"system": "You are a patient tutor for middle-school students. Explain ideas with one everyday "
                    "analogy and end every answer with a one-sentence summary that starts with 'In short:'.",
          "prompt": "Why is the sky blue?"}
inputs = encode_record(tokenizer, record, perturb_mode="neutral_system", max_length=1024)
output_ids, stats = syrup_generate(model, tokenizer, **inputs, lambda_reward=0.05, lambda_delta=2.0, top_k=10)
print(tokenizer.decode(output_ids[0, inputs["prompt_ids"].size(1):], skip_special_tokens=True))
```

Default hyper-parameters follow the paper:

| | value |
|---|---|
| base models | Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct (fp16) |
| reward-head training | AdamW, lr 1e-4, 2 epochs, grad-accum 8, warmup 3%, weight decay 0, β = 1, seed 42; per-GPU batch 4 (Qwen 1.5B), 2 (Llama 3B), 8 (Llama 8B) |
| decoding | greedy, top-k = 10, λ_r = 0.05, λ_d = 2.0, neutral system prompt, repetition penalty 1.03, max 1024 new tokens |
| inputs | truncated to 1024 tokens (RealGuardrails, SysBench: 4096) |
| per-benchmark | IHEval: max 256 new tokens; SysBench: first 100 or 250 turns (`--end_idx`) |
| judge | gpt-4o-mini, seeds 42/43/44 |

Ablations: `--lambda_reward 0` (contrast only), `--lambda_delta 0` (reward head only), `--top_k`,
`--perturb_mode {neutral_system,no_system,empty_system}`.

## Repository Layout

```
main.py                 entry point (--stage train | generate)
syrup/
  config.py             arguments and paper defaults
  paths.py              run-directory layout (every file read or written)
  data.py               preference pairs (training) and benchmark records (decoding)
  modeling.py           model / tokenizer loading, chat templates, reward head, Bradley-Terry loss
  train.py              reward-head training (single GPU or torchrun)
  generate.py           reward-guided decoding with the two KV-cache streams
  inference.py          decoding over a benchmark file (resumable)
preprocess/             benchmark converters -> data/benchmarks/*.json
evaluation/
  judge.py              Multifaceted-Bench LLM judge, one run per seed
  aggregate.py          mean ± std over judge runs
  ifeval.py             IFEval / IHEval with the official IFEval verifier
  realguardrails.py     RealGuardrails guardrail-compliance judge
  sysbench.py           SysBench constraint judge (ISR / CSR / SSR)
  llm.py                async OpenAI / OpenRouter client used by the judges
scripts/run_syrup.sh    full pipeline
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kim2026syrup,
    title={{SyRuP}: Enhancing System-Prompt Following via Reward-Guided Prediction in {LLM} Decoding},
    author={Seoyeon Kim and Minjae Kang and Jaehyung Kim},
    booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
    year={2026},
    url={https://aclanthology.org/2026.emnlp-main.1537/}
}
```

## License

The code is released under the [MIT License](LICENSE). The IFEval verifier is downloaded from
Google Research (Apache License 2.0). The base models and the datasets are subject to their own
licenses and terms of use.
