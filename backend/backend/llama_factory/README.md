---
BACKEND_DESCRIPTIONS
Local LLaMA-Factory fine-tuning workspace (source tree + training configs).
Invoke: activate the env, `cd llama_factory/`, then
`llamafactory-cli train ../config/train_sft_gsm8k.yaml` (full; LoRA 已下线).
Dataset names must be registered in `llama_factory/data/dataset_info.json`;
the run writes a checkpoint under the config's output_dir.
---

# LLaMA-Factory: Fine-Tuning Workspace

This directory contains everything needed to fine-tune models with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): source code, training configs, launch scripts, and documentation.

---

## Environment

All LLaMA-Factory commands must run inside the dedicated conda/virtual environment:

```bash
source <repo>/public/llama_factory_env/bin/activate
```

---

## Directory Layout

```
llama_factory/
├── config/           # Training YAML configs — add your own configs here
├── scripts/          # Convenience launch scripts
├── llama_factory/    # LLaMA-Factory source code (run commands from this directory)
├── documents/        # Official LLaMA-Factory documentation (RST format)
└── README.md         # This file
```

### `config/` — Training Configurations

Contains YAML config files for `llamafactory-cli train`. Create new configs here for each training run.

| File | Description |
|---|---|
| `train_sft_gsm8k.yaml` | Example: 全参 SFT on Qwen3-1.7B using `identity` + `alpaca_en_demo` datasets |

Each config specifies model path, fine-tuning method (`full` — LoRA/QLoRA 已下线 2026-08-20), dataset, template, output dir, and training hyperparameters.

### `scripts/` — Launch Scripts

Convenience shell scripts that activate the environment and invoke `llamafactory-cli`. Use these as templates for your own runs.

| File | Description |
|---|---|
| `train.sh` | Launches SFT training using `config/train_sft_gsm8k.yaml` |
| `llama_factory_data_preflight.py` | Pre-GPU data validation (see [Data Preflight](#data-preflight)) — catches `dataset_info.json` column-registration mistakes before a GPU is occupied |
| `llama_factory_template_preflight.py` | Pre-GPU **advisory** template↔eos check (see [Template-EOS Preflight](#template-eos-preflight)) — since 2026-08-19 it no longer refuses templates; it prints the turn-ender vs base-eos comparison (exit 0 for every valid template, exit 2 only for config/registry errors) |
| `check_eos_alignment.py` | Post-training eos self-check (see [Loop-output self-check](#loop-output-self-check)) — verifies the checkpoint's tokenizer.eos is a real, emit-able vocab token and (with `--samples`) diagnoses `length`-loop output as training-side vs infra |

### `llama_factory/` — Source Code

The LLaMA-Factory source tree. **All `llamafactory-cli` commands must be run from this directory** so relative imports resolve correctly. Key subdirectories:

| Directory | Purpose |
|---|---|
| `src/llamafactory/cli.py` | CLI entry point |
| `src/llamafactory/train/` | Training loop, trainers |
| `src/llamafactory/data/` | Dataset loading, preprocessing, templates |
| `src/llamafactory/model/` | Model loading, PEFT adapters (LoRA 已下线, 训练不走) |
| `src/llamafactory/hparams/` | Argument parsing & hyperparameter models |
| `src/llamafactory/eval/` | Evaluation routines |
| `src/llamafactory/chat/` | Chat/inference interface |
| `src/llamafactory/api/` | REST API server |
| `src/llamafactory/webui/` | Gradio Web UI |

### `documents/` — Documentation Index

Official LLaMA-Factory docs in RST format (for reference, not read at runtime):

```
documents/
├── index.rst                          # Document root
├── getting_started/
│   ├── installation.rst               # Environment setup & dependencies
│   ├── data_preparation.rst           # Dataset format & registration
│   ├── sft.rst                        # Supervised fine-tuning
│   ├── inference.rst                  # Model inference / generation
│   ├── eval.rst                       # Benchmark evaluation
│   ├── merge_lora.rst                 # Merging LoRA weights (LoRA 已下线)
│   └── webui.rst                      # Gradio Web UI usage
├── advanced/
│   ├── arguments.rst                  # Full CLI argument reference
│   ├── trainers.rst                   # Trainer types & selection
│   ├── tuning_algorithms.rst          # Full SFT / DPO / etc. (LoRA/QLoRA 已下线)
│   ├── model_support.rst              # Supported model matrix
│   ├── acceleration.rst               # DeepSpeed, FSDP, multi-GPU
│   ├── distributed.rst                # Ray, multi-node
│   ├── quantization.rst               # GPTQ, AWQ, bitsandbytes
│   ├── monitor.rst                    # WandB, TensorBoard, SwanLab, MLflow
│   ├── extras.rst                     # Additional features / tips
│   └── best_practice/
│       ├── index.rst                  # Best-practice index
│       └── gpt-oss.rst                # GPT-OSS integration
└── multibackend/
    └── npu/
        ├── index.rst                  # NPU (Ascend) guide
        ├── npu_installation.rst
        └── npu_training.rst
```

---

## Quick Start

### 1. Activate environment

```bash
source <repo>/public/llama_factory_env/bin/activate
```

### 2. Enter the source directory

```bash
cd <repo>/backend/backend/llama_factory/llama_factory
```

### 3. Run training

```bash
# Using the convenience script:
bash ../scripts/train.sh

# Or directly:
llamafactory-cli train ../config/train_sft_gsm8k.yaml
```

---

## Creating a New Training Run

### Step 1: Write a config

Copy and modify the example config:

```bash
cp config/train_sft_gsm8k.yaml config/train_sft_myexperiment.yaml
```

Key config sections:

| Section | Important Fields |
|---|---|
| `model` | `model_name_or_path`, `trust_remote_code` |
| `method` | `stage` (sft/dpo/pt), `finetuning_type` (`full`/`freeze`; LoRA 已下线), 全参 hyperparams (`learning_rate`/`num_train_epochs`) |
| `dataset` | `dataset` (comma-separated names), `template`, `cutoff_len`, `max_samples` |
| `output` | `output_dir`, `save_steps`, `logging_steps`, `report_to` (none/wandb/tensorboard) |
| `train` | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `bf16` |
| `eval` | `eval_dataset`, `val_size`, `eval_strategy`, `eval_steps` (commented out by default) |

### Step 2: Write a launch script (optional)

```bash
#!/bin/bash
cd <repo>/backend/backend/llama_factory/llama_factory
source <repo>/public/llama_factory_env/bin/activate
llamafactory-cli train <repo>/backend/backend/llama_factory/config/train_sft_myexperiment.yaml
```

### Step 3: Run

```bash
bash scripts/train_myexperiment.sh
```

---

## Common Commands

```bash
# SFT training (full-parameter; LoRA 已下线)
llamafactory-cli train config/train_sft_gsm8k.yaml

# Chat with a fine-tuned model
llamafactory-cli chat config/chat.yaml

# Evaluate on benchmarks
llamafactory-cli eval config/eval.yaml

# Launch Web UI
llamafactory-cli webui
```

---

## Data Preflight

**Run this before launching any GPU training.** It validates the registered dataset using the *same* `llamafactory.data.parser` the trainer uses, so a registration mistake is caught in seconds instead of after `llamafactory-cli train` has grabbed a GPU card and then failed.

### What it catches

1. **Wrong `columns` keys (the classic)** — `dataset_info.json` `columns:` keys must be LLaMA-Factory **internal names** (`prompt`/`query`/`response`/...), **not** your row field names. Using `instruction`/`input`/`output` as keys is silently ignored → every row's prompt/query/response is `None` → stage-1 dies with `Cannot find valid samples`.
2. **Missing mapped field** — `columns: {"query": "input"}` but the file has no `input` key → the converter raises `KeyError: 'input'` on the first row.
3. **Empty content** — all rows empty after mapping → every row dropped as an invalid example.

### Usage

```bash
source <repo>/public/llama_factory_env/bin/activate
python scripts/llama_factory_data_preflight.py \
    --train-config config/train_sft_gsm8k.yaml \
    --dataset-dir <dir-containing-dataset_info.json>
```

Exit codes: `0` pass · `1` rejected (clear reason printed) · `2` config/environment error.

The ExpOS pipeline runs this automatically as **stage-0** (before the SFT stage, which is the first GPU step). A rejection marks the experiment `failed` with `stage-0 (data preflight) ...` before any GPU is burned. If the script file is missing on a machine (stale checkout), stage-0 logs a warning and continues rather than blocking the whole loop.

### The column convention (alpaca)

Your row fields may be named anything; what must line up is the `columns` mapping:

```json
{ "my_sft_data": {
    "file_name": "train.jsonl",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
} }
```

- `columns` keys = LLaMA-Factory internal names; values = your file's field names.
- Every mapped value must exist in **every** row (preflight enforces this).
- No `columns` block at all means the file must already carry `instruction`/`input`/`output` keys (LLaMA-Factory's alpaca defaults).
- Register one **distinct dataset name per file** — do not overwrite an existing entry.

## Template-EOS Preflight

**Since 2026-08-19 this check is *advisory*, not a gate.** It parses the train
template's assistant turn-ender, compares it with the base model's eos, and
prints the result — aligned → `OK`; split → a `NOTE` explaining the expected
tokenizer/config divergence. It **never refuses a template** (exit `0` for every
valid template); the only failures left are config/environment errors (exit `2`).

### The 0817 bug that started this (context)

When fine-tuning a **base** model (e.g. `Qwen3-4B-Base`) with the chat templates
`qwen3` / `qwen3_nothink`, every assistant turn ends with `<|im_end|>` (**151645**).
LLaMA-Factory writes that into the trained tokenizer's `eos_token` but **never**
into `model.config` / `generation_config` — those keep the base eos
`<|endoftext|>` (**151643**). The checkpoint is self-contradictory. In 0817 the
pipeline served it through an old vLLM env that built its stop set from the
*config* eos (151643) while the model was trained to terminate with 151645, so
vLLM never stopped: 250/250 samples `length`, median output ~13.6k chars, scores
crushed.

### Why it was relaxed (2026-08-19 six-template experiment)

> **LoRA 已下线 (2026-08-20)。** 本段的 LoRA 数据是历史实证 (也是下线的主要依据),
> 不再作为可选项 —— 训练只走全参 SFT (`finetuning_type: full`)。

The hard invariant was **falsified** by a controlled experiment (same harness,
GSM8K `limit=200`, one template variable at a time):

| template × method | finish_reason | accuracy |
|---|---|---|
| `default` × LoRA (r16/1ep, 历史) | 162 length / 38 stop | 0.010 |
| `qwen3_nothink` × LoRA (历史) | 200 length / 0 stop | 0.060 |
| `qwen3` × LoRA (历史) | 200 length / 0 stop | 0.560* |
| `default` × full-SFT | 0 length / 200 stop | 0.830 |
| `qwen3_nothink` × full-SFT | 0 length / 200 stop | 0.825 |
| `qwen3` × full-SFT | 0 length / 200 stop | 0.805 |

\* inflated by an answer extractor that greps `ANSWER: X` out of the repetition loop.

The **template is not the differentiator; the training method is.** Every full-SFT
run stopped 200/200; every historical LoRA (rank 16, 1 epoch) run degenerated into
a length-loop regardless of template — the stop-token learning difference that
led to LoRA's decommission. Root cause of the split's harmlessness: vLLM
unconditionally adds **tokenizer.eos** to the stop set (`--generation-config vllm`
only neutralizes sampling defaults like temperature/top_p/max_new_tokens — the
checkpoint `generation_config`'s eos ids are still unioned in, which merely adds
extra stop tokens). So a tokenizer/config eos split is benign — as
long as `tokenizer.eos` is a real vocab token the model can emit, vLLM stops the
moment it appears. `qwen3` / `qwen3_nothink` on a Qwen3 base are therefore **valid
choices**.

### Two lessons the 0819 run taught us

1. **全参 SFT 学会发 stop token —— 这是 LoRA 下线的主因之一。** Same data and
   template, historical LoRA rank16/1ep never emitted a stop token in a long
   few-shot context and looped; full-SFT stopped 200/200. LoRA 已下线, 训练只走
   全参 SFT (`finetuning_type: full`); 想省显存只能降数据行数/epochs。
2. **If you see loop output (`finish_reason=length` dominant / repetition),
   self-check eos-token alignment first** with
   [`check_eos_alignment.py`](#loop-output-self-check). It tells you whether
   `tokenizer.eos` is emit-able (→ the loop is **training-side**: data/epochs/
   capacity, not infra) or genuinely missing from the vocab (→ infra, the model
   literally cannot emit its own terminator).

### Usage

```bash
source <repo>/public/llama_factory_env/bin/activate
python scripts/llama_factory_template_preflight.py --train-config config/train_sft_gsm8k_example.yaml
```

Exit codes: `0` pass (with either an `OK` or a `NOTE` for a benign split) · `2`
config/environment error (YAML unreadable / template not registered /
llamafactory not importable / base eos unresolvable). The ExpOS pipeline runs
this automatically as **stage-0b**, right after the data preflight and before any
GPU is occupied. Since it no longer rejects templates, it cannot block a run; it
only surfaces the turn-ender↔base-eos comparison for the log.

## Loop-output self-check

When a run's eval output loops (`finish_reason=length` dominates, responses are
tens of thousands of chars, `ANSWER:` blocks repeat), run this to separate the two
possible causes:

```bash
# any python, no llamafactory dependency:
python scripts/check_eos_alignment.py \
    --checkpoint-dir <ckpt> \
    [--samples <samples.jsonl>]    # optional: benchmark suite eval output
```

What it does:

1. **ckpt eos alignment** — reads `tokenizer_config.json` / `config.json` /
   `generation_config.json` eos values and decodes the ids to tokens (merging
   `added_tokens_decoder` → `tokenizer.json.added_tokens` → `model.vocab`, since a
   saved checkpoint may drop any of these). The key verdict: is `tokenizer.eos`
   a **real vocab token the model can emit**? Yes → the model *can* stop; the
   tokenizer/config split (e.g. `im_end` vs `endoftext`) is benign under the
   current serve path. No → **infra** problem (the model literally cannot emit its
   own terminator); fix the training side (vocab resize / template special tokens).
2. **sample scan (with `--samples`)** — counts `finish_reason`, response lengths,
   `im_end` presence, and an `ANSWER:`-repetition heuristic. Combined with layer 1:
   emit-able `tokenizer.eos` **and** length-dominated output → the loop is
   **training-side** (data / epochs / capacity), not an eos bug.

Exit codes: `0` aligned (or benign split) / `1` real problem (un-emittable eos,
or confirmed training-side loop) / `2` config/file error.

Real-world output on the 0819 runs:

```
# exp_tpl_qwen3_lora_r3 (历史 broken case — correct diagnosis, exit 1; LoRA 已下线):
== ckpt eos 三处 ==
  tokenizer_config   eos_token   = '<|im_end|>'
  config             eos_token_id = 151643 -> '<|endoftext|>'
  generation_config  eos_token_id = 151643 -> '<|endoftext|>'
  (tokenizer eos id = 151645 -> '<|im_end|>')
NOTE: tokenizer 与 config/generation_config 的 eos 分叉 ... 当前 serve 路径下无害 ...
== eval samples 循环信号 ==
  finish_reason = {'length': 200}
  im_end=0      = 200/200
  复读(ANSWER x>=3) = 191/200
FAIL: ... 训练侧问题, 不是 eos 对齐.

# exp_tpl_qwen3_full_r2 (the healthy case, exit 0):
== ckpt eos 三处 ==
  tokenizer_config   eos_token   = '<|im_end|>'
  config             eos_token_id = 151645 -> '<|im_end|>'
  generation_config  eos_token_id = 151645 -> '<|im_end|>'
OK: tokenizer / config / generation_config 三处 eos 一致.
== eval samples 循环信号 ==
  finish_reason = {'stop': 200}
  复读(ANSWER x>=3) = 0/200
OK: stop 主导, 未见明显循环.
```

Note: full-SFT checkpoints are self-consistent at `im_end` because full
fine-tuning updates `config.eos_token_id`. (Historical LoRA checkpoints kept the
base `endoftext` in config while the tokenizer said `im_end` — LoRA 已下线, no
longer relevant.) Either is fine under the current serve path, and the script says
so.

## Key Notes

- **Source directory matters**: always `cd` into `llama_factory/` before running any `llamafactory-cli` command.
- **Environment first**: the `source .../llama_factory_env/bin/activate` line is required in every new shell session.
- **Config paths**: use absolute paths or paths relative to the `llama_factory/` source directory.
- **Template names**: the `template` field in the config must match a registered template in LLaMA-Factory's `src/llamafactory/data/template.py`. For Qwen3: `default` (assistant turn-ender = `{"eos_token"}` = tokenizer eos), `qwen3_nothink`, `qwen3`, ... are all valid for a base model — the 0819 experiment showed the template is not the differentiator (see [Template-EOS Preflight](#template-eos-preflight)). **LoRA 已下线 (2026-08-20), 全参 SFT 是唯一训练方式** — `finetuning_type: full`; 0819 实证 full-SFT 学会发 stop token, 而历史 LoRA 低 rank/epochs 在长 few-shot 里循环。
- **Dataset registration**: dataset names in the config (e.g. `identity`, `alpaca_en_demo`) must exist in LLaMA-Factory's `data/dataset_info.json` or be custom-registered.
