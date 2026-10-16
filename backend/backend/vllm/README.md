# vLLM serving environment

The existing deployment reports `vllm==0.27.1+cu129`, `torch==2.13.0+cu129`,
`transformers==5.15.1`, and `numpy==2.3.5` (metadata inspected on 2026-10-15).
This is an observed deployment, not a clean-install or GPU validation result.
Keep this environment separate from slime, whose build scripts pin a different
PyTorch version and NumPy 1.x.

## Install on a Linux NVIDIA GPU host

Run from the repository root. A supported NVIDIA driver and sufficient disk and
GPU memory are required; consult the [upstream GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
The following is a candidate installation procedure pending a fresh GPU run.
It intentionally lets vLLM resolve its compatible PyTorch build.

```bash
uv venv --python 3.12.3 public/vllm_env_0819
uv pip install --python public/vllm_env_0819/bin/python \
  'vllm==0.27.1' 'transformers==5.15.1' --torch-backend=cu129
uv pip check --python public/vllm_env_0819/bin/python
source public/vllm_env_0819/bin/activate
python -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__); assert torch.cuda.is_available()"
uv pip freeze --python public/vllm_env_0819/bin/python > vllm-environment.freeze.txt
```

If the resolver cannot find these versions or reports conflicting dependencies,
retain the error and resolve it before labeling this combination reproducible.
Do not force installation with `--no-deps`. The inspected `+cu129` deployment
may require a wheel index or wheel artifacts that are not recorded in this repo.

## Verify serving and generation

Use an existing local HF model. Start in a foreground terminal on an allocated
GPU; this command does not stop other services.

```bash
source public/vllm_env_0819/bin/activate
export MODEL_DIR=/absolute/path/to/hf-model
vllm serve "$MODEL_DIR" --served-model-name smoke-model \
  --host 127.0.0.1 --port 18101 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 --max-model-len 4096
```

In another terminal, wait for readiness and request actual generated tokens:

```bash
curl --fail http://127.0.0.1:18101/v1/models
curl --fail http://127.0.0.1:18101/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

Acceptance requires a loaded model, a successful response containing generated
text, no CUDA errors, and a recorded GPU/driver/package inventory. A metadata
check or `/v1/models` response alone does not verify inference.
Stop the foreground server with Ctrl-C after the smoke run.

## Pipeline integration

The default pipeline activates `<repo>/public/vllm_env_0819/bin/activate`.
The `eval_all.vllm_env` configuration can override it. Slime torch-dist checkpoints
must first be converted to HF format; validate tokenizer, EOS, and rope settings
using the repository's serving guard before evaluating. A successful 4K smoke
run does not verify 35B training or 32K memory limits.

The bundled `docs/` snapshot is labeled vLLM 0.11.0 and is older than the observed
deployment. Use the matching upstream documentation and installed
`vllm serve --help` to verify flags.
