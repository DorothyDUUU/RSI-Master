---
BACKEND_DESCRIPTIONS
vLLM 0.27.1 local serving workspace — OpenAI-compatible API.
Invoke: `vllm serve <model_dir> --port 8101` (engine args in docs/configuration).
Convenience launcher: `bash scripts/run_qwen_3-1.8.sh` (Qwen3-1.7B, port 8101).
---

# vLLM 0.27.1 — Local Serving Workspace

vLLM inference engine for local model serving. **Version: 0.27.1**.

## Environment

```bash
source <repo>/public/vllm_env_0819/bin/activate
```

---

## Directory Layout

```
vllm/
├── docs/       # Official vLLM 0.11.0 documentation
├── scripts/    # Launch scripts for serving models
└── README.md
```

### `scripts/` — Launch Scripts

| File | Description |
|---|---|
| `run_qwen_3-1.8.sh` | Serve Qwen3-1.7B on port 8101 |

Usage:
```bash
bash scripts/run_qwen_3-1.8.sh
```

### `docs/` — Documentation Index

Official vLLM 0.11.0 docs. Key entry points:

```
docs/
├── README.md                              # Docs overview
├── getting_started/
│   └── quickstart.md                      # Installation & first launch
├── configuration/
│   ├── engine_args.md                     # EngineArgs reference (--gpu-memory-utilization, --max-model-len, etc.)
│   ├── serve_args.md                      # `vllm serve` CLI arguments
│   ├── optimization.md                    # Performance tuning (chunked prefill, prefix caching, etc.)
│   ├── conserving_memory.md               # KV cache offloading, quantization, etc.
│   └── env_vars.md                        # Environment variable reference
├── serving/
│   ├── openai_compatible_server.md        # OpenAI-compatible API (chat/completions, /models)
│   ├── offline_inference.md               # Python API for batch inference (LLM.generate)
│   ├── parallelism_scaling.md             # Tensor/pipeline/data parallel scaling
│   ├── data_parallel_deployment.md        # Multi-node data parallel serving
│   └── expert_parallel_deployment.md      # Expert parallel for MoE models
├── features/
│   ├── automatic_prefix_caching.md        # APC — reduces redundant KV computation
│   ├── lora.md                            # LoRA adapter serving
│   ├── spec_decode.md                     # Speculative decoding
│   ├── structured_outputs.md              # Guided decoding (JSON schema, regex, grammar)
│   ├── tool_calling.md                    # Function calling / tool use
│   ├── multimodal_inputs.md               # Image, video, audio inputs
│   ├── reasoning_outputs.md               # Reasoning parser (e.g. DeepSeek-R1)
│   ├── disagg_prefill.md                  # Disaggregated prefill & decode
│   ├── prompt_embeds.md                   # Prompt embedding inputs
│   └── sleep_mode.md                      # Idle weight offloading
├── usage/
│   ├── faq.md                             # Frequently asked questions
│   ├── troubleshooting.md                 # Common issues & solutions
│   ├── metrics.md                         # Prometheus metrics reference
│   ├── reproducibility.md                 # Deterministic generation
│   └── v1_guide.md                        # V1 engine migration guide
├── models/
│   ├── supported_models.md                # Full supported model list
│   ├── generative_models.md               # Text generation models
│   └── pooling_models.md                  # Embedding / pooling models
├── deployment/
│   ├── docker.md                          # Docker deployment
│   ├── k8s.md                             # Kubernetes deployment
│   └── nginx.md                           # Nginx reverse proxy
├── design/
│   ├── arch_overview.md                   # Architecture overview
│   ├── paged_attention.md                 # PagedAttention mechanism
│   ├── prefix_caching.md                  # Automatic prefix caching design
│   ├── plugin_system.md                   # Plugin architecture
│   └── ...                                # (more design docs)
├── training/
│   ├── rlhf.md                            # RLHF with vLLM
│   └── trl.md                             # TRL integration
├── cli/
│   ├── serve.md                           # `vllm serve` command
│   ├── chat.md                            # `vllm chat` command
│   ├── complete.md                        # `vllm complete` command
│   └── run-batch.md                       # `vllm run-batch` command
├── api/
│   └── README.md                          # API reference
├── contributing/
│   └── README.md                          # Contribution guide
├── community/
│   ├── contact_us.md                      # Discord, GitHub, etc.
│   └── sponsors.md                        # Sponsors
└── examples/
    └── README.md                          # Example scripts
```

---

## Quick Reference

```bash
# Serve a model (OpenAI-compatible API)
vllm serve /path/to/model --port 8101

# Common serve options:
#   --gpu-memory-utilization 0.90    GPU memory fraction (default: 0.90)
#   --max-model-len 4096             Max context length
#   --tensor-parallel-size 2         Split model across 2 GPUs
#   --enable-prefix-caching          Enable automatic prefix caching
#   --enable-lora                    Enable LoRA adapter support
#   --quantization awq               Load AWQ/GPTQ quantized model
#
# See: docs/configuration/serve_args.md for the full list.

# Offline batch inference (Python)
python -c "from vllm import LLM; llm = LLM('/path/to/model'); print(llm.generate(['Hello!']))"

# Interactive chat
vllm chat /path/to/model
```
