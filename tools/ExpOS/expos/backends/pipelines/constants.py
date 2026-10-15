"""Cluster paths / ports / benchmark-eval config shared by the pipeline drivers.

从 ``expos.backends.pipeline_llama_factory_gsm8k`` 单文件平移而来, 作为
``pipelines/`` 包的常量层。

路径默认从本文件位置推导 (``<repo>/tools/ExpOS/expos/backends/pipelines/``), 仓库
不在该位置时用 ``$RSI_MASTER_ROOT`` 覆盖; 仓库之外的第三方安装 (slime 等) 通过各自
的环境变量提供 (构建方法见 ``backend/backend/slime/docs/env_construct/README.md``)。
未设置时渲染仍可用, 但生成的 pipeline.sh 会在执行前断言失败 —— 不静默换解释器。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# 仓库根: 优先 $RSI_MASTER_ROOT, 否则由本文件位置上溯 5 层
# (pipelines -> backends -> expos -> ExpOS -> tools -> <repo>)。
_ROOT = os.environ.get("RSI_MASTER_ROOT") or str(Path(__file__).resolve().parents[5])
_LLAMA_FACTORY_ENV = f"{_ROOT}/public/llama_factory_env/bin/activate"
_LLAMA_FACTORY_CWD = f"{_ROOT}/backend/backend/llama_factory/llama_factory"
# 训练数据预检脚本 (stage-0): 在占住 GPU 前用真实 llamafactory DatasetAttr 校验
# dataset_info.json 的列名注册。脚本缺失 = 集群 checkout 未更新 -> stage-0 降级
# 为 warning 放行 (部署滞后不该卡死整轮)。
_LLAMA_FACTORY_PREFLIGHT = f"{_ROOT}/backend/backend/llama_factory/scripts/llama_factory_data_preflight.py"
# 模板-eos 预检脚本 (stage-0b, H6): 在占 GPU 前拦截"训练 turn-ender != base eos"
# 的模板配置 (qwen3_nothink 等), 这是 0817 两 run 4096 退化的根因. 缺失同样降级
# warning 放行 (部署滞后不卡轮).
_LLAMA_FACTORY_TEMPLATE_PREFLIGHT = f"{_ROOT}/backend/backend/llama_factory/scripts/llama_factory_template_preflight.py"
# serve 用的 vllm env: vllm_env_0819 (transformers 5.15.1 / vllm 0.27.1), 与训练
# env (llama_factory_env, transformers 5.8.0) 同属 5.x, tokenizer 格式兼容, 训练
# 产物原样 serve, 无 tokenizer 恢复步骤。可用 eval_all 的 ``vllm_env`` 覆盖。
_VLLM_ENV = f"{_ROOT}/public/vllm_env_0819/bin/activate"
# serve 前对齐守卫 (stage-2 开头): 拦截 checkpoint 的 rope 配置格式与 serve env
# transformers 版本不兼容 (0819 六模板实验根因: 旧 env 4.57 读不懂 5.x 的
# rope_parameters -> 静默回落 rope_theta=10000 差 100x)。缺失降级 warning 放行
# (部署滞后不卡轮), 与其余 preflight 一致。
_VLLM_SERVE_GUARD = f"{_ROOT}/backend/backend/llama_factory/scripts/vllm_serve_guard.py"
# vLLM 端口在同一台机器上可能被复用, 用 experiment_id hash 挑一个稳定但唯一
# 的高位端口, 避免多个实验同时跑时 8101 相互踩踏。
_VLLM_PORT_BASE = 18100
_VLLM_PORT_MOD = 800

# slime (THUDM Megatron-LM) —— 只在配置层接入: 预检用 slime conda 环境解释器,
# 训练委托 backend/backend/slime/scripts/launch_sft.sh (env 覆盖), 不碰安装/源码。
# slime 是仓库之外的独立安装, 三个路径都从环境变量取 (构建方法见
# backend/backend/slime/docs/env_construct/README.md)。未设置时预检跳过, 训练阶段
# 会在生成脚本里断言失败并指向该文档。
_SLIME_ENV = os.environ.get("SLIME_ENV", "")
# Empty when SLIME_ENV is unset — the rendered pipeline.sh asserts on it
# and aborts with a pointer to the build docs, rather than silently
# running the wrong interpreter.
_SLIME_PYTHON = f"{_SLIME_ENV}/bin/python3" if _SLIME_ENV else ""
_SLIME_SCRIPTS_DIR = f"{_ROOT}/backend/backend/slime/scripts"
_SLIME_PREFLIGHT = f"{_SLIME_SCRIPTS_DIR}/slime_data_preflight.py"
_SLIME_LAUNCH_SFT = f"{_SLIME_SCRIPTS_DIR}/launch_sft.sh"
# slime 训练产物是 Megatron torch_dist, vLLM 不认; stage-1.5 用 slime 自带的
# convert_torch_dist_to_hf.py 离线转成 HF safetensors (slime conda env 解释器,
# slime / megatron_core 均为该 env 的 editable 安装; PYTHONPATH 指向 Megatron-LM).
_SLIME_MEGATRON_LM = os.environ.get("SLIME_MEGATRON_LM", "")
_SLIME_CONVERT_TO_HF = os.environ.get("SLIME_CONVERT_TO_HF", "")

# 各 benchmark 的 stage-3 评测配置。每个条目给出:
#   eval_dir   : 评测脚本所在目录 (cd 进去再跑, 只允许 backend/backend/eval_*)
#   eval_py    : 评测入口 python 文件
#   metric_key : 从 eval/metrics.json 里提取哪个键作为最终 score
#   extra_args : 追加到 python 命令后的固定参数 (dataset/revision/fewshot 等);
#                dataset 一律是 public/datasets 下的本地绝对路径, 不允许远程 HF.
#   max_tokens/max_workers: 该 benchmark 的评测默认值 (对齐各自 eval.sh).
#   judge_env  : arena_hard_writing / healthbench_easy 需要 LLM judge
#                (OPENAI_BASE_URL + OPENAI_API_KEY 指向 judge sidecar).
#                渲染侧如果已有这两个环境变量, 就把它们 export 进 pipeline.sh,
#                供评测脚本的 openai SDK 使用 (生成仍走本地 vLLM BASE_URL).
#                eval_all 模式 (bench_id 非空) 按 bench_id 自动匹配这里的
#                judge_env, 也可用 config ``evaluation.eval_all.judge_env``
#                显式覆盖; 凭据放 ``evaluation.eval_all.judge.{base_url,api_key}``.
# 常量与 backend/backend/{llama_factory, vllm, eval_*}/scripts 里的绝对路径保持一致。
_BENCHMARK_EVAL: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_gsm8k",
        "eval_py": "gsm8k_eval.py",
        "metric_key": "accuracy",
        "extra_args": (
            f'--dataset "{_ROOT}/public/datasets/gsm8k" '
            f'--revision "cc7b047b6e5bb11b4f1af84efc572db110a51b3c" '
            f'--fewshot 10'
        ),
        "max_tokens": 3000,
        "max_workers": 30,
        "judge_env": False,
    },
    "aime_2025": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_aime2025",
        "eval_py": "aime2025_eval.py",
        "metric_key": "accuracy",
        "extra_args": f'--dataset "{_ROOT}/public/datasets/aime2025"',
        "max_tokens": 16000,
        "max_workers": 10,
        "judge_env": False,
    },
    "gpqa_main": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_gpqa",
        "eval_py": "gpqa_eval.py",
        "metric_key": "accuracy",
        "extra_args": f'--dataset "{_ROOT}/public/datasets/gpqa"',
        "max_tokens": 16000,
        "max_workers": 6,
        "judge_env": False,
    },
    "bfcl": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_bfcl",
        "eval_py": "bfcl_eval.py",
        "metric_key": "accuracy",
        "extra_args": "",
        "max_tokens": 16000,
        "max_workers": 6,
        "judge_env": False,
    },
    "human_eval": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_humaneval",
        "eval_py": "humaneval_eval.py",
        "metric_key": "pass_at_1",
        "extra_args": "",
        "max_tokens": 4000,
        "max_workers": 4,
        "judge_env": False,
    },
    "arena_hard_writing": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_arenahardwriting",
        "eval_py": "arenahard_eval.py",
        "metric_key": "accuracy",
        "extra_args": "",
        "max_tokens": 16384,
        "max_workers": 10,
        "judge_env": True,
    },
    "healthbench_easy": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_healthbench",
        "eval_py": "healthbench_eval.py",
        "metric_key": "accuracy",
        "extra_args": "",
        "max_tokens": 16384,
        "max_workers": 10,
        "judge_env": True,
    },
    # 统一评测 backend: 当 ``request.bench_id`` (benchmark suite bench_id) 非空时,
    # stage-3 从上面这些 per-benchmark 分立评测切换到 eval_all.py, 由 benchmark suite
    # 框架按 bench_id 评测任意 bench。``extra_args`` 会被 build_pipeline_script 里
    # 组装的 EVAL_ARGS 整体覆盖 (bench-id / split / trial-name / bmk-root)。
    # max_tokens/max_workers 默认 0 = 交给 eval_all.py / benchmark suite 自身 config;
    # eval_all 的 limit 由 eval_all config 计算, 永不传 -1。
    "eval_all": {
        "eval_dir": f"{_ROOT}/backend/backend/eval_all",
        "eval_py": "eval_all.py",
        "metric_key": "accuracy",
        "extra_args": "",
        "max_tokens": 0,
        "max_workers": 0,
        "judge_env": False,
    },
}

# The benchmark suite's canonical id omits the underscores.  Keep the historical
# per-benchmark name above for backwards compatibility, but make eval_all's
# automatic judge detection resolve the canonical id as well.  Without this
# alias a rendered Arena pipeline silently omitted OPENAI_* judge credentials
# and failed only after training and generation had completed.
_BENCHMARK_EVAL["arenahardwriting"] = _BENCHMARK_EVAL["arena_hard_writing"]

_DEFAULT_LIMIT = -1
# eval_all 的评测采样数默认值 —— 必须是一个正数 (run_eval.py 的 generic runner
# 把 limit 用作 `len(rows) >= limit` 的终止条件, -1 会导致一个样本都不跑)。
_DEFAULT_EVAL_ALL_LIMIT = 500

# stage-2 vLLM serve 默认值 (任务 #79)。实测: 共享 H20 上默认 gpu_memory_utilization
# 0.9 要 125 GiB, 而并发训练后只剩 ~97 GiB free -> serve 必 OOM; 默认 max-model-len
# 8192 < AIME eval 的 max_tokens 16000 -> "max_tokens too large" 400 错误。所以:
#   - gpu-memory-utilization 默认 0.6 (留足余量, 单个 1.7B~7B serve 足够)。
#   - max-model-len 默认取 eval 的 max_tokens + 提示词余量; eval_all 模式
#     (max_tokens 未知, 交给 benchmark suite) 按模型尺寸兜底: 大模型 (35B/MoE,
#     原生 256K) 用 262144, 小模型 (4B/8B, 原生 32K) 用 32768。
# 两者都可用 config ``evaluation.eval_all.serve_gpu_memory_utilization`` /
# ``serve_max_model_len`` 覆盖 (大模型如 32B teacher 需要显式调低 util 或调高 len)。
_DEFAULT_SERVE_GPU_MEM_UTIL = 0.6
_DEFAULT_SERVE_MAX_MODEL_LEN = 32768            # 小模型兜底 (= Qwen3 原生 32K)
_DEFAULT_SERVE_MAX_MODEL_LEN_LARGE = 262144     # 大模型兜底 (= Qwen3.5-35B-A3B 原生 256K)
_SERVE_MAX_MODEL_LEN_PROMPT_SLACK = 4096
