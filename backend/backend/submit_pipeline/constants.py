"""Cluster paths / ports / benchmark-eval config shared by the pipeline drivers.

从 rsi_master ExpOS ``expos.backends.pipelines.constants`` 移植。

仓库内的路径默认由本文件位置推导 (``<repo>/backend/backend/submit_pipeline/``),
仓库不在该位置时用 ``$RSI_MASTER_ROOT`` 覆盖; 仓库之外的第三方安装 (llama_factory
env / vllm env / slime) 一律从 ``public/`` 或各自的环境变量取。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# 仓库根: 优先 $RSI_MASTER_ROOT, 否则由本文件位置上溯 4 层
# (submit_pipeline -> backend -> backend -> <repo>)。
_ROOT = os.environ.get("RSI_MASTER_ROOT") or str(Path(__file__).resolve().parents[3])
# llama_factory_env 由 `public/` 构建 (见 README), 不在本仓库源码树内。
_LLAMA_FACTORY_ENV = f"{_ROOT}/public/llama_factory_env/bin/activate"
_LLAMA_FACTORY_PYTHON = f"{_ROOT}/public/llama_factory_env/bin/python"
_LLAMA_FACTORY_CWD = f"{_ROOT}/backend/backend/llama_factory/llama_factory"
# 训练数据预检脚本 (stage-0): 在占住 GPU 前用真实 llamafactory DatasetAttr 校验
# dataset_info.json 的列名注册。脚本缺失 = 集群 checkout 未更新 -> stage-0 降级
# 为 warning 放行 (部署滞后不该卡死整轮)。
_LLAMA_FACTORY_PREFLIGHT = f"{_ROOT}/backend/backend/llama_factory/scripts/llama_factory_data_preflight.py"
# 模板-eos 预检脚本 (stage-0b, 咨询性): 打印 turn-ender vs base eos 比对。
# 缺失同样降级 warning 放行 (部署滞后不卡轮).
_LLAMA_FACTORY_TEMPLATE_PREFLIGHT = f"{_ROOT}/backend/backend/llama_factory/scripts/llama_factory_template_preflight.py"
# serve 用的 vllm env: 本仓库 public/vllm_env_0819 (transformers 5.15.1 / vllm
# 0.27.1), 与训练 env (llama_factory_env, transformers 5.x) 同属 5.x, 读得懂训练
# 产物写的新式 rope_parameters 字段; tokenizer 格式兼容, 训练产物原样 serve。
# 注意: public/vllm_env 是旧 env (transformers 4.57.6), 只认 legacy rope_theta,
# 会把新式 rope_parameters 静默回落到默认 10000 (差 100x) —— serve 对齐守卫
# (vllm_serve_guard.py) 会在 stage-2 拒绝该组合。render.py 的注释即约定默认应
# 指向 vllm_env_0819, 此常量此前是 stale 值 (2026-08-27 修正)。
_VLLM_ENV = f"{_ROOT}/public/vllm_env_0819/bin/activate"
# serve 前对齐守卫 (stage-2 开头): 拦截 checkpoint 的 rope 配置格式与 serve env
# transformers 版本不兼容。缺失降级 warning 放行 (部署滞后不卡轮)。
_VLLM_SERVE_GUARD = f"{_ROOT}/backend/backend/llama_factory/scripts/vllm_serve_guard.py"
# vLLM 端口在同一台机器上可能被复用, 高位端口区间内随机挑, 避免并发实验互踩。
_VLLM_PORT_BASE = 18100
_VLLM_PORT_MOD = 800

# benchmark suite 根目录: $BENCHMARK_ROOT, 否则用仓库内自带的 bmk_everything/。
BENCHMARK_ROOT = os.environ.get("BENCHMARK_ROOT", "").strip() or f"{_ROOT}/bmk_everything"

# 去污 (decontamination) 运行时 stage-0 入口 —— 在 GPU 节点上用 heavy python 跑,
# 读原始 TRAIN_DATA_PATH, 剔除命中评测 query 的行, 写净化副本 + 报告 + (LF 路径)
# llama_factory_data 注册表。纯 python 脚本, 进程启动即全文加载, 无 ESTALE 风险。
_DECONTAM_SCRIPT = f"{_ROOT}/backend/backend/submit_pipeline/decontam_stage.py"

# slime (THUDM Megatron-LM) —— 只在配置层接入: 预检用 slime conda 环境解释器,
# 训练委托 backend/backend/slime/scripts/launch_sft.sh (env 覆盖), 不碰安装/源码。
# slime 是仓库之外的独立安装, 路径从环境变量取 (构建方法见
# backend/backend/slime/docs/env_construct/README.md)。未设置时渲染仍可用, 但生成的
# pipeline.sh 会在执行前断言失败并指向该文档 —— 不静默换解释器。
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
# PYTHONPATH 指向 Megatron-LM).
_SLIME_MEGATRON_LM = os.environ.get("SLIME_MEGATRON_LM", "")
_SLIME_CONVERT_TO_HF = os.environ.get("SLIME_CONVERT_TO_HF", "")

# stage-3 评测配置。rollout 恒走 benchmark suite 统一评测 (``bench_id`` 非空 ->
# eval_all 模式): 只保留 ``eval_all`` 条目。per-benchmark 分立评测的
# eval_gsm8k/eval_aime2025/eval_gpqa 等模块在本仓库不存在, 不移植。
#   eval_dir   : 评测脚本所在目录 (cd 进去再跑)
#   eval_py    : 评测入口 python 文件
#   metric_key : 从 eval/metrics.json 里提取哪个键作为最终 score
#   extra_args : 会被 build_pipeline_script 里组装的 EVAL_ARGS 整体覆盖
#   max_tokens/max_workers: 默认 0 = 交给 eval_all.py / benchmark suite 自身 config
_BENCHMARK_EVAL: dict[str, dict[str, Any]] = {
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

# 兼容旧 import (serve_eval.py 引用; eval_all 模式不使用, 恒为正数).
_DEFAULT_LIMIT = -1
# eval_all 的评测采样数默认值 —— 必须是一个正数 (eval_all.py 的 generic runner
# 把 limit 用作 `len(rows) >= limit` 的终止条件, -1 会导致一个样本都不跑)。
_DEFAULT_EVAL_ALL_LIMIT = 500

# stage-2 vLLM serve 默认值: gpu-memory-utilization=0.6 留足显存余量;
# max-model-len 默认取 eval max_tokens + 提示词余量, eval_all 模式用 32768 兜底。
# 两者都可用 ``eval_all.serve_gpu_memory_utilization`` / ``serve_max_model_len`` 覆盖。
_DEFAULT_SERVE_GPU_MEM_UTIL = 0.6
_DEFAULT_SERVE_MAX_MODEL_LEN = 32768
_SERVE_MAX_MODEL_LEN_PROMPT_SLACK = 4096
