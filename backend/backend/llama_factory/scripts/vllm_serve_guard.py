#!/usr/bin/env python3
"""vLLM serve 前对齐守卫 —— checkpoint 的 rope 配置格式 vs serve env 的 transformers 版本。

0819 六模板实验定位出的根因: LLaMA-Factory (transformers 5.8) 把 checkpoint 的
RoPE base 写进**新式**字段 ``rope_parameters`` (如 ``{"rope_theta": 1000000}``);
旧 serve env (transformers 4.57 / vllm 0.11) 只认**旧式**字段 ``rope_theta``,
读不懂 ``rope_parameters`` -> **静默**回落默认 ``rope_theta=10000`` (差 100x),
位置编码全错 -> attention / 停止行为全变 -> 同一 checkpoint 在两个 env 下得到
看似正常实则错误的分数 (同批数据: 0.626 vs 0.01, 且 "好的" 那个其实是错 rope
的 artifact)。

本脚本在 vllm serve 启动前 (stage-2 开头) 拦截这类不兼容: 用 **serve env 自己的
python** 读 checkpoint 的 config.json 并自报 transformers 版本, 判定格式是否
可被正确读取, 不兼容就 **fail loudly** (不静默 serve 错模型)。对齐通过时打印
生效的 rope_theta, 供 pipeline log 审计。

判定规则:
  - config.json 缺失                          -> 错误 (2)
  - checkpoint 有 ``rope_parameters`` (5.x 格式):
      * serve env transformers >= 5.0        -> 通过 (5.x 读 ``rope_parameters``)
      * serve env transformers <  5.0        -> 拒绝 (1)  <- 0819 事故根因
  - checkpoint 只有旧式 ``rope_theta`` (4.x 格式):
      * 任何版本都读一致                      -> 通过; transformers < 5 时打一行提示
  - 两者都没有                                 -> 通过 + WARN (将用加载器默认, 需人工确认)

退出码: 0 通过 / 1 拒绝 (不兼容) / 2 配置或环境错误。

用法 (stage-2, serve env 已 activate):
  python vllm_serve_guard.py --serve-dir <checkpoint-* 目录>
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _serve_env_transformers_version() -> tuple[int, ...] | None:
    """返回当前 python (即 serve env) 的 transformers 主/次版本, 不可导入则 None."""
    try:
        import transformers  # noqa: PLC0415

        return tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:
        return None


def _parse_version(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    return tuple(int(x) for x in parts[:2])


def _load_config(serve_dir: str) -> dict:
    cfg_path = os.path.join(serve_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"checkpoint 缺少 config.json: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config.json 不是 JSON object: {cfg_path}")
    return cfg


def _rope_fields(cfg: dict) -> tuple[dict | None, object]:
    """返回 (rope_parameters, 旧式 rope_theta)."""
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict):
        return rp, cfg.get("rope_theta")
    return None, cfg.get("rope_theta")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve-dir", required=True, help="待 serve 的 checkpoint 目录")
    parser.add_argument(
        "--transformers-version",
        default=None,
        help="显式指定 serve env 的 transformers 版本 (测试/诊断用; 默认自动探测)",
    )
    args = parser.parse_args()

    ver = (
        _parse_version(args.transformers_version)
        if args.transformers_version
        else _serve_env_transformers_version()
    )
    try:
        cfg = _load_config(args.serve_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[serve_guard] ERROR: {exc}", file=sys.stderr)
        return 2

    rp, legacy = _rope_fields(cfg)
    ver_str = ".".join(map(str, ver)) if ver else "未知(transformers 不可导入)"
    print(f"[serve_guard] serve env transformers = {ver_str}")
    print(f"[serve_guard] checkpoint: rope_parameters={rp!r} legacy rope_theta={legacy!r}")

    # 5.x 格式字段存在 -> 必须由 transformers>=5 的 serve env 读取 (0819 根因).
    if isinstance(rp, dict) and "rope_theta" in rp:
        eff = rp["rope_theta"]
        if ver is None:
            print("[serve_guard] ERROR: 无法读取 serve env 的 transformers 版本, 拒绝继续", file=sys.stderr)
            return 2
        if ver >= (5, 0):
            print(f"[serve_guard] OK: rope_parameters.rope_theta={eff} (5.x 格式) 由 transformers {ver_str} 正确读取")
            return 0
        print(
            "[serve_guard] REJECT: checkpoint 使用 transformers-5.x 的 rope_parameters 字段 "
            f"(rope_theta={eff}), 但 serve env 是 transformers {ver_str}, 只认旧字段 rope_theta, "
            "会**静默回落默认 10000** (差 100x, 0819 六模板实验根因). "
            "请改用 transformers>=5 的 serve env (public/vllm_env_0819).",
            file=sys.stderr,
        )
        return 1

    # 只有旧式字段 (4.x 格式) / 或两者都没有: 各版本读取一致, 放行.
    if legacy is not None:
        print(f"[serve_guard] OK: 旧式 rope_theta={legacy} 由 transformers {ver_str} 读取")
    else:
        print(
            f"[serve_guard] WARN: checkpoint 没有 rope_theta / rope_parameters, "
            f"将使用加载器默认值 (transformers {ver_str}); 请确认与训练 env 一致"
        )
    if ver is not None and ver < (5, 0):
        print(
            "[serve_guard] NOTE: serve env 是 transformers <5 (旧 env), checkpoint 为 4.x 格式 "
            "故读取一致; 但新 checkpoint 一律走 public/vllm_env_0819 以保持一致。"
        )
    print("[serve_guard] OK: serve env 能正确读取 checkpoint 的 rope 配置")
    return 0


if __name__ == "__main__":
    sys.exit(main())
