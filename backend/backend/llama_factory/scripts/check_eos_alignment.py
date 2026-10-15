#!/usr/bin/env python3
"""检查训练产物 (ckpt) 的 eos 对齐 + 可选扫描 eval samples 的循环信号。

何时用: 评测结果出现**循环输出** (finish_reason=length 主导 / response 极长 /
复读) 时, 先自检 eos 是否对齐, 再判断循环是"训练侧没学会发 stop token"还是
"infra 真坏了"。

检查两个层面:

1. **ckpt 三处 eos 是否一致** (tokenizer / config / generation_config)。0819 六模板
   实验实证: LLaMA-Factory 的 ``replace_eos=True`` 模板 (qwen3 / qwen3_nothink) 只会
   把模板 turn-ender 写进 **tokenizer** (``eos_token=<|im_end|>`` 151645), config /
   generation_config 仍是 base eos (``<|endoftext|>`` 151643)。这个"分叉"在
   **当前 serve 路径** (vllm_env_0819 = vllm 0.27 + ``--generation-config vllm``) 下
   **无害**: vLLM 每次请求都**无条件**把 tokenizer.eos 加进 stop set
   (``input_processor`` -> ``_all_stop_token_ids.add(tokenizer.eos)``), 而模型在训练里
   被教会发的 turn-ender 正是 tokenizer.eos (模板 replace_eos) —— 模型一发出来 vllm
   就停。``--generation-config vllm`` 只中和**采样默认值** (temperature/top_p/
   max_new_tokens); checkpoint generation_config 的 eos ids 仍会被 union 进 stop 集,
   但那只是多几个 stop token, 无害。真正会坏的是: tokenizer.eos 的 token **不在词表里**
   (模型根本发不出自己的终止符) -> 必 loop。

2. **(可选) --samples** 给定 eval 的 samples.jsonl, 统计 finish_reason 分布 / im_end
   出现 / 复读特征, 给出循环诊断。结合第 1 层: 若 tokenizer.eos 可发但 samples 仍
   length 主导 -> **训练侧** (数据/epochs/容量没教会模型在目标上下文发 stop token),
   不是 infra。

退出码: 0 对齐 (或可发的良性分叉) / 1 真实问题 (tokenizer.eos 不在词表 -> 必 loop;
或 samples 扫描确认循环) / 2 配置/文件错误。

用法 (任意 python, 不依赖 llamafactory):
  python scripts/check_eos_alignment.py --checkpoint-dir <ckpt>
  python scripts/check_eos_alignment.py --checkpoint-dir <ckpt> --samples <samples.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

IM_END = "<|im_end|>"


class CheckError(Exception):
    """配置/文件错误 (exit 2)。"""


def _load_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"cannot read {path}: {exc}") from exc
    if not isinstance(d, dict):
        raise CheckError(f"{path} is not a JSON object")
    return d


def _tokenizer_eos_id(cp_dir: Path, tok_cfg: dict, added_decoder: dict) -> int | None:
    """从 tokenizer_config 反解 eos_token 字符串对应的 id (0 也在意).

    merged ckpt 常缺 added_tokens_decoder, 所以字符串也要落到 tokenizer.json 的
    added_tokens 数组反查.
    """
    content = tok_cfg.get("eos_token")
    if isinstance(content, dict):
        content = content.get("content")
    if not isinstance(content, str) or not content:
        return None
    # added_tokens_decoder: {id: {content, ...}} -> 反查
    for sid, info in (added_decoder or {}).items():
        if isinstance(info, dict) and info.get("content") == content:
            return int(sid)
    # tokenizer.json.added_tokens: [{"id": ..., "content": ...}] -> 反查
    tok_path = cp_dir / "tokenizer.json"
    if tok_path.is_file():
        try:
            tok = json.loads(tok_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tok = None
        if tok:
            for a in tok.get("added_tokens") or []:
                if isinstance(a, dict) and a.get("content") == content:
                    return int(a["id"])
    # tokenizer_config 自带的 eos_token_id (部分 tokenizer 写)
    eid = tok_cfg.get("eos_token_id")
    if isinstance(eid, int):
        return eid
    return None


def _id_to_token(cp_dir: Path, eid, added_decoder: dict) -> str | None:
    """id -> token 字符串. 合并三个来源 (LLaMA-Factory 合并后各来源都可能残缺):
    tokenizer_config.added_tokens_decoder -> tokenizer.json.added_tokens
    -> tokenizer.json.model.vocab 反查. 只查 tokenizer_config 会漏 (merged ckpt
    常常把 added_tokens_decoder 丢掉), 只查 model.vocab 也会漏 (im_end 等特殊
    token 走 added_tokens, 不进 vocab). 兼容 eos_token_id 是 int 或 list."""
    if eid is None:
        return None
    # list (generation_config 允许多个 eos) -> 只显示第一个
    if isinstance(eid, list):
        if not eid:
            return None
        eid = eid[0]
    info = (added_decoder or {}).get(str(eid))
    if isinstance(info, dict) and info.get("content"):
        return str(info["content"])
    tok_path = cp_dir / "tokenizer.json"
    if not tok_path.is_file():
        return None
    try:
        tok = json.loads(tok_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # added_tokens: [{"id": ..., "content": ...}, ...]
    for a in tok.get("added_tokens") or []:
        if isinstance(a, dict) and a.get("id") == int(eid) and a.get("content"):
            return str(a["content"])
    # model.vocab: {token: id}
    vocab = tok.get("model", {}).get("vocab", {})
    rev = {v: k for k, v in vocab.items()}
    return rev.get(int(eid))


def check_checkpoint(cp_dir: Path) -> int:
    tok_cfg = _load_json(cp_dir / "tokenizer_config.json")
    added_decoder = tok_cfg.get("added_tokens_decoder") or {}
    cfg = _load_json(cp_dir / "config.json")
    gen_cfg = _load_json(cp_dir / "generation_config.json")

    tok_eos_str = tok_cfg.get("eos_token")
    if isinstance(tok_eos_str, dict):
        tok_eos_str = tok_eos_str.get("content")
    cfg_eos_id = cfg.get("eos_token_id")
    gen_eos_id = gen_cfg.get("eos_token_id")
    # 兼容 list 形式的 eos_token_id: 取第一个作为"对齐比较"基准
    if isinstance(cfg_eos_id, list):
        cfg_eos_id = cfg_eos_id[0] if cfg_eos_id else None
    if isinstance(gen_eos_id, list):
        gen_eos_id = gen_eos_id[0] if gen_eos_id else None

    tok_eos_id = _tokenizer_eos_id(cp_dir, tok_cfg, added_decoder)
    tok_eos_decoded = _id_to_token(cp_dir, tok_eos_id, added_decoder)

    print("== ckpt eos 三处 ==")
    print(f"  tokenizer_config   eos_token   = {tok_eos_str!r}")
    print(f"  config             eos_token_id = {cfg_eos_id!r} -> {_id_to_token(cp_dir, cfg_eos_id, added_decoder)!r}")
    print(f"  generation_config  eos_token_id = {gen_eos_id!r} -> {_id_to_token(cp_dir, gen_eos_id, added_decoder)!r}")
    print(f"  (tokenizer eos id = {tok_eos_id!r} -> {tok_eos_decoded!r})")

    # 模型能不能发出 tokenizer.eos? 关键判定.
    emit_able = tok_eos_decoded is not None
    if not emit_able:
        print(
            "FAIL: tokenizer.eos 对应的 token 不在词表 (added_tokens_decoder / "
            "tokenizer.json vocab 都查不到) -> 模型**根本发不出**自己的终止符, "
            "生成永远停不下来. 这是 infra 级问题: 检查训练是否 resize_vocab / "
            "模板 fix_special_tokens 是否把 eos 加进了词表.",
            file=sys.stderr,
        )
        return 1

    split = cfg_eos_id != tok_eos_id or gen_eos_id != tok_eos_id
    if split:
        print(
            "NOTE: tokenizer 与 config/generation_config 的 eos 分叉 "
            f"(tokenizer={tok_eos_decoded!r} vs config={cfg_eos_id!r}). 这是 "
            "replace_eos=True 模板 (qwen3/qwen3_nothink) 的正常产物. 在**当前 serve "
            "路径** (vllm_env_0819 = vllm 0.27 + `--generation-config vllm`) 下**无害**: "
            "vllm 无条件把 tokenizer.eos 加入 stop set (`--generation-config vllm` 只中和"
            "采样默认值, gen_config 的 eos ids 仍会被 union —— 多几个 stop token, 无害), "
            "且该 token 在词表里、模型能发."
        )
    else:
        print("OK: tokenizer / config / generation_config 三处 eos 一致.")

    print(
        "结论: tokenizer.eos 可发 -> 模型**能**停止. 若评测仍 length 主导, 循环是"
        "**训练侧** (数据/epochs/容量没教会模型在目标上下文发 stop token), 不是 infra."
    )
    return 0


def check_samples(path: Path) -> int:
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise CheckError(f"{path}: line {i + 1} is not valid JSON")
    if not rows:
        raise CheckError(f"{path}: empty samples.jsonl")
    n = len(rows)
    reasons = collections.Counter(r.get("finish_reason", "?") for r in rows)
    lens = [len(r.get("response", "")) for r in rows]
    n_im0 = sum(1 for r in rows if r.get("response", "").count(IM_END) == 0)
    # 复读特征: "ANSWER:" 出现 >=3 次, 说明在循环输出答案块
    n_repeat = sum(1 for r in rows if r.get("response", "").count("ANSWER:") >= 3)
    med = sorted(lens)[n // 2]

    print("== eval samples 循环信号 ==")
    print(f"  n             = {n}")
    print(f"  finish_reason = {dict(reasons)}")
    print(f"  response 长度 = med {med} / max {max(lens)} (chars)")
    print(f"  im_end=0      = {n_im0}/{n}")
    print(f"  复读(ANSWER x>=3) = {n_repeat}/{n}")

    length_frac = reasons.get("length", 0) / n
    if length_frac > 0.5 and n_im0 == n:
        print(
            "FAIL: length 主导且所有样本都没发 im_end -> 模型在评测上下文里没发 "
            "stop token. 结合 ckpt 检查: 若 tokenizer.eos 可发 (上一节), 这是训练侧 "
            "问题 (数据/epochs/容量不够 / 数据格式与评测上下文差太远), 不是 eos 对齐.",
            file=sys.stderr,
        )
        return 1
    if length_frac > 0.5:
        print(
            "WARN: length 主导但部分样本发了 im_end -> 停止习惯不稳定, 训练侧倾向 "
            "(加 epochs / 提数据质量; LoRA 已下线, 全参 SFT 唯一)."
        )
    else:
        print("OK: stop 主导, 未见明显循环.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="ckpt 目录")
    parser.add_argument(
        "--samples",
        default=None,
        help="可选: eval 的 samples.jsonl (benchmark suite output), 扫描循环信号",
    )
    args = parser.parse_args(argv)

    cp_dir = Path(args.checkpoint_dir)
    if not cp_dir.is_dir():
        raise CheckError(f"checkpoint dir not found: {cp_dir}")
    for name in ("tokenizer_config.json", "config.json", "generation_config.json"):
        if not (cp_dir / name).is_file():
            raise CheckError(f"{cp_dir} 缺 {name} (不是有效 checkpoint)")

    rc = check_checkpoint(cp_dir)
    if args.samples:
        if not Path(args.samples).is_file():
            raise CheckError(f"samples file not found: {args.samples}")
        rc = rc or check_samples(Path(args.samples))
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as exc:
        print(f"EOS-CHECK CONFIG ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
