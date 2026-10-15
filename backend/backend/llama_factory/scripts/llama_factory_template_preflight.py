#!/usr/bin/env python3
"""LLaMA-Factory template-eos preflight check —— 2026-08-19 实证后为**咨询性**
(不再硬拒绝模板)。

历史: 这份检查曾是 H6 硬不变式 "训练模板 turn-ender == base eos" 的闸门, 拦的是
0817 两 run (CTI/Earth) 的核心测量 bug。当时推理: qwen3/qwen3_nothink 模板
(turn-ender = ``<|im_end|>`` 151645) 训 base Qwen3-4B-Base (eos =
``<|endoftext|>`` 151643) 时, LLaMA-Factory 的 ``fix_special_tokens`` 只把模板
eos 写进 tokenizer、从不写 model.config / generation_config, 训练产物自身三处
eos 分叉, 而 vLLM 从 serve 的 tokenizer/config 建 stop 集合时可能不含 151645 ->
模型学到发 151645 却不被识别为 stop -> 每 sample 跑满 max_tokens ->
last-ANSWER grader 读到末尾噪声, 系统性压分。

**0819 六模板对比实验推翻了这个硬不变式**: 同一 serve 路径下, ``default`` /
``qwen3_nothink`` / ``qwen3`` × full-SFT **三个模板全部 200/200 stop、0.80-0.83**。
(同实验的 LoRA rank16/1ep 三模板全部退化 length loop —— 这组对比正是 LoRA 下线
2026-08-20 的依据之一, 不再作为可选项。) 差别是训练容量, **不是模板**。
原因: 当前 serve env (``vllm_env_0819`` = vllm 0.27 + ``--generation-config
vllm``) 无条件把 **tokenizer.eos** 加进 stop 集合、并**完全忽略** checkpoint 的
generation_config。于是 tokenizer(config 分叉 (im_end vs endoftext) **无害**:
只要 tokenizer.eos 是词表里的真实 token (模型能发), 模型发出来 vllm 就停。
→ 模板不再需要等于 base eos, qwen3 / qwen3_nothink 都是**合法选择**。

因此本检查降级为: **解析模板、算出 turn-ender, 与 base eos 比对后打印**
—— 对齐就报 OK; 分叉就报预期分裂 (并提示训后用 ``check_eos_alignment.py``
自证 tokenizer.eos 在词表、serve 路径下无害)。**不再 exit 1 拒绝任何模板**。
仍保留的拦截是配置/环境错误 (exit 2): 读不到 YAML / 模板不在 registry /
llamafactory 不可导入 / base eos 无法解析。

训练侧真问题 (比模板对齐重要得多):
  1. LoRA 已下线 (2026-08-20), 全参 SFT 是唯一训练方式; 数据/epochs/容量不足
     的模型在长 few-shot 评测上下文里 **不发任何 stop token** -> length loop。
     0819 实证 full-SFT 三模板 200/200 stop (0.80-0.83)。
  2. 发现循环输出 (finish_reason=length 主导 / 复读) 时, 先跑
     ``scripts/check_eos_alignment.py --checkpoint-dir <ckpt> --samples
     samples.jsonl`` 自查: tokenizer.eos 在词表 -> 训练侧; 不在 -> infra。

判定来源与真实训练同源: 用 ``llamafactory.data.template.TEMPLATES`` (vendored
source 的 editable install, 与 stage-1 的 ``llamafactory-cli train`` 同一份
registry) 解析模板, 从 ``format_assistant`` 的最后一个 slot 提取回合末 token,
与 base 模型的 eos 字符串比对。

退出码:
  0  通过 (含"turn-ender != base eos"的**预期分裂** —— 打印 NOTE 不拒绝)
  2  配置/环境错误 (读不到 YAML / 模板不在 registry / llamafactory 不可导入 /
     base eos 无法解析)

用法:
  python llama_factory_template_preflight.py \
      --train-config <train.yaml>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# format_assistant slot 里的占位符 dict ``{"eos_token"}`` —— 语义是"追加
# tokenizer 当前的 eos" (即 base eos, 对齐成立)。
_SENTINEL_BASE_EOS = "<__base_tokenizer_eos__>"

# Qwen 风格回合末 token 的形状: "<|im_end|>" "<|endoftext|>" 等。
_SPECIAL_TOKEN_RE = re.compile(r"(<\|[^|<]+\|>)\s*$")


class ConfigError(Exception):
    """配置/环境不可读 (exit 2)。"""


def _load_training_config(path: Path) -> dict:
    try:
        import yaml

        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"cannot read training config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"training config {path} is not a YAML mapping")
    return cfg


def _base_eos_string(base_dir: Path) -> str:
    """读 base 模型的 eos token 字符串。

    优先级: tokenizer_config.json eos_token -> special_tokens_map.json eos_token
    -> config/generation_config eos_token_id -> (added_tokens_decoder / vocab)
    反解出 content。读不到返回 ""。
    """
    tc_path = base_dir / "tokenizer_config.json"
    tc = {}
    if tc_path.is_file():
        try:
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tc = {}

    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        f = base_dir / name
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        v = d.get("eos_token")
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str) and v:
            return v

    # eos_token_id -> content
    eid = None
    for name in ("generation_config.json", "config.json"):
        f = base_dir / name
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        eid = d.get("eos_token_id")
        if eid is not None:
            break
    if eid is None:
        return ""
    atd = tc.get("added_tokens_decoder") or {}
    info = atd.get(str(eid))
    if isinstance(info, dict) and info.get("content"):
        return str(info["content"])
    tok_path = base_dir / "tokenizer.json"
    if tok_path.is_file():
        try:
            tok = json.loads(tok_path.read_text(encoding="utf-8"))
            vocab = tok.get("model", {}).get("vocab", {})
            rev = {v: k for k, v in vocab.items()}
            token = rev.get(int(eid))
            if token:
                return str(token)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return ""


def _assistant_terminator(template) -> Optional[str]:
    """提取 assistant 回合结束 token 的字符串。

    规则 (从最后一个 slot 往回收):
      - dict ``{"eos_token"}``        -> 哨兵 ``_SENTINEL_BASE_EOS`` (= base eos)
      - dict ``{"token": "..."}``     -> 该字面 token
      - str 尾部是 ``<|...|>``        -> 该字面 token
      - 其它 (callable / 纯文本)      -> None (无法静态判定)
    注意只认 Qwen 风格的 ``<|...|>`` 包裹 token; 其它风格 (如 chatglm 的
    ``<|assistant|>``) 不在本期护栏内, 返回 None -> warning 放行。
    """
    formatter = template.format_assistant
    slots = getattr(formatter, "slots", None)
    if not slots:
        return None
    for slot in reversed(slots):
        if isinstance(slot, dict):
            if "eos_token" in slot:
                return _SENTINEL_BASE_EOS
            if "token" in slot:
                return str(slot["token"])
            continue
        if isinstance(slot, str):
            stripped = slot.rstrip()
            m = _SPECIAL_TOKEN_RE.search(stripped)
            if m:
                return m.group(1)
            if stripped:
                return None  # 尾部是普通文本, 不是单 token
            continue
        continue  # callable / 其它 slot 类型
    return None


def check_invariant(template_name: str, template, base_eos: str) -> None:
    """核心判定: 解析 turn-ender 与 base eos 比对, 打印**咨询性**结论。

    不再拒绝任何模板 (0819 实证: 分叉在当前 serve 路径下无害)。对齐 -> OK;
    分叉 -> NOTE 说明这是预期分裂、训后用 check_eos_alignment.py 自证无害。
    """
    term = _assistant_terminator(template)
    if term == _SENTINEL_BASE_EOS:
        term = base_eos
    if term is None:
        print(
            "WARN: cannot statically determine assistant turn-ender from "
            "format_assistant (non Qwen-style slot); skipping turn-ender note. "
            "For Qwen3 base models `default` / `qwen3` / `qwen3_nothink` are all "
            "valid — see the 0819 results.",
            file=sys.stderr,
        )
        return
    if term == base_eos:
        print(
            f"template-eos aligned: {template_name!r} turn-ender "
            f"({term!r}) == base eos. Checkpoint will be self-consistent.",
            file=sys.stderr,
        )
    else:
        print(
            f"NOTE (expected split, NOT a rejection): {template_name!r} "
            f"turn-ender {term!r} != base eos {base_eos!r}. LLaMA-Factory writes "
            "the template eos only into the trained tokenizer; config / "
            "generation_config keep the base eos. Under the current serve path "
            "(vllm_env_0819 = vllm 0.27 + `--generation-config vllm`) this split "
            "is BENIGN: vllm adds tokenizer.eos to the stop set unconditionally "
            "(`--generation-config vllm` only neutralizes sampling defaults; the "
            "checkpoint generation_config's eos ids are still unioned in, which "
            "merely adds extra stop tokens), so as long as tokenizer.eos is a "
            "real vocab token (model can emit it) generation stops the moment it "
            "is emitted. Verified 2026-08-19: qwen3 / "
            "qwen3_nothink / default x full-SFT all 200/200 stop, 0.80-0.83.",
            file=sys.stderr,
        )
        print(
            "  Post-training self-check: python "
            "scripts/check_eos_alignment.py --checkpoint-dir <ckpt> "
            "-> it confirms tokenizer.eos is emit-able (then any length-loop is "
            "training-side; LoRA 已下线, 全参 SFT 唯一 —— 提 rank/epochs/数据质量)。",
            file=sys.stderr,
        )
    if template.replace_eos and template.stop_words:
        sw = template.stop_words[0]
        if sw != base_eos:
            print(
                f"  NOTE: replace_eos=True writes tokenizer.eos={sw!r} "
                f"(stop_words[0]); config/generation_config keep {base_eos!r}. "
                "Same benign split as above under the current serve path.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, help="training YAML path")
    args = parser.parse_args(argv)

    train_cfg = _load_training_config(Path(args.train_config))

    ft = str(train_cfg.get("finetuning_type") or "").strip().lower()
    if ft in ("lora", "qlora"):
        # advisory only (exit 0, 不 block): 提交/渲染层已硬拒 lora/qlora, 这里
        # 只提前提示, 让 agent 在 GPU 入队前改 YAML。
        print(
            f"WARN: train YAML finetuning_type={ft!r} is decommissioned "
            "(2026-08-20, LoRA 停用) — the submit side will hard-reject this. "
            "Use finetuning_type: full (full-parameter SFT).",
            file=sys.stderr,
        )

    template_name = str(train_cfg.get("template") or "").strip()
    if not template_name:
        print(
            "WARN: train YAML has no `template` key (LLaMA-Factory auto-detect). "
            "Cannot statically verify the turn-ender. 0819 实证后 default / "
            "qwen3 / qwen3_nothink 都是合法选择; 显式写 template 可让本预检给出 "
            "turn-ender 与 base eos 的比对说明.",
            file=sys.stderr,
        )
        return 0

    base_model = str(train_cfg.get("model_name_or_path") or "").strip()
    base_dir = Path(base_model)
    if not base_dir.is_dir():
        print(
            f"WARN: base model {base_model!r} is not a local directory; cannot "
            "read its eos token to verify the invariant. Cluster runs use local "
            "model paths.",
            file=sys.stderr,
        )
        return 0

    base_eos = _base_eos_string(base_dir)
    if not base_eos:
        raise ConfigError(
            f"cannot resolve eos token of base model {base_dir} "
            "(no tokenizer_config.json/special_tokens_map.json with eos_token, "
            "and no config/generation_config eos_token_id)"
        )

    try:
        from llamafactory.data.template import TEMPLATES  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "llamafactory is not importable in this python; run under the "
            "llama_factory_env (source public/llama_factory_env/bin/activate)"
        ) from exc

    template = TEMPLATES.get(template_name)
    if template is None:
        raise ConfigError(
            f"template {template_name!r} is not registered in this llamafactory "
            f"registry (have: {sorted(TEMPLATES)}). stage-1 would fail the same "
            "way, so this is caught before GPU launch."
        )

    check_invariant(template_name, template, base_eos)

    print(
        f"template-eos preflight OK: template={template_name} base_eos={base_eos!r} "
        f"turn-ender={'<sentinel: base eos>' if _assistant_terminator(template) == _SENTINEL_BASE_EOS else _assistant_terminator(template)!r}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"TEMPLATE-EOS PREFLIGHT CONFIG ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
