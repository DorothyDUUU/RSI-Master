#!/usr/bin/env python3
"""LLaMA-Factory training-data preflight check.

在**任何 GPU 训练启动前**运行, 拦截数据集注册错误 —— 这些错误原本要等
``llamafactory-cli train`` 真正跑起来(占住一张卡)才暴露:

  mode 1  ``dataset_info.json`` 的 ``columns:`` 键写成了行字段名而不是
          LLaMA-Factory 读的内部名(例如 ``instruction/input/output`` 而应
          ``prompt/query/response``)。未知键被静默忽略 -> prompt/query/response
          全为 None -> 每行 user 内容为空 -> stage-1 报 "Cannot find valid
          samples"。
  mode 2  映射到的字段在数据行里不存在(例如 ``columns: {"query": "input"}``
          但文件没有 ``input`` 键) -> 转换器在第一行就抛 ``KeyError: 'input'``。

检查与真实训练同源: 用 ``llamafactory.data.parser.get_dataset_list`` 解析
``dataset_info.json``(列名映射语义与训练完全一致), 再读数据文件逐行验证
每个映射字段都存在且内容非空。

判定策略 (2026-08-20 放宽, 与产品指示一致 —— 结构拦截保留, 行级降级为告警):
  - **结构拦截 (exit 1, 硬拒)**: C1 ``columns``/``tags`` 键不是 LLaMA-Factory
    内部名; C2 必需内部列为 None; C3 映射字段在**全部**行缺失 (整体缺失必
    KeyError); sharegpt 消息 role/content tag 在**全部**消息缺失。
  - **行级告警 (WARN, 不拒绝)**: C3 映射字段仅**部分**行缺失; C4 内容字段
    全空 (stage-1 会自行报 "Cannot find valid samples"); 数据文件为空。
  - 硬拒绝的 LoRA 相关检查放在提交/渲染层 (``WorkerService.submit`` /
    ``build_pipeline_script``), 不在此 preflight 里做 —— 保持脚本宽松。

退出码:
  0  数据通过(结构合法; 行级问题只打印 WARN)
  1  预检拒绝该数据(结构拦截, pipeline 在 GPU 入队前标 failed)
  2  配置/环境错误(读不到 YAML / registry / 数据文件)

用法:
  python llama_factory_data_preflight.py \
      --train-config <train.yaml> --dataset-dir <含 dataset_info.json 的目录>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# LLaMA-Factory 的 DatasetAttr 从 ``columns`` 里只读这些内部键 —— 任何其它键
# 都会被静默忽略(见 llamafactory/data/parser.py::DatasetAttr.join)。这里的
# 列表必须与该版本保持一致: vendored 0.9.6.dev0。
_VALID_COLUMN_KEYS = frozenset(
    {
        "prompt",
        "query",
        "response",
        "history",
        "messages",
        "system",
        "tools",
        "images",
        "videos",
        "audios",
        "chosen",
        "rejected",
        "kto_tag",
    }
)
_VALID_TAG_KEYS = frozenset(
    {
        "role_tag",
        "content_tag",
        "user_tag",
        "assistant_tag",
        "observation_tag",
        "function_tag",
        "system_tag",
    }
)

# 每个 formatting 必需的内部列; 缺了 -> mode 1 的直接后果。
_REQUIRED_COLUMNS_BY_FORMATTING = {
    "alpaca": ("prompt", "response"),  # query 可选
    "sharegpt": ("messages",),
    "openai": ("messages",),
}

# 按 formatting 决定"内容载体"字段, 用于全空检查。
_CONTENT_FIELDS_BY_FORMATTING = {
    "alpaca": ("prompt", "query", "response"),
    "sharegpt": ("messages",),
    "openai": ("messages",),
}

# 每个 formatting 里会被 converter **逐行解引用** 的内部列(DatasetAttr 对
# alpaca 有默认列 prompt/query/response = instruction/input/output, 所以
# 没写 columns 映射时这些键也必须存在于每一行)。注意 tag 字段只在 sharegpt
# 里被解引用, alpaca 下 role_tag 等是未用的默认值, 不能查。
_ALPACA_ROW_FIELDS = (
    "history",
    "prompt",
    "query",
    "kto_tag",
    "chosen",
    "rejected",
    "response",
    "system",
    "tools",
    "images",
    "videos",
    "audios",
)
_SHAREGPT_ROW_FIELDS = (
    "messages",
    "system",
    "tools",
    "images",
    "videos",
    "audios",
    "kto_tag",
)
_OPENAI_ROW_FIELDS = ("messages", "system", "tools", "images", "videos", "audios")


class PreflightError(Exception):
    """数据被预检拒绝 (exit 1, 结构拦截)。"""


class ConfigError(Exception):
    """配置/环境不可读 (exit 2)。"""


def _warn(message: str) -> None:
    """行级告警: 打印 stderr, **不**改退出码 (预检宽松策略, 2026-08-20)。"""
    print(f"WARN: {message}", file=sys.stderr)


def _load_training_config(path: Path) -> dict:
    try:
        import yaml

        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"cannot read training config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"training config {path} is not a YAML mapping")
    return cfg


def _dataset_name(cfg: dict) -> str:
    dataset = cfg.get("dataset")
    if isinstance(dataset, str):
        names = [name.strip() for name in dataset.split(",") if name.strip()]
    elif isinstance(dataset, list):
        names = [str(n) for n in dataset]
    else:
        raise ConfigError("training YAML has no `dataset` key")
    if len(names) != 1:
        raise ConfigError(
            f"preflight requires exactly one named dataset, got {names!r}"
        )
    return names[0]


def _parse_dataset_attr(dataset_name: str, dataset_dir: Path):
    """用真实 LLaMA-Factory 解析器构建 DatasetAttr —— 列映射语义与训练一致。"""
    try:
        from llamafactory.data.parser import get_dataset_list
    except ImportError as exc:
        raise ConfigError(
            "llamafactory is not importable in this python; run under the "
            "llama_factory_env (source public/llama_factory_env/bin/activate)"
        ) from exc
    try:
        attrs = get_dataset_list([dataset_name], str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"cannot parse dataset {dataset_name!r} from {dataset_dir}: {exc}"
        ) from exc
    if not attrs:
        raise ConfigError(f"dataset {dataset_name!r} not found in {dataset_dir}")
    return attrs[0]


def _load_registry(dataset_dir: Path) -> dict:
    path = dataset_dir / "dataset_info.json"
    if not path.is_file():
        raise ConfigError(f"dataset registry not found: {path}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(registry, dict):
        raise ConfigError(f"dataset registry {path} is not a JSON object")
    return registry


def check_columns_keys(entry: dict, dataset_name: str) -> None:
    """C1: ``columns``/``tags`` 里的键必须是 LLaMA-Factory 内部名。"""
    columns = entry.get("columns")
    if isinstance(columns, dict):
        bad = sorted(set(columns) - _VALID_COLUMN_KEYS)
        if bad:
            raise PreflightError(
                f"dataset {dataset_name!r}: `columns` keys are not valid "
                f"LLaMA-Factory internal names: {bad}. Valid keys: "
                f"{sorted(_VALID_COLUMN_KEYS)}. "
                "Note: keys must be the INTERNAL names (prompt/query/response), "
                "NOT the row field names (instruction/input/output); wrong "
                "keys are silently ignored and every row becomes empty "
                "(\"Cannot find valid samples\" at stage-1)."
            )
    tags = entry.get("tags")
    if isinstance(tags, dict):
        bad = sorted(set(tags) - _VALID_TAG_KEYS)
        if bad:
            raise PreflightError(
                f"dataset {dataset_name!r}: `tags` keys are not valid sharegpt "
                f"tag names: {bad}. Valid keys: {sorted(_VALID_TAG_KEYS)}."
            )


def check_required_columns(attr, dataset_name: str) -> None:
    """C2: 按 formatting 必需的内部列不能为 None (mode 1 的直接后果)。"""
    required = _REQUIRED_COLUMNS_BY_FORMATTING.get(attr.formatting)
    if required is None:
        raise ConfigError(f"unsupported formatting {attr.formatting!r}")
    missing = [c for c in required if getattr(attr, c) is None]
    if missing:
        raise PreflightError(
            f"dataset {dataset_name!r} (formatting={attr.formatting}): required "
            f"column(s) {missing} are None after parsing dataset_info.json. "
            "This almost always means `columns` used row field names "
            "(instruction/input/output) instead of internal names "
            "(prompt/query/response) -> every row is dropped at stage-1 "
            "(\"Cannot find valid samples\")."
        )


def _read_rows(file_path: Path, dataset_name: str) -> list[dict]:
    suffix = file_path.suffix.lower()
    rows: list[dict] = []
    if suffix == ".jsonl":
        with file_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        f"invalid JSONL in {file_path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ConfigError(
                        f"expected JSON object at {file_path}:{line_no}"
                    )
                rows.append(row)
    elif suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        if not all(isinstance(r, dict) for r in rows):
            raise ConfigError(f"{file_path} is not a JSON array of objects")
    else:
        raise ConfigError(
            f"dataset {dataset_name!r}: unsupported data file {file_path} for "
            "preflight (support .jsonl/.json); the pipeline's decontaminated "
            "artifact is always jsonl"
        )
    if not rows:
        # 行级告警 (放宽, 2026-08-20): 与 C4 全空同精神 —— stage-1 会自行报
        # "Cannot find valid samples", 不必在 preflight 硬拦。
        _warn(
            f"dataset {dataset_name!r}: data file {file_path} is empty; "
            "stage-1 will report 'Cannot find valid samples'"
        )
    return rows


def _mapped_fields(attr) -> dict[str, str]:
    """内部列名 -> 行字段名, 只含被当前 formatting 逐行解引用的字段。

    见 ``data/converter.py``: alpaca 用 ``example[attr.prompt]`` 等(默认
    instruction/input/output), sharegpt 用 ``example[messages]`` + 每条消息
    里的 ``role_tag/content_tag``; tag 字段的默认值(from/value/...) 只在
    sharegpt 下才会被读。
    """
    if attr.formatting == "alpaca":
        names = _ALPACA_ROW_FIELDS
    elif attr.formatting == "sharegpt":
        names = _SHAREGPT_ROW_FIELDS
    elif attr.formatting == "openai":
        names = _OPENAI_ROW_FIELDS
    else:
        return {}
    fields: dict[str, str] = {}
    for name in names:
        value = getattr(attr, name, None)
        if isinstance(value, str) and value:
            fields[name] = value
    return fields


def check_field_presence(attr, rows: list[dict], dataset_name: str) -> None:
    """C3: 映射字段存在性 —— 放宽策略 (2026-08-20):

    - **全部**行缺该字段 -> 硬拒 (PreflightError, exit 1): 转换器第一行就
      KeyError, 结构性问题;
    - **部分**行缺 -> WARN (stderr, 不拒绝): 坏行在 stage-1 才 KeyError,
      由下游自行暴露, preflight 保持宽松。
    sharegpt 额外检查每条消息里的 role/content 标签键 (同规则)。
    """
    fields = _mapped_fields(attr)
    for internal, field in fields.items():
        missing = [i for i, row in enumerate(rows) if field not in row]
        if not missing:
            continue
        first = missing[0]
        if len(missing) >= len(rows):
            raise PreflightError(
                f"dataset {dataset_name!r}: `{internal}` maps to row field "
                f"{field!r}, but **all** {len(rows)} rows lack it "
                f"(first at row index {first}). stage-1 would raise "
                f"KeyError: {field!r} on the first such row. Check the file's "
                "actual keys (e.g. `head`) and fix the `columns` mapping "
                "(or drop `query` if the file has no `input` key)."
            )
        _warn(
            f"dataset {dataset_name!r}: `{internal}` maps to row field "
            f"{field!r}, but {len(missing)}/{len(rows)} rows lack it "
            f"(first at row index {first}). Those rows will raise "
            f"KeyError: {field!r} at stage-1. Partial gap -> WARN, not rejected."
        )
    if attr.formatting == "sharegpt":
        _check_sharegpt_message_tags(attr, rows, dataset_name)


def _check_sharegpt_message_tags(attr, rows: list[dict], dataset_name: str) -> None:
    """sharegpt: ``message[role_tag]`` / ``message[content_tag]`` 每条消息必须
    有(默认 from/value)。消息缺失/为空时跳过 —— converter 只在有消息时解引用。

    放宽策略 (2026-08-20, 与 C3 一致): 消息不是 JSON object 是**结构错误**, 仍
    硬拒; tag 缺失按比例: **全部**消息缺 -> 硬拒 (整体缺失必 KeyError), **部分**
    消息缺 -> WARN。
    """
    role = getattr(attr, "role_tag", None) or "from"
    content = getattr(attr, "content_tag", None) or "value"
    msgs_field = getattr(attr, "messages", None)
    if not isinstance(msgs_field, str) or not msgs_field:
        return
    total_messages = 0
    missing_by_internal: dict[str, int] = {}
    for row_idx, row in enumerate(rows):
        msgs = row.get(msgs_field)
        if not isinstance(msgs, list):
            continue
        for msg_idx, message in enumerate(msgs):
            if not isinstance(message, dict):
                raise PreflightError(
                    f"dataset {dataset_name!r}: {msgs_field}[{row_idx}][{msg_idx}] "
                    "is not a JSON object; the sharegpt converter dereferences "
                    "role/content keys on every message."
                )
            total_messages += 1
            for internal, tag in (("role_tag", role), ("content_tag", content)):
                if tag not in message:
                    missing_by_internal[internal] = missing_by_internal.get(internal, 0) + 1
    for internal, tag in (("role_tag", role), ("content_tag", content)):
        missing = missing_by_internal.get(internal, 0)
        if missing == 0:
            continue
        if missing >= total_messages:
            raise PreflightError(
                f"dataset {dataset_name!r}: **all** {missing}/{total_messages} "
                f"messages lack {internal} key {tag!r}; the sharegpt converter "
                f"raises KeyError: {tag!r} on the first such message."
            )
        _warn(
            f"dataset {dataset_name!r}: {missing}/{total_messages} messages lack "
            f"{internal} key {tag!r}; those messages will KeyError at stage-1. "
            "Partial gap -> WARN, not rejected."
        )


def _is_empty_content(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def check_content_nonempty(
    attr, rows: list[dict], dataset_name: str
) -> None:
    """C4: 内容载体字段全空 -> WARN (放宽, 2026-08-20)。

    全空时每行都被当作 invalid example 丢弃, stage-1 会报 "Cannot find valid
    samples" —— 这个错误在训练侧自会暴露, preflight 不硬拦, 只打告警。
    """
    mapped = _mapped_fields(attr)
    content_fields = [
        mapped[internal]
        for internal in _CONTENT_FIELDS_BY_FORMATTING.get(attr.formatting, ())
        if internal in mapped
    ]
    if not content_fields:
        return
    valid = 0
    for row in rows:
        if any(not _is_empty_content(row.get(f)) for f in content_fields):
            valid += 1
    if valid == 0:
        _warn(
            f"dataset {dataset_name!r} (formatting={attr.formatting}): all "
            f"{len(rows)} rows have empty content for fields {content_fields} "
            "-- every row would be dropped as an invalid example at stage-1 "
            "(\"Cannot find valid samples\"). Check the `columns` mapping and "
            "the data file."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, help="training YAML path")
    parser.add_argument("--dataset-dir", required=True, help="dir with dataset_info.json")
    args = parser.parse_args(argv)

    train_cfg = _load_training_config(Path(args.train_config))
    dataset_name = _dataset_name(train_cfg)
    dataset_dir = Path(args.dataset_dir)

    attr = _parse_dataset_attr(dataset_name, dataset_dir)
    registry = _load_registry(dataset_dir)
    if dataset_name not in registry:
        raise PreflightError(
            f"dataset {dataset_name!r} missing from {dataset_dir / 'dataset_info.json'}"
        )
    entry = registry[dataset_name]

    check_columns_keys(entry, dataset_name)          # C1: 内部键必须合法
    check_required_columns(attr, dataset_name)       # C2: 必需列不能 None

    file_name = entry.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        raise ConfigError(f"dataset {dataset_name!r} has no `file_name` in dataset_info.json")
    rows = _read_rows(dataset_dir / file_name, dataset_name)

    check_field_presence(attr, rows, dataset_name)   # C3: 字段存在性
    check_content_nonempty(attr, rows, dataset_name)  # C4: 内容非空

    print(
        f"preflight OK: dataset={dataset_name} formatting={attr.formatting} "
        f"rows={len(rows)} file={file_name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PreflightError as exc:
        print(f"PREFLIGHT REJECTED: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConfigError as exc:
        print(f"PREFLIGHT CONFIG ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
