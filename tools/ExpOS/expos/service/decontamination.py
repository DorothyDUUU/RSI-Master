"""Exact-query decontamination for training data submitted to ExpOS."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# Repo root: $RSI_MASTER_ROOT, else derived from this file's location
# (service -> expos -> ExpOS -> tools -> <repo>).
_ROOT = Path(os.environ.get("RSI_MASTER_ROOT") or Path(__file__).resolve().parents[4])
_DATASETS_PYTHON = _ROOT / "public/llama_factory_env/bin/python"


def _benchmark_root() -> Path:
    """Benchmark-suite root: ``$BENCHMARK_ROOT``, else the in-repo checkout.

    Decontamination must read the *same* question source the evaluator uses, so
    the default is the ``bmk_everything/`` checkout that ships with this repo —
    point ``$BENCHMARK_ROOT`` elsewhere to use a different one.
    """
    raw = os.environ.get("BENCHMARK_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _ROOT / "bmk_everything"


def _benchmark_datasets(*parts: str) -> Path:
    """``$BENCHMARK_ROOT/data/datasets/<parts...>``."""
    return _benchmark_root().joinpath("data", "datasets", *parts)


@dataclass(frozen=True)
class DecontaminationReport:
    benchmark: str
    original_rows: int
    removed_rows: int
    retained_rows: int
    contamination_rate: float
    output_path: str
    removed_query_hashes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_query(value: Any) -> str:
    """Return a stable representation without weakening exact-match semantics."""
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "role" in item for item in value):
            value = "\n".join(
                f"{item.get('role', '')}: {item.get('content', '')}" for item in value
            )
        else:
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def benchmark_query_texts(value: Any) -> list[str]:
    """Extract query text using the same unit used by submitted train data.

    A plain benchmark row is already one query.  Structured rows may contain a
    multi-turn conversation (HealthBench) or a list of conversation cells
    (BFCL).  Training decontamination compares the final user turn of each
    ``messages`` row, so benchmark-side extraction must use that same unit.
    """

    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text", block.get("content", ""))
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return ""

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        # One conversation: match training_query_from_messages and keep only
        # its final user turn.  A nested list is treated as multiple cells.
        if value and all(isinstance(item, dict) and "role" in item for item in value):
            users = [
                _content_text(item.get("content"))
                for item in value
                if str(item.get("role") or "") == "user"
            ]
            return [users[-1]] if users and users[-1].strip() else []
        texts: list[str] = []
        for item in value:
            texts.extend(benchmark_query_texts(item))
        return texts

    if isinstance(value, dict):
        if "role" in value:
            if str(value.get("role") or "") != "user":
                return []
            text = _content_text(value.get("content"))
            return [text] if text.strip() else []
        for key in ("messages", "conversation", "query", "question", "prompt", "problem", "instruction"):
            if key in value:
                return benchmark_query_texts(value[key])
    return []


def training_query(row: dict[str, Any]) -> str:
    if "instruction" not in row:
        raise ValueError("training row must contain an 'instruction' field")
    instruction = str(row.get("instruction") or "")
    input_text = normalize_query(row.get("input"))
    return input_text or normalize_query(instruction)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def load_benchmark_queries(
    benchmark: str,
    *,
    bench_id: str | None = None,
    eval_all: dict[str, Any] | None = None,
) -> list[Any]:
    """Load raw benchmark queries from the same sources used by evaluation."""
    # Normalize legacy/internal names to the benchmark-suite id. This
    # keeps decontamination on the exact same question source as evaluation and
    # avoids relying on worker-local copies of PostTrainBench data.
    benchmark_aliases = {
        "arena_hard_writing": "arenahardwriting",
        "if_bench": "ifbench",
        "instruction_following_bench": "ifbench",
        "super_gpqa": "supergpqa",
        "super-gpqa": "supergpqa",
    }
    benchmark = benchmark_aliases.get(benchmark, benchmark)
    if bench_id:
        bench_id = benchmark_aliases.get(bench_id, bench_id)
    elif benchmark == "arenahardwriting":
        bench_id = benchmark

    if bench_id:
        cfg = eval_all or {}
        # eval_all.bmk_root wins; otherwise $BENCHMARK_ROOT.
        _cfg_root = str(cfg.get("bmk_root") or "").strip()
        bmk_root = Path(_cfg_root).expanduser() if _cfg_root else _benchmark_root()
        if not bmk_root.is_dir():
            raise ValueError("eval_all decontamination requires an existing bmk_root")
        bench_dir = bmk_root / "format" / bench_id
        candidates = sorted(bench_dir.glob("*.jsonl"))
        split = str(cfg.get("split") or "")
        if split:
            candidates = [p for p in candidates if split in p.stem] or candidates
        # Skip broken symlinks / unreadable candidates (e.g. a stale
        # format/<id>/default.jsonl -> dead host path) so we fall through to the
        # benchmark-driven source instead of crashing with FileNotFoundError.
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            rows = [row for path in candidates for row in _read_jsonl(path)]
            fields = ("query", "question", "prompt", "problem", "instruction")
            raw_queries = [next((row[k] for k in fields if k in row), None) for row in rows]
            queries = [text for query in raw_queries for text in benchmark_query_texts(query)]
            if queries:
                return queries
        # 该 bench 的 format/<id>/ 没有 query jsonl (如 gsm8k: 数据集走
        # custom.dataset_path, 不在 format/ 下放问题文件) —— 回落到 benchmark 驱动的
        # query 源 (下方 jsonl_specs / disk_specs / gsm8k datasets 分支)。若该
        # benchmark 也没有独立源, 最终 raise 会给出清晰的 unsupported 报错。

    # ARC-AGI: eval queries are LLM prompts rendered by format/<bench>/arc_runner.prompt()
    # from data/datasets/<bench>/default.jsonl (train/test grids). The rows have no
    # "query"/"prompt" field, so rebuild the exact eval prompt per task.
    if benchmark in ("arc-agi-1", "arc-agi-2"):
        path = _benchmark_datasets(benchmark, "default.jsonl")
        if not path.is_file():
            raise ValueError(f"benchmark data not found: {path}")

        def _arc_prompt(task: dict[str, Any]) -> str:
            parts = [
                "Solve this ARC-AGI task. Infer the exact transformation from the "
                "training examples and apply it to each test input. Return only a "
                "JSON array containing one output grid per test input. Do not include "
                "Markdown fences or explanations."
            ]
            for i, x in enumerate(task["train"], 1):
                parts += [
                    f'Training example {i} input:\n{json.dumps(x["input"])}\n'
                    f'Training example {i} output:\n{json.dumps(x["output"])}'
                ]
            for i, x in enumerate(task["test"], 1):
                parts += [f'Test input {i}:\n{json.dumps(x["input"])}']
            return "\n\n".join(parts)

        return [_arc_prompt(row) for row in _read_jsonl(path)]


    # conversation-style benchmarks whose eval queries live in the LAST user turn
    # of ``conversation.messages`` (benchmark suite data/datasets layout).
    conv_jsonl_specs: dict[str, Path] = {
        "healthbench_professional": _benchmark_datasets("healthbench_professional", "test.jsonl"),
    }
    if benchmark in conv_jsonl_specs:
        path = conv_jsonl_specs[benchmark]
        if not path.is_file():
            raise ValueError(f"benchmark data not found: {path}")
        queries: list[Any] = []
        for row in _read_jsonl(path):
            msgs = (row.get("conversation") or {}).get("messages") or []
            users = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            if users:
                queries.append(users[-1])
        return queries

    jsonl_specs: dict[str, tuple[Path, str]] = {
        "ifbench": (_benchmark_datasets("ifbench", "train.jsonl"), "prompt"),
        "supergpqa": (_benchmark_datasets("supergpqa", "train.jsonl"), "question"),
        "imo-answerbench": (_benchmark_datasets("imo-answerbench", "default.jsonl"), "Problem"),
        "aime_2025": (_ROOT / "public/datasets/aime2025/test.jsonl", "problem"),
        "bbeh": (_benchmark_datasets("bbeh", "train.jsonl"), "input"),
        "medxpertqa": (_benchmark_datasets("medxpertqa", "test.jsonl"), "question"),
        "gpqa_main": (_ROOT / "public/datasets/gpqa/gpqa_main.jsonl", "Question"),
        "healthbench_easy": (_benchmark_datasets("healthbench_easy", "default.jsonl"), "prompt"),
    }
    if benchmark in jsonl_specs:
        path, field = jsonl_specs[benchmark]
        if not path.is_file():
            raise ValueError(f"benchmark data not found: {path}")
        return [
            text
            for row in _read_jsonl(path)
            if field in row
            for text in benchmark_query_texts(row[field])
        ]

    # HuggingFace save_to_disk datasets used by the corresponding evaluators.
    disk_specs = {
        "bfcl": (_ROOT / "public/datasets/bfcl_exec_simple", "question"),
        "human_eval": (_ROOT / "public/datasets/human_eval", "prompt"),
        # alias: runs whose state.benchmark is the benchmark-suite id "humaneval"
        # (bench_id namespace) hit the same HF save_to_disk human_eval dataset.
        "humaneval": (_ROOT / "public/datasets/human_eval", "prompt"),
    }
    if benchmark == "gsm8k":
        path = _ROOT / "public/datasets/gsm8k"
        script = """
import json, sys
from datasets import load_dataset
dataset = load_dataset(sys.argv[1], 'default', split='test')
seen = set()
for row in dataset:
    query = row['question']
    if query not in seen:
        seen.add(query)
        print(json.dumps(query, ensure_ascii=False))
"""
        proc = subprocess.run(
            [str(_DATASETS_PYTHON), "-c", script, str(path)],
            check=True, capture_output=True, text=True,
        )
        raw_queries = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        return [text for query in raw_queries for text in benchmark_query_texts(query)]
    if benchmark in disk_specs:
        path, field = disk_specs[benchmark]
        if not path.exists():
            raise ValueError(f"benchmark data not found: {path}")
        if not _DATASETS_PYTHON.is_file():
            raise ValueError(f"benchmark loader Python not found: {_DATASETS_PYTHON}")
        script = """
import json, sys
from datasets import load_from_disk
dataset = load_from_disk(sys.argv[1])
for row in dataset:
    value = row[sys.argv[2]]
    if sys.argv[3] == 'bfcl' and isinstance(value, list) and len(value) == 1:
        value = value[0]
    print(json.dumps(value, ensure_ascii=False))
"""
        proc = subprocess.run(
            [str(_DATASETS_PYTHON), "-c", script, str(path), field, benchmark],
            check=True,
            capture_output=True,
            text=True,
        )
        # Flatten structured cells (BFCL conversation lists) to the same user-text
        # unit the training matcher compares, mirroring the jsonl_specs/gsm8k
        # branches above. human_eval prompts are plain strings -> pass through.
        raw_queries = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        return [text for query in raw_queries for text in benchmark_query_texts(query)]
    raise ValueError(f"unsupported benchmark for decontamination: {benchmark!r}")


def decontaminate_jsonl(
    source: Path,
    destination: Path,
    benchmark_queries: Iterable[Any],
    *,
    benchmark: str,
) -> DecontaminationReport:
    rows = _read_jsonl(source)
    blocked = {normalize_query(query) for query in benchmark_queries}
    blocked.discard("")
    if not blocked:
        raise ValueError(f"benchmark {benchmark!r} yielded no queries")
    retained: list[dict[str, Any]] = []
    removed_hashes: list[str] = []
    for row in rows:
        query = training_query(row)
        if query in blocked:
            removed_hashes.append(hashlib.sha256(query.encode()).hexdigest())
        else:
            retained.append(row)
    if not retained:
        raise ValueError("decontamination removed every training row")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return DecontaminationReport(
        benchmark=benchmark,
        original_rows=len(rows),
        removed_rows=len(removed_hashes),
        retained_rows=len(retained),
        contamination_rate=(len(removed_hashes) / len(rows)) if rows else 0.0,
        output_path=str(destination),
        removed_query_hashes=removed_hashes,
    )


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a parquet file into a list of plain dicts (pyarrow, lazy import).

    供 ``decontaminate_parquet`` / 预检测试复用。pyarrow 只在调用时才 import,
    这样不带 pyarrow 的环境 (如仅跑 jsonl 路径) 仍能 import 本模块。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise ImportError("decontamination.read_parquet_rows requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def training_query_from_messages(row: dict[str, Any]) -> str:
    """从 OpenAI chat 的 ``messages`` 列提取可比对文本 (slime 训练数据格式).

    slime 训练数据是 ``messages`` 列。取最后一个 user 轮次的 content 作为污染
    比对文本 (没有 user 轮次时退回最后一个非空轮次), 与 :func:`training_query`
    一样做 ``normalize_query`` 归一化, 与 benchmark 侧保持一致。
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("training row must contain a non-empty 'messages' list")
    user_turns = [
        str(turn.get("content") or "")
        for turn in messages
        if isinstance(turn, dict) and str(turn.get("role") or "") == "user"
    ]
    if user_turns:
        return normalize_query(user_turns[-1])
    last = messages[-1]
    if isinstance(last, dict) and str(last.get("content") or "").strip():
        return normalize_query(last["content"])
    raise ValueError("training row 'messages' has no extractable user content")


def decontaminate_parquet(
    source: Path,
    destination: Path,
    benchmark_queries: Iterable[Any],
    *,
    benchmark: str,
) -> DecontaminationReport:
    """污染检查的 parquet 变体: 读取 ``source``, 剔除命中 benchmark 的行, 写净化
    副本到 ``destination`` (保留原始 schema, 用 ``table.take`` 只过滤不重建)。

    与 :func:`decontaminate_jsonl` 同语义: ``blocked`` 是归一化后的 benchmark
    query 集合, 训练侧用 ``training_query_from_messages`` 提取可比对文本。
    """
    try:
        import pyarrow as pa  # noqa: F401  (类型注解表, 实际使用 take/write_table)
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise ImportError("decontamination.decontaminate_parquet requires pyarrow") from exc

    table = pq.read_table(source)
    rows = table.to_pylist()
    blocked = {normalize_query(query) for query in benchmark_queries}
    blocked.discard("")
    if not blocked:
        raise ValueError(f"benchmark {benchmark!r} yielded no queries")
    keep_indices: list[int] = []
    removed_hashes: list[str] = []
    for i, row in enumerate(rows):
        query = training_query_from_messages(row)
        if query in blocked:
            removed_hashes.append(hashlib.sha256(query.encode()).hexdigest())
        else:
            keep_indices.append(i)
    if not keep_indices:
        raise ValueError("decontamination removed every training row")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(keep_indices), destination)
    return DecontaminationReport(
        benchmark=benchmark,
        original_rows=len(rows),
        removed_rows=len(removed_hashes),
        retained_rows=len(keep_indices),
        contamination_rate=(len(removed_hashes) / len(rows)) if rows else 0.0,
        output_path=str(destination),
        removed_query_hashes=removed_hashes,
    )
