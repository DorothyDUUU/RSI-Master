"""Data pool service — CLI backbone for shared training data assets.

数据池目录结构 (每一份数据一个文件夹, 使 worker 可以自由存放脚本 / 备份等):

    data_pool/{data_id}/
        README.md
        config.json     (:class:`DataPoolRecord`)
        <原始 jsonl / 分析脚本 / ...>

Worker / Reviewer 共用相同的读取接口, 只有 Worker 才能写 (add_data / update_readme
/ 通过 submit 落盘)。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .contracts import DataMemoryRecommendation, DataPoolRecord, dump_model
from .experiment_os import ExperimentOS
from .io_utils import count_jsonl_rows, ensure_dir, read_json, sha256_of_file, write_json


class DataPoolService:
    """Service for the shared ``data_pool/`` directory."""

    def __init__(self, os_: ExperimentOS):
        self.os = os_

    # ── Listing / reading ────────────────────────────────────────────────────

    def list_data(self) -> list[dict[str, Any]]:
        """Return summarized :class:`DataPoolRecord` entries for every data_id."""
        out: list[dict[str, Any]] = []
        if not self.os.data_pool_dir.exists():
            return out
        for entry in sorted(self.os.data_pool_dir.iterdir()):
            if not entry.is_dir():
                continue
            record = self._load_record(entry.name)
            if record is None:
                continue
            summary = dump_model(record)
            summary["path"] = str(entry)
            summary["files"] = [p.name for p in entry.iterdir() if p.is_file()]
            out.append(summary)
        return out

    def inspect_data(self, data_id: str) -> dict[str, Any]:
        """Return the full record plus README + file listing for ``data_id``."""
        record = self._load_record(data_id)
        if record is None:
            raise KeyError(f"unknown data_id: {data_id}")
        data_dir = self.os.data_pool_dir / data_id
        readme_path = data_dir / "README.md"
        readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        files = []
        for entry in sorted(data_dir.iterdir()):
            files.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        return {
            "record": dump_model(record),
            "path": str(data_dir),
            "readme": readme,
            "files": files,
        }

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_data(
        self,
        *,
        worker_id: str | None = None,
        source_experiment_id: str | None = None,
        benchmark: str = "",
        artifact_path: str | None = None,
        artifact_type: str = "train_data",
        num_rows: int | None = None,
        score: float | None = None,
        tags: list[str] | None = None,
        readme: str | None = None,
        descriptions: str = "",
        data_id: str | None = None,
        synthesis_funnel: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a new ``data_pool/{data_id}/`` folder, optionally copying in data.

        If ``artifact_path`` is given, the file is copied into the folder as
        ``raw{ext}`` and its sha256 + row count are recorded automatically.

        ``descriptions`` is a free-text note (source, composition, intended use)
        that lands in config.json / README.md and in the global ``data.jsonl``.

        ``synthesis_funnel`` (optional) carries the data-synthesis funnel counts
        (generated / struct_ok / exec_pass / decontam_kept / dedup_kept) into the
        ``data_added`` event payload, so the small-batch-validate -> scale funnel
        is machine-scanable instead of only living in the README prose (traj_analysis
        metric D4).  Purely additive — the event schema stays backward compatible.
        """
        data_id = data_id or self.os.new_data_id()
        data_dir = ensure_dir(self.os.data_pool_dir / data_id)

        artifact_final_path = ""
        sha = ""
        rows = num_rows
        if artifact_path:
            src = Path(artifact_path).expanduser()
            if not src.exists() or not src.is_file():
                raise FileNotFoundError(f"artifact_path does not exist: {src}")
            dst = data_dir / f"raw{src.suffix or '.dat'}"
            shutil.copy2(src, dst)
            artifact_final_path = str(dst)
            sha = sha256_of_file(dst)
            if rows is None and dst.suffix == ".jsonl":
                rows = count_jsonl_rows(dst)

        record = DataPoolRecord(
            data_id=data_id,
            source_worker_id=worker_id,
            source_experiment_id=source_experiment_id,
            benchmark=benchmark,
            artifact_type=artifact_type,
            artifact_path=artifact_final_path,
            sha256=sha,
            num_rows=rows,
            score=score,
            promotion_status="candidate",
            tags=tags or [],
            descriptions=descriptions,
        )
        write_json(data_dir / "config.json", dump_model(record))
        readme_path = data_dir / "README.md"
        if not readme_path.exists() or readme is not None:
            readme_path.write_text(
                readme
                or (
                    f"# Data `{data_id}`\n\n"
                    f"- benchmark: `{benchmark}`\n"
                    f"- source_worker_id: `{worker_id}`\n"
                    f"- source_experiment_id: `{source_experiment_id}`\n"
                    f"- artifact_type: `{artifact_type}`\n"
                    f"- descriptions: {descriptions or '(fill in the source and how this data should be used)'}\n"
                ),
                encoding="utf-8",
            )
        self._update_state_index(record)
        payload: dict[str, Any] = {
            "artifact_type": artifact_type,
            "benchmark": benchmark,
            "rows": rows,
            "descriptions": descriptions,
        }
        if synthesis_funnel:
            payload["synthesis_funnel"] = synthesis_funnel
        self.os.events.emit(
            "data_added",
            worker_id=worker_id,
            data_id=data_id,
            payload=payload,
        )
        return {"data_id": data_id, "path": str(data_dir), "record": dump_model(record)}

    def update_readme(self, data_id: str, update_summary: str) -> dict[str, Any]:
        """Append ``update_summary`` to the data's README.

        模型本身可以直接编辑对应的 README.md, 这里 CLI 只做**记账**并落事件。
        """
        data_dir = self.os.data_pool_dir / data_id
        readme_path = data_dir / "README.md"
        if not data_dir.exists():
            raise KeyError(f"unknown data_id: {data_id}")
        existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        appended = existing.rstrip() + "\n\n## Update\n\n" + update_summary.strip() + "\n"
        readme_path.write_text(appended, encoding="utf-8")
        self.os.events.emit(
            "data_readme_updated",
            data_id=data_id,
            payload={"chars": len(update_summary)},
        )
        return {"data_id": data_id, "readme_path": str(readme_path), "length": len(appended)}

    def apply_memory_recommendation(
        self,
        data_id: str,
        recommendation: DataMemoryRecommendation,
        *,
        reviewer_id: str | None = None,
        review_verdict: str | None = None,
    ) -> dict[str, Any]:
        """Merge a reviewer's :class:`DataMemoryRecommendation` into the record."""
        record = self._load_record(data_id)
        if record is None:
            raise KeyError(f"unknown data_id: {data_id}")
        payload = dump_model(record)
        payload["memory_recommendation"] = dump_model(recommendation)
        if review_verdict is not None:
            payload["review_verdict"] = review_verdict
        payload["what_to_preserve"] = list(recommendation.what_to_preserve)
        payload["what_to_change"] = list(recommendation.what_to_change)
        if recommendation.action == "promote":
            payload["promotion_status"] = "promoted"
        elif recommendation.action == "discard":
            payload["promotion_status"] = "discarded"
        new_record = DataPoolRecord(**payload)
        write_json(self.os.data_pool_dir / data_id / "config.json", dump_model(new_record))
        self._update_state_index(new_record)
        self.os.events.emit(
            "data_memory_recommendation_applied",
            data_id=data_id,
            reviewer_id=reviewer_id,
            payload={"action": recommendation.action, "verdict": review_verdict},
        )
        return {"data_id": data_id, "record": dump_model(new_record)}

    def record_reuse(self, data_id: str, experiment_id: str) -> None:
        record = self._load_record(data_id)
        if record is None:
            return
        if experiment_id in record.reused_by_experiments:
            return
        payload = dump_model(record)
        payload["reused_by_experiments"] = list(record.reused_by_experiments) + [experiment_id]
        new_record = DataPoolRecord(**payload)
        write_json(self.os.data_pool_dir / data_id / "config.json", dump_model(new_record))
        self._update_state_index(new_record)
        self.os.events.emit(
            "data_reused",
            data_id=data_id,
            experiment_id=experiment_id,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _load_record(self, data_id: str) -> DataPoolRecord | None:
        config_path = self.os.data_pool_dir / data_id / "config.json"
        payload = read_json(config_path, default=None)
        if not isinstance(payload, dict):
            return None
        try:
            return DataPoolRecord(**payload)
        except Exception:
            return None

    def _update_state_index(self, record: DataPoolRecord) -> None:
        """Refresh the compact ``data_pool_index`` inside ``state.json``."""
        state = self.os.load_state()
        index: list[dict[str, Any]] = list(state.get("data_pool_index") or [])
        index = [entry for entry in index if entry.get("data_id") != record.data_id]
        index.append(
            {
                "data_id": record.data_id,
                "benchmark": record.benchmark,
                "artifact_type": record.artifact_type,
                "promotion_status": record.promotion_status,
                "score": record.score,
                "source_experiment_id": record.source_experiment_id,
                "source_worker_id": record.source_worker_id,
                "num_rows": record.num_rows,
                "descriptions": record.descriptions,
            }
        )
        state["data_pool_index"] = index
        self.os.save_state(state)
        self._sync_global_data_jsonl()

    def _sync_global_data_jsonl(self) -> None:
        """Rewrite ``data_pool/data.jsonl`` — one compact record per line, all entries.

        全局索引: 每次池子变更后从磁盘重建, 每个 data_id 一行 JSON
        (完整 :class:`DataPoolRecord`), 让 agent / 脚本一次性读到整个池,
        按 data_id 排序保证输出稳定。
        """
        lines: list[str] = []
        if self.os.data_pool_dir.exists():
            for entry in sorted(self.os.data_pool_dir.iterdir()):
                if not entry.is_dir() or entry.name.startswith("raw"):
                    continue
                record = self._load_record(entry.name)
                if record is None:
                    continue
                lines.append(json.dumps(dump_model(record), ensure_ascii=False, sort_keys=True))
        path = self.os.data_pool_dir / "data.jsonl"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def reconcile(self) -> list[str]:
        """Rebuild ``state.data_pool_index`` + global ``data.jsonl`` from disk.

        兜底同步: 覆盖绕过 service 手工/旧进程创建的数据目录, 保证 DAG 里
        不会出现 data 孤儿。幂等, 返回索引里所有 data_id。
        """
        index: list[dict[str, Any]] = []
        if self.os.data_pool_dir.exists():
            for entry in sorted(self.os.data_pool_dir.iterdir()):
                if not entry.is_dir() or entry.name.startswith("raw"):
                    continue
                record = self._load_record(entry.name)
                if record is None:
                    continue
                index.append(
                    {
                        "data_id": record.data_id,
                        "benchmark": record.benchmark,
                        "artifact_type": record.artifact_type,
                        "promotion_status": record.promotion_status,
                        "score": record.score,
                        "source_experiment_id": record.source_experiment_id,
                        "source_worker_id": record.source_worker_id,
                        "num_rows": record.num_rows,
                        "descriptions": record.descriptions,
                    }
                )
        state = self.os.load_state()
        state["data_pool_index"] = index
        self.os.save_state(state)
        self._sync_global_data_jsonl()
        return [e["data_id"] for e in index]
