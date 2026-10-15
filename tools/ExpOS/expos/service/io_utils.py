"""Filesystem helpers shared by every core service.

统一封装 JSON / JSONL 的原子读写以及 sha256 摘要, 避免各服务模块散落自己
的 IO 实现。所有落盘操作都通过这里进入, 便于将来替换成 fcntl 锁或 sqlite。

NFS 稳健性 (2026-08-12 任务 #6 aime 15h 崩溃根因):
state.json 每次写入都用 os.replace 原子替换 → 生成全新 inode。NFS 客户端
可能缓存陈旧文件句柄, 后续 open/read 偶发返回 ESTALE (Errno 116) —— 旧代码
read_json 不捕获 OSError, 一个瞬时 ESTALE 直接把整个 run 杀掉; 更糟的是
陈旧句柄上的 read 可能硬挂起数小时 (本 run 主循环卡了约 6.5h)。

这里的策略:
  1. 对瞬态 OSError (ESTALE / EIO / 网络瞬断等) 做有限次重试 + 退避,
     每次重试都是全新 open, 会重新 LOOKUP, 拿到新句柄后即可恢复;
  2. read_json 在重试耗尽后返回 default (绝不因 IO 抛死调度循环);
  3. write_json 在重试耗尽后抛最后一次异常 (写失败必须让调用方知道),
     但同样会重试, 而不是一次 ESTALE 就丢弃整次写。
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

# 瞬态 OSError: 重试通常能恢复。ESTALE(116) 是 NFS 陈旧句柄 —— 本次崩溃的真凶。
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.ESTALE,        # 116  Stale file handle (NFS 原子替换后旧句柄)
        errno.EIO,           # 5    I/O error (NFS 端瞬时故障)
        errno.EAGAIN,        # 11   resource temporarily unavailable
        errno.EWOULDBLOCK,   # = EAGAIN
        errno.ENETDOWN,      # 100  network is down
        errno.ENETUNREACH,   # 101  network is unreachable
        errno.ECONNRESET,    # 104  connection reset by peer
        errno.ETIMEDOUT,     # 110  connection timed out
        errno.EHOSTDOWN,     # 112  host is down
        errno.EHOSTUNREACH,  # 113  no route to host
        errno.ENOBUFS,       # 105  buffer space available
    }
)

# 重试参数: 读 5 次 / 写 5 次, 退避 0.5s * attempt (0.5/1.0/1.5/2.0/2.5s)。
# 每次重试都是全新 open → 重新 LOOKUP → 拿新句柄, 无需上层感知。
_READ_RETRIES = 5
_WRITE_RETRIES = 5
_RETRY_BACKOFF_S = 0.5


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_transient_oserror(exc: OSError) -> bool:
    return exc.errno in _TRANSIENT_ERRNOS


def _backoff(attempt: int) -> None:
    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))


def _read_text(path: Path) -> str:
    """Read *path*'s text with bounded retry on transient NFS OSErrors.

    Raises the last OSError once retries are exhausted; FileNotFoundError /
    JSONDecodeError are handled by the caller (read_json).
    """
    last_exc: OSError | None = None
    for attempt in range(_READ_RETRIES):
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            last_exc = exc
            if not _is_transient_oserror(exc):
                raise
            _backoff(attempt)
    assert last_exc is not None
    raise last_exc


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, returning *default* on any non-fatal error.

    NFS 上 state.json 被频繁原子替换, 客户端可能缓存陈旧句柄 → 偶发 ESTALE。
    这里重试瞬态 OSError, 重试耗尽或遇到非瞬态 OSError 也返回 default ——
    调度循环的例行 sync 读必须「不崩不挂」, 宁可读到旧数据也不杀死整个 run。
    """
    try:
        return json.loads(_read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return default
    except OSError:
        # 重试耗尽仍失败 (NFS 持续异常): 返回 default, 不让单次读崩 run。
        return default


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text* (temp file + os.replace).

    fsync 后再 replace, 保证 readers 看到的是完整文件。写失败会清理临时文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        # NamedTemporaryFile 默认 mode 0600 → os.replace 后目标文件会变成 0600,
        # 其他进程/脚本读不到 (2026-08-12 确认所有 run 的 state.json 都是 600)。
        # 这里显式 chmod 0644, 与仓库里其它落盘文件一致。
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise


def write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as pretty JSON.

    Writes to a sibling temp file then :func:`os.replace` to avoid corrupting
    a reader in the middle of a partial write. 对瞬态 NFS OSError (ESTALE 等)
    做有限重试; 重试耗尽后抛最后一次异常, 让调用方决定如何处理。
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    last_exc: OSError | None = None
    for attempt in range(_WRITE_RETRIES):
        try:
            _write_text_atomic(path, text)
            return
        except OSError as exc:
            last_exc = exc
            if not _is_transient_oserror(exc):
                raise
            _backoff(attempt)
    assert last_exc is not None
    raise last_exc


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON-encoded line to a jsonl file (retry transient OSErrors)."""
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    last_exc: OSError | None = None
    for attempt in range(_WRITE_RETRIES):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return
        except OSError as exc:
            last_exc = exc
            if not _is_transient_oserror(exc):
                raise
            _backoff(attempt)
    assert last_exc is not None
    raise last_exc


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object in a jsonl file, skipping blank/malformed lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 digest of *path*; empty string if unreadable."""
    if not path.exists() or not path.is_file():
        return ""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return ""
    return hasher.hexdigest()


def count_jsonl_rows(path: Path) -> int | None:
    """Count non-blank lines in a jsonl file, returning ``None`` on error."""
    try:
        if not path.exists() or not path.is_file():
            return None
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    except OSError:
        return None


def slug(value: Any) -> str:
    """Sanitize an arbitrary string into a safe filename slug."""
    text = str(value or "").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")).strip("_") or "unknown"
