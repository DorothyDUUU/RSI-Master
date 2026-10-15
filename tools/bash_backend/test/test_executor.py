"""命令执行器测试。"""

from __future__ import annotations

import time

import pytest

from tools.bash_backend.executor import CommandExecutor, parse_memory_limit_to_bytes


# ---------------------------------------------------------------------------
# parse_memory_limit_to_bytes
# ---------------------------------------------------------------------------


class TestParseMemoryLimitToBytes:
    """内存限制字符串解析测试。"""

    def test_none(self):
        assert parse_memory_limit_to_bytes(None) is None

    def test_empty(self):
        assert parse_memory_limit_to_bytes("") is None

    def test_bytes(self):
        assert parse_memory_limit_to_bytes("1073741824") == 1073741824

    def test_kilobytes(self):
        assert parse_memory_limit_to_bytes("1024k") == 1024 * 1024
        assert parse_memory_limit_to_bytes("1024K") == 1024 * 1024

    def test_megabytes(self):
        assert parse_memory_limit_to_bytes("512m") == 512 * 1024**2
        assert parse_memory_limit_to_bytes("512M") == 512 * 1024**2

    def test_gigabytes(self):
        assert parse_memory_limit_to_bytes("4g") == 4 * 1024**3
        assert parse_memory_limit_to_bytes("4G") == 4 * 1024**3

    def test_terabytes(self):
        assert parse_memory_limit_to_bytes("1t") == 1024**4

    def test_float_value(self):
        assert parse_memory_limit_to_bytes("1.5g") == int(1.5 * 1024**3)

    def test_invalid(self):
        assert parse_memory_limit_to_bytes("abc") is None


# ---------------------------------------------------------------------------
# CommandExecutor
# ---------------------------------------------------------------------------


class TestCommandExecutor:
    """命令执行器测试。"""

    def test_simple_command(self):
        executor = CommandExecutor()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="echo hello",
            timeout=10,
            workdir="/tmp",
        )
        assert exit_code == 0
        assert "hello" in stdout

    def test_command_with_error(self):
        executor = CommandExecutor()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="ls /nonexistent_path_12345",
            timeout=10,
            workdir="/tmp",
        )
        assert exit_code != 0

    def test_timeout(self):
        executor = CommandExecutor()
        start = time.time()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="sleep 30",
            timeout=1,
            workdir="/tmp",
        )
        elapsed = time.time() - start
        assert exit_code == -1
        assert "超时" in stderr
        assert elapsed < 5  # 应该很快返回

    def test_chained_commands(self):
        executor = CommandExecutor()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="echo hello && echo world",
            timeout=10,
            workdir="/tmp",
        )
        assert exit_code == 0
        assert "hello" in stdout
        assert "world" in stdout

    def test_pgid_set(self):
        executor = CommandExecutor()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="echo test",
            timeout=10,
            workdir="/tmp",
        )
        assert proc is not None
        assert pgid is not None

    def test_working_directory(self, tmp_path):
        executor = CommandExecutor()
        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command="pwd",
            timeout=10,
            workdir=str(tmp_path),
        )
        assert exit_code == 0
        assert str(tmp_path) in stdout.strip()
