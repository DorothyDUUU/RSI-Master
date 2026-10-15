"""命令解析和路径提取测试。"""

from __future__ import annotations

import os
import pytest

from tools.bash_backend.command_parser import (
    _looks_like_path,
    extract_paths_from_segment,
    is_path_in_allowed_dirs,
    parse_command_segments,
    resolve_path,
)


# ---------------------------------------------------------------------------
# parse_command_segments
# ---------------------------------------------------------------------------


class TestParseCommandSegments:
    """命令拆分测试。"""

    def test_simple_command(self):
        assert parse_command_segments("echo hello") == ["echo hello"]

    def test_and_separator(self):
        result = parse_command_segments("ls -la && echo done")
        assert result == ["ls -la", "echo done"]

    def test_or_separator(self):
        result = parse_command_segments("cat file || echo failed")
        assert result == ["cat file", "echo failed"]

    def test_semicolon_separator(self):
        result = parse_command_segments("cd /tmp; ls")
        assert result == ["cd /tmp", "ls"]

    def test_pipe_separator(self):
        result = parse_command_segments("cat file | grep pattern")
        assert result == ["cat file", "grep pattern"]

    def test_mixed_separators(self):
        result = parse_command_segments("a && b || c ; d | e")
        assert result == ["a", "b", "c", "d", "e"]

    def test_quoted_separators_preserved(self):
        result = parse_command_segments('echo "a && b" && echo done')
        assert result == ['echo "a && b"', "echo done"]

    def test_single_quoted_preserved(self):
        result = parse_command_segments("echo 'a | b' && echo done")
        assert result == ["echo 'a | b'", "echo done"]

    def test_empty_segments_filtered(self):
        result = parse_command_segments("a &&  && b")
        assert "" not in result

    def test_whitespace_handling(self):
        result = parse_command_segments("  ls  &&  echo hi  ")
        assert result == ["ls", "echo hi"]


# ---------------------------------------------------------------------------
# _looks_like_path
# ---------------------------------------------------------------------------


class TestLooksLikePath:
    """路径识别测试。"""

    def test_absolute_path(self):
        assert _looks_like_path("/etc/passwd") is True

    def test_home_tilde(self):
        assert _looks_like_path("~/Documents") is True

    def test_relative_with_slash(self):
        assert _looks_like_path("./file.txt") is True
        assert _looks_like_path("../file.txt") is True

    def test_relative_subdir(self):
        assert _looks_like_path("subdir/file.txt") is True

    def test_not_a_path(self):
        assert _looks_like_path("hello") is False
        assert _looks_like_path("-v") is False
        assert _looks_like_path("") is False

    def test_url_not_path(self):
        assert _looks_like_path("http://example.com") is False
        assert _looks_like_path("https://example.com") is False


# ---------------------------------------------------------------------------
# extract_paths_from_segment
# ---------------------------------------------------------------------------


class TestExtractPathsFromSegment:
    """路径提取测试。"""

    def test_simple_path_command(self):
        paths = extract_paths_from_segment("ls /etc")
        assert "/etc" in paths

    def test_cat_with_path(self):
        paths = extract_paths_from_segment("cat /tmp/test.txt")
        assert "/tmp/test.txt" in paths

    def test_flags_skipped(self):
        paths = extract_paths_from_segment("ls -la /tmp")
        assert "/tmp" in paths
        assert "-la" not in paths

    def test_option_equals_path(self):
        paths = extract_paths_from_segment("python --output=/tmp/out.py script.py")
        assert "/tmp/out.py" in paths

    def test_relative_path(self):
        paths = extract_paths_from_segment("cat ./file.txt")
        assert "./file.txt" in paths

    def test_no_paths(self):
        paths = extract_paths_from_segment("echo hello world")
        assert paths == []

    def test_empty_segment(self):
        paths = extract_paths_from_segment("")
        assert paths == []

    def test_git_command(self):
        paths = extract_paths_from_segment("git add /tmp/file.txt")
        assert "/tmp/file.txt" in paths


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    """路径解析测试。"""

    def test_absolute_path_unchanged(self):
        result = resolve_path("/etc/passwd", "/workspace")
        assert result == "/etc/passwd"

    def test_relative_to_workspace(self):
        result = resolve_path("./file.txt", "/workspace")
        assert result == "/workspace/file.txt"

    def test_dotdot_resolution(self):
        result = resolve_path("../file.txt", "/workspace/sub")
        assert result == "/workspace/file.txt"

    def test_tilde_expansion(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/test")
        result = resolve_path("~/file.txt", "/workspace")
        assert result == "/home/test/file.txt"


# ---------------------------------------------------------------------------
# is_path_in_allowed_dirs
# ---------------------------------------------------------------------------


class TestIsPathInAllowedDirs:
    """路径白名单检查测试。"""

    def test_exact_match(self):
        assert is_path_in_allowed_dirs("/workspace", ["/workspace"]) is True

    def test_subdirectory(self):
        assert is_path_in_allowed_dirs("/workspace/sub/file.txt", ["/workspace"]) is True

    def test_not_allowed(self):
        assert is_path_in_allowed_dirs("/etc/passwd", ["/workspace"]) is False

    def test_multiple_allowed_dirs(self):
        allowed = ["/workspace", "/tmp"]
        assert is_path_in_allowed_dirs("/workspace/file", allowed) is True
        assert is_path_in_allowed_dirs("/tmp/file", allowed) is True
        assert is_path_in_allowed_dirs("/etc/file", allowed) is False

    def test_partial_match_rejected(self):
        # /workspace2 不应该是 /workspace 的子目录
        assert is_path_in_allowed_dirs("/workspace2/file", ["/workspace"]) is False
