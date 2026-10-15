"""Shell 命令解析和路径提取。

提供命令拆分、路径提取、路径解析等功能，供过滤器使用。
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from .constants import PATH_COMMANDS

# 匹配 shell 分隔符的正则
_SHELL_SEPARATORS = re.compile(r"(?:&&|\|\||;|\|)")

# 匹配引号内的字符串
_QUOTED_STRING = re.compile(r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""")


def parse_command_segments(command: str) -> list[str]:
    """将复合命令按 shell 分隔符拆分为独立段。

    处理 &&, ||, ;, | 但不拆分引号内的内容。
    """
    # 先替换引号内内容为占位符，避免误拆分
    placeholders: list[str] = []

    def _replace_quoted(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"__QUOTE_{len(placeholders) - 1}__"

    sanitized = _QUOTED_STRING.sub(_replace_quoted, command)
    segments = _SHELL_SEPARATORS.split(sanitized)

    # 还原引号内容
    result = []
    for seg in segments:
        for i, orig in enumerate(placeholders):
            seg = seg.replace(f"__QUOTE_{i}__", orig)
        result.append(seg.strip())

    return [s for s in result if s]


def extract_paths_from_segment(segment: str) -> list[str]:
    """从一个命令段中提取可能的文件路径。

    策略：
    1. 用 shlex 分词
    2. 跳过 flag（以 - 开头）
    3. 对于已知的带路径参数的命令，提取非 flag 参数
    4. 对于 option=value 形式（如 --output=/path），提取 = 后的部分
    """
    try:
        tokens = shlex.split(segment)
    except ValueError:
        # shlex 解析失败时用简单 split
        tokens = segment.split()

    if not tokens:
        return []

    paths: list[str] = []
    cmd_name = Path(tokens[0]).name  # 处理 /usr/bin/python3 这种情况

    i = 1
    while i < len(tokens):
        token = tokens[i]

        # 跳过 flag
        if token.startswith("-"):
            # 处理 --output=/path 形式
            if "=" in token:
                _, value = token.split("=", 1)
                if _looks_like_path(value):
                    paths.append(value)
            # 处理 -o /path 形式（flag 后紧跟路径）
            elif len(token) == 2 and cmd_name in PATH_COMMANDS:
                # 下一个 token 可能是路径值
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    i += 1
                    if _looks_like_path(tokens[i]):
                        paths.append(tokens[i])
            i += 1
            continue

        # 对于路径命令，非 flag 的参数可能是路径
        if cmd_name in PATH_COMMANDS:
            if _looks_like_path(token):
                paths.append(token)
        elif _looks_like_path(token):
            # 任何命令中的看起来像路径的参数
            paths.append(token)

        i += 1

    return paths


def _looks_like_path(s: str) -> bool:
    """判断一个字符串是否看起来像文件路径。"""
    if not s or s.startswith("-"):
        return False
    # 绝对路径
    if s.startswith("/"):
        return True
    # 以 ~ 开头
    if s.startswith("~"):
        return True
    # 包含路径分隔符（排除 URL）
    if "/" in s and not s.startswith("http://") and not s.startswith("https://"):
        return True
    # 相对路径标记
    if s.startswith("./") or s.startswith("../"):
        return True
    return False


def resolve_path(path_str: str, workspace: str) -> str:
    """将路径解析为绝对路径。

    处理 ~, 相对路径等。
    """
    # 展开 ~
    if path_str.startswith("~"):
        path_str = os.path.expanduser(path_str)

    p = Path(path_str)

    # 如果是相对路径，基于 workspace 解析
    if not p.is_absolute():
        p = Path(workspace) / p

    # 规范化（解析 .., . 等）
    try:
        return str(p.resolve())
    except (OSError, ValueError):
        return str(p)


def is_path_in_allowed_dirs(resolved_path: str, allowed_dirs: list[str]) -> bool:
    """检查路径是否在允许的目录下。

    支持精确匹配和前向前缀匹配（目录级）。
    不做反向前缀匹配：白名单目录的「祖先」目录不会被放行 —— 否则白名单里
    一个 /a/b/public 就会放行整个 /a/b，沙箱形同虚设。
    """
    for allowed in allowed_dirs:
        allowed_resolved = str(Path(allowed).resolve())
        # 精确匹配
        if resolved_path == allowed_resolved:
            return True
        # 前向前缀匹配（必须是子目录，需要 / 分隔）
        if resolved_path.startswith(allowed_resolved + "/"):
            return True
    return False
