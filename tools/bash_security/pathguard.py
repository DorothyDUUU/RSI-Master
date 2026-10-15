"""路径边界检查 —— check_paths。

移植 pathValidation.ts 的 validatePath/isPathAllowed + filesystem.ts 的
pathInAllowedWorkingPath。策略:
- 只对 PATH_COMMANDS 命令集抽取路径操作数,避免 `echo /etc/passwd` 误伤。
- 重定向目标(> >> 2> &> 等)单独抽取并做 create 级校验。
- 关键逃逸防御: -- 结束选项、-o/--output= 值、grep/rg pattern 分割、
  mv/cp 任何 flag → reject、cd 后相对写路径 → fail-closed。
- python/python3 段跳过(脚本内部文件访问无法静态分析)。
- 危险删除路径(isDangerousRemovalPath)、shell 展开字符、glob 写操作、~变体。
"""

from __future__ import annotations

import os
import posixpath
import re

from . import patterns as P
from .result import SecurityResult
from .shelllex import (
    Token,
    extract_heredocs,
    extract_redirect_targets,
    get_positional_words,
    split_segments,
    tokenize,
)

# ---------------------------------------------------------------------------
# 命令集与写/删除分类
# ---------------------------------------------------------------------------
PATH_COMMANDS = frozenset({
    # 源码 PATH_EXTRACTORS 命令集 (pathValidation.ts 附近)
    "cd", "ls", "find", "mkdir", "touch", "rm", "rmdir", "mv", "cp", "cat",
    "head", "tail", "sort", "uniq", "wc", "cut", "paste", "column", "tr",
    "file", "stat", "diff", "awk", "strings", "hexdump", "od", "base64", "nl",
    "grep", "rg", "sed", "git", "jq", "sha256sum", "sha1sum", "md5sum",
    # 本地 bash_backend/constants.py 的命令集
    "chmod", "chown", "ln", "readlink", "realpath", "du", "df", "tar", "zip",
    "unzip", "gzip", "gunzip", "python", "python3", "pip", "pip3", "bash",
    "sh", "source", ".", "docker", "docker-compose", "tee", "xargs", "vim",
    "nano", "emacs", "less", "more", "tree",
})

# 跳过整段路径检查的命令(脚本内部访问无法静态分析)
SKIP_PATH_CHECK_COMMANDS = frozenset({"python", "python3"})

# 写操作命令(glob 写路径被禁、写入路径必须解析到允许目录)
WRITE_COMMANDS = frozenset({
    "touch", "mkdir", "mv", "cp", "ln", "chmod", "chown", "tee", "gzip",
    "gunzip", "unzip", "vim", "nano", "emacs",
})
# 删除操作命令(危险删除路径检查)
REMOVE_COMMANDS = frozenset({"rm", "rmdir"})

# mv/cp 带任何 flag → reject(--target-directory / -t 绕过防御)
FLAG_REJECT_COMMANDS = frozenset({"mv", "cp"})

# grep/rg: 第一个非 flag 参数(或 -e 值)是 pattern,不是路径
PATTERN_FIRST_COMMANDS = frozenset({"grep", "rg"})

# 带路径值的 flag(值即路径,需检查)
PATH_VALUE_FLAGS = frozenset({
    "--output", "-o", "--file", "--from-file", "--rawfile", "--slurpfile",
    "--library-path", "--target-directory", "--reference", "--chdir",
    "--prefix", "-P", "-C",
})
# 仅对部分命令生效的 -f FILE(其它命令的 -f 是 flag,如 rm -f)
PATTERN_FILE_FLAGS = frozenset({"grep", "rg", "jq", "sed"})

# 重定向目标直接放行(安全的伪设备)
SAFE_REDIRECT_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

_ABS_RE = re.compile(r"^/|^[A-Za-z]:[\\/]|^~")
_TILDE_VARIANT_RE = re.compile(r"^~[^/]")


def _expand_tilde(path: str) -> str:
    if path == "~" or path.startswith("~/"):
        return os.path.expanduser(path)
    return path


def _path_within_allowed(allowed: str, resolved: str) -> bool:
    if resolved == allowed:
        return True
    return resolved.startswith(allowed.rstrip("/") + "/")


def _is_dangerous_removal_path(resolved: str) -> bool:
    """移植 isDangerousRemovalPath。"""
    fw = resolved.replace("\\", "/")
    fw = re.sub(r"/+", "/", fw)
    if fw == "*" or fw.endswith("/*"):
        return True
    norm = fw if fw == "/" else fw.rstrip("/")
    if norm == "/":
        return True
    if P.WINDOWS_DRIVE_ROOT_REGEX.match(norm):
        return True
    home = os.path.expanduser("~").replace("\\", "/")
    home = re.sub(r"/+", "/", home)
    if norm == home:
        return True
    parent = posixpath.dirname(norm)
    if parent == "/":
        return True
    if P.WINDOWS_DRIVE_CHILD_REGEX.match(norm):
        return True
    return False


def _validate_path(
    path_str: str,
    cwd: str,
    allowed_dirs: list[str],
    op: str,
) -> SecurityResult:
    """校验单个路径。op ∈ read / write / remove。"""
    clean = path_str.strip()
    if not clean or clean == "-":
        return SecurityResult.allow()

    # UNC 网络路径(可能泄露凭据)
    if clean.startswith("\\\\") or clean.startswith("//"):
        return SecurityResult.reject("path_unc", f"UNC 网络路径不允许: {clean}")

    # ~ 变体 (~user / ~+ / ~- / ~N) —— 展开与 bash 不一致,TOCTOU
    if clean.startswith("~") and not (clean == "~" or clean.startswith("~/")):
        return SecurityResult.reject(
            "path_tilde_variant", f"Tilde 变体路径不允许: {clean}"
        )

    # shell 展开语法($ / % / 词首 =) —— 校验时保留字面、执行时展开,TOCTOU
    if any(ch in clean for ch in P.SHELL_EXPANSION_CHARS) or clean.startswith("="):
        return SecurityResult.reject(
            "path_expansion", f"路径含 shell 展开语法,无法静态验证: {clean}"
        )

    # 写操作: glob 模式禁止(写工具不会展开 glob,校验的是字面路径)
    if op in ("write", "create") and P.GLOB_PATTERN_REGEX.search(clean):
        return SecurityResult.reject(
            "path_glob_write", f"写路径不允许含 glob 模式: {clean}"
        )

    # 解析为绝对 + realpath(符号链接感知)
    expanded = _expand_tilde(clean)
    abs_path = (
        expanded
        if expanded.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", expanded)
        else os.path.join(cwd, expanded)
    )
    resolved = os.path.realpath(abs_path)

    # 删除操作: 危险删除路径
    if op == "remove" and _is_dangerous_removal_path(resolved):
        return SecurityResult.reject(
            "path_dangerous_removal", f"危险的删除路径: {resolved}"
        )

    # 允许目录边界(目录级前向,精确到 dir/ 边界)
    for allowed in allowed_dirs:
        if _path_within_allowed(allowed, resolved):
            return SecurityResult.allow()
    return SecurityResult.reject(
        "path_outside",
        f"路径 '{path_str}' (解析为 '{resolved}') 不在允许的目录范围内。允许的目录: {allowed_dirs}",
    )


# ---------------------------------------------------------------------------
# 参数/路径抽取
# ---------------------------------------------------------------------------
def _is_flag(token_text: str) -> bool:
    return len(token_text) > 1 and token_text.startswith("-")


def _extract_operands(
    base: str,
    tokens: list[Token],
) -> tuple[list[str], list[Token]]:
    """返回 (path_operands, 剩余未消费 token)。

    处理: 跳过前导 env 赋值与命令名; -- 结束选项; --flag=value; flag 后接路径值;
    grep/rg 的 pattern。
    """
    from .interceptor import _is_env_assignment

    path_values: list[str] = []
    pattern_mode = base in PATTERN_FIRST_COMMANDS
    pattern_consumed = False
    i = 0
    n = len(tokens)
    operands: list[str] = []
    # 跳过前导 env 赋值(VAR=val)与命令名本身
    command_seen = False
    while i < n:
        t = tokens[i]
        if t.kind == "word" and _is_env_assignment(t.text):
            i += 1
            continue
        if not command_seen and t.kind == "word":
            command_seen = True
            i += 1
            break
        i += 1
    while i < n:
        t = tokens[i]
        if t.kind == "subst":
            # 参数位置出现替换 —— 保留给上层判 rm 之类,这里不进 path
            operands.append("")
            i += 1
            continue
        if t.kind != "word":
            i += 1
            continue
        text = t.text
        if text == "--":
            # 剩余全部是路径操作数
            for j in range(i + 1, n):
                tj = tokens[j]
                if tj.kind == "word" and tj.text:
                    operands.append(tj.text)
            break
        if _is_flag(text):
            # --flag=value
            if text.startswith("--") and "=" in text:
                flag, _, val = text.partition("=")
                if flag in PATH_VALUE_FLAGS and val:
                    path_values.append(val)
                i += 1
                continue
            if base in PATTERN_FILE_FLAGS and text in ("-f", "--file"):
                if i + 1 < n and tokens[i + 1].kind == "word":
                    path_values.append(tokens[i + 1].text)
                    i += 2
                    continue
            if text in ("-e", "--regexp", "-E"):
                if base in PATTERN_FIRST_COMMANDS and i + 1 < n:
                    i += 2  # pattern 值,不是路径
                    pattern_consumed = True
                    continue
            if text in PATH_VALUE_FLAGS:
                if i + 1 < n and tokens[i + 1].kind == "word":
                    path_values.append(tokens[i + 1].text)
                    i += 2
                    continue
            # 其它 flag 直接跳过
            i += 1
            continue
        # 普通 word
        if pattern_mode and not pattern_consumed and not operands:
            # 第一个普通参数是 pattern(grep PAT file)
            pattern_consumed = True
            i += 1
            continue
        operands.append(text)
        i += 1
    return path_values + operands, tokens


def _write_classified(base: str, flag_texts: set[str], operands: list[str]) -> bool:
    """该段是否有写操作。"""
    if base in WRITE_COMMANDS:
        return True
    if base == "tar" and any(f in flag_texts for f in ("c", "x")):
        return True
    if base == "sed" and "-i" in flag_texts:
        return True
    if base == "sort" and "-o" in flag_texts:
        return True
    return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def check_paths(
    command: str,
    *,
    cwd: str | None = None,
    allowed_dirs: list[str] | None = None,
) -> SecurityResult:
    """路径边界检查。cwd 默认 '.';allowed_dirs 为空 → 任何绝对路径都算越界。"""
    if not command or not command.strip():
        return SecurityResult.allow()
    cwd = cwd or "."
    allowed = [os.path.realpath(d) for d in (allowed_dirs or []) if d]
    if not allowed:
        # 空 allowed → 只有相对路径(解析到 cwd)合法
        allowed = [os.path.realpath(cwd)]

    # 1. 抽 heredoc(全部替换为占位符,tokenizer 无需处理主体)
    processed, heredocs = extract_heredocs(command, quoted_only=False)

    # 1.5 命令替换递归 ($() `${}` `$[]` 反引号) —— 内部命令也要走路径边界。
    # 单独跑 check_command 不传 allowed_dirs 时递归里跳过路径检查,`echo $(cat
    # /etc/passwd)` 这种非 PATH_COMMANDS 的壳会从 check_paths 漏过;这里补上。
    from .interceptor import _check_substitutions

    subst_result = _check_substitutions(
        processed, cwd=cwd, allowed_dirs=allowed, depth=0
    )
    if not subst_result.allowed:
        return subst_result

    # 2. tokenize + 分段
    tokens = tokenize(processed)
    segments = [s for s in split_segments(tokens) if s]

    # 2.5 python 内联代码扫描 —— `python -c CODE` / `python <<'EOF'` heredoc 主体。
    # python 段在 shell 级跳过路径检查,但轨迹里的逃逸(读 eval 集、写 /etc、
    # os.system 套 shell)都藏在可见的 -c/heredoc 代码里,这里静态扫一遍。
    py_result = _check_python_segments(command, processed, heredocs, cwd=cwd, allowed_dirs=allowed)
    if not py_result.allowed:
        return py_result

    # 2.6 xargs 注入 —— 管道前段输出的白名单外字面路径会被 xargs 当操作数,
    # 而 echo/printf 等非 PATH_COMMANDS 段不经过普通路径检查,补 fail-closed 扫描。
    xg_result = _check_xargs_pipeline(segments, allowed_dirs=allowed)
    if not xg_result.allowed:
        return xg_result

    cd_seen = False
    for segment in segments:
        r = _check_segment(
            segment, cwd=cwd, allowed_dirs=allowed, cd_seen=cd_seen
        )
        if not r.allowed:
            return r
        base = _segment_base(segment)
        if base == "cd":
            cd_seen = True
    return SecurityResult.allow()


def _segment_base(segment: list[Token]) -> str:
    from .interceptor import extract_base_command

    return extract_base_command(segment)


def _check_segment(
    segment: list[Token],
    *,
    cwd: str,
    allowed_dirs: list[str],
    cd_seen: bool,
) -> SecurityResult:
    # 处理子 shell (...) 递归
    if any(t.kind == "operator" and t.text in "()" for t in segment):
        r = _check_subshells(segment, cwd=cwd, allowed_dirs=allowed_dirs)
        if not r.allowed:
            return r

    base = _segment_base(segment)

    # python/python3 段跳过(脚本内部访问无法静态分析)
    if base in SKIP_PATH_CHECK_COMMANDS:
        return SecurityResult.allow()

    # mv/cp 带任何 flag → reject(--target-directory / -t / --backup 绕过)
    if base in FLAG_REJECT_COMMANDS:
        for t in segment[1:]:
            if t.kind == "word" and _is_flag(t.text) and t.text != "--":
                return SecurityResult.reject(
                    "path_mv_cp_flags",
                    f"{base} 带 flag 无法验证目标路径(--target-directory 等绕过): {t.text}",
                )

    # find -exec/-execdir/-ok/-okdir/-delete → 在 find 内执行/删除任意文件,无法静态验证
    if base == "find":
        find_flags = [t.text for t in segment if t.kind == "word"]
        for dangerous in ("-exec", "-execdir", "-ok", "-okdir", "-delete"):
            if dangerous in find_flags:
                return SecurityResult.reject(
                    "path_find_exec",
                    f"find {dangerous} 会在 find 内执行/删除文件,无法静态验证",
                )

    # 1. 重定向目标(create 级校验)
    redirects = extract_redirect_targets(segment)
    write_redirect_present = False
    for op, target, is_fd in redirects:
        if op in (">", ">>", ">|", "&>", "2>", "1>", ">&") and not is_fd:
            write_redirect_present = True
        if not target or is_fd:
            continue
        if target in SAFE_REDIRECT_TARGETS:
            continue
        if op in (">", ">>", ">|", "&>") and not is_fd:
            # 写重定向
            r = _validate_path(target, cwd, allowed_dirs, "create")
            if not r.allowed:
                return r
        elif op in ("<", "<<", "<<<", "<>") and not is_fd:
            # 读/输入重定向
            if op == "<<":
                continue  # heredoc 分隔符(主体已抽取)
            r = _validate_path(target, cwd, allowed_dirs, "read")
            if not r.allowed:
                return r

    # cd 之后的写路径/重定向无法验证(fail-closed)
    if cd_seen and (write_redirect_present or (base in WRITE_COMMANDS and _segment_has_operands(segment))):
        return SecurityResult.reject(
            "path_cd_write", "cd 改变工作目录后出现写操作/重定向,目标无法静态验证"
        )

    # 2. 只对 PATH_COMMANDS 抽取路径操作数
    if base not in PATH_COMMANDS:
        return SecurityResult.allow()

    operands, _rest = _extract_operands(base, segment)

    # 3. bash -c 'script' / sh -c 递归
    if base in ("bash", "sh", "zsh", "ksh"):
        script = _extract_c_script(segment)
        if script:
            from .interceptor import check_command

            cr = check_command(script)
            if not cr.allowed:
                return cr
            pr = check_paths(script, cwd=cwd, allowed_dirs=allowed_dirs)
            if not pr.allowed:
                return pr

    # 4. 逐路径校验
    if base in REMOVE_COMMANDS:
        op = "remove"
    elif _write_classified(base, {t.text for t in segment if t.kind == "word"}, operands):
        op = "write"
    else:
        op = "read"

    for path_str in operands:
        if not path_str:
            # 空 = 参数位置是替换(subst token),交由 interceptor 递归处理
            continue
        r = _validate_path(path_str, cwd, allowed_dirs, op)
        if not r.allowed:
            return r
    return SecurityResult.allow()


def _segment_has_operands(segment: list[Token]) -> bool:
    from .interceptor import extract_base_command

    base = extract_base_command(segment)
    operands, _ = _extract_operands(base, segment)
    return any(operands)


def _extract_c_script(segment: list[Token]) -> str | None:
    """bash/sh -c 'script' 时返回脚本字符串。"""
    for i, t in enumerate(segment):
        if t.kind == "word" and t.text == "-c":
            for nxt in segment[i + 1 :]:
                if nxt.kind == "word" and nxt.text:
                    return nxt.text
                break
    return None


# ---------------------------------------------------------------------------
# python 内联代码扫描 + xargs 注入扫描
# ---------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(r"^__HEREDOC_(\d+)_bashsec__$")


def _check_python_segments(
    command: str,
    processed: str,
    heredocs: list[dict],
    *,
    cwd: str,
    allowed_dirs: list[str],
) -> SecurityResult:
    """对命令中的 python 段做内联代码扫描。

    覆盖三种代码来源:
    - `python -c CODE` —— 从原始命令按 Python 字符串语义抽取(代码里常含
      ( ) ; |,shell tokenizer 会切碎);
    - `python <<'EOF'` / `python - <<'EOF'` heredoc 主体 —— 从替换后的
      placeholder 反查(heredocs 按 content_end 降序,placeholder
      `__HEREDOC_k_bashsec__` ↔ matches[len-1-k]);

    脚本文件(`python script.py`)内容无法静态读取,仍跳过 —— 由 OS 级沙箱兜底。
    """
    from .python_scan import check_python_code, extract_python_c_codes

    codes: list[str] = extract_python_c_codes(command)

    if heredocs:
        placeholder_to_content: dict[str, str] = {}
        for idx, h in enumerate(heredocs):
            k = len(heredocs) - 1 - idx
            placeholder_to_content[f"__HEREDOC_{k}_bashsec__"] = command[h["content_start"] : h["content_end"]]

        tokens = tokenize(processed)
        segments = [s for s in split_segments(tokens) if s]
        for segment in segments:
            if _segment_base(segment) not in SKIP_PATH_CHECK_COMMANDS:
                continue
            for t in segment:
                if t.kind == "word" and _PLACEHOLDER_RE.match(t.text):
                    content = placeholder_to_content.get(t.text)
                    if content:
                        codes.append(content)

    for code in codes:
        r = check_python_code(code, cwd=cwd, allowed_dirs=allowed_dirs)
        if not r.allowed:
            return r
    return SecurityResult.allow()


_ABS_PATH_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9_.\-+])((/[A-Za-z0-9_.\-+]+){2,})")


def _check_xargs_pipeline(
    segments: list[list[Token]],
    *,
    allowed_dirs: list[str],
) -> SecurityResult:
    """xargs 注入: 管道前段输出的白名单外字面路径会被 xargs 当操作数。"""
    for i, seg in enumerate(segments):
        if _segment_base(seg) != "xargs":
            continue
        for prev in segments[:i]:
            text = " ".join(t.text for t in prev)
            for m in _ABS_PATH_IN_TEXT_RE.finditer(text):
                p = m.group(1)
                if p.startswith("/dev/"):
                    continue
                if any(_path_within_allowed(a, p) for a in allowed_dirs):
                    continue
                return SecurityResult.reject(
                    "xargs_path",
                    f"xargs 前段输出白名单外路径 '{p}',无法静态验证 xargs 的操作数",
                )
    return SecurityResult.allow()


def _check_subshells(
    segment: list[Token],
    *,
    cwd: str,
    allowed_dirs: list[str],
) -> SecurityResult:
    """处理 `( ... )` 子 shell: 递归 check_paths 组内文本。"""
    depth = 0
    start = -1
    i = 0
    n = len(segment)
    while i < n:
        t = segment[i]
        if t.kind == "operator" and t.text == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif t.kind == "operator" and t.text == ")":
            if depth == 1 and start != -1:
                inner = segment[start:i]
                if inner:
                    inner_text = " ".join(t2.text for t2 in inner if t2.kind != "operator")
                    r = check_paths(inner_text, cwd=cwd, allowed_dirs=allowed_dirs)
                    if not r.allowed:
                        return r
                depth = 0
                start = -1
            elif depth > 1:
                depth -= 1
        i += 1
    return SecurityResult.allow()
