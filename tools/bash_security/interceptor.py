"""命令形状拦截器 —— check_command。

移植 bashSecurity.ts 的校验器电池。策略取舍(相对源码):
- 源码对 `$()` / `${}` / `$[]` 一律 ask;我们**递归分析内部命令**,允许
  `echo $(date)` 这类良性用法,但 `echo $(cat /etc/passwd)`(路径越界)或
  `rm -rf $(pwd)`(rm 操作数含替换)仍会被拦。
- 本框架无人在环,源码的 ask/deny 统一收敛为 reject(见 result.py)。

pathguard.check_paths 负责路径边界;这里负责命令形状(替换/混淆/危险命令)。
"""

from __future__ import annotations

import re

from . import patterns as P
from .result import SecurityResult
from .shelllex import (
    extract_heredocs,
    extract_quoted_content,
    has_unescaped_char,
    strip_safe_redirections,
)

MAX_SUBSTITUTION_DEPTH = 3

# ---------------------------------------------------------------------------
# 基础命令抽取 (validateZshDangerousCommands + stripSafeWrappers 语义)
# ---------------------------------------------------------------------------
# 包装命令: 剥掉后看真正执行的 base command
_WRAPPERS = frozenset({"nohup", "time", "timeout", "nice", "env", "setsid", "sudo", "doas", "envrun"})
# Zsh precommand modifiers
_PRECOMMANDS = frozenset({"command", "builtin", "noglob", "nocorrect"})

_EVAL_LIKE_BUILTINS = frozenset({
    "eval", "trap", "enable", "hash", "coproc", "mapfile", "readarray",
    "bind", "complete", "compgen", "let", "builtin", "exec", "fc", "command",
})


def _is_env_assignment(token: str) -> bool:
    return bool(re.match(r"^[A-Za-z_]\w*=.*$", token)) and "=" in token and not token.startswith("=")


def extract_base_command(tokens: list) -> str:
    """从 token 列表抽 base command: 跳过环境赋值、precommand 修饰符、包装命令。"""
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        text = getattr(t, "text", t)
        if t.kind == "subst":
            # 命令位置出现替换 (e.g. $(x) ...) —— 无法静态判定 base,保守返回
            return ""
        if t.kind != "word":
            i += 1
            continue
        if _is_env_assignment(text):
            i += 1
            continue
        if text in _PRECOMMANDS:
            i += 1
            continue
        if text in _WRAPPERS:
            # 消费包装命令及其参数: timeout N, nice -n N, env -i
            i += 1
            j = i
            while j < n and j < i + 2:
                t2 = tokens[j]
                if t2.kind == "word" and (t2.text.startswith("-") or t2.text.isdigit()):
                    j += 1
                    continue
                break
            i = j
            continue
        # 路径形式命令名: /usr/bin/python3 -> python3
        return text.rsplit("/", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# 反斜杠转义检测器 (与 bashSecurity.ts 的 scanner 语义一致)
# ---------------------------------------------------------------------------
def _has_backslash_escaped_whitespace(command: str) -> bool:
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "\\" and not in_single:
            if not in_double and i + 1 < len(command) and command[i + 1] in (" ", "\t"):
                return True
            i += 2  # 跳过转义字符(双引号内 \\ 也是转义对)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        i += 1
    return False


def _has_backslash_escaped_operator(command: str) -> bool:
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "\\" and not in_single:
            if not in_double and i + 1 < len(command) and command[i + 1] in P.SHELL_OPERATORS:
                return True
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        i += 1
    return False


def _has_carriage_return_outside_double_quote(command: str) -> bool:
    in_single = in_double = escaped = False
    for ch in command:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and not in_single:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "\r" and not in_double:
            return True
    return False


# ---------------------------------------------------------------------------
# brace expansion (移植 validateBraceExpansion 的安全子集)
# ---------------------------------------------------------------------------
def _is_escaped_at(content: str, pos: int) -> bool:
    count = 0
    i = pos - 1
    while i >= 0 and content[i] == "\\":
        count += 1
        i -= 1
    return count % 2 == 1


def _has_brace_expansion(content: str) -> str | None:
    """返回 None 或命中原因。"""
    unescaped_open = sum(
        1 for i, ch in enumerate(content) if ch == "{" and not _is_escaped_at(content, i)
    )
    unescaped_close = sum(
        1 for i, ch in enumerate(content) if ch == "}" and not _is_escaped_at(content, i)
    )
    if unescaped_open > 0 and unescaped_close > unescaped_open:
        return "excess closing braces after quote stripping"

    # 引号内单个花括号 + 未引号花括号 → 混淆尝试
    if unescaped_open > 0 and re.search(r"['\"][{}]['\"]", content):
        return "quoted brace character inside brace context"

    # 深度匹配找外层逗号 / ..
    for i, ch in enumerate(content):
        if ch != "{" or _is_escaped_at(content, i):
            continue
        depth = 1
        close = -1
        for j in range(i + 1, len(content)):
            c = content[j]
            if c == "{" and not _is_escaped_at(content, j):
                depth += 1
            elif c == "}" and not _is_escaped_at(content, j):
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close == -1:
            continue
        inner_depth = 0
        for k in range(i + 1, close):
            c = content[k]
            if c == "{" and not _is_escaped_at(content, k):
                inner_depth += 1
            elif c == "}" and not _is_escaped_at(content, k):
                inner_depth -= 1
            elif inner_depth == 0:
                if c == "," or (c == "." and k + 1 < close and content[k + 1] == "."):
                    return "brace expansion could alter command parsing"
    return None


# ---------------------------------------------------------------------------
# 校验器电池
# ---------------------------------------------------------------------------
def check_command(
    command: str,
    *,
    cwd: str | None = None,
    allowed_dirs: list[str] | None = None,
    _depth: int = 0,
) -> SecurityResult:
    """检查命令形状。命中任何危险模式 → reject。

    allowed_dirs/cwd 提供时,`$()`/反引号内部命令还会递归做路径边界检查
    (由 pathguard.check_paths 完成,循环导入延迟到函数体内)。
    """
    if _depth > MAX_SUBSTITUTION_DEPTH:
        return SecurityResult.reject(
            "substitution_depth",
            "命令替换嵌套过深,无法静态验证内部命令",
        )

    original = command
    if not command or not command.strip():
        return SecurityResult.allow()

    # 1. 控制字符 (先于一切,避免污染其它校验器)
    if P.CONTROL_CHAR_RE.search(original):
        return SecurityResult.reject(
            "control_chars", "命令包含非打印控制字符,可能绕过安全检查"
        )
    if P.UNICODE_WS_RE.search(original):
        return SecurityResult.reject(
            "unicode_whitespace", "命令包含 Unicode 空白,shell 解析可能不一致"
        )

    # 2. 反斜杠转义 空白 / 运算符
    if _has_backslash_escaped_whitespace(original):
        return SecurityResult.reject(
            "backslash_whitespace", "命令含反斜杠转义的空白,可改变命令解析"
        )
    if _has_backslash_escaped_operator(original):
        return SecurityResult.reject(
            "backslash_operator", "命令含反斜杠转义的 shell 运算符(; | & < >),可隐藏命令结构"
        )

    # 3. Zsh 混淆: =cmd 展开 / 进程替换 / 直接判死的展开形态
    if P.ZSH_EQUALS_EXPANSION_RE.search(original):
        return SecurityResult.reject("zsh_equals", "命令含 Zsh equals 展开 (=cmd)")
    for pat, label in P.PROCESS_SUBSTITUTION_PATTERNS:
        if pat.search(original):
            return SecurityResult.reject("process_substitution", f"命令含 {label}")
    for pat, label in P.DANGEROUS_EXPANSION_PATTERNS:
        if pat.search(original):
            return SecurityResult.reject("dangerous_expansion", f"命令含 {label}")

    # 4. heredoc 抽取(只抽引号/转义分隔符;未引号主体保留给校验器)
    processed, _ = extract_heredocs(original, quoted_only=True)

    # 5. 换行检测 (用 strip 前的 fullyUnquoted,防 `>/dev/null` 制造假续行)
    _, fully_unquoted, keep_quote_chars = extract_quoted_content(processed)
    if P.NEWLINE_SPLIT_COMMAND_RE.search(fully_unquoted):
        return SecurityResult.reject(
            "newline", "命令含换行,可能分隔多条命令(除反斜杠续行)"
        )

    # 6. 回车符在双引号外 → 解析差异
    if _has_carriage_return_outside_double_quote(original):
        return SecurityResult.reject(
            "carriage_return", "命令含回车符(\\r),bash 与解析器切分不一致"
        )

    # 7. 未转义反引号
    with_double_quotes, _, _ = extract_quoted_content(processed)
    if has_unescaped_char(with_double_quotes, "`"):
        return SecurityResult.reject("backtick", "命令含未转义反引号(`),用于命令替换")

    # 8. 混淆引号
    if P.ANSI_C_QUOTING_RE.search(original):
        return SecurityResult.reject("obfuscated_quoting", "命令含 ANSI-C 引号($'...'),可隐藏字符")
    if P.LOCALE_QUOTING_RE.search(original):
        return SecurityResult.reject("obfuscated_quoting", "命令含 locale 引号($\"...\"),可隐藏字符")
    if P.EMPTY_SPECIAL_QUOTE_DASH_RE.search(original) or P.EMPTY_QUOTE_DASH_RE.search(original):
        return SecurityResult.reject("obfuscated_quoting", "命令含空引号接短横线(潜在 flag 混淆)")
    if P.QUOTED_DASH_OBFUSCATION_RE.search(original):
        return SecurityResult.reject("obfuscated_quoting", "命令含引号拼接短横线(潜在 flag 混淆)")

    # 9. 引号内 shell 元字符 (find -name/-path/-regex 注入)
    if P.SHELL_METACHARS_IN_QUOTES_RE.search(with_double_quotes):
        return SecurityResult.reject(
            "shell_metachars_quoted", "命令引号参数内含 shell 元字符(; | &)"
        )
    for pat in P.SHELL_METACHARS_IN_GLOB_PATTERNS:
        if pat.search(with_double_quotes):
            return SecurityResult.reject(
                "shell_metachars_quoted", "命令 find 谓词引号内藏 shell 元字符"
            )

    # 10. 变量紧邻重定向/管道
    stripped = strip_safe_redirections(fully_unquoted)
    for pat in P.DANGEROUS_VARIABLES_PATTERNS:
        if pat.search(stripped):
            return SecurityResult.reject(
                "dangerous_variables", "命令含变量紧邻重定向或管道(<>|)"
            )

    # 11. /proc/*/environ
    if P.PROC_ENVIRON_ACCESS_RE.search(original):
        return SecurityResult.reject(
            "proc_environ", "命令访问 /proc/*/environ,可能泄露环境变量"
        )

    # 12. IFS 注入
    if P.IFS_INJECTION_RE.search(original):
        return SecurityResult.reject("ifs_injection", "命令使用 IFS 变量,可能绕过校验")

    # 13. 中间词 # (解析差异)
    if re.search(r"\S(?<!\$\{)#", keep_quote_chars):
        return SecurityResult.reject("mid_word_hash", "命令含词中间 #,解析器与 bash 处理不一致")

    # 14. brace expansion 混淆
    brace_reason = _has_brace_expansion(fully_unquoted)
    if brace_reason:
        return SecurityResult.reject("brace_expansion", f"命令含 brace expansion({brace_reason})")

    # 15. base command 相关检查
    base = _extract_base_for_checks(original)
    if base in P.ZSH_DANGEROUS_COMMANDS:
        return SecurityResult.reject(
            "zsh_dangerous", f"命令使用 Zsh 专有命令 '{base}',可绕过安全检查"
        )
    if base == "fc" and re.search(r"\s-\S*e", original):
        return SecurityResult.reject(
            "zsh_dangerous", "命令使用 'fc -e',可通过编辑器执行任意命令"
        )
    if base == "jq":
        if P.JQ_SYSTEM_FUNCTION_RE.search(original):
            return SecurityResult.reject("jq_danger", "jq 命令含 system() 函数,执行任意命令")
        after_jq = original[original.find("jq") + 2 :].strip()
        if P.JQ_DANGEROUS_FLAGS_RE.search(after_jq):
            return SecurityResult.reject("jq_danger", "jq 命令含危险 flag,可读任意文件")

    if base in _EVAL_LIKE_BUILTINS:
        # 源码同款豁免: 存在性查询 / 历史列表 / 全补全
        if base == "command" and re.search(r"\bcommand\s+-[vV]\b", original):
            pass
        elif base == "fc" and re.search(r"\bfc\s+-[ln]", original):
            pass
        elif base == "compgen" and re.search(r"\bcompgen\s+-[cfv]\b", original):
            pass
        else:
            return SecurityResult.reject(
                "eval_like", f"命令 '{base}' 是 eval 类内建,可执行任意代码"
            )

    # 16. 现有 dangerous.py 的 9 条规则(超集)
    for pat, label in P.DANGEROUS_COMMAND_PATTERNS:
        if pat.search(original):
            return SecurityResult.reject("dangerous_command", f"命令匹配危险模式: {label}")

    # 17. 命令替换递归分析 ($() `${}` `$[]` 反引号)
    subst_result = _check_substitutions(original, cwd=cwd, allowed_dirs=allowed_dirs, depth=_depth)
    if not subst_result.allowed:
        return subst_result

    return SecurityResult.allow()


def _extract_base_for_checks(command: str) -> str:
    """抽 base command(处理 env 赋值 / precommand / 包装命令)。"""
    from .shelllex import split_segments, tokenize

    tokens = tokenize(command)
    segments = [s for s in split_segments(tokens) if s]
    if not segments:
        return ""
    return extract_base_command(segments[0])


# ---------------------------------------------------------------------------
# 命令替换递归分析
# ---------------------------------------------------------------------------
def _find_substitution_spans(command: str) -> list[tuple[str, str]]:
    """返回 [(kind, inner)] —— 用状态机扫描,避免正则嵌套误配。kind ∈ $() ${} $[] `。"""
    spans: list[tuple[str, str]] = []
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "$" and i + 1 < n:
            nxt = command[i + 1]
            if nxt == "(":
                close = _scan_parens(command, i + 1, ")", "(")
                if close != -1:
                    spans.append(("$()", command[i + 2 : close]))
                    i = close + 1
                    continue
            elif nxt == "{":
                close = _scan_balanced(command, i + 1, "{", "}")
                if close != -1:
                    spans.append(("${}", command[i + 2 : close]))
                    i = close + 1
                    continue
            elif nxt == "[":
                close = _scan_balanced(command, i + 1, "[", "]")
                if close != -1:
                    spans.append(("$[]", command[i + 2 : close]))
                    i = close + 1
                    continue
        if ch == "`":
            close = command.find("`", i + 1)
            if close != -1:
                spans.append(("backtick", command[i + 1 : close]))
                i = close + 1
                continue
        i += 1
    return spans


def _scan_parens(command: str, start: int, close_ch: str, open_ch: str) -> int:
    """扫描到配对 close_ch,处理嵌套与引号。返回 close 下标或 -1。

    从 start(通常是开括号本身)起扫: 首字符 open_ch 把 depth 推到 1,
    匹配的 close_ch 把 depth 减回 0 时返回(必须在 depth==0 判断**之后**减,
    否则最外层闭合括号会漏)。
    """
    depth = 0
    in_s = in_d = False
    dq_esc = False
    k = start
    while k < len(command):
        ch = command[k]
        if in_s:
            if ch == "'":
                in_s = False
            k += 1
            continue
        if in_d:
            if dq_esc:
                dq_esc = False
            elif ch == "\\":
                dq_esc = True
            elif ch == '"':
                in_d = False
            k += 1
            continue
        if ch == "'":
            in_s = True
        elif ch == '"':
            in_d = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            if depth <= 1:
                return k
            depth -= 1
        k += 1
    return -1


def _scan_balanced(command: str, start: int, open_ch: str, close_ch: str) -> int:
    return _scan_parens(command, start, close_ch, open_ch)


def _check_substitutions(
    command: str,
    *,
    cwd: str | None,
    allowed_dirs: list[str] | None,
    depth: int,
) -> SecurityResult:
    spans = _find_substitution_spans(command)
    if not spans:
        return SecurityResult.allow()

    for kind, inner in spans:
        inner = inner.strip()
        if not inner:
            continue
        # 递归: 先命令形状,再(若给了目录)路径边界
        r = check_command(inner, cwd=cwd, allowed_dirs=allowed_dirs, _depth=depth + 1)
        if not r.allowed:
            return SecurityResult.reject(
                "command_substitution",
                f"{kind} 内部命令被拦截: {r.reason}",
            )
        if allowed_dirs is not None:
            from .pathguard import check_paths

            pr = check_paths(inner, cwd=cwd or ".", allowed_dirs=allowed_dirs)
            if not pr.allowed:
                return SecurityResult.reject(
                    "command_substitution",
                    f"{kind} 内部路径越界: {pr.reason}",
                )
    return SecurityResult.allow()
