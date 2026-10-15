"""shell 词法工具 —— 引号剥离 / heredoc 抽取 / 顶层 tokenizer。

移植自 claude_code_source 的 extractQuotedContent、stripSafeRedirections、
hasUnescapedChar、heredoc.ts 的关键语义。纯 stdlib,不依赖 shlex 的解析差异
(刻意用自己的状态机,因为 shlex 对 `$()`/heredoc/进程替换的处理与 bash 不一致)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# heredoc 占位符(带随机盐,防与命令字面量碰撞)
_HEREDOC_SALT = "bashsec"


def _next_placeholder(i: int) -> str:
    return f"__HEREDOC_{i}_{_HEREDOC_SALT}__"


# ---------------------------------------------------------------------------
# 引号内容抽取 (extractQuotedContent)
# ---------------------------------------------------------------------------
def extract_quoted_content(command: str, is_jq: bool = False) -> tuple[str, str, str]:
    """返回 (with_double_quotes, fully_unquoted, unquoted_keep_quote_chars)。

    语义与 bashSecurity.ts 的 extractQuotedContent 一致:
    - with_double_quotes: 保留双引号内容、剥掉单引号(双引号内不转义自身)。
    - fully_unquoted: 单双引号内容全部剥离。
    - unquoted_keep_quote_chars: 剥内容但保留引号分隔符(用于检测引号相邻的 #)。
    """
    with_double_quotes = ""
    fully_unquoted = ""
    keep_quote_chars = ""
    in_single = False
    in_double = False
    escaped = False

    for ch in command:
        if escaped:
            escaped = False
            if not in_single:
                with_double_quotes += ch
            if not in_single and not in_double:
                fully_unquoted += ch
                keep_quote_chars += ch
            continue

        if ch == "\\" and not in_single:
            escaped = True
            if not in_single:
                with_double_quotes += ch
            if not in_single and not in_double:
                fully_unquoted += ch
                keep_quote_chars += ch
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            keep_quote_chars += ch
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            keep_quote_chars += ch
            if not is_jq:
                continue

        if not in_single:
            with_double_quotes += ch
        if not in_single and not in_double:
            fully_unquoted += ch
            keep_quote_chars += ch

    return with_double_quotes, fully_unquoted, keep_quote_chars


# ---------------------------------------------------------------------------
# 安全重定向剥离 (stripSafeRedirections)
# ---------------------------------------------------------------------------
def strip_safe_redirections(content: str) -> str:
    """剥离 `2>&1` / `>/dev/null` / `</dev/null`。

    三条正则都必须带结尾边界 (?=\\\\s|$),否则 `> /dev/nullo` 会误剥前缀,让
    `echo hi > /dev/nullo` 逃过重定向检测。
    """
    content = re.sub(r"\s+2\s*>&\s*1(?=\s|$)", "", content)
    content = re.sub(r"[012]?\s*>\s*\/dev\/null(?=\s|$)", "", content)
    content = re.sub(r"\s*<\s*\/dev\/null(?=\s|$)", "", content)
    return content


# ---------------------------------------------------------------------------
# 未转义字符检测 (hasUnescapedChar)
# ---------------------------------------------------------------------------
def has_unescaped_char(content: str, char: str) -> bool:
    """内容中是否存在未转义的单个字符。反斜杠转义对跳过。"""
    if len(char) != 1:
        raise ValueError("has_unescaped_char only works with single characters")
    i = 0
    while i < len(content):
        if content[i] == "\\" and i + 1 < len(content):
            i += 2  # 跳过反斜杠 + 被转义字符
            continue
        if content[i] == char:
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# heredoc 抽取 (heredoc.ts 的简化但安全的移植)
# ---------------------------------------------------------------------------
# 完整版: /(?<!<)<<(?!<)(-)?[ \t]*(?:(['"])(\\?\w+)\2|\\?(\w+))/
_HEREDOC_START_FULL_RE = re.compile(
    r"(?<!<)<<(?!<)(-)?[ \t]*(?:(['\"])(\\?\w+)\2|\\?(\w+))"
)


def _advance_scan(command: str, up_to: int, start: int = 0):
    """增量扫描到 up_to,返回 (in_single, in_double, in_comment, pending_backslashes)。"""
    in_single = False
    in_double = False
    in_comment = False
    pending = 0
    dq_escape = False
    for i in range(start, up_to):
        ch = command[i]
        if ch == "\n":
            in_comment = False
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if dq_escape:
                dq_escape = False
                continue
            if ch == "\\":
                dq_escape = True
                continue
            if ch == '"':
                in_double = False
            continue
        # unquoted
        if ch == "\\":
            pending += 1
            continue
        escaped = pending % 2 == 1
        pending = 0
        if escaped:
            continue
        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif not in_comment and ch == "#":
            in_comment = True
    return in_single, in_double, in_comment, pending


def extract_heredocs(command: str, quoted_only: bool = True) -> tuple[str, list[dict]]:
    """抽取 heredoc 并替换为占位符。

    quoted_only=True  (interceptor 用): 只抽引号/反斜杠转义分隔符的 heredoc
      (主体是字面量,不展开);未引号 heredoc 的主体保留在命令里,让校验器能看到
      $() 等可执行替换。
    quoted_only=False (pathguard 用): 全部替换为占位符,tokenizer 无需处理主体。

    无法安全解析时返回原命令(安全方向 —— 原样内容会走完整校验链,多行命令会
    被 newline 校验拦下)。
    """
    if "<< " not in command and "<<\n" not in command and not command.endswith("<<"):
        # 快速路径:没有任何 heredoc 形态
        if "<<" not in command:
            return command, []
    if "<<" not in command:
        return command, []

    # 安全预检(继承 heredoc.ts):
    # 1) ANSI-C / locale 引号会让我们的引号跟踪器失同步
    if re.search(r"\$['\"]", command):
        return command, []
    # 2) 第一个 << 之前的反引号(反引号嵌套解析复杂,且是 shell_eof_token)
    first_heredoc = command.find("<<")
    if first_heredoc > 0 and "`" in command[:first_heredoc]:
        return command, []
    # 3) 第一个 << 之前未配对的 (( —— << 可能是算术位移而不是 heredoc
    if first_heredoc > 0:
        before = command[:first_heredoc]
        if before.count("((") > before.count("))"):
            return command, []

    matches: list[dict] = []
    skipped_ranges: list[tuple[int, int]] = []

    for m in _HEREDOC_START_FULL_RE.finditer(command):
        start_index = m.start()
        in_single, in_double, in_comment, pending = _advance_scan(command, start_index)
        if in_single or in_double or in_comment:
            continue
        if pending % 2 == 1:  # \<< 不是 heredoc
            continue

        # 是否落在已跳过的(未引号)heredoc 主体内
        if any(s < start_index < e for s, e in skipped_ranges):
            continue

        is_dash = m.group(1) == "-"
        quote_char = m.group(2)
        delimiter = m.group(3) or m.group(4) or ""
        full_match = m.group(0)
        operator_end = start_index + len(full_match)

        # 校验 1: 引号分隔符必须被完整匹配(防止 <<"EO F" 这类提前截断)
        if quote_char and (operator_end - 1 >= len(command) or command[operator_end - 1] != quote_char):
            continue

        is_quoted_or_escaped = bool(quote_char) or ("\\" in full_match)

        # 校验 2: 下一个字符必须是 bash 元字符(词终止符)
        if operator_end < len(command):
            nxt = command[operator_end]
            if not re.match(r"^[ \t\n|&;()<>]$", nxt):
                continue

        # 找到逻辑行上第一个"不在引号内"的换行
        first_nl = -1
        in_s2 = in_d2 = False
        dq_esc = False
        k = operator_end
        while k < len(command):
            ch = command[k]
            if in_s2:
                if ch == "'":
                    in_s2 = False
                k += 1
                continue
            if in_d2:
                if dq_esc:
                    dq_esc = False
                elif ch == "\\":
                    dq_esc = True
                elif ch == '"':
                    in_d2 = False
                k += 1
                continue
            if ch == "\n":
                first_nl = k - operator_end
                break
            # unquoted: 反斜杠转义检测
            bc = 0
            j = k - 1
            while j >= operator_end and command[j] == "\\":
                bc += 1
                j -= 1
            if bc % 2 == 1:
                k += 1
                continue
            if ch == "'":
                in_s2 = True
            elif ch == '"':
                in_d2 = True
            k += 1
        if first_nl == -1:
            continue

        # 同行内容尾部奇数个反斜杠 → 续行,抽取会误判,bail
        same_line = command[operator_end : operator_end + first_nl]
        trailing_bs = len(same_line) - len(same_line.rstrip("\\"))
        if trailing_bs % 2 == 1:
            continue

        content_start = operator_end + first_nl
        after_nl = command[content_start + 1 :]
        content_lines = after_nl.split("\n")

        # 找结束分隔符行
        closing = -1
        for i, line in enumerate(content_lines):
            stripped = line.replace("\t" * 100, "", 1) if is_dash else line
            if is_dash:
                stripped = re.sub(r"^\t*", "", line)
            else:
                stripped = line
            if stripped == delimiter:
                closing = i
                break
            # PST_EOFTOKEN 式提前闭合检测: 分隔符后跟 shell 元字符 → bail
            if len(stripped) > len(delimiter) and stripped.startswith(delimiter):
                after_d = stripped[len(delimiter)]
                if re.match(r"^[)}`|&;(<>]$", after_d):
                    closing = -1
                    break
        if closing == -1:
            continue

        content_end = content_start + 1 + len("\n".join(content_lines[: closing + 1]))

        # quoted_only 模式下,未引号 heredoc 记录范围但跳过
        if quoted_only and not is_quoted_or_escaped:
            skipped_ranges.append((content_start, content_end))
            continue

        matches.append(
            {
                "start": start_index,
                "operator_end": operator_end,
                "content_start": content_start,
                "content_end": content_end,
                "delimiter": delimiter,
            }
        )

    if not matches:
        return command, []

    # 过滤嵌套 heredoc + 去重 content_start(同行使索引错乱)
    top = [
        h for h in matches
        if not any(
            h["start"] > o["content_start"] and h["start"] < o["content_end"]
            for o in matches
            if o is not h
        )
    ]
    content_starts = {h["content_start"] for h in top}
    if not top or len(content_starts) < len(top):
        return command, []

    top.sort(key=lambda h: h["content_end"], reverse=True)
    processed = command
    for idx, h in enumerate(top):
        placeholder = _next_placeholder(len(top) - 1 - idx)
        processed = (
            processed[: h["start"]]
            + placeholder
            + processed[h["operator_end"] : h["content_start"]]
            + processed[h["content_end"] :]
        )
    return processed, top


# ---------------------------------------------------------------------------
# 顶层 tokenizer(quote / $() / ${} / 反引号 / 进程替换感知)
# ---------------------------------------------------------------------------
@dataclass
class Token:
    kind: str  # 'word' | 'redirect' | 'operator' | 'subst'
    text: str
    # redirect 专用: 目标是否为 fd 数字(如 2>&1 的 1)
    fd_target: bool = False


def _scan_balanced(command: str, i: int, open_ch: str, close_ch: str) -> int:
    """从 i(指向开括号)扫描到配对的 close_ch,返回 close 下标(含)。跳过引号。"""
    depth = 0
    in_s = in_d = False
    dq_esc = False
    k = i
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
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def tokenize(command: str) -> list[Token]:
    """把命令切成一串 token。

    - word:     普通参数(已去引号)
    - subst:    $() / ${} / $[] / 反引号 —— 路径检查跳过(interceptor 递归处理)
    - redirect: > >> < << 2> &> >& 等
    - operator: && || | & ; ( )  以及换行(以 \n 形式存在)
    """
    tokens: list[Token] = []
    i = 0
    n = len(command)
    word_buf: list[str] = []

    def flush_word():
        if word_buf:
            tokens.append(Token("word", "".join(word_buf)))
            word_buf.clear()

    while i < n:
        ch = command[i]

        # --- 空白 ---
        if ch in " \t":
            flush_word()
            i += 1
            continue
        if ch == "\n":
            flush_word()
            tokens.append(Token("operator", "\n"))
            i += 1
            continue

        # --- 反斜杠转义 ---
        if ch == "\\":
            if i + 1 < n and command[i + 1] == "\n":
                # 续行: 跳过两个字符
                i += 2
                continue
            if i + 1 < n:
                word_buf.append(command[i + 1])  # 去转义
                i += 2
                continue
            word_buf.append("\\")
            i += 1
            continue

        # --- 单引号 ---
        if ch == "'":
            flush_word()
            j = i + 1
            buf: list[str] = []
            while j < n and command[j] != "'":
                buf.append(command[j])
                j += 1
            word_buf.append("".join(buf))
            i = j + 1 if j < n else n
            continue

        # --- 双引号 ---
        if ch == '"':
            flush_word()
            j = i + 1
            buf = []
            while j < n and command[j] != '"':
                if command[j] == "\\" and j + 1 < n and command[j + 1] in '"\\$`':
                    buf.append(command[j + 1])
                    j += 2
                    continue
                buf.append(command[j])
                j += 1
            word_buf.append("".join(buf))
            i = j + 1 if j < n else n
            continue

        # --- 命令替换 / 参数替换 ---
        if ch == "$" and i + 1 < n and command[i + 1] in "( { [":
            flush_word()
            if command[i + 1] == "(" and i + 2 < n and command[i + 2] == "(":
                # $(( 算术展开 —— 需要匹配 ))
                close = _scan_balanced(command, i, "(", ")")
                if close == -1:
                    tokens.append(Token("subst", command[i:]))
                    i = n
                    continue
                # 检查是否真的以 )) 结尾
                if command[close - 1 : close + 1] == "))" or close >= 1 and command[close] == ")":
                    tokens.append(Token("subst", command[i : close + 1]))
                    i = close + 1
                    continue
                # 保守: 不识别为完整替换,交给后续处理
                tokens.append(Token("subst", command[i:close+1]))
                i = close + 1
                continue
            open_ch = "(" if command[i + 1] == "(" else ("{" if command[i + 1] == "{" else "[")
            close_ch = ")" if open_ch == "(" else ("}" if open_ch == "{" else "]")
            close = _scan_balanced(command, i + 1, open_ch, close_ch)
            if close == -1:
                tokens.append(Token("subst", command[i:]))
                i = n
                continue
            tokens.append(Token("subst", command[i : close + 1]))
            i = close + 1
            continue

        # --- 反引号 ---
        if ch == "`":
            flush_word()
            j = i + 1
            buf = []
            while j < n and command[j] != "`":
                if command[j] == "\\" and j + 1 < n and command[j + 1] == "`":
                    buf.append("`")
                    j += 2
                    continue
                buf.append(command[j])
                j += 1
            tokens.append(Token("subst", "`" + "".join(buf) + "`"))
            i = j + 1 if j < n else n
            continue

        # --- 重定向 ---
        if ch == ">" or ch == "<":
            flush_word()
            if ch == "<" and i + 1 < n and command[i + 1] == "<":
                if i + 2 < n and command[i + 2] == "<":
                    tokens.append(Token("redirect", "<<<"))
                    i += 3
                    continue
                tokens.append(Token("redirect", "<<"))
                i += 2
                continue
            if ch == "<" and i + 1 < n and command[i + 1] == ">":
                tokens.append(Token("redirect", "<>"))
                i += 2
                continue
            if ch == ">" and i + 1 < n and command[i + 1] == ">":
                tokens.append(Token("redirect", ">>"))
                i += 2
                continue
            if ch == ">" and i + 1 < n and command[i + 1] == "&":
                tokens.append(Token("redirect", ">&"))
                i += 2
                continue
            if ch == ">" and i + 1 < n and command[i + 1] == "|":
                tokens.append(Token("redirect", ">|"))
                i += 2
                continue
            tokens.append(Token("redirect", ch))
            i += 1
            continue

        # --- fd 前缀重定向 (2> 3<) ---
        if ch.isdigit():
            j = i
            while j < n and command[j].isdigit():
                j += 1
            if j < n and command[j] in "<>":
                flush_word()
                if command[j] == "<" and j + 1 < n and command[j + 1] == "<":
                    tokens.append(Token("redirect", command[i : j + 2]))
                    i = j + 2
                    continue
                tokens.append(Token("redirect", command[i : j + 1]))
                i = j + 1
                continue
            # 否则当作普通 word 继续
            word_buf.append(ch)
            i += 1
            continue

        # --- 运算符 ---
        if ch in ";|&":
            flush_word()
            two = command[i : i + 2]
            if two in ("&&", "||", "|&", ";;"):
                tokens.append(Token("operator", two))
                i += 2
                continue
            tokens.append(Token("operator", ch))
            i += 1
            continue

        if ch in "()":
            flush_word()
            tokens.append(Token("operator", ch))
            i += 1
            continue

        # --- 普通 word 字符 ---
        word_buf.append(ch)
        i += 1

    flush_word()
    return tokens


def split_segments(tokens: list[Token]) -> list[list[Token]]:
    """按顶层运算符把 token 切成分段(&& || ; | & 以及换行)。() 保留在段内。"""
    segments: list[list[Token]] = []
    current: list[Token] = []
    for t in tokens:
        if t.kind == "operator" and t.text in ("&&", "||", ";", "|", "&", ";;", "\n"):
            if current:
                segments.append(current)
                current = []
            continue
        current.append(t)
    if current:
        segments.append(current)
    return segments


def extract_redirect_targets(tokens: list[Token]) -> list[tuple[str, str, bool]]:
    """返回 [(operator, target_text, is_fd_target)]。

    operator ∈ {>, >>, 2>, 3>, &>, >&, <, <<, <<<, <>}。
    target 为 fd 数字(如 `2>&1` 的 `1`)时 is_fd_target=True,调用方跳过。
    """
    out: list[tuple[str, str, bool]] = []
    for idx, t in enumerate(tokens):
        if t.kind != "redirect":
            continue
        # 找下一个 word / subst token 作为目标
        target = ""
        for nxt in tokens[idx + 1 :]:
            if nxt.kind in ("word", "subst"):
                target = nxt.text
                break
            if nxt.kind == "redirect":
                break
            # operator 或空 —— 继续找(如 `> file > file2`)
        is_fd = target.isdigit() and t.text in (">&", ">&2", ">&1", "2>", "1>")
        out.append((t.text, target, is_fd))
    return out


def get_positional_words(tokens: list[Token]) -> list[str]:
    """返回分段内非重定向目标的 word token(未解析 flag 语义,供 pathguard 精排)。"""
    redirect_indices = [idx for idx, t in enumerate(tokens) if t.kind == "redirect"]
    target_indices: set[int] = set()
    for idx in redirect_indices:
        for nxt_idx in range(idx + 1, len(tokens)):
            nxt = tokens[nxt_idx]
            if nxt.kind in ("word", "subst"):
                target_indices.add(nxt_idx)
                break
            if nxt.kind == "redirect" or (nxt.kind == "operator" and nxt.text in "();&|"):
                break
    return [t.text for idx, t in enumerate(tokens) if t.kind == "word" and idx not in target_indices]
