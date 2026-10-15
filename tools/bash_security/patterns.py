"""正则常量 —— 移植自 claude_code_source/src/tools/BashTool/bashSecurity.ts。

只放「常量」,不放逻辑;消费方(interceptor/pathguard)按需组合。纯 stdlib。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 命令替换 / 进程替换 / Zsh 混淆模式 (COMMAND_SUBSTITUTION_PATTERNS)
# ---------------------------------------------------------------------------
# 注意: `$(`、`${`、`$[` 在这里**不**被直接判死 —— interceptor 会递归分析其内部
# 命令(允许 echo $(date) 这类良性用法)。本表只保留进程替换与 Zsh 特有的混淆形态。
PROCESS_SUBSTITUTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<\("), "process substitution <()"),
    (re.compile(r">\("), "process substitution >()"),
    (re.compile(r"=\("), "Zsh process substitution =()"),
]

# Zsh equals expansion: 词首 `=cmd` 展开为 `$(which cmd)`。
# 只匹配词首 `=` 后跟命令名字符(VAR=val 赋值不匹配)。
ZSH_EQUALS_EXPANSION_RE = re.compile(r"(?:^|[\s;&|])=[a-zA-Z_]")

# 直接判死的替换/展开形态(其余如 ~[]、glob qualifier、always block、PowerShell 注释)
DANGEROUS_EXPANSION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"~\["), "Zsh-style parameter expansion"),
    (re.compile(r"\(e:"), "Zsh-style glob qualifiers"),
    (re.compile(r"\(\+"), "Zsh glob qualifier with command execution"),
    (re.compile(r"\}\s*always\s*\{"), "Zsh always block (try/always construct)"),
    # Defense in depth: 即使我们不在 PowerShell 里执行也拦截,防止未来引入
    (re.compile(r"<#"), "PowerShell comment syntax"),
]

# ---------------------------------------------------------------------------
# Zsh 危险命令 (ZSH_DANGEROUS_COMMANDS)
# ---------------------------------------------------------------------------
ZSH_DANGEROUS_COMMANDS = frozenset({
    "zmodload", "emulate", "sysopen", "sysread", "syswrite", "sysseek",
    "zpty", "ztcp", "zsocket", "mapfile",
    "zf_rm", "zf_mv", "zf_ln", "zf_chmod", "zf_chown", "zf_mkdir",
    "zf_rmdir", "zf_chgrp",
})

ZSH_PRECOMMAND_MODIFIERS = frozenset({"command", "builtin", "noglob", "nocorrect"})

# ---------------------------------------------------------------------------
# 控制字符 / Unicode 空白 (bash 静默丢弃、但会迷惑我们的校验器)
# ---------------------------------------------------------------------------
# 非打印控制字符: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F。排除 tab(0x09)、
# LF(0x0A)、CR(0x0D) —— 它们由各自校验器处理。
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# shell-quote 当作词分隔符、bash 却当作字面词内容的 Unicode 空白。
UNICODE_WS_RE = re.compile("[\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF]")

# ---------------------------------------------------------------------------
# 反斜杠转义 shell 操作符 (validateBackslashEscapedOperators)
# ---------------------------------------------------------------------------
SHELL_OPERATORS = frozenset({";", "|", "&", "<", ">"})

# ---------------------------------------------------------------------------
# 换行检测 (validateNewlines): LF/CR 后跟非空白 = 疑似多命令,除非是 `\` 续行。
# JS 源: /(?<![\s]\\)[\n\r]\s*\S/  —— 负向后瞻(空白+反斜杠)。
# ---------------------------------------------------------------------------
NEWLINE_SPLIT_COMMAND_RE = re.compile(r"(?<![\s]\\\n)[\r\n]\s*\S")

# ---------------------------------------------------------------------------
# 变量危险上下文 (validateDangerousVariables): $VAR 紧邻 <>| 重定向/管道
# ---------------------------------------------------------------------------
DANGEROUS_VARIABLES_PATTERNS = [
    re.compile(r"[<>|]\s*\$[A-Za-z_]"),
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\s*[|<>]"),
]

# ---------------------------------------------------------------------------
# 引号内 shell 元字符 (validateShellMetacharacters): 引号参数里藏 ; | &
# ---------------------------------------------------------------------------
SHELL_METACHARS_IN_QUOTES_RE = re.compile(r'(?:^|\s)["\'][^"\']*[;&][^"\']*["\'](?:\s|$)')
SHELL_METACHARS_IN_GLOB_PATTERNS = [
    re.compile(r'-name\s+["\'][^"\']*[;|&][^"\']*["\']'),
    re.compile(r'-path\s+["\'][^"\']*[;|&][^"\']*["\']'),
    re.compile(r'-iname\s+["\'][^"\']*[;|&][^"\']*["\']'),
    re.compile(r'-regex\s+["\'][^"\']*[;&][^"\']*["\']'),
]

# ---------------------------------------------------------------------------
# IFS 注入 (validateIFSInjection)
# ---------------------------------------------------------------------------
IFS_INJECTION_RE = re.compile(r"\$IFS|\$\{[^}]*IFS")

# ---------------------------------------------------------------------------
# /proc/*/environ 访问 (validateProcEnvironAccess)
# ---------------------------------------------------------------------------
PROC_ENVIRON_ACCESS_RE = re.compile(r"/proc/.*/environ")

# ---------------------------------------------------------------------------
# jq 危险用法 (validateJqCommand)
# ---------------------------------------------------------------------------
JQ_SYSTEM_FUNCTION_RE = re.compile(r"\bsystem\s*\(")
JQ_DANGEROUS_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:-f\b|--from-file|--rawfile|--slurpfile|-L\b|--library-path)"
)

# ---------------------------------------------------------------------------
# 混淆引号 (validateObfuscatedFlags)
# ---------------------------------------------------------------------------
ANSI_C_QUOTING_RE = re.compile(r"\$'[^']*'")          # $'...'
LOCALE_QUOTING_RE = re.compile(r'\$"[^"]*"')           # $"..."
EMPTY_SPECIAL_QUOTE_DASH_RE = re.compile(r"\$['\"]{2}\s*-")
EMPTY_QUOTE_DASH_RE = re.compile(r"(?:^|\s)(?:''|\"\")+\s*-")
QUOTED_DASH_OBFUSCATION_RE = re.compile(r"(?:^|\s)(?:''|\"\")+['\"]-")

# ---------------------------------------------------------------------------
# 现有 dangerous.py 的 9 条规则(保留,作为超集的一部分)
# ---------------------------------------------------------------------------
DANGEROUS_COMMAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/\s*$"), "rm -rf /"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/\s"), "rm -rf /<path>"),
    (re.compile(r"\bdd\s+.*of=/dev/[a-z]"), "dd of=/dev/..."),
    (re.compile(r"\bmkfs\b"), "mkfs"),
    (re.compile(r":\(\)\s*\{.*:.*\|.*:.*\}.*;.*:"), "fork bomb"),
    (re.compile(r"\bchmod\s+(-[a-zA-Z]*R[a-zA-Z]*|)\s*777\s+/"), "chmod -R 777 /"),
    (re.compile(r"\bshutdown\b"), "shutdown"),
    (re.compile(r"\breboot\b"), "reboot"),
    (re.compile(r"\binit\s+0"), "init 0"),
]

# ---------------------------------------------------------------------------
# 路径边界 (utils/permissions/pathValidation.ts)
# ---------------------------------------------------------------------------
GLOB_PATTERN_REGEX = re.compile(r"[*?[\]{}]")

WINDOWS_DRIVE_ROOT_REGEX = re.compile(r"^[A-Za-z]:\/?$")
WINDOWS_DRIVE_CHILD_REGEX = re.compile(r"^[A-Za-z]:\/[^/]+$")

# 危险的删除目标: `*`、`/*`、根目录、根的直接子目录等。前缀匹配后按段判定。
DANGEROUS_REMOVAL_PREFIXES = ("*", "/*")

# 写路径中禁止出现的 shell 展开语法(validatePath): $, %, 词首 =
SHELL_EXPANSION_CHARS = ("$", "%")
