"""python 内联代码扫描 —— `python -c 'CODE'` / `python <<'EOF'` heredoc 主体。

python 段在 shell 级跳过路径检查(脚本内部文件访问无法静态分析)是既有设计;
但 0817 轨迹里大量逃逸正是经由 python 完成(读 PostTrainBench eval 集、写 /etc、
os.system 套 shell)。这里对「可见的」内联代码做静态扫描,捕获三类:

1. **文件访问字面路径**: open()/Path()/shutil.*/os.remove 等的第一个字符串参数。
   参数是白名单外的绝对路径 → 拒绝(python_inline_path)。
2. **子进程 shell**: os.system / subprocess(..., shell=True) / Popen 等的命令串,
   递归走 check_command + check_paths(python_shell)。
3. **eval/exec 字面代码**: 递归扫内层代码(python_eval)。

变量间接寻址(W/"x" 拼接、open(p) 中 p=变量)无法静态追踪 —— 由 OS 级沙箱
(bwrap/landlock)兜底,本层不误伤。
"""

from __future__ import annotations

import re

from .result import SecurityResult

# ---------------------------------------------------------------------------
# 文件访问 API 字面路径。kind 由调用方按函数名 + mode 判定。
# ---------------------------------------------------------------------------
_PY_FILE_ACCESS_RE = re.compile(
    r"""\b(?P<fn>(?:io\.)?open|os\.open|Path|PosixPath|os\.(?:remove|unlink|rmdir|rename|replace|makedirs|mkdir|symlink)|shutil\.(?:rmtree|copyfile|copy|move|remove|unlink|rmdir))\s*\(\s*(?P<q>['"])(?P<path>[^'"]*)(?P=q)"""
)

# Path(...).write_text/.touch 等 —— 写通过方法发生,单靠 Path() 创建对象看不到
_PY_WRITE_METHOD_RE = re.compile(
    r"""\bPath\s*\(\s*(['"])(?P<path>[^'"]*)\1\s*\)\s*\.\s*(?:write_text|write_bytes|touch|unlink|rmdir|rename|replace|symlink_to)\s*\("""
)

# 目录/glob 读: glob.glob('/etc/*') / os.walk('/etc') 等 —— 0817 轨迹实测逃逸通道
_PY_DIR_READ_RE = re.compile(
    r"""\b(?:glob\.(?:glob|iglob)|os\.(?:listdir|scandir|walk))\s*\(\s*(?P<q>['"])(?P<path>[^'"]*)(?P=q)"""
)

_GLOB_METACHARS = frozenset("*?[")

# 无 shell 语义时是整串 argv(不执行),有 shell 语义/元字符才是命令 —— 见 check_python_code
_PY_SUBPROCESS_RE = re.compile(
    r"""\bsubprocess\.(?:run|call|check_call|check_output|Popen)\s*\(\s*(?P<q>['"])(?P<shell>[^'"]*)(?P=q)"""
)

# 恒走 shell 的系统调用
_PY_OS_SHELL_RE = re.compile(
    r"""\b(?:os\.system|os\.popen|commands\.getoutput|subprocess\.(?:getoutput|getstatusoutput|check_output|Popen))\s*\(\s*(?P<q>['"])(?P<shell>[^'"]*)(?P=q)"""
)

# subprocess 列表形式: subprocess.run(['cat', '/etc/passwd']) —— 引号项拼成命令行检查
_PY_SUBPROCESS_LIST_RE = re.compile(
    r"""\bsubprocess\.\w+\s*\(\s*\[(?P<items>[^\[\]]*?)\]"""
)

_PY_EVAL_INVOCATION_RE = re.compile(r"\b(?:eval|exec)\s*\(")

def _eval_arg(code: str, m: re.Match) -> str | None:
    """取 eval/exec 调用后的首个字符串实参(单/双引号 + 反斜杠转义)。

    内层代码常含与包裹层不同的引号(exec('os.system("rm -rf /")')),
    简单 `[^'"]*` 无法跨越 —— 必须按字符串字面量语义解析。
    """
    k = m.end()
    n = len(code)
    while k < n and code[k] in " \t":
        k += 1
    if k >= n or code[k] not in "'\"":
        return None
    q = code[k]
    buf: list[str] = []
    k += 1
    while k < n:
        c = code[k]
        if c == "\\" and k + 1 < n:
            buf.append(code[k + 1])
            k += 2
            continue
        if c == q:
            return "".join(buf)
        buf.append(c)
        k += 1
    return None

# shell 元字符: 字符串里出现它们才需要按 shell 语义拦截
_SHELL_METACHARS = (";", "|", "&", "<", ">", "$", "`", "\n")

_WRITE_MODES = frozenset("wax+")
_ALWAYS_WRITE_FN = frozenset({
    "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
    "os.makedirs", "os.mkdir", "os.symlink",
    "shutil.rmtree", "shutil.copyfile", "shutil.copy", "shutil.move",
    "shutil.remove", "shutil.unlink", "shutil.rmdir",
})
_OPEN_FN = frozenset({"open", "io.open", "os.open"})


# ---------------------------------------------------------------------------
# `python -c CODE` 抽取 —— 必须针对原始命令做(内联代码里常含 ( ) ; | 等,
# shell tokenizer 会切碎;-c 的代码是一个 Python 字符串,按引号/转义语义抽取)。
# ---------------------------------------------------------------------------
_PY_INVOCATION_RE = re.compile(r"\bpython3?\b")

def _find_c_flag(command: str, start: int) -> int | None:
    """从 start 起在本段(到 ; | & 换行)内找独立 `-c` flag,返回其下标。"""
    seg_end = len(command)
    for delim in ";\n|&":
        p = command.find(delim, start)
        if p != -1:
            seg_end = min(seg_end, p)
    sub = command[start:seg_end]
    for mm in re.finditer(r"\s-c(?=\s|$)", sub):
        return start + mm.start() + 1
    return None


def extract_python_c_codes(command: str) -> list[str]:
    """提取所有 `python -c CODE` 的 CODE(支持引号与反斜杠转义)。"""
    codes: list[str] = []
    i = 0
    n = len(command)
    while True:
        m = _PY_INVOCATION_RE.search(command, i)
        if not m:
            break
        c_idx = _find_c_flag(command, m.end())
        if c_idx is None:
            i = m.end()
            continue
        k = c_idx + 2
        while k < n and command[k] in " \t":
            k += 1
        if k >= n:
            i = m.end()
            continue
        ch = command[k]
        if ch in "'\"":
            q = ch
            buf: list[str] = []
            k += 1
            while k < n:
                c = command[k]
                if c == "\\" and k + 1 < n and command[k + 1] in (q, "\\"):
                    buf.append(command[k + 1])
                    k += 2
                    continue
                if c == q:
                    break
                buf.append(c)
                k += 1
            codes.append("".join(buf))
            i = k + 1
        else:
            e = k
            while e < n and command[e] not in " \t\n;|&":
                e += 1
            codes.append(command[k:e])
            i = e
    return codes


def _check_shell(shell: str, *, cwd: str, allowed_dirs: list[str]) -> SecurityResult:
    """递归: 把 python 内的 shell 字符串当一条命令查。"""
    from .interceptor import check_command
    from .pathguard import check_paths

    if not shell or not shell.strip():
        return SecurityResult.allow()
    r = check_command(shell)
    if not r.allowed:
        return r
    return check_paths(shell, cwd=cwd, allowed_dirs=allowed_dirs)


def check_python_code(code: str, *, cwd: str, allowed_dirs: list[str]) -> SecurityResult:
    """扫描一段 python 内联代码。cwd/allowed_dirs 语义与 check_paths 一致。"""
    if not code or not code.strip():
        return SecurityResult.allow()

    from .pathguard import _validate_path

    # 1. eval/exec 字面代码 —— 任意执行,取首个字符串实参递归扫内层
    for m in _PY_EVAL_INVOCATION_RE.finditer(code):
        inner = _eval_arg(code, m)
        if inner is None:
            continue
        r = check_python_code(inner, cwd=cwd, allowed_dirs=allowed_dirs)
        if not r.allowed:
            return SecurityResult.reject("python_eval", f"python eval/exec 内联代码被拦截: {r.reason}")

    # 2. 文件访问字面路径
    for m in _PY_FILE_ACCESS_RE.finditer(code):
        fn, path = m.group("fn"), m.group("path")
        if not path:
            continue
        if fn in _ALWAYS_WRITE_FN:
            op = "write"
        elif fn in _OPEN_FN:
            # open(p, mode): 调用内首个引号参数含写字符 → write
            rest = code[m.end() : m.end() + 80]
            mm = re.search(r"['\"]([^'\"]*)['\"]", rest)
            op = "write" if (mm and any(c in _WRITE_MODES for c in mm.group(1))) else "read"
        else:  # Path / PosixPath —— 建对象本身是读;写走方法(regex 3)
            op = "read"
        r = _validate_path(path, cwd, allowed_dirs, op)
        if not r.allowed:
            return SecurityResult.reject(
                "python_inline_path", f"python 内联代码访问路径越界: {r.reason}"
            )

    # 3. Path(...).write_text/.touch 等写方法
    for m in _PY_WRITE_METHOD_RE.finditer(code):
        r = _validate_path(m.group("path"), cwd, allowed_dirs, "write")
        if not r.allowed:
            return SecurityResult.reject(
                "python_inline_path", f"python 内联代码 Path 写越界: {r.reason}"
            )

    # 3.5 glob 目录读: glob.glob('/etc/*') 等。取首个 glob 元字符前的目录前缀做
    #     校验 —— `/workspace/*.py` 前缀在 allowed 内放行,`/etc/*` 拦截。
    for m in _PY_DIR_READ_RE.finditer(code):
        pat = m.group("path")
        if not pat:
            continue
        prefix = pat
        cut = next((i for i, ch in enumerate(pat) if ch in _GLOB_METACHARS), None)
        if cut is not None:
            prefix = pat[:cut]
        if not prefix:
            continue  # 纯 glob 模式(如 '*/*.json')解析到 cwd,放行
        r = _validate_path(prefix, cwd, allowed_dirs, "read")
        if not r.allowed:
            return SecurityResult.reject(
                "python_inline_path", f"python 内联代码 glob 读越界: {r.reason}"
            )

    # 4. os.system 等恒 shell 的系统调用
    for m in _PY_OS_SHELL_RE.finditer(code):
        shell = m.group("shell")
        if not shell:
            continue
        r = _check_shell(shell, cwd=cwd, allowed_dirs=allowed_dirs)
        if not r.allowed:
            return SecurityResult.reject("python_shell", f"python 子进程 shell 被拦截: {r.reason}")

    # 5. subprocess(..., shell=True) / 字符串含 shell 元字符
    for m in _PY_SUBPROCESS_RE.finditer(code):
        shell = m.group("shell")
        if not shell:
            continue
        rest = code[m.end() : m.end() + 80]
        shell_yes = re.search(r"shell\s*=\s*True", rest)
        if not shell_yes and not any(c in shell for c in _SHELL_METACHARS):
            continue  # 无 shell 语义的整串 argv,运行时不执行
        r = _check_shell(shell, cwd=cwd, allowed_dirs=allowed_dirs)
        if not r.allowed:
            return SecurityResult.reject("python_shell", f"python 子进程 shell 被拦截: {r.reason}")

    # 6. subprocess 列表形式: 引号项拼接成命令行检查
    for m in _PY_SUBPROCESS_LIST_RE.finditer(code):
        joined = " ".join(
            x for x in re.findall(r"['\"]([^'\"]*)['\"]", m.group("items")) if x.strip()
        )
        if joined.strip():
            r = _check_shell(joined, cwd=cwd, allowed_dirs=allowed_dirs)
            if not r.allowed:
                return SecurityResult.reject("python_shell", f"python 子进程列表被拦截: {r.reason}")

    return SecurityResult.allow()
