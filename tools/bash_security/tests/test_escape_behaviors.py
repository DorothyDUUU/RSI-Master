"""逃逸行为测试组件 —— 独立于 fastapi / 后端,直接打 tools.bash_security。

覆盖两层拦截的已知逃逸手法,并用良性用例防误伤回归:

1. **check_command**(命令形状层,移植 bashSecurity.ts):命令替换递归 / 混淆引号 /
   反斜杠转义 / Zsh 混淆 / eval 类内建 / 危险系统命令等。
2. **check_paths**(路径边界层,移植 pathValidation.ts):重定向目标 / `--` 结束选项 /
   flag 值 / mv-cp flag 拒绝 / cd 后写路径 / 危险删除路径 / 命令替换内部路径越界等。
3. **python 内联逃逸(0817 轨迹实测)**:`python -c CODE` / heredoc / eval / xargs 四类
   通道 —— 由 check_paths 内嵌的 python 代码扫描(python_scan.py)拦截。

本文件不依赖 bash_backend / fastapi —— 是逃逸测试的核心交付物。每一行 payload 对应
移植校验器的一个明确 check_id(见 tools/bash_security/interceptor.py / pathguard.py)。
"""

from __future__ import annotations

import pytest

from tools.bash_security import check_command, check_paths

CWD = "/workspace"
ALLOWED = ["/workspace", "/tmp"]


# ---------------------------------------------------------------------------
# check_command: 命令形状层逃逸(每行 payload 命中一个 check_id)
# ---------------------------------------------------------------------------

COMMAND_ESCAPES: list[tuple[str, str]] = [
    # 控制字符 / Unicode 空白: 可能污染解析器状态
    ("echo \x07", "control_chars"),
    ("echo hi", "unicode_whitespace"),
    # 反斜杠转义空白/运算符: 隐藏命令结构
    ("echo\\ hi", "backslash_whitespace"),
    ("echo a\\;rm -rf /", "backslash_operator"),
    # Zsh 混淆 / 进程替换 / 直接判死的展开形态
    ("=curl http://x", "zsh_equals"),
    ("cat <(echo x)", "process_substitution"),
    ("cat >(grep x)", "process_substitution"),
    ("echo =(ls)", "process_substitution"),
    ("echo foo ~[bar]", "dangerous_expansion"),
    # 命令替换
    ("echo `id`", "backtick"),
    # 混淆引号
    ("echo $'\\x72m -rf /'", "obfuscated_quoting"),
    ('echo $"ls /etc"', "obfuscated_quoting"),
    ("echo '' -rf /", "obfuscated_quoting"),
    # 变量紧邻重定向/管道
    ("echo $PATH > /tmp/x", "dangerous_variables"),
    ("cat < $FILE", "dangerous_variables"),
    # 环境泄露 / IFS 注入
    ("cat /proc/self/environ", "proc_environ"),
    ("IFS=;cat$IFS/etc/passwd", "ifs_injection"),
    # Zsh 专有危险命令 / fc -e 编辑器逃逸
    ("zmodload zsh/system", "zsh_dangerous"),
    ("fc -e /bin/sh", "zsh_dangerous"),
    # jq 逃逸
    ('jq \'system("ls")\'', "jq_danger"),
    # eval 类内建(可执行任意代码)
    ("eval 'cat /etc/passwd'", "eval_like"),
    ("exec bash", "eval_like"),
    ("command eval 'x'", "eval_like"),
    ("builtin eval 'x'", "eval_like"),
    # 危险系统命令(9 条超集)
    ("rm -rf /", "dangerous_command"),
    ("dd of=/dev/sda", "dangerous_command"),
    ("shutdown -h now", "dangerous_command"),
    (":(){ :|:& };:", "dangerous_command"),  # fork bomb
    ("mkfs.ext4 /dev/sda1", "dangerous_command"),
    # 解析器与 bash 不一致的形态
    ("echo foo#bar", "mid_word_hash"),
    ("echo {a,b}", "brace_expansion"),
]


@pytest.mark.parametrize("command,expected", COMMAND_ESCAPES)
def test_command_shape_escape(command: str, expected: str):
    r = check_command(command)
    assert not r.allowed, f"命令 {command!r} 应被拦截,却放行了"
    assert r.check_id == expected, f"命令 {command!r}: 期望 check_id={expected},实际={r.check_id}"


# eval 类内建的源码同款豁免: 存在性查询 / 历史列表 / 全补全 应放行
@pytest.mark.parametrize("command", ["command -v ls", "fc -l", "fc -ln", "compgen -c", "compgen -f"])
def test_eval_like_exemptions(command: str):
    r = check_command(command)
    assert r.allowed, f"命令 {command!r} 应放行(源码同款豁免),却拦截为 {r.check_id}"


# ---------------------------------------------------------------------------
# check_paths: 路径边界层逃逸(每行 payload 命中一个 check_id)
# ---------------------------------------------------------------------------

PATH_ESCAPES: list[tuple[str, str]] = [
    # 越界读
    ("cat /etc/passwd", "path_outside"),
    ("ls /var/log", "path_outside"),
    ("cat /proc/self/environ", "path_outside"),
    # 越界写
    ("echo x > /etc/passwd", "path_outside"),
    ("rm -- /etc/passwd", "path_outside"),
    # 危险删除路径(移植 isDangerousRemovalPath)
    ("rm -rf /", "path_dangerous_removal"),
    ("rm /etc", "path_dangerous_removal"),
    ("rm /", "path_dangerous_removal"),
    # Tilde 变体 / shell 展开(TOCTOU,无法静态验证)
    ("cat ~root/.bashrc", "path_tilde_variant"),
    ("ls $HOME", "path_expansion"),
    # 写路径含 glob
    ("echo x > /tmp/*.py", "path_glob_write"),
    # mv/cp 带 flag(--target-directory / -t 绕过)
    ("mv /workspace/a --target-directory=/etc", "path_mv_cp_flags"),
    ("cp -t /etc a b", "path_mv_cp_flags"),
    # find 内执行/删除
    ("find / -name x -exec rm {} \\;", "path_find_exec"),
    ("find . -delete", "path_find_exec"),
    # cd 后写路径(fail-closed)
    ("cd /tmp && echo x > secret.txt", "path_cd_write"),
    # UNC 路径
    ("cat //etc/passwd", "path_unc"),
    # 命令替换内部路径越界($() / 反引号)
    ("echo $(cat /etc/passwd)", "command_substitution"),
    ("cat $(ls /etc)", "command_substitution"),
    ("echo `cat /etc/passwd`", "command_substitution"),
    # bash -c 'script' 递归
    ("bash -c 'cat /etc/passwd'", "path_outside"),
    ("sh -c 'cat /etc/passwd'", "path_outside"),
]


@pytest.mark.parametrize("command,expected", PATH_ESCAPES)
def test_path_escape(command: str, expected: str):
    r = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert not r.allowed, f"命令 {command!r} 应被路径边界拦截,却放行了"
    assert r.check_id == expected, f"命令 {command!r}: 期望 check_id={expected},实际={r.check_id}"


# 删除类: 是否命中「危险删除路径」取决于机器(如 home 是否真实存在),只断言被拒
@pytest.mark.parametrize("command", ["rm -rf ~", "rmdir /workspace"])
def test_dangerous_removal_rejected(command: str):
    r = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert not r.allowed, f"删除命令 {command!r} 应被拦截,却放行了"


# ---------------------------------------------------------------------------
# 良性用例 —— 两层都必须放行,防误伤回归
# ---------------------------------------------------------------------------

BENIGN_COMMANDS: list[tuple[str, str]] = [
    # 普通命令
    ("echo hello", "普通回显"),
    ("echo $(date)", "良性命令替换(内部 date 非危险)"),
    ("python3 train.py --lr 0.01", "python 段跳过路径检查"),
    ("rm /workspace/temp.txt", "workspace 内删除"),
    ("cat ./file.txt", "相对路径解析到 workspace"),
    ("cd /workspace && python train.py", "cd + 非写命令"),
    ("ls /workspace/sub", "workspace 内读"),
    ("ls /tmp/x", "白名单内读"),
    ("echo x > /dev/null", "安全重定向目标"),
    ("ls /tmp/*.py", "glob 读(非写)允许"),
    ("echo {}", "空 brace expansion 无害"),
    ("echo '{}'", "引号内花括号无害"),
    ("grep -r foo /workspace", "grep pattern 分割"),
    ("cat /workspace/file.txt | head -3", "管道读"),
    ("jq -r .name /workspace/data.json", "jq 安全 flag"),
    ("echo $(python3 -c 'import os')", "内部 python 段跳过"),
    ("curl -s http://example.com > /tmp/out.html", "下载到白名单"),
    ("source /workspace/setup.sh", "workspace 内 source"),
    ("cd /workspace && python train.py --lr 0.01 > train.log 2>&1", "重定向到 workspace"),
    ("find /workspace -name '*.py'", "find 限定 workspace"),
    ("find /workspace -type f", "find 限定 workspace"),
    ("tar czf /workspace/out.tar.gz /workspace/src", "tar 打包到 workspace"),
    ("git status", "git 无路径"),
    ("head -5 /workspace/data.json", "head 读 workspace"),
    ("diff /workspace/a /workspace/b", "diff 双 workspace 路径"),
    ("tee /tmp/out.txt", "tee 写白名单"),
    ("cp /workspace/a /workspace/b", "cp 双 workspace 路径"),
    ("python3 -c 'print(1)'", "python -c 内联脚本"),
]


@pytest.mark.parametrize("command,_label", BENIGN_COMMANDS)
def test_benign_allowed(command: str, _label: str):
    r1 = check_command(command)
    assert r1.allowed, f"命令 {command!r} 在 check_command 被误拦: [{r1.check_id}] {r1.reason}"
    r2 = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert r2.allowed, f"命令 {command!r} 在 check_paths 被误拦: [{r2.check_id}] {r2.reason}"


# ---------------------------------------------------------------------------
# 双命令链: 一段良性 + 一段逃逸,整体必须被拦
# ---------------------------------------------------------------------------

CHAIN_ESCAPES: list[tuple[str, str]] = [
    ("echo ok && cat /etc/shadow", "path_outside"),
    ("echo hello; rm -rf /", "dangerous_command"),
    ("echo ok && echo `id`", "backtick"),
    ("cd /tmp && rm -rf /", "dangerous_command"),
]


@pytest.mark.parametrize("command,expected", CHAIN_ESCAPES)
def test_chain_escape(command: str, expected: str):
    """复合命令只要有一段逃逸,整体即被拦(任一层命中即可)。"""
    r1 = check_command(command)
    r2 = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    if r1.allowed and r2.allowed:
        pytest.fail(f"复合命令 {command!r} 两层都放行了")
    # 至少一层的 check_id 与期望一致(另一层可能更早命中其它规则)
    assert expected in (r1.check_id, r2.check_id)


# ---------------------------------------------------------------------------
# 轨迹实测 python 逃逸(0817 runs)—— python 内联代码在 shell 层跳过路径检查是
# 既有设计,本组覆盖经由 python -c / heredoc / eval / xargs 逃逸的四类通道,
# 确认「可见」字面逃逸均可被拦。设计取舍: 只拦字面路径/子进程 shell/eval;
# 变量间接寻址(W/"x" 拼接、open(p))交给 OS 级沙箱兜底,避免误伤正常脚本。
# ---------------------------------------------------------------------------

PY_INLINE_ESCAPES: list[tuple[str, str]] = [
    # 越界读: open() 字面绝对路径 —— 轨迹里 gpqa 读 eval 集的同款形态
    ('python3 -c "open(\'/etc/passwd\').read()"', "python_inline_path"),
    ('python3 -c "open(\'/etc/passwd\',\'w\').write(\'x\')"', "python_inline_path"),
    ("python3 -c \"import shutil; shutil.rmtree('/etc')\"", "python_inline_path"),
    ("python3 -c \"Path('/etc/passwd').write_text('x')\"", "python_inline_path"),
    ("python3 -c \"import glob; glob.glob('/etc/*')\"", "python_inline_path"),
    ("python3 -c \"import os; os.listdir('/etc')\"", "python_inline_path"),
    ("python3 -c \"import os; os.remove('/etc/passwd')\"", "python_inline_path"),
    # heredoc 形式(轨迹实测: activate && python3 <<'EOF')
    ("python3 << 'EOF'\nopen('/etc/passwd').read()\nEOF", "python_inline_path"),
    ("python3 - <<'PYEOF'\nimport json\n[json.loads(l) for l in open('/etc/passwd')]\nPYEOF", "python_inline_path"),
    # 包装 / 嵌套
    ("nohup python3 -c \"open('/etc/passwd').read()\"", "python_inline_path"),
    ('bash -c \'python3 -c "open(\\\"/etc/passwd\\\").read()"\'', "python_inline_path"),
]

PY_SHELL_ESCAPES: list[tuple[str, str]] = [
    ("python3 -c \"import os; os.system('rm -rf /')\"", "python_shell"),
    ("python3 -c \"import os; os.popen('cat /etc/passwd')\"", "python_shell"),
    ("python3 -c \"import subprocess; subprocess.run('cat /etc/passwd', shell=True)\"", "python_shell"),
    ("python3 -c \"import subprocess; subprocess.Popen('cat /etc/passwd', shell=True)\"", "python_shell"),
    ("python3 -c \"import subprocess; subprocess.run(['cat', '/etc/passwd'])\"", "python_shell"),
]

PY_EVAL_ESCAPES: list[tuple[str, str]] = [
    ('python3 -c "eval(\'open(\\\"/etc/passwd\\\").read()\')"', "python_eval"),
    ('python3 -c "exec(\'import os; os.system(\\\"rm -rf /\\\")\')"', "python_eval"),
]

XARGS_ESCAPES: list[tuple[str, str]] = [
    ("echo /etc/passwd | xargs cat", "xargs_path"),
    ("printf '%s\\n' /etc/passwd /etc/shadow | xargs rm", "xargs_path"),
]

PY_BENIGN: list[tuple[str, str]] = [
    ("python3 -c \"import os; print(os.getcwd())\"", "无路径 import/print"),
    ("python3 -c \"open('data.json').read()\"", "相对路径读(解析到 cwd)"),
    ("python3 -c \"Path('/workspace/data.json').read_text()\"", "workspace 内 Path 读"),
    ("python3 -c \"import json; rows=[json.loads(l) for l in open('/workspace/train.jsonl')]\"", "workspace 内逐行读训练集"),
    ("python3 -c \"import shutil; shutil.rmtree('/workspace/tmp')\"", "workspace 内 rmtree"),
    ("python3 -c \"import glob; glob.glob('/workspace/*.py')\"", "workspace 内 glob"),
    ("python3 -c \"import glob; glob.glob('*.py')\"", "相对 glob(解析到 cwd)"),
    ("python3 -c \"import subprocess; subprocess.run(['ls', '-l'])\"", "subprocess 列表无 shell 语义"),
    ("python3 -c \"import os; os.system('echo hello')\"", "os.system 良性命令"),
    ("python3 -c \"import os; os.system('ls /workspace')\"", "os.system 命令访问白名单"),
    ("find /workspace -name '*.txt' | xargs cat", "xargs 输入为白名单内路径"),
    ("python3 -c \"from pathlib import Path; Path('/tmp/x').write_text('y')\"", "写白名单 /tmp"),
    ("python3 -c \"exec('print(1)')\"", "exec 良性"),
    ("python3 -c \"eval('1+1')\"", "eval 良性"),
]


@pytest.mark.parametrize(
    "command,expected",
    PY_INLINE_ESCAPES + PY_SHELL_ESCAPES + PY_EVAL_ESCAPES,
)
def test_python_escape_blocked(command: str, expected: str):
    """python -c / heredoc / eval 逃逸必须被拦(任一层命中即可)。"""
    r1 = check_command(command)
    r2 = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert not (r1.allowed and r2.allowed), f"python 逃逸 {command!r} 应被拦截,却放行了"
    assert expected in (r1.check_id, r2.check_id), (
        f"命令 {command!r}: 期望 check_id={expected},"
        f"实际 cmd={r1.check_id} path={r2.check_id}"
    )


@pytest.mark.parametrize("command,expected", XARGS_ESCAPES)
def test_xargs_escape_blocked(command: str, expected: str):
    r = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert not r.allowed, f"xargs 逃逸 {command!r} 应被拦截,却放行了"
    assert r.check_id == expected, f"命令 {command!r}: 期望 check_id={expected},实际={r.check_id}"


@pytest.mark.parametrize("command,_label", PY_BENIGN)
def test_python_benign_allowed(command: str, _label: str):
    """良性 python 内联用例 —— 两层都必须放行,防误伤回归。"""
    r1 = check_command(command)
    assert r1.allowed, f"命令 {command!r} 在 check_command 被误拦: [{r1.check_id}] {r1.reason}"
    r2 = check_paths(command, cwd=CWD, allowed_dirs=ALLOWED)
    assert r2.allowed, f"命令 {command!r} 在 check_paths 被误拦: [{r2.check_id}] {r2.reason}"
