"""BashTool 兜底路径安全拦截测试。

后端服务不可用时,BashTool 会 fallback 到 session.exec_bash → local_exec → 裸
subprocess.run(shell=True)。兜底路径在落入该路径前必须先跑共享拦截模块
(tools/bash_security,与后端过滤器同源)。本测试:
- 把 BASH_BACKEND_URL 置空,强制走兜底路径;
- 断言危险命令被拦(返回含 [BLOCKED by security])且 session.exec_bash **未被调用**;
- 断言良性命令正常放行(session.exec_bash 被调用)。

参照 test_agent_context.py 的 MockSession 模式。
"""

# Add the project root directory to the Python path so that evomaster / tools 可被导入
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent  # evomaster/agent/tools → repo root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json

import pytest

from evomaster.agent.session.base import BaseSession, SessionConfig
from evomaster.agent.tools.builtin.bash import (
    BASH_BACKEND_URL,
    BashTool,
    configure_backend,
)


class MockSession(BaseSession):
    """记录 exec_bash 调用的假 session。"""

    def __init__(self, config: SessionConfig | None = None):
        super().__init__(config or SessionConfig(workspace_path="/workspace"))
        self._is_open = True
        self.exec_calls: list[tuple] = []

    def open(self):
        self._is_open = True

    def close(self):
        self._is_open = False

    def exec_bash(self, command: str, timeout=None, is_input=False):
        self.exec_calls.append((command, timeout, is_input))
        return {
            "stdout": f"ran: {command}",
            "stderr": "",
            "exit_code": 0,
            "working_dir": self.config.workspace_path,
        }

    def upload(self, local_path: str, remote_path: str):
        pass

    def download(self, remote_path: str, timeout=None):
        return b""


@pytest.fixture
def bash_tool():
    return BashTool()


@pytest.fixture
def session():
    return MockSession()


@pytest.fixture(autouse=True)
def _force_fallback():
    """测试期间禁用后端,强制走兜底路径;结束后恢复原 URL。"""
    orig = BASH_BACKEND_URL
    configure_backend("")  # 空 URL → execute() 跳过后端分支
    yield
    configure_backend(orig)


def _call(tool, session, command: str, is_input: str = "false"):
    """调用 BashTool.execute,返回 (obs, info)。"""
    return tool.execute(session, json.dumps({"command": command, "is_input": is_input}))


# ---------------------------------------------------------------------------
# 危险命令: 必须被兜底拦截,且绝不落到 session.exec_bash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",                             # 危险系统命令
        "cat /etc/passwd",                      # 越界读
        "echo `id`",                            # 反引号命令替换
        "echo $(cat /etc/passwd)",              # $() 内部越界读
        "cat /proc/self/environ",               # proc environ 泄露
        "IFS=;cat$IFS/etc/passwd",              # IFS 注入
        "mv /tmp/a --target-directory=/etc",    # mv flag 绕过
        "find / -name x -exec rm {} \\;",       # find 内执行
        "cd /tmp && echo x > secret.txt",       # cd 后写路径
        "cat <(echo x)",                        # 进程替换
        "eval 'cat /etc/passwd'",               # eval 类内建
        "rm /etc",                              # 危险删除路径
        # --- python 内联逃逸(python_scan,与后端过滤器同源) ---
        "python3 -c \"open('/etc/passwd').read()\"",      # -c 越界读
        "python3 -c \"import shutil; shutil.rmtree('/etc')\"",  # -c rmtree
        "python3 -c \"import os; os.system('rm -rf /')\"",      # -c 子进程 shell
        "python3 -c \"exec('import os; os.system(\\\"rm -rf /\\\")')\"",  # -c exec 嵌套
        "python3 << 'EOF'\nopen('/etc/passwd').read()\nEOF",  # heredoc 越界读
        "echo /etc/passwd | xargs cat",          # xargs stdin 注入字面路径
    ],
)
def test_dangerous_blocked(bash_tool, session, command):
    obs, info = _call(bash_tool, session, command)
    assert "[BLOCKED by security]" in obs, f"命令 {command!r} 未被兜底拦截, obs={obs!r}"
    assert info["exit_code"] == -1
    assert session.exec_calls == [], (
        f"命令 {command!r} 已被拦截,却仍落到了 session.exec_bash —— 兜底拦截失效"
    )


def test_blocked_message_includes_escape_guidance(bash_tool, session):
    """拦截返回信息必须附带英文引导,明确禁止逃逸出允许目录。"""
    obs, _info = _call(bash_tool, session, "cat /etc/passwd")
    assert "Do not attempt to escape-run commands outside the allowed folders at any time." in obs, (
        f"拦截返回缺少逃逸引导, obs={obs!r}"
    )
    assert session.exec_calls == [], "被拦命令不应落到 session.exec_bash"


# ---------------------------------------------------------------------------
# 良性命令: 应正常放行,走 session.exec_bash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "ls /tmp",                              # 兜底白名单内含 /tmp
        "echo $(date)",                         # 良性命令替换
        "cat /workspace/file.txt",              # workspace 内读
        "command -v ls",                        # eval 类豁免
        "cd /workspace && python train.py",     # cd + 非写命令
    ],
)
def test_benign_allowed(bash_tool, session, command):
    obs, info = _call(bash_tool, session, command)
    assert "[BLOCKED by security]" not in obs, f"良性命令 {command!r} 被误拦: obs={obs!r}"
    assert len(session.exec_calls) == 1, f"命令 {command!r} 应落到 session.exec_bash"
    assert session.exec_calls[0][0] == command
    assert info["exit_code"] == 0


# ---------------------------------------------------------------------------
# is_input=true 时不做拦截(输入发给运行中进程,不是新命令)
# ---------------------------------------------------------------------------


def test_is_input_skips_security(bash_tool, session):
    obs, info = _call(bash_tool, session, "C-c", is_input="true")
    # is_input 命令不经过拦截,直接交给 session.exec_bash 发 STDIN
    assert len(session.exec_calls) == 1
    assert session.exec_calls[0][0] == "C-c"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
