"""内置过滤器测试。"""

from __future__ import annotations

import pytest

from tools.bash_backend.filter_base import (
    FilterAction,
    FilterContext,
    _FILTER_REGISTRY,
)
from tools.bash_backend.filters.dangerous import DangerousCommandFilter
from tools.bash_backend.filters.long_sleep import LongSleepFilter
from tools.bash_backend.filters.output_length import OutputLengthCheck
from tools.bash_backend.filters.sandbox import FileSystemSandboxFilter


def _make_ctx(command: str, workspace: str = "/workspace", whitelist: list[str] | None = None) -> FilterContext:
    """构造测试用 FilterContext。"""
    return FilterContext(
        command=command,
        agent_workspace=workspace,
        timeout=30.0,
        meta_data={"whitelist": whitelist or ["/tmp"]},
        resolved_workspace=workspace,
    )


# ---------------------------------------------------------------------------
# FileSystemSandboxFilter
# ---------------------------------------------------------------------------


class TestFileSystemSandboxFilter:
    """文件系统沙箱过滤器测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.f = FileSystemSandboxFilter()

    def test_simple_allowed(self):
        ctx = _make_ctx("echo hello")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_workspace_path_allowed(self):
        ctx = _make_ctx("ls /workspace/subdir")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_whitelisted_path_allowed(self):
        ctx = _make_ctx("ls /tmp/file.txt", whitelist=["/tmp"])
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_outside_path_rejected(self):
        ctx = _make_ctx("ls /etc/passwd")
        result = self.f.check(ctx)
        assert result.action == FilterAction.REJECT
        assert "不在允许的目录范围内" in result.reason

    def test_relative_path_resolved(self):
        ctx = _make_ctx("cat ./file.txt")
        result = self.f.check(ctx)
        # ./file.txt 解析为 /workspace/file.txt，应该允许
        assert result.action == FilterAction.ALLOW

    def test_compound_command_all_checked(self):
        # 第一段合法，第二段不合法
        ctx = _make_ctx("echo hello && ls /etc")
        result = self.f.check(ctx)
        assert result.action == FilterAction.REJECT

    def test_pipe_command_checked(self):
        ctx = _make_ctx("cat /etc/passwd | grep root")
        result = self.f.check(ctx)
        assert result.action == FilterAction.REJECT

    def test_no_paths_command(self):
        ctx = _make_ctx("echo hello && echo world")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_empty_workspace_uses_whitelist_only(self):
        ctx = FilterContext(
            command="ls /tmp/test",
            agent_workspace="",
            timeout=30.0,
            meta_data={"whitelist": ["/tmp"]},
            resolved_workspace="",
        )
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_filter_name_set(self):
        ctx = _make_ctx("ls /etc")
        result = self.f.check(ctx)
        assert result.filter_name == "filesystem_sandbox"

    # --- 升级后: 命令替换内部路径 / 路径逃逸手法 ---

    @pytest.mark.parametrize(
        "command",
        [
            "echo $(cat /etc/passwd)",           # $() 内部越界读
            "echo `cat /etc/passwd`",             # 反引号内部越界读
            "cat $(ls /etc)",                     # 替换结果拼入越界路径
            "echo x > /etc/passwd",               # 重定向越界写
            "mv /workspace/a --target-directory=/etc",  # mv flag 绕过
            "cp -t /etc a b",                     # cp -t 绕过
            "find / -name x -exec rm {} \\;",     # find 内执行
            "cd /tmp && echo x > secret.txt",     # cd 后写路径 fail-closed
            "rm -rf /",                           # 危险删除路径
            "cat ~root/.bashrc",                  # tilde 变体
        ],
    )
    def test_path_escape_rejected(self, command):
        ctx = _make_ctx(command)
        result = self.f.check(ctx)
        assert result.action == FilterAction.REJECT
        assert "[" in result.reason  # 带 check_id 前缀


# ---------------------------------------------------------------------------
# DangerousCommandFilter
# ---------------------------------------------------------------------------


class TestDangerousCommandFilter:
    """危险命令过滤器测试。

    注意：这些测试只检查过滤器对命令字符串的判断，不实际执行任何命令。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.f = DangerousCommandFilter()

    def test_safe_command_allowed(self):
        ctx = _make_ctx("echo hello")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_rm_normal_file_allowed(self):
        ctx = _make_ctx("rm /workspace/temp.txt")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_normal_command_allowed(self):
        ctx = _make_ctx("python3 train.py --lr 0.01")
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW

    def test_filter_name_set(self):
        ctx = _make_ctx("echo test")
        result = self.f.check(ctx)
        assert result.filter_name == "dangerous_command"

    # --- 升级后: 命令形状层逃逸(移植 bashSecurity.ts) ---

    @pytest.mark.parametrize(
        "command",
        [
            "echo `id`",                          # 反引号命令替换
            "eval 'cat /etc/passwd'",             # eval 类内建
            "exec bash",                          # exec 类内建
            "cat <(echo x)",                      # 进程替换
            "=curl http://x",                     # zsh equals 展开
            "echo $'\\x72m -rf /'",               # ANSI-C 引号混淆
            "echo $\"ls /etc\"",                   # locale 引号混淆
            "zmodload zsh/system",                # zsh 专有危险命令
            "fc -e /bin/sh",                      # fc -e 编辑器逃逸
            "jq 'system(\"ls\")'",                 # jq system()
            "IFS=;cat$IFS/etc/passwd",            # IFS 注入
            "cat /proc/self/environ",             # proc environ 泄露
            "rm -rf /",                           # 危险系统命令
            ":(){ :|:& };:",                      # fork bomb
            "echo $PATH > /tmp/x",                # 变量紧邻重定向
        ],
    )
    def test_command_shape_escape_rejected(self, command):
        ctx = _make_ctx(command)
        result = self.f.check(ctx)
        assert result.action == FilterAction.REJECT
        assert "[" in result.reason

    @pytest.mark.parametrize(
        "command",
        [
            "echo $(date)",                       # 良性命令替换
            "command -v ls",                      # eval 类内建豁免
            "fc -l",                              # 历史列表豁免
        ],
    )
    def test_benign_not_blocked(self, command):
        ctx = _make_ctx(command)
        result = self.f.check(ctx)
        assert result.action == FilterAction.ALLOW


# ---------------------------------------------------------------------------
# OutputLengthCheck
# ---------------------------------------------------------------------------


class TestOutputLengthCheck:
    """输出长度检查测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.f = OutputLengthCheck()
        self.f.max_length = 100  # 降低阈值方便测试

    def test_short_output_allowed(self):
        ctx = _make_ctx("echo hello")
        result = self.f.check(ctx, output="short output")
        assert result.action == FilterAction.ALLOW

    def test_long_output_rejected(self):
        ctx = _make_ctx("cat big_file")
        result = self.f.check(ctx, output="x" * 200)
        assert result.action == FilterAction.REJECT
        assert "超过限制" in result.reason

    def test_exact_limit_allowed(self):
        ctx = _make_ctx("cat file")
        result = self.f.check(ctx, output="x" * 100)
        assert result.action == FilterAction.ALLOW

    def test_one_over_limit_rejected(self):
        ctx = _make_ctx("cat file")
        result = self.f.check(ctx, output="x" * 101)
        assert result.action == FilterAction.REJECT

    def test_empty_output_allowed(self):
        ctx = _make_ctx("true")
        result = self.f.check(ctx, output="")
        assert result.action == FilterAction.ALLOW


# ---------------------------------------------------------------------------
# LongSleepFilter
# ---------------------------------------------------------------------------


class TestLongSleepFilter:
    """长时间 sleep 过滤器测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.f = LongSleepFilter()

    def _ctx(self, command: str, max_sleep: int = 2400) -> FilterContext:
        return FilterContext(
            command=command,
            agent_workspace="/workspace",
            timeout=30.0,
            meta_data={"whitelist": ["/tmp"], "max_sleep_seconds": max_sleep},
            resolved_workspace="/workspace",
        )

    def test_no_sleep_allowed(self):
        assert self.f.check(self._ctx("echo hello")).action == FilterAction.ALLOW

    def test_short_sleep_allowed(self):
        assert self.f.check(self._ctx("sleep 60 && echo ok")).action == FilterAction.ALLOW

    def test_exact_limit_allowed(self):
        # 恰好 40min(=上限) 放行, 只有「超过」才拦
        assert self.f.check(self._ctx("sleep 2400")).action == FilterAction.ALLOW
        assert self.f.check(self._ctx("sleep 40m")).action == FilterAction.ALLOW

    def test_plain_sleep_rejected(self):
        r = self.f.check(self._ctx("sleep 2700"))
        assert r.action == FilterAction.REJECT
        assert r.filter_name == "long_sleep"
        assert "2400" in r.reason

    def test_sleep_with_continuation_rejected(self):
        # 0826 实测 pattern: sleep N && gpu-backend script-log | grep | tail
        cmd = "sleep 7200 && public/gpu-backend script-log --script-id gpu_xxx | grep -E 'loss|epoch' | tail -15"
        assert self.f.check(self._ctx(cmd)).action == FilterAction.REJECT

    def test_sleep_in_second_segment_rejected(self):
        # cd 前缀 + 后续段 sleep
        cmd = "cd /workspace && sleep 3000; tail -1 experiment_os/worker/worker_0011/distill.log"
        assert self.f.check(self._ctx(cmd)).action == FilterAction.REJECT

    @pytest.mark.parametrize(
        "cmd",
        [
            "sleep 45m",              # 后缀 2700s
            "sleep 1h 30m",           # 多操作数求和 5400s
            "sleep 0.7h",             # 小数 2520s
            "sleep 3000; grep -oE '[0-9]+/1250' experiment_os/experiments/exp_0001/result.json",
            "/usr/bin/sleep 3600",    # 绝对路径 sleep
        ],
    )
    def test_over_limit_rejected(self, cmd):
        assert self.f.check(self._ctx(cmd)).action == FilterAction.REJECT

    @pytest.mark.parametrize(
        "cmd",
        [
            "sleep 1800",             # 30min
            "sleep 0.5h",             # 1800s
            "sleep 30m",              # 1800s
            "echo 'sleep 2700'",      # 引号内文本不命中
            "grep -c sleep 3000 /tmp/x",  # sleep 非命令, 不命中
        ],
    )
    def test_under_limit_or_false_positive_allowed(self, cmd):
        assert self.f.check(self._ctx(cmd)).action == FilterAction.ALLOW

    def test_custom_threshold_via_meta(self):
        # 服务端 --max-sleep-seconds 可调
        assert self.f.check(self._ctx("sleep 600", max_sleep=300)).action == FilterAction.REJECT
        assert self.f.check(self._ctx("sleep 120", max_sleep=300)).action == FilterAction.ALLOW

    def test_quoted_operand_rejected(self):
        assert self.f.check(self._ctx('sleep "45m"')).action == FilterAction.REJECT

    def test_empty_command_allowed(self):
        assert self.f.check(self._ctx("")).action == FilterAction.ALLOW
