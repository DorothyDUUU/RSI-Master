"""FastAPI 应用集成测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tools.bash_backend.app import create_app


@pytest.fixture
def app():
    return create_app(
        default_workspace="/tmp",
        whitelist=["/tmp", "/var"],
        max_output_length=500,
    )


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# /execute
# ---------------------------------------------------------------------------


class TestExecuteEndpoint:
    """命令执行端点测试。"""

    def test_simple_command(self, client):
        resp = client.post("/execute", json={
            "command": "echo hello",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]
        assert data["filtered"] is False
        assert data["process_id"] is not None

    def test_command_with_error(self, client):
        resp = client.post("/execute", json={
            "command": "ls /nonexistent_abc_12345",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] != 0

    def test_chained_commands(self, client):
        resp = client.post("/execute", json={
            "command": "echo hello && echo world",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]
        assert "world" in data["stdout"]

    def test_pipe_command(self, client):
        resp = client.post("/execute", json={
            "command": "echo hello | grep hello",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]

    def test_working_directory(self, client):
        resp = client.post("/execute", json={
            "command": "pwd",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0
        assert data["working_dir"] == "/tmp"

    def test_duration_tracked(self, client):
        resp = client.post("/execute", json={
            "command": "echo test",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["duration_seconds"] >= 0

    def test_empty_command(self, client):
        resp = client.post("/execute", json={
            "command": "true",
            "agent_workspace": "/tmp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0

    def test_default_workspace_used(self, client):
        """当 agent_workspace 为空时使用默认 workspace。"""
        resp = client.post("/execute", json={
            "command": "pwd",
            "agent_workspace": "",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["working_dir"] == "/tmp"


# ---------------------------------------------------------------------------
# Sandbox Filter Integration
# ---------------------------------------------------------------------------


class TestSandboxFilterIntegration:
    """沙箱过滤器集成测试。"""

    def test_allowed_path(self, client):
        resp = client.post("/execute", json={
            "command": "ls /tmp | head -1",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False

    def test_rejected_path(self, client):
        resp = client.post("/execute", json={
            "command": "ls /etc/passwd",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is True
        assert data["exit_code"] == -1
        assert "FILTERED" in data["stderr"]

    def test_compound_rejected(self, client):
        resp = client.post("/execute", json={
            "command": "echo ok && cat /etc/shadow",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is True

    def test_relative_path_allowed(self, client):
        resp = client.post("/execute", json={
            "command": "echo test > /tmp/test_bash_backend.txt && cat /tmp/test_bash_backend.txt",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False
        assert data["exit_code"] == 0

    def test_whitelisted_path(self, client):
        resp = client.post("/execute", json={
            "command": "ls /var",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False


# ---------------------------------------------------------------------------
# Output Length Check Integration
# ---------------------------------------------------------------------------


class TestOutputLengthCheckIntegration:
    """输出长度检查集成测试。"""

    def test_short_output_passes(self, client):
        resp = client.post("/execute", json={
            "command": "echo short",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False

    def test_long_output_rejected(self, client):
        """生成超长输出，应被 post-check 拦截。"""
        # max_output_length=500 in fixture
        resp = client.post("/execute", json={
            "command": "python3 -c \"print('x' * 1000)\"",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is True
        assert "POST-CHECK" in data["stderr"] or "超过限制" in data["stderr"]


# ---------------------------------------------------------------------------
# Dangerous Command Integration
# ---------------------------------------------------------------------------


class TestDangerousCommandIntegration:
    """危险命令过滤集成测试。"""

    def test_normal_command_not_filtered(self, client):
        resp = client.post("/execute", json={
            "command": "echo hello world",
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False
        assert data["exit_code"] == 0

    # --- 升级后: 逃逸手法端到端拦截 ---

    @pytest.mark.parametrize(
        "command",
        [
            "cat /etc/passwd",                     # 越界读
            "echo x > /tmp/../etc/passwd",         # 相对路径穿越逃逸
            "echo `id`",                           # 反引号命令替换
            "echo $(cat /etc/passwd)",             # $() 内部越界读
            "rm -rf /",                            # 危险删除
            "mv /tmp/a --target-directory=/etc",   # mv flag 绕过
            "find / -name x -exec rm {} \\;",      # find 内执行
            "cd /tmp && echo x > secret.txt",      # cd 后写路径
        ],
    )
    def test_escape_filtered(self, client, command):
        resp = client.post("/execute", json={
            "command": command,
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is True
        assert data["exit_code"] == -1
        assert "FILTERED" in data["stderr"]

    @pytest.mark.parametrize(
        "command",
        [
            "echo $(date)",                         # 良性命令替换
            "command -v ls",                        # eval 类豁免
            "ls /tmp | head -1",                    # 白名单读
        ],
    )
    def test_benign_not_filtered(self, client, command):
        resp = client.post("/execute", json={
            "command": command,
            "agent_workspace": "/tmp",
        })
        data = resp.json()
        assert data["filtered"] is False


# ---------------------------------------------------------------------------
# /processes
# ---------------------------------------------------------------------------


class TestProcessesEndpoint:
    """进程列表端点测试。"""

    def test_list_empty(self, client):
        resp = client.get("/processes")
        assert resp.status_code == 200
        # 可能有之前测试留下的进程
        assert isinstance(resp.json(), list)

    def test_list_after_execute(self, client):
        client.post("/execute", json={
            "command": "echo test",
            "agent_workspace": "/tmp",
        })
        resp = client.get("/processes")
        procs = resp.json()
        assert len(procs) >= 1
        # 检查结构
        p = procs[-1]
        assert "process_id" in p
        assert "command" in p
        assert "status" in p

    def test_filter_by_status(self, client):
        client.post("/execute", json={
            "command": "echo test",
            "agent_workspace": "/tmp",
        })
        resp = client.get("/processes?status=completed")
        assert resp.status_code == 200
        for p in resp.json():
            assert p["status"] == "completed"

    def test_invalid_status(self, client):
        resp = client.get("/processes?status=invalid_status")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /kill and /kill_all
# ---------------------------------------------------------------------------


class TestKillEndpoints:
    """Kill 端点测试。"""

    def test_kill_nonexistent(self, client):
        resp = client.post("/kill/nonexistent-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_kill_all_empty(self, client):
        resp = client.post("/kill_all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["killed_count"] == 0


# ---------------------------------------------------------------------------
# /filters
# ---------------------------------------------------------------------------


class TestFiltersEndpoint:
    """过滤器端点测试。"""

    def test_list_filters(self, client):
        resp = client.get("/filters")
        assert resp.status_code == 200
        filters = resp.json()
        assert len(filters) >= 3  # sandbox, dangerous, output_length

        names = {f["name"] for f in filters}
        assert "filesystem_sandbox" in names
        assert "dangerous_command" in names
        assert "output_length" in names

    def test_toggle_filter(self, client):
        # 先获取当前状态
        filters = client.get("/filters").json()
        sandbox = next(f for f in filters if f["name"] == "filesystem_sandbox")
        original_state = sandbox["enabled"]

        # 切换
        new_state = not original_state
        resp = client.post("/filters/filesystem_sandbox/toggle", json={
            "enabled": new_state,
        })
        assert resp.status_code == 200
        assert resp.json()["enabled"] == new_state

        # 验证已更改
        filters = client.get("/filters").json()
        sandbox = next(f for f in filters if f["name"] == "filesystem_sandbox")
        assert sandbox["enabled"] == new_state

        # 恢复
        client.post("/filters/filesystem_sandbox/toggle", json={
            "enabled": original_state,
        })

    def test_toggle_nonexistent(self, client):
        resp = client.post("/filters/nonexistent_filter/toggle", json={
            "enabled": True,
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    """系统状态端点测试。"""

    def test_system_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_usage_percent" in data
        assert "memory" in data
        assert "active_processes" in data
        assert "total_tracked_processes" in data
