"""共享 fixtures。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture
def app():
    """创建测试用 FastAPI 应用。"""
    from tools.bash_backend.app import create_app
    return create_app(
        default_workspace="/tmp/test_workspace",
        whitelist=["/tmp"],
        max_output_length=1000,
    )


@pytest.fixture
def client(app):
    """创建测试客户端。"""
    return TestClient(app)


@pytest.fixture
def sample_workspace(tmp_path):
    """创建临时工作目录并写入测试文件。"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "hello.txt").write_text("hello world")
    (ws / "subdir").mkdir()
    (ws / "subdir" / "nested.txt").write_text("nested content")
    return str(ws)
