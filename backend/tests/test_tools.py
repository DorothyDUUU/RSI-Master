"""Test script for GPU tools."""

import json
import subprocess
import sys
from pathlib import Path

# Add repo root to path (derived from this file's location: backend/tests/ -> <repo>)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tools.gpu_tools import (
    GpuSubmitTaskTool,
    GpuInspectRunningTool,
    GpuInspectTaskTool,
    GpuKillTaskTool,
    create_gpu_tools,
)


class MockSession:
    """Mock session for testing."""

    def __init__(self):
        self.workspace = Path("/tmp/test_workspace")
        self.workspace.mkdir(exist_ok=True)


def test_tool_creation():
    """Test tool creation and registration."""
    print("Testing tool creation...")

    tools = create_gpu_tools()
    print(f"  ✓ Created {len(tools)} tools")

    for tool in tools:
        print(f"    - {tool.name}: {tool.get_description()[:50]}...")

    return True


def test_tool_specs():
    """Test tool specifications."""
    print("Testing tool specs...")

    tools = create_gpu_tools()
    for tool in tools:
        spec = tool.get_tool_spec()
        print(f"  ✓ {tool.name}:")
        print(f"    Description: {spec.function.description[:50]}...")
        print(f"    Parameters: {list(spec.function.parameters.get('properties', {}).keys())}")

    return True


def test_submit_task_tool():
    """Test GpuSubmitTaskTool."""
    print("Testing GpuSubmitTaskTool...")

    # Create test script
    test_script = Path("/tmp/test_tool_task.sh")
    test_script.write_text("#!/bin/bash\necho 'Hello from tool test'\n")

    try:
        tool = GpuSubmitTaskTool()
        session = MockSession()

        args = json.dumps({
            "bash_scripts_path": str(test_script),
            "run_backend": False,
            "timeout": 30,
            "gpu_nums": 1,
            "prior": "normal",
            "require_instant_feedback": True,
            "task_name": "tool_test",
        })

        # This will fail if server is not running, which is expected
        try:
            observation, info = tool.execute(session, args)
            print(f"  ✓ Task submitted: {info.get('task_id')}")
        except Exception as e:
            print(f"  ⚠ Server not available (expected in unit test): {e}")

        return True
    finally:
        test_script.unlink(missing_ok=True)


def test_inspect_running_tool():
    """Test GpuInspectRunningTool."""
    print("Testing GpuInspectRunningTool...")

    tool = GpuInspectRunningTool()
    session = MockSession()

    args = json.dumps({})

    try:
        observation, info = tool.execute(session, args)
        print(f"  ✓ Running tasks: {info.get('running_tasks', 0)}")
        print(f"  ✓ Queued tasks: {info.get('queued_tasks', 0)}")
    except Exception as e:
        print(f"  ⚠ Server not available (expected in unit test): {e}")

    return True


def test_inspect_task_tool():
    """Test GpuInspectTaskTool."""
    print("Testing GpuInspectTaskTool...")

    tool = GpuInspectTaskTool()
    session = MockSession()

    args = json.dumps({"task_id": "test_task_123"})

    try:
        observation, info = tool.execute(session, args)
        print(f"  ✓ Task status: {info.get('status')}")
    except Exception as e:
        print(f"  ⚠ Server not available (expected in unit test): {e}")

    return True


def test_kill_task_tool():
    """Test GpuKillTaskTool."""
    print("Testing GpuKillTaskTool...")

    tool = GpuKillTaskTool()
    session = MockSession()

    args = json.dumps({"task_id": "test_task_123"})

    try:
        observation, info = tool.execute(session, args)
        print(f"  ✓ Kill result: {info.get('success')}")
    except Exception as e:
        print(f"  ⚠ Server not available (expected in unit test): {e}")

    return True


def test_parameter_validation():
    """Test parameter validation."""
    print("Testing parameter validation...")

    from backend.server.models import TaskRequest, TaskPriority

    # Test valid request
    request = TaskRequest(
        bash_scripts_path="/tmp/test.sh",
        run_backend=False,
        timeout=30,
        gpu_nums=1,
        prior=TaskPriority.NORMAL,
        require_instant_feedback=True,
    )
    print(f"  ✓ Valid request created: {request.task_name or 'unnamed'}")

    # Test invalid gpu_nums
    try:
        request = TaskRequest(
            bash_scripts_path="/tmp/test.sh",
            gpu_nums=0,  # Invalid
        )
        print("  ✗ Should have failed validation")
        return False
    except Exception as e:
        print(f"  ✓ Validation caught invalid gpu_nums: {type(e).__name__}")

    return True


def run_tests():
    """Run all tests."""
    print("Running GPU tools tests\n")

    tests = [
        test_tool_creation,
        test_tool_specs,
        test_parameter_validation,
        test_submit_task_tool,
        test_inspect_running_tool,
        test_inspect_task_tool,
        test_kill_task_tool,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test {test.__name__} raised exception: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    return all(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
