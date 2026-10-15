"""进程跟踪器测试。"""

from __future__ import annotations

import time

import pytest

from tools.bash_backend.tracker import ProcessInfo, ProcessStatus, ProcessTracker


def _make_info(process_id: str = "test-1", **kwargs) -> ProcessInfo:
    """构造测试用 ProcessInfo。"""
    defaults = dict(
        process_id=process_id,
        pgid=12345,
        pid=12345,
        command="echo hello",
        agent_workspace="/workspace",
        start_time=time.time(),
    )
    defaults.update(kwargs)
    return ProcessInfo(**defaults)


class TestProcessTracker:
    """进程跟踪器测试。"""

    def test_register_and_get(self):
        tracker = ProcessTracker()
        info = _make_info("p1")
        tracker.register(info)

        got = tracker.get("p1")
        assert got is not None
        assert got.process_id == "p1"
        assert got.command == "echo hello"

    def test_get_nonexistent(self):
        tracker = ProcessTracker()
        assert tracker.get("nonexistent") is None

    def test_update_status(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))

        tracker.update("p1", status=ProcessStatus.COMPLETED, exit_code=0)
        info = tracker.get("p1")
        assert info.status == ProcessStatus.COMPLETED
        assert info.exit_code == 0

    def test_update_nonexistent(self):
        tracker = ProcessTracker()
        # 不应抛异常
        tracker.update("nonexistent", status=ProcessStatus.COMPLETED)

    def test_get_all(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))
        tracker.register(_make_info("p2"))

        all_procs = tracker.get_all()
        assert len(all_procs) == 2

    def test_get_all_with_status_filter(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))
        tracker.register(_make_info("p2"))
        tracker.update("p1", status=ProcessStatus.COMPLETED)

        completed = tracker.get_all(ProcessStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].process_id == "p1"

        running = tracker.get_all(ProcessStatus.RUNNING)
        assert len(running) == 1
        assert running[0].process_id == "p2"

    def test_get_active(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))
        tracker.register(_make_info("p2"))
        tracker.update("p2", status=ProcessStatus.COMPLETED)

        active = tracker.get_active()
        assert len(active) == 1
        assert active[0].process_id == "p1"

    def test_kill_process_not_found(self):
        tracker = ProcessTracker()
        ok, msg = tracker.kill_process("nonexistent")
        assert ok is False
        assert "不存在" in msg

    def test_kill_process_already_finished(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))
        tracker.update("p1", status=ProcessStatus.COMPLETED)

        ok, msg = tracker.kill_process("p1")
        assert ok is False
        assert "已结束" in msg

    def test_kill_all_empty(self):
        tracker = ProcessTracker()
        killed, messages = tracker.kill_all()
        assert killed == 0
        assert messages == []

    def test_cleanup_old(self):
        tracker = ProcessTracker()
        info = _make_info("p1")
        info.end_time = time.time() - 7200  # 2 hours ago
        tracker.register(info)
        tracker.update("p1", status=ProcessStatus.COMPLETED, end_time=info.end_time)

        cleaned = tracker.cleanup_old(max_age_seconds=3600)
        assert cleaned == 1
        assert tracker.get("p1") is None

    def test_cleanup_old_keeps_recent(self):
        tracker = ProcessTracker()
        tracker.register(_make_info("p1"))
        tracker.update("p1", status=ProcessStatus.COMPLETED, end_time=time.time())

        cleaned = tracker.cleanup_old(max_age_seconds=3600)
        assert cleaned == 0
        assert tracker.get("p1") is not None

    def test_thread_safety(self):
        """多线程并发注册和更新不应崩溃。"""
        import threading

        tracker = ProcessTracker()
        errors = []

        def worker(i: int):
            try:
                pid = f"p-{i}"
                tracker.register(_make_info(pid))
                tracker.update(pid, status=ProcessStatus.COMPLETED, exit_code=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker.get_all()) == 100
