"""D-state-hard subprocess runner.

``subprocess.run(timeout=N)`` cannot reliably bound a child stuck in
uninterruptible sleep (D-state — e.g. a wedged NVIDIA driver or a blocked NFS
mount): SIGKILL is deferred until the kernel unblocks, so ``wait()`` blocks the
caller for minutes.  That was the 0826 disconnect root cause — the VRAM scanner
held ``_state_lock`` across ``subprocess.run(["nvidia-smi"], ...)``, and one
D-state child pinned the lock (and every lock-taking endpoint, incl. /health)
for up to ~2h.

``run_hard`` runs each subprocess on a daemon thread and waits on a queue with
an independent wall-clock bound: on expiry we *abandon* the thread (daemon — it
can never block process exit) and return the :data:`HUNG` sentinel.  Callers
treat ``HUNG`` as "unknown/failed" and must not hold locks across a call.
"""

from __future__ import annotations

import logging
import subprocess
import threading

logger = logging.getLogger(__name__)

# Sentinel for an abandoned (wedged / D-state) subprocess.  ``returncode`` of
# -1 is what this codebase already uses for "killed / unknown" exit paths.
HUNG = subprocess.CompletedProcess(args=[], returncode=-1, stdout="", stderr="")

# Bound the number of abandoned daemon threads a prolonged wedge can leak.
MAX_IN_FLIGHT = 16
_in_flight = 0
_in_flight_lock = threading.Lock()


def run_hard(
    args: list[str],
    timeout: int = 10,
    wall_ms: int = 15000,
) -> subprocess.CompletedProcess:
    """Run *args* bounded by an independent wall clock that survives D-state.

    Returns the real ``CompletedProcess`` (any return code) on success, or the
    :data:`HUNG` sentinel if the child didn't finish within ``wall_ms`` — the
    daemon thread is abandoned, never the caller.
    """
    global _in_flight
    with _in_flight_lock:
        if _in_flight >= MAX_IN_FLIGHT:
            logger.warning("run_hard: %d wedged subprocesses already — returning HUNG",
                           _in_flight)
            return HUNG
        _in_flight += 1

    q: "queue.Queue[subprocess.CompletedProcess]" = __import__("queue").Queue(maxsize=1)

    def _run() -> None:
        try:
            q.put(subprocess.run(args, capture_output=True, text=True, timeout=timeout))
        except subprocess.TimeoutExpired:
            q.put(HUNG)
        except Exception:  # noqa: BLE001 — FileNotFoundError etc.
            q.put(HUNG)
        finally:
            global _in_flight
            with _in_flight_lock:
                _in_flight -= 1

    threading.Thread(target=_run, daemon=True, name="gpu-subproc").start()
    try:
        return q.get(timeout=wall_ms / 1000)
    except Exception:  # queue.Empty
        logger.warning("subprocess %s hung past %dms wall (D-state?) — abandoned",
                       args[0] if args else "?", wall_ms)
        return HUNG
