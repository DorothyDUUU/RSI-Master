"""GPU resource monitoring — nvidia-smi + psutil (+ optional nvitop)."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any

import psutil

from .models import GPUDevice, GPUInfo
from .subproc import run_hard

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors GPU devices and system resources.

    Thresholds are enforced so the scheduler never over-commits a device.

    Snapshot cache: every device query runs on a background sampler thread
    (``start_sampler``) and callers read a cached ``GPUInfo`` — never a
    subprocess on their own path.  A wedged NVIDIA driver (a D-state
    nvidia-smi child that ``subprocess.run(timeout=...)`` cannot kill) therefore
    cannot stall any endpoint or lock holder; it only leaves the snapshot
    *stale*, which the scheduler treats as "no free GPUs" (safe) and /health
    reports as "degraded".
    """

    MAX_MEMORY_UTILIZATION = 95.0   # percent
    MAX_GPU_UTILIZATION = 99.0      # percent
    # Wall-clock bound for the nvitop fallback — its NVML calls have no native
    # timeout and a wedged driver can hang them for minutes.
    NVITOP_TIMEOUT = 15             # seconds
    SAMPLE_INTERVAL = 5.0           # seconds between background refreshes
    SAMPLE_WALL_MS = 15000          # hard wall for one nvidia-smi subprocess

    def __init__(self) -> None:
        self._nvidia_smi_available = self._check_nvidia_smi()
        self._nvitop_available = self._check_nvitop()
        self._cache: GPUInfo | None = None
        self._cache_ts: float = 0.0            # 0 = never sampled
        self._last_refresh_ok: bool | None = None  # None = never refreshed
        self._sample_in_flight = False
        self._cache_lock = threading.Lock()
        self._sampler_thread: threading.Thread | None = None

    # -- capability checks ----------------------------------------------------

    @staticmethod
    def _check_nvidia_smi() -> bool:
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("nvidia-smi not available — GPU monitoring disabled")
            return False

    @staticmethod
    def _check_nvitop() -> bool:
        try:
            import nvitop  # noqa: F401
            return True
        except ImportError:
            logger.debug("nvitop not installed — falling back to nvidia-smi only")
            return False

    # -- background snapshot sampler -----------------------------------------

    def start_sampler(self) -> None:
        """Start the background VRAM sampler thread (idempotent).

        Takes one synchronous sample first so callers never see an empty cache
        (and the server never mis-schedules on missing data at boot).  Called by
        the server lifespan; direct-construction callers (unit tests) skip it
        and get bounded one-shot samples instead.
        """
        if self._sampler_thread is not None and self._sampler_thread.is_alive():
            return
        self._refresh_once()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, name="gpu-vram-sampler", daemon=True
        )
        self._sampler_thread.start()
        logger.info("GPU monitor sampler thread started")

    def _sampler_loop(self) -> None:
        while True:
            time.sleep(self.SAMPLE_INTERVAL)
            self._refresh_once()

    def _refresh_once(self) -> GPUInfo | None:
        """Take one bounded sample and store it as the snapshot.

        Runs on the sampler thread (or a one-shot caller when no sampler is
        started).  Never blocks beyond the subprocess wall clocks.  A failed
        sample (wedged driver) keeps the previous snapshot and marks it stale —
        an empty result is only stored when it is genuinely valid.
        """
        with self._cache_lock:
            if self._sample_in_flight:
                return self._cache
            self._sample_in_flight = True
        try:
            devices, ok = self._collect_devices()
            with self._cache_lock:
                self._last_refresh_ok = ok
            if ok:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                info = GPUInfo(
                    devices=devices,
                    total_devices=len(devices),
                    available_devices=[
                        d.device_id for d in devices
                        if d.vram_usage_percent < self.MAX_MEMORY_UTILIZATION
                        and d.gpu_utilization < self.MAX_GPU_UTILIZATION
                    ],
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_total_gb=memory.total / (1024 ** 3),
                    memory_used_gb=memory.used / (1024 ** 3),
                )
                with self._cache_lock:
                    self._cache = info
                    self._cache_ts = time.time()
            return self._cache
        except Exception:  # noqa: BLE001
            logger.exception("GPU sample refresh failed")
            return self._cache
        finally:
            with self._cache_lock:
                self._sample_in_flight = False

    # -- device collection (bounded) -----------------------------------------

    def _collect_devices(self) -> tuple[list[GPUDevice], bool]:
        """Collect device metrics.  Returns (devices, ok); ok=False means the
        sample failed (e.g. wedged driver) and the caller must keep the
        previous snapshot.

        The nvidia-smi subprocess runs under ``run_hard``'s independent wall
        clock, so a D-state child abandons after ``SAMPLE_WALL_MS`` instead of
        blocking the sampler thread (and any lock) for minutes.
        """
        if not self._nvidia_smi_available:
            fallback = self._get_devices_via_nvitop()
            return fallback, bool(fallback)

        result = run_hard(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            wall_ms=self.SAMPLE_WALL_MS,
        )
        if result.returncode != 0:  # includes HUNG (-1) and genuine failures
            logger.warning("nvidia-smi query failed (rc=%d) — falling back to nvitop",
                           result.returncode)
            fallback = self._get_devices_via_nvitop()
            return fallback, bool(fallback)
        return self._parse_nvidia_smi(result.stdout), True

    @staticmethod
    def _parse_nvidia_smi(stdout: str) -> list[GPUDevice]:
        devices: list[GPUDevice] = []
        for line in stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                devices.append(GPUDevice(
                    device_id=int(parts[0]),
                    name=parts[1],
                    total_memory_mb=int(parts[2]),
                    used_memory_mb=int(parts[3]),
                    free_memory_mb=int(parts[4]),
                    gpu_utilization=float(parts[5]),
                    memory_utilization=float(parts[6]),
                    temperature=float(parts[7]) if len(parts) > 7 and parts[7] else None,
                ))
            except (ValueError, IndexError):
                continue
        return devices

    def _get_devices_via_nvitop(self) -> list[GPUDevice]:
        """Fallback: collect device info via nvitop.

        Bounded: a wedged driver can hang nvitop's NVML calls with no
        Python-side timeout, so the query runs on a daemon thread and we wait at
        most ``NVITOP_TIMEOUT``.  A stall here costs one worker thread, never the
        event loop or a whole VRAM scan.
        """
        if not self._nvitop_available:
            return []
        import queue as _queue

        result_q: _queue.Queue = _queue.Queue(maxsize=1)

        def _run() -> None:
            try:
                result_q.put(self._collect_nvitop_devices())
            except Exception:  # noqa: BLE001
                logger.exception("nvitop device query failed")
                result_q.put(None)

        worker = threading.Thread(target=_run, name="nvitop-query", daemon=True)
        worker.start()
        try:
            devices = result_q.get(timeout=self.NVITOP_TIMEOUT)
        except _queue.Empty:
            logger.warning("nvitop device query timed out after %ds", self.NVITOP_TIMEOUT)
            return []
        return devices or []

    @staticmethod
    def _collect_nvitop_devices() -> list[GPUDevice]:
        from nvitop import Device
        devices: list[GPUDevice] = []
        for dev in Device.all():
            mem = dev.memory_stats()
            devices.append(GPUDevice(
                device_id=dev.index,
                name=dev.name(),
                total_memory_mb=mem.get("total", 0),
                used_memory_mb=mem.get("used", 0),
                free_memory_mb=mem.get("free", 0),
                gpu_utilization=dev.utilization(),
                memory_utilization=mem.get("utilization", 0.0),
                temperature=dev.temperature(),
            ))
        return devices

    # -- aggregated info ------------------------------------------------------

    def get_gpu_devices(self) -> list[GPUDevice]:
        """Return the latest per-device metrics from the snapshot cache.

        Never spawns a subprocess once the sampler has produced a snapshot; the
        pre-sampler fallback is a single bounded sample.
        """
        snapshot = self._snapshot()
        if snapshot is not None:
            return snapshot.devices
        devices, _ = self._collect_devices()
        return devices

    def get_gpu_info(self) -> GPUInfo:
        """Return the latest snapshot (never blocks past the subprocess walls).

        A stale snapshot (wedged driver) advertises no free devices so the
        scheduler stops placing work on unknown GPU state.
        """
        info = self._snapshot()
        if info is None:
            info = self._refresh_once()
        if info is None:
            info = self._empty_gpu_info()
        if self.snapshot_stale():
            return self._stale_view(info)
        return info

    def get_available_devices(self, num_devices: int = 1) -> list[int]:
        """Return up to *num_devices* free GPU ids, sorted by free memory desc."""
        devices = self.get_gpu_devices()
        available = [
            d for d in devices
            if d.vram_usage_percent < self.MAX_MEMORY_UTILIZATION
            and d.gpu_utilization < self.MAX_GPU_UTILIZATION
        ]
        available.sort(key=lambda d: d.free_memory_mb, reverse=True)

        if len(available) < num_devices:
            logger.warning("Requested %d GPU(s) but only %d available", num_devices, len(available))
            return []

        return [d.device_id for d in available[:num_devices]]

    # -- snapshot staleness ---------------------------------------------------

    def _snapshot(self) -> GPUInfo | None:
        with self._cache_lock:
            return self._cache

    def snapshot_age(self) -> float:
        """Seconds since the last successful sample (-1 before any — JSON-safe)."""
        if self._cache_ts <= 0:
            return -1.0
        return time.time() - self._cache_ts

    def snapshot_stale(self) -> bool:
        """True when nvidia-smi is expected to work but the latest sample failed
        (wedged driver).  On machines without nvidia-smi this is always False —
        an empty device set is the honest answer there."""
        return bool(self._nvidia_smi_available and self._last_refresh_ok is False)

    @staticmethod
    def _stale_view(info: GPUInfo) -> GPUInfo:
        return GPUInfo(
            devices=[], total_devices=0, available_devices=[],
            cpu_percent=info.cpu_percent, memory_percent=info.memory_percent,
            memory_total_gb=info.memory_total_gb, memory_used_gb=info.memory_used_gb,
        )

    @staticmethod
    def _empty_gpu_info() -> GPUInfo:
        return GPUInfo(
            devices=[], total_devices=0, available_devices=[],
            cpu_percent=0.0, memory_percent=0.0,
            memory_total_gb=0.0, memory_used_gb=0.0,
        )

    # -- health check ---------------------------------------------------------

    def check_resources(self) -> dict[str, Any]:
        """Return a health report: healthy flag + any threshold violations.

        Pure cache read — never spawns a subprocess, so a wedged driver cannot
        block the probe.
        """
        snapshot = self._snapshot()
        devices = snapshot.devices if snapshot else []
        stale = self.snapshot_stale()

        issues: list[str] = []
        for d in devices:
            if d.vram_usage_percent >= self.MAX_MEMORY_UTILIZATION:
                issues.append(
                    f"GPU {d.device_id}: VRAM {d.vram_usage_percent:.1f}% >= "
                    f"{self.MAX_MEMORY_UTILIZATION}%"
                )
            if d.gpu_utilization >= self.MAX_GPU_UTILIZATION:
                issues.append(
                    f"GPU {d.device_id}: util {d.gpu_utilization:.1f}% >= "
                    f"{self.MAX_GPU_UTILIZATION}%"
                )
        if stale:
            age = self.snapshot_age()
            age_s = "never" if age < 0 else f"{int(age)}s"
            issues.append(f"GPU snapshot stale (driver wedged?) — last sample {age_s} ago")

        cpu_percent = snapshot.cpu_percent if snapshot else 0.0
        memory_percent = snapshot.memory_percent if snapshot else 0.0
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_devices": [
                {
                    "id": d.device_id,
                    "vram_usage_percent": round(d.vram_usage_percent, 1),
                    "gpu_util": d.gpu_utilization,
                }
                for d in devices
            ],
        }
