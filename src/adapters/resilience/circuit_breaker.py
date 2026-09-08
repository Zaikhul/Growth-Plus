"""Process-local circuit with atomic admission and generation-fenced results."""

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from src.domain.errors import DomainError


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(DomainError):
    def __init__(self, name: str, retry_after_seconds: float) -> None:
        super().__init__(
            message=f"Circuit '{name}' cannot admit this operation",
            code="CIRCUIT_BREAKER_OPEN",
            details={"circuit_name": name, "retry_after_seconds": retry_after_seconds},
        )


@dataclass(frozen=True, slots=True)
class _Permit:
    generation: int
    probe: bool
    acquired_at: float = 0.0


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout_seconds: float | None = None,
        half_open_success_threshold: int = 1,
        probe_timeout_seconds: float | None = None,
    ) -> None:
        from src.config.settings import get_settings

        cfg = get_settings().resilience
        fail_thresh = (
            failure_threshold if failure_threshold is not None else cfg.circuit_failure_threshold
        )
        recov_to = (
            recovery_timeout_seconds
            if recovery_timeout_seconds is not None
            else cfg.circuit_recovery_timeout_seconds
        )
        probe_to = (
            probe_timeout_seconds
            if probe_timeout_seconds is not None
            else cfg.circuit_probe_timeout_seconds
        )

        if fail_thresh < 1 or not 60 <= recov_to <= 900:
            raise ValueError("Invalid circuit policy")
        if half_open_success_threshold != 1:
            raise ValueError("PRD recovery uses one half-open probe")
        self._name = name
        self._failure_threshold = fail_thresh
        self._base_timeout = float(recov_to)
        self._recovery_timeout = self._base_timeout
        self._probe_timeout = float(probe_to)
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._generation = 0
        self._opened_at = 0.0
        self._probe_inflight = False
        self._results: deque[tuple[float, bool]] = deque()
        self._permits: dict[tuple[int, int], _Permit] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _owner() -> tuple[int, int]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), id(task) if task is not None else 0

    @property
    def name(self) -> str:
        return self._name

    def _refresh(self, now: float) -> None:
        while self._results and self._results[0][0] <= now - 60:
            self._results.popleft()
        if self._state == CircuitState.OPEN and now - self._opened_at >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._probe_inflight = False

        # DEFECT-15: Reclaim abandoned permits
        stale_owners = [
            owner
            for owner, p in self._permits.items()
            if now - p.acquired_at > (self._probe_timeout if p.probe else 120.0)
        ]
        for owner in stale_owners:
            p = self._permits.pop(owner)
            if p.probe and p.generation == self._generation:
                # Abandoned probe should re-open circuit
                self._open(now, failed_probe=True)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._refresh(time.monotonic())
            return self._state

    def _check_admissibility(self, now: float) -> tuple[bool, float]:
        """Check if operation can be admitted. Returns (allowed, retry_after)."""
        owner = self._owner()
        if owner in self._permits:
            return False, 1.0
        if len(self._permits) >= 1024:
            return False, 1.0
        if self._state == CircuitState.OPEN:
            remaining = max(0.01, self._recovery_timeout - (now - self._opened_at))
            return False, remaining
        probe = self._state == CircuitState.HALF_OPEN
        if probe and self._probe_inflight:
            return False, 1.0
        return True, 0.0

    def can_execute(self) -> bool:
        """Reserve admission; each True must be paired with one result/cancel call."""
        with self._lock:
            now = time.monotonic()
            self._refresh(now)
            allowed, _ = self._check_admissibility(now)
            if not allowed:
                return False
            probe = self._state == CircuitState.HALF_OPEN
            self._probe_inflight = self._probe_inflight or probe
            self._permits[self._owner()] = _Permit(self._generation, probe, acquired_at=now)
            return True

    def check_permission(self) -> None:
        """Reserve admission or raise CircuitBreakerOpenError (DEFECT-46)."""
        with self._lock:
            now = time.monotonic()
            self._refresh(now)
            allowed, retry_after = self._check_admissibility(now)
            if not allowed:
                raise CircuitBreakerOpenError(self._name, retry_after_seconds=retry_after)
            probe = self._state == CircuitState.HALF_OPEN
            self._probe_inflight = self._probe_inflight or probe
            self._permits[self._owner()] = _Permit(self._generation, probe, acquired_at=now)

    @asynccontextmanager
    async def guarded(self) -> AsyncIterator[None]:
        """Context manager ensuring permit release on completion or error (DEFECT-15)."""
        self.check_permission()
        try:
            yield
        except asyncio.CancelledError:
            self.record_cancelled()
            raise
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()

    def _open(self, now: float, failed_probe: bool) -> None:
        self._state = CircuitState.OPEN
        self._generation += 1
        self._opened_at = now
        self._probe_inflight = False
        if failed_probe:
            self._recovery_timeout = min(900.0, self._recovery_timeout * 2)

    def _record(self, succeeded: bool) -> None:
        with self._lock:
            permit = self._permits.pop(self._owner(), None)
            if permit is None:
                raise RuntimeError("Circuit result has no matching admission")
            if permit.generation != self._generation:
                return  # An old in-flight operation cannot change a newer generation.
            now = time.monotonic()
            self._refresh(now)
            if permit.probe:
                self._probe_inflight = False
                if succeeded:
                    self._state = CircuitState.CLOSED
                    self._generation += 1
                    self._consecutive_failures = 0
                    self._results.clear()
                    self._recovery_timeout = self._base_timeout
                else:
                    self._open(now, failed_probe=True)
                return
            self._results.append((now, succeeded))
            self._consecutive_failures = 0 if succeeded else self._consecutive_failures + 1
            rate_exceeded = len(self._results) >= 20 and sum(
                not ok for _, ok in self._results
            ) * 2 > len(self._results)
            if self._consecutive_failures >= self._failure_threshold or rate_exceeded:
                self._open(now, failed_probe=False)

    def record_success(self) -> None:
        self._record(True)

    def record_failure(self) -> None:
        self._record(False)

    def record_cancelled(self) -> None:
        with self._lock:
            permit = self._permits.pop(self._owner(), None)
            if permit is None:
                raise RuntimeError("Cancellation has no matching admission")
            if permit.probe and permit.generation == self._generation:
                self._open(time.monotonic(), failed_probe=True)
