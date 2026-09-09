"""Data rights policies, commercial entitlements, and legal boundaries.

Enforces Section 3.2 and Section 5.3 of PRD:
Unlicensed providers (Binance, Coinbase, Farside, FRED, Reddit, X)
must not be used for ML training, persistence, or derived signal distribution
without affirmative commercial entitlement. UNKNOWN denies the operation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from src.domain.errors import RightsViolationError
from src.domain.time import ensure_utc


class DataOperation(StrEnum):
    """Granular operations requiring independent legal entitlement."""

    COLLECT = "COLLECT"
    RETAIN_RAW = "RETAIN_RAW"
    COMPUTE_FEATURES = "COMPUTE_FEATURES"
    TRAIN_ML = "TRAIN_ML"
    INFER_ML = "INFER_ML"
    SERVE_DISPLAY = "SERVE_DISPLAY"
    DISTRIBUTE_SIGNAL = "DISTRIBUTE_SIGNAL"
    EXPORT_RAW = "EXPORT_RAW"


class EntitlementStatus(StrEnum):
    """Entitlement status. UNKNOWN must always deny the operation."""

    PERMITTED = "PERMITTED"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RightsEvaluationResult:
    """Result of an operation rights evaluation."""

    allowed: bool
    operation: DataOperation
    source_id: str
    dataset_id: str
    reason_code: str
    policy_id: str

    def assert_permitted(self) -> None:
        """Raise RightsViolationError if operation is denied."""
        if not self.allowed:
            msg = (
                f"Data rights denied for operation '{self.operation.value}' on "
                f"{self.source_id}/{self.dataset_id}: {self.reason_code}"
            )
            raise RightsViolationError(
                msg,
                details={
                    "operation": self.operation.value,
                    "source_id": self.source_id,
                    "dataset_id": self.dataset_id,
                    "reason_code": self.reason_code,
                    "policy_id": self.policy_id,
                },
            )


@dataclass(frozen=True, slots=True)
class RightsPolicy:
    """Immutable data rights policy specification for a source dataset."""

    policy_id: str
    source_id: str
    dataset_id: str
    version: str
    effective_from: datetime
    effective_to: datetime | None = None
    entitlements: Mapping[DataOperation, EntitlementStatus] = field(default_factory=dict)
    rationale: str = ""

    def evaluate(
        self,
        operation: DataOperation,
        eval_time: datetime | None = None,
    ) -> RightsEvaluationResult:
        """Evaluate an operation against this policy.

        Strict rule: If the operation is not present or is UNKNOWN, deny it.
        Also verifies effective_from and effective_to time bounds.
        """
        check_time = ensure_utc(eval_time) if eval_time is not None else datetime.now(tz=UTC)
        if check_time < ensure_utc(self.effective_from):
            return RightsEvaluationResult(
                allowed=False,
                operation=operation,
                source_id=self.source_id,
                dataset_id=self.dataset_id,
                reason_code=f"POLICY_NOT_YET_EFFECTIVE: Starts {self.effective_from.isoformat()}",
                policy_id=self.policy_id,
            )
        if self.effective_to is not None and check_time >= ensure_utc(self.effective_to):
            return RightsEvaluationResult(
                allowed=False,
                operation=operation,
                source_id=self.source_id,
                dataset_id=self.dataset_id,
                reason_code=f"POLICY_EXPIRED: Ended {self.effective_to.isoformat()}",
                policy_id=self.policy_id,
            )

        status = self.entitlements.get(operation, EntitlementStatus.UNKNOWN)
        if status == EntitlementStatus.PERMITTED:
            return RightsEvaluationResult(
                allowed=True,
                operation=operation,
                source_id=self.source_id,
                dataset_id=self.dataset_id,
                reason_code="AUTHORIZED",
                policy_id=self.policy_id,
            )

        reason = (
            "UNAUTHORIZED_UNKNOWN_PERMISSION"
            if status == EntitlementStatus.UNKNOWN
            else "OPERATION_EXPLICITLY_DENIED"
        )
        if self.rationale:
            reason = f"{reason}: {self.rationale}"

        return RightsEvaluationResult(
            allowed=False,
            operation=operation,
            source_id=self.source_id,
            dataset_id=self.dataset_id,
            reason_code=reason,
            policy_id=self.policy_id,
        )
