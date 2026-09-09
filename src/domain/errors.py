"""Domain-specific error hierarchy for Growth+.

Pure domain exceptions with structured error codes and zero external dependencies.
"""

from collections.abc import Mapping
from typing import Any


class DomainError(Exception):
    """Base exception for all Growth+ domain errors."""

    def __init__(self, message: str, code: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class PointInTimeViolationError(DomainError):
    """Raised when data availability violates Point-in-Time cutoff rules."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="POINT_IN_TIME_VIOLATION", details=details)


class RightsViolationError(DomainError):
    """Raised when an operation is executed on data lacking valid legal/commercial rights."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="RIGHTS_VIOLATION", details=details)


class InvariantViolationError(DomainError):
    """Raised when a core domain invariant is violated (e.g. probabilities do not sum to 1)."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="INVARIANT_VIOLATION", details=details)


class StaleDataError(DomainError):
    """Raised when essential market or reference data exceeds hard freshness stop limits."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="STALE_DATA_LIMIT_EXCEEDED", details=details)


class InvalidEntityIdError(DomainError):
    """Raised when an entity identifier fails domain format or normalization rules."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="INVALID_ENTITY_IDENTIFIER", details=details)


class SecurityError(DomainError):
    """Raised when cryptographic verification, tampering, or authorization fails."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="SECURITY_VIOLATION", details=details)


class EmptyPartitionError(DomainError):
    """Raised when an empirical cross-validation or purging partition contains zero samples."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message=message, code="EMPTY_PARTITION_ERROR", details=details)
