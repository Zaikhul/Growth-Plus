"""Abstract port definition for data rights authorization."""

from typing import Protocol

from src.domain.rights import DataOperation, RightsEvaluationResult


class RightsAuthorizer(Protocol):
    """Port for verifying data operation entitlements against active policies."""

    def authorize(
        self,
        source_id: str,
        dataset_id: str,
        operation: DataOperation,
    ) -> RightsEvaluationResult:
        """Evaluate if operation is legally and contractually authorized."""
        ...

    def is_source_enabled(self, source_id: str) -> bool:
        """Check if source is enabled in current configuration."""
        ...
