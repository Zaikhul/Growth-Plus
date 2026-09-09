"""Configuration-driven data rights authorizer.

Enforces Section 3.2 and Section 5.3 of PRD:
Loads defaults and source-level rights policies. Denies unentitled operations
and disabled providers.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.domain.rights import (
    DataOperation,
    EntitlementStatus,
    RightsEvaluationResult,
    RightsPolicy,
)


class ConfigRightsAuthorizer:
    """Evaluates rights policies based on configuration."""

    def __init__(
        self,
        policies: Mapping[str | tuple[str, str], RightsPolicy] | None = None,
        source_enabled_flags: Mapping[str, bool] | None = None,
        disabled_reasons: Mapping[str, str] | None = None,
    ) -> None:
        self._policies: dict[tuple[str, str], RightsPolicy] = {}
        if policies:
            for k, p in policies.items():
                if isinstance(k, tuple):
                    self._policies[k] = p
                else:
                    self._policies[(k, p.dataset_id)] = p
                    self._policies[(k, "*")] = p
        self._enabled_flags: dict[str, bool] = dict(source_enabled_flags or {})
        self._disabled_reasons: dict[str, str] = dict(disabled_reasons or {})

    @classmethod
    def from_yaml_file(cls, config_path: str | Path | None = None) -> "ConfigRightsAuthorizer":
        """Load source flags and build authorizer from YAML config file."""
        if config_path is None:
            from src.config.settings import get_settings

            path = get_settings().app.config_path
        else:
            path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        sources_cfg: dict[str, Any] = data.get("sources", {})
        enabled_flags: dict[str, bool] = {}
        disabled_reasons: dict[str, str] = {}
        policies: dict[str | tuple[str, str], RightsPolicy] = {}

        now = datetime.now(UTC)

        for source_id, s_conf in sources_cfg.items():
            is_enabled = bool(s_conf.get("enabled", False))
            enabled_flags[source_id] = is_enabled
            if not is_enabled:
                disabled_reasons[source_id] = str(s_conf.get("reason", "DISABLED_BY_POLICY"))
            else:
                # Default permitted operations for enabled official public sources
                policy = RightsPolicy(
                    policy_id=f"policy_{source_id}_default",
                    source_id=source_id,
                    dataset_id="default",
                    version="1.0.0",
                    effective_from=now,
                    entitlements={
                        DataOperation.COLLECT: EntitlementStatus.PERMITTED,
                        DataOperation.RETAIN_RAW: EntitlementStatus.PERMITTED,
                        DataOperation.COMPUTE_FEATURES: EntitlementStatus.PERMITTED,
                        DataOperation.TRAIN_ML: EntitlementStatus.PERMITTED,
                        DataOperation.INFER_ML: EntitlementStatus.PERMITTED,
                        DataOperation.DISTRIBUTE_SIGNAL: EntitlementStatus.PERMITTED,
                    },
                )
                policies[(source_id, "default")] = policy
                policies[(source_id, "*")] = policy

        return cls(
            policies=policies,
            source_enabled_flags=enabled_flags,
            disabled_reasons=disabled_reasons,
        )

    def is_source_enabled(self, source_id: str) -> bool:
        """Check if source is enabled in configuration."""
        return self._enabled_flags.get(source_id, False)

    def register_policy(self, policy: RightsPolicy) -> None:
        """Register or override a rights policy for a source and dataset."""
        self._policies[(policy.source_id, policy.dataset_id)] = policy
        self._policies[(policy.source_id, "*")] = policy

    def authorize(
        self,
        source_id: str,
        dataset_id: str,
        operation: DataOperation,
        eval_time: datetime | None = None,
    ) -> RightsEvaluationResult:
        """Evaluate if operation on source/dataset is permitted."""
        if not self.is_source_enabled(source_id):
            reason = self._disabled_reasons.get(
                source_id,
                f"Source '{source_id}' is disabled in configuration",
            )
            return RightsEvaluationResult(
                allowed=False,
                operation=operation,
                source_id=source_id,
                dataset_id=dataset_id,
                reason_code=f"SOURCE_DISABLED: {reason}",
                policy_id="config_defaults",
            )

        policy = (
            self._policies.get((source_id, dataset_id))
            or self._policies.get((source_id, "*"))
            or self._policies.get((source_id, "default"))
        )
        if policy is None:
            return RightsEvaluationResult(
                allowed=False,
                operation=operation,
                source_id=source_id,
                dataset_id=dataset_id,
                reason_code="UNAUTHORIZED_UNKNOWN_PERMISSION: No policy registered",
                policy_id="missing_policy",
            )

        return policy.evaluate(operation, eval_time=eval_time)
