"""In-memory tenant repositories with tenant isolation and optimistic concurrency."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from src.domain.errors import InvariantViolationError
from src.ports.repositories import TenantAlertRule, TenantWatchlist


class InMemoryWatchlistRepository:
    """In-memory watchlist repository with strict tenant isolation and optimistic locking."""

    def __init__(self) -> None:
        # Keyed by (tenant_id, watchlist_id)
        self._store: dict[tuple[uuid.UUID, uuid.UUID], TenantWatchlist] = {}

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[TenantWatchlist]:
        """Fetch all watchlists owned by tenant."""
        return [wl for (t_id, _), wl in self._store.items() if t_id == tenant_id]

    async def get_by_id(
        self, tenant_id: uuid.UUID, watchlist_id: uuid.UUID
    ) -> TenantWatchlist | None:
        """Fetch watchlist by ID scoped strictly to tenant."""
        return self._store.get((tenant_id, watchlist_id))

    async def save(
        self, watchlist: TenantWatchlist, expected_version: int | None = None
    ) -> TenantWatchlist:
        """Create or update watchlist with optimistic locking."""
        key = (watchlist.tenant_id, watchlist.id)
        existing = self._store.get(key)

        if existing is not None:
            if expected_version is not None and existing.version != expected_version:
                raise InvariantViolationError(
                    f"Optimistic concurrency conflict: expected version {expected_version}, "
                    f"found version {existing.version}"
                )
            new_version = existing.version + 1
        else:
            new_version = 1

        updated = TenantWatchlist(
            id=watchlist.id,
            tenant_id=watchlist.tenant_id,
            name=watchlist.name,
            markets=tuple(watchlist.markets),
            version=new_version,
            updated_at=datetime.now(tz=UTC),
        )
        self._store[key] = updated
        return updated

    async def delete(self, tenant_id: uuid.UUID, watchlist_id: uuid.UUID) -> bool:
        """Delete watchlist scoped to tenant."""
        key = (tenant_id, watchlist_id)
        if key in self._store:
            del self._store[key]
            return True
        return False


class InMemoryAlertRuleRepository:
    """In-memory alert rule repository with strict tenant isolation and optimistic locking."""

    def __init__(self) -> None:
        # Keyed by (tenant_id, rule_id)
        self._store: dict[tuple[uuid.UUID, uuid.UUID], TenantAlertRule] = {}

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[TenantAlertRule]:
        """Fetch all alert rules owned by tenant."""
        return [rule for (t_id, _), rule in self._store.items() if t_id == tenant_id]

    async def get_by_id(
        self, tenant_id: uuid.UUID, rule_id: uuid.UUID
    ) -> TenantAlertRule | None:
        """Fetch alert rule by ID scoped strictly to tenant."""
        return self._store.get((tenant_id, rule_id))

    async def save(
        self, rule: TenantAlertRule, expected_version: int | None = None
    ) -> TenantAlertRule:
        """Create or update alert rule with optimistic locking."""
        key = (rule.tenant_id, rule.id)
        existing = self._store.get(key)

        now = datetime.now(tz=UTC)
        if existing is not None:
            if expected_version is not None and existing.version != expected_version:
                raise InvariantViolationError(
                    f"Optimistic concurrency conflict: expected version {expected_version}, "
                    f"found version {existing.version}"
                )
            new_version = existing.version + 1
            created_at = existing.created_at or now
        else:
            new_version = 1
            created_at = rule.created_at or now

        updated = TenantAlertRule(
            id=rule.id,
            tenant_id=rule.tenant_id,
            name=rule.name,
            markets=tuple(rule.markets),
            horizons=tuple(rule.horizons),
            labels=tuple(rule.labels),
            min_confidence=rule.min_confidence,
            cooldown_seconds=rule.cooldown_seconds,
            target_channels=tuple(rule.target_channels),
            destination=rule.destination,
            destination_status=rule.destination_status,
            is_active=rule.is_active,
            consent_at=rule.consent_at or now,
            version=new_version,
            created_at=created_at,
            updated_at=now,
        )
        self._store[key] = updated
        return updated

    async def delete(self, tenant_id: uuid.UUID, rule_id: uuid.UUID) -> bool:
        """Delete alert rule scoped to tenant."""
        key = (tenant_id, rule_id)
        if key in self._store:
            del self._store[key]
            return True
        return False
