"""PostgreSQL repositories for tenant-isolated models with Row-Level Security."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.persistence.postgres.models import (
    TenantAlertRuleModel,
    TenantWatchlistModel,
)
from src.adapters.persistence.postgres.tenant_context import set_session_tenant
from src.domain.errors import InvariantViolationError
from src.ports.repositories import TenantAlertRule, TenantWatchlist


class PostgresWatchlistRepository:
    """PostgreSQL watchlist repository executing within tenant RLS boundary."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[TenantWatchlist]:
        """Fetch all watchlists owned by tenant under RLS."""
        await set_session_tenant(self._session, tenant_id)
        stmt = (
            select(TenantWatchlistModel)
            .where(TenantWatchlistModel.tenant_id == tenant_id)
            .order_by(TenantWatchlistModel.name)
        )
        result = await self._session.execute(stmt)
        return [
            TenantWatchlist(
                id=row.id,
                tenant_id=row.tenant_id,
                name=row.name,
                markets=tuple(row.markets),
                version=int(row.version),
            )
            for row in result.scalars()
        ]

    async def get_by_id(
        self, tenant_id: uuid.UUID, watchlist_id: uuid.UUID
    ) -> TenantWatchlist | None:
        """Fetch watchlist by ID scoped to tenant."""
        await set_session_tenant(self._session, tenant_id)
        stmt = select(TenantWatchlistModel).where(
            TenantWatchlistModel.tenant_id == tenant_id,
            TenantWatchlistModel.id == watchlist_id,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return TenantWatchlist(
            id=row.id,
            tenant_id=row.tenant_id,
            name=row.name,
            markets=tuple(row.markets),
            version=int(row.version),
        )

    async def save(
        self, watchlist: TenantWatchlist, expected_version: int | None = None
    ) -> TenantWatchlist:
        """Create or update watchlist with optimistic concurrency check."""
        await set_session_tenant(self._session, watchlist.tenant_id)
        existing = await self.get_by_id(watchlist.tenant_id, watchlist.id)

        if existing is None:
            # Create new row
            row = TenantWatchlistModel(
                id=watchlist.id,
                tenant_id=watchlist.tenant_id,
                name=watchlist.name,
                markets=list(watchlist.markets),
                version=1,
            )
            self._session.add(row)
            await self._session.flush()
            return TenantWatchlist(
                id=row.id,
                tenant_id=row.tenant_id,
                name=row.name,
                markets=tuple(row.markets),
                version=int(row.version),
                updated_at=datetime.now(tz=UTC),
            )

        # Update existing row with optimistic locking
        if expected_version is not None and existing.version != expected_version:
            raise InvariantViolationError(
                f"Optimistic concurrency conflict: expected version {expected_version}, "
                f"found version {existing.version}"
            )

        new_version = existing.version + 1
        stmt = (
            update(TenantWatchlistModel)
            .where(
                TenantWatchlistModel.id == watchlist.id,
                TenantWatchlistModel.tenant_id == watchlist.tenant_id,
                TenantWatchlistModel.version == existing.version,
            )
            .values(
                name=watchlist.name,
                markets=list(watchlist.markets),
                version=new_version,
            )
        )
        res = await self._session.execute(stmt)
        cursor = cast(CursorResult[Any], res)
        if cursor.rowcount == 0:
            raise InvariantViolationError(
                "Concurrent modification conflict during watchlist update"
            )
        await self._session.flush()
        return TenantWatchlist(
            id=watchlist.id,
            tenant_id=watchlist.tenant_id,
            name=watchlist.name,
            markets=tuple(watchlist.markets),
            version=new_version,
            updated_at=datetime.now(tz=UTC),
        )

    async def delete(self, tenant_id: uuid.UUID, watchlist_id: uuid.UUID) -> bool:
        """Delete watchlist scoped to tenant."""
        await set_session_tenant(self._session, tenant_id)
        stmt = delete(TenantWatchlistModel).where(
            TenantWatchlistModel.tenant_id == tenant_id,
            TenantWatchlistModel.id == watchlist_id,
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        cursor = cast(CursorResult[Any], res)
        return bool(cursor.rowcount > 0)


class PostgresAlertRuleRepository:
    """PostgreSQL alert rule repository executing within tenant RLS boundary."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[TenantAlertRule]:
        """Fetch all alert rules owned by tenant under RLS."""
        await set_session_tenant(self._session, tenant_id)
        stmt = (
            select(TenantAlertRuleModel)
            .where(TenantAlertRuleModel.tenant_id == tenant_id)
            .order_by(TenantAlertRuleModel.id)
        )
        result = await self._session.execute(stmt)
        items: list[TenantAlertRule] = []
        for row in result.scalars():
            spec = row.specification or {}
            items.append(
                TenantAlertRule(
                    id=row.id,
                    tenant_id=row.tenant_id,
                    name=spec.get("name", ""),
                    markets=tuple(spec.get("markets", ())),
                    horizons=tuple(spec.get("horizons", ())),
                    labels=tuple(spec.get("labels", ())),
                    min_confidence=float(spec.get("min_confidence", 0.60)),
                    cooldown_seconds=int(spec.get("cooldown_seconds", 900)),
                    target_channels=tuple(spec.get("target_channels", ("webhook",))),
                    destination=spec.get("destination", ""),
                    destination_status=spec.get("destination_status", "PENDING_VERIFICATION"),
                    is_active=bool(row.enabled),
                    consent_at=row.consent_at,
                    version=int(row.version),
                    created_at=datetime.fromisoformat(spec["created_at"])
                    if "created_at" in spec
                    else None,
                    updated_at=datetime.fromisoformat(spec["updated_at"])
                    if "updated_at" in spec
                    else None,
                )
            )
        return items

    async def get_by_id(
        self, tenant_id: uuid.UUID, rule_id: uuid.UUID
    ) -> TenantAlertRule | None:
        """Fetch alert rule by ID scoped to tenant."""
        await set_session_tenant(self._session, tenant_id)
        stmt = select(TenantAlertRuleModel).where(
            TenantAlertRuleModel.tenant_id == tenant_id,
            TenantAlertRuleModel.id == rule_id,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        spec = row.specification or {}
        return TenantAlertRule(
            id=row.id,
            tenant_id=row.tenant_id,
            name=spec.get("name", ""),
            markets=tuple(spec.get("markets", ())),
            horizons=tuple(spec.get("horizons", ())),
            labels=tuple(spec.get("labels", ())),
            min_confidence=float(spec.get("min_confidence", 0.60)),
            cooldown_seconds=int(spec.get("cooldown_seconds", 900)),
            target_channels=tuple(spec.get("target_channels", ("webhook",))),
            destination=spec.get("destination", ""),
            destination_status=spec.get("destination_status", "PENDING_VERIFICATION"),
            is_active=bool(row.enabled),
            consent_at=row.consent_at,
            version=int(row.version),
            created_at=datetime.fromisoformat(spec["created_at"])
            if "created_at" in spec
            else None,
            updated_at=datetime.fromisoformat(spec["updated_at"])
            if "updated_at" in spec
            else None,
        )

    async def save(
        self, rule: TenantAlertRule, expected_version: int | None = None
    ) -> TenantAlertRule:
        """Create or update alert rule with optimistic concurrency check."""
        await set_session_tenant(self._session, rule.tenant_id)
        existing = await self.get_by_id(rule.tenant_id, rule.id)
        now = datetime.now(tz=UTC)
        created_at = rule.created_at or now

        spec = {
            "name": rule.name,
            "markets": list(rule.markets),
            "horizons": list(rule.horizons),
            "labels": list(rule.labels),
            "min_confidence": rule.min_confidence,
            "cooldown_seconds": rule.cooldown_seconds,
            "target_channels": list(rule.target_channels),
            "destination": rule.destination,
            "destination_status": rule.destination_status,
            "created_at": created_at.isoformat(),
            "updated_at": now.isoformat(),
        }

        if existing is None:
            consent_at = rule.consent_at or now
            row = TenantAlertRuleModel(
                id=rule.id,
                tenant_id=rule.tenant_id,
                specification=spec,
                consent_at=consent_at,
                enabled=rule.is_active,
                version=1,
            )
            self._session.add(row)
            await self._session.flush()
            return TenantAlertRule(
                id=row.id,
                tenant_id=row.tenant_id,
                name=rule.name,
                markets=rule.markets,
                horizons=rule.horizons,
                labels=rule.labels,
                min_confidence=rule.min_confidence,
                cooldown_seconds=rule.cooldown_seconds,
                target_channels=rule.target_channels,
                destination=rule.destination,
                destination_status=rule.destination_status,
                is_active=rule.is_active,
                consent_at=consent_at,
                version=1,
                created_at=created_at,
                updated_at=now,
            )

        if expected_version is not None and existing.version != expected_version:
            raise InvariantViolationError(
                f"Optimistic concurrency conflict: expected version {expected_version}, "
                f"found version {existing.version}"
            )

        new_version = existing.version + 1
        stmt = (
            update(TenantAlertRuleModel)
            .where(
                TenantAlertRuleModel.id == rule.id,
                TenantAlertRuleModel.tenant_id == rule.tenant_id,
                TenantAlertRuleModel.version == existing.version,
            )
            .values(
                specification=spec,
                enabled=rule.is_active,
                version=new_version,
            )
        )
        res = await self._session.execute(stmt)
        cursor = cast(CursorResult[Any], res)
        if cursor.rowcount == 0:
            raise InvariantViolationError(
                "Concurrent modification conflict during alert rule update"
            )
        await self._session.flush()

        return TenantAlertRule(
            id=rule.id,
            tenant_id=rule.tenant_id,
            name=rule.name,
            markets=rule.markets,
            horizons=rule.horizons,
            labels=rule.labels,
            min_confidence=rule.min_confidence,
            cooldown_seconds=rule.cooldown_seconds,
            target_channels=rule.target_channels,
            destination=rule.destination,
            destination_status=rule.destination_status,
            is_active=rule.is_active,
            consent_at=rule.consent_at or existing.consent_at,
            version=new_version,
            created_at=existing.created_at or created_at,
            updated_at=now,
        )

    async def delete(self, tenant_id: uuid.UUID, rule_id: uuid.UUID) -> bool:
        """Delete alert rule scoped to tenant."""
        await set_session_tenant(self._session, tenant_id)
        stmt = delete(TenantAlertRuleModel).where(
            TenantAlertRuleModel.tenant_id == tenant_id,
            TenantAlertRuleModel.id == rule_id,
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        cursor = cast(CursorResult[Any], res)
        return bool(cursor.rowcount > 0)
