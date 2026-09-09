"""Market watchlist management endpoints."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import (
    VerifiedTenant,
    get_verified_tenant,
    get_watchlist_repository,
)
from src.ports.repositories import TenantWatchlist, WatchlistRepository

router = APIRouter(prefix="/v1/watchlists", tags=["Watchlists"])


class WatchlistDTO(BaseModel):
    id: str
    name: str = Field(..., min_length=1, max_length=128)
    markets: list[str] = Field(default_factory=list, max_length=20)
    updated_at: datetime


class WatchlistCreateRequest(BaseModel):
    id: str | None = None
    name: str = Field(..., min_length=1, max_length=128)
    markets: list[str] = Field(default_factory=list, max_length=20)


@router.get("", response_model=list[WatchlistDTO])
async def list_watchlists(
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> list[WatchlistDTO]:
    """Retrieve all watchlists owned by the authenticated tenant."""
    items = await repo.list_by_tenant(tenant.tenant_id)
    if not items:
        # Provide default institutional watchlist if none created yet
        now = datetime.now(tz=UTC)
        default_id = uuid.uuid5(tenant.tenant_id, "core-institutional-spot")
        default_wl = TenantWatchlist(
            id=default_id,
            tenant_id=tenant.tenant_id,
            name="Core Institutional Spot",
            markets=["binance:BTCUSDT", "binance:ETHUSDT", "coinbase:BTC-USD"],
            version=1,
            updated_at=now,
        )
        await repo.save(default_wl)
        return [
            WatchlistDTO(
                id=str(default_wl.id),
                name=default_wl.name,
                markets=list(default_wl.markets),
                updated_at=now,
            )
        ]

    return [
        WatchlistDTO(
            id=str(item.id),
            name=item.name,
            markets=list(item.markets),
            updated_at=item.updated_at or datetime.now(tz=UTC),
        )
        for item in items
    ]


@router.post("", response_model=WatchlistDTO, status_code=status.HTTP_201_CREATED)
async def create_or_update_watchlist(
    payload: WatchlistCreateRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> WatchlistDTO:
    """Create or overwrite a tenant market watchlist (maximum 20 markets)."""
    if len(payload.markets) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 20 markets per watchlist reached (PRD §4.10)",
        )

    wl_id: uuid.UUID
    if payload.id:
        try:
            wl_id = uuid.UUID(payload.id)
        except ValueError:
            wl_id = uuid.uuid5(tenant.tenant_id, payload.id)
    else:
        wl_id = uuid.uuid4()

    now = datetime.now(tz=UTC)
    entity = TenantWatchlist(
        id=wl_id,
        tenant_id=tenant.tenant_id,
        name=payload.name,
        markets=payload.markets,
        version=1,
        updated_at=now,
    )
    saved = await repo.save(entity)

    return WatchlistDTO(
        id=str(saved.id),
        name=saved.name,
        markets=list(saved.markets),
        updated_at=saved.updated_at or now,
    )


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watchlist(
    watchlist_id: str,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> None:
    """Delete a watchlist owned by the authenticated tenant."""
    try:
        wl_uuid = uuid.UUID(watchlist_id)
    except ValueError:
        wl_uuid = uuid.uuid5(tenant.tenant_id, watchlist_id)

    deleted = await repo.delete(tenant.tenant_id, wl_uuid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watchlist '{watchlist_id}' not found",
        )
