"""Market watchlist management endpoints."""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import VerifiedTenant, get_verified_tenant

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


# In-memory storage by tenant for non-persistent mode, or fallback repository
WATCHLIST_STORE: dict[str, dict[str, Any]] = {}


@router.get("", response_model=list[WatchlistDTO])
async def list_watchlists(
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> list[WatchlistDTO]:
    """Retrieve all watchlists owned by the authenticated tenant."""
    tenant_str = str(tenant.tenant_id)
    items = [
        WatchlistDTO(**data)
        for data in WATCHLIST_STORE.values()
        if data.get("tenant_id") == tenant_str
    ]
    if not items:
        # Provide default institutional watchlist if none created yet
        default_wl = WatchlistDTO(
            id=f"wl-{tenant_str[:8]}",
            name="Core Institutional Spot",
            markets=["binance:BTCUSDT", "binance:ETHUSDT", "coinbase:BTC-USD"],
            updated_at=datetime.now(tz=UTC),
        )
        return [default_wl]
    return items


@router.post("", response_model=WatchlistDTO, status_code=status.HTTP_201_CREATED)
async def create_or_update_watchlist(
    payload: WatchlistCreateRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> WatchlistDTO:
    """Create or overwrite a tenant market watchlist (maximum 20 markets)."""
    if len(payload.markets) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 20 markets per watchlist reached (PRD §4.10)",
        )

    wl_id = payload.id or f"wl-{uuid.uuid4()}"
    now = datetime.now(tz=UTC)
    record = {
        "id": wl_id,
        "tenant_id": str(tenant.tenant_id),
        "name": payload.name,
        "markets": payload.markets,
        "updated_at": now,
    }
    WATCHLIST_STORE[wl_id] = record
    return WatchlistDTO(
        id=wl_id,
        name=payload.name,
        markets=payload.markets,
        updated_at=now,
    )


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watchlist(
    watchlist_id: str,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> None:
    """Delete a watchlist owned by the authenticated tenant."""
    record = WATCHLIST_STORE.get(watchlist_id)
    if record is None or record.get("tenant_id") != str(tenant.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watchlist '{watchlist_id}' not found",
        )
    del WATCHLIST_STORE[watchlist_id]
