"""Signal delivery REST endpoints.

Enforces PRD Section 4.5 & 4.6:
- Latest signal delivery with ETags and cache validation
- Historical signal queries by ID and paginated search
- Strict RFC 7807 problem details on errors
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response

from src.api.dependencies import get_signal_repository
from src.api.schemas import SignalListResponse, SignalResponse
from src.domain.identity import HorizonId, MarketId
from src.domain.signals import SignalStatus
from src.ports.repositories import SignalRepository

router = APIRouter(prefix="/v1/signals", tags=["Signals"])


@router.get("/latest", response_model=SignalResponse)
async def get_latest_signal(
    request: Request,
    response: Response,
    market: str = Query("binance:BTCUSDT", description="Standard market identifier"),
    horizon: str = Query("swing_24h", description="Forecast horizon"),
    if_none_match: str | None = Header(None, alias="If-None-Match"),
    repo: SignalRepository = Depends(get_signal_repository),
) -> Any:
    """Retrieve the latest valid directional signal for a given market and horizon."""
    try:
        m_id = MarketId(market)
        h_id = HorizonId(horizon)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "https://errors.growthplus.ai/invalid-parameters",
                "title": "Invalid Parameters",
                "status": 400,
                "detail": str(err),
            },
        ) from err

    signal = await repo.get_latest_signal(m_id, h_id)
    if not signal:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "https://errors.growthplus.ai/signal-not-found",
                "title": "Signal Not Found",
                "status": 404,
                "detail": f"No signal available for market '{market}' and horizon '{horizon}'",
            },
        )

    # Compute ETag from immutable record attributes
    etag = f'"{signal.signal_id}-{signal.sequence}"'

    # Check for conditional cache match (HTTP 304)
    if if_none_match and if_none_match.strip() == etag:
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": "public, max-age=5"},
        )

    # Check expiration relative to tradable clock
    now = datetime.now(tz=UTC)
    if signal.is_expired_at(now) and signal.status == SignalStatus.READY:
        # Note: expired signals remain historically readable, but status is EXPIRED
        object.__setattr__(signal, "status", SignalStatus.EXPIRED)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "public, max-age=5, must-revalidate"
    return SignalResponse.from_domain(signal)


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal_by_id(
    signal_id: uuid.UUID,
    repo: SignalRepository = Depends(get_signal_repository),
) -> Any:
    """Fetch an immutable historical signal by its unique identifier."""
    signal = await repo.get_signal(signal_id)
    if not signal:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "https://errors.growthplus.ai/signal-not-found",
                "title": "Signal Not Found",
                "status": 404,
                "detail": f"Signal '{signal_id}' was not found",
            },
        )
    return SignalResponse.from_domain(signal)


@router.get("", response_model=SignalListResponse)
async def list_signals(
    market: str = Query("binance:BTCUSDT", description="Market filter"),
    horizon: str = Query("swing_24h", description="Horizon filter"),
    limit: int = Query(50, ge=1, le=200),
    repo: SignalRepository = Depends(get_signal_repository),
) -> SignalListResponse:
    """Query historical signals with pagination controls."""
    try:
        m_id = MarketId(market)
        h_id = HorizonId(horizon)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "https://errors.growthplus.ai/invalid-parameters",
                "title": "Invalid Parameters",
                "status": 400,
                "detail": str(err),
            },
        ) from err

    items = await repo.list_signals(market_id=m_id, horizon=h_id, limit=limit)
    responses = [SignalResponse.from_domain(s) for s in items]

    return SignalListResponse(
        items=responses,
        total=len(responses),
        limit=limit,
        offset=0,
    )
