"""Server-Sent Events (SSE) streaming endpoint.

Enforces PRD Section 4.5:
- Streaming signal updates via text/event-stream
- 15-second heartbeat ping
- Graceful client disconnect handling
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_signal_repository
from src.api.schemas import SignalResponse
from src.domain.identity import HorizonId, MarketId
from src.ports.repositories import SignalRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/stream", tags=["Streaming"])


async def signal_event_generator(
    request: Request,
    repo: SignalRepository,
    market_id: MarketId = MarketId("binance:BTCUSDT"),
    horizon_id: HorizonId = HorizonId.SWING_24H,
    heartbeat_interval_seconds: float = 15.0,
    max_events: int | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for newly published signals with durable cursor and keep-alive pings."""
    last_signal_id: str | None = None
    events_emitted = 0

    try:
        while True:
            # Check latest signal for requested market and horizon (DEFECT-39)
            try:
                latest = await repo.get_latest_signal(
                    market_id=market_id,
                    horizon=horizon_id,
                )
                if latest and str(latest.signal_id) != last_signal_id:
                    last_signal_id = str(latest.signal_id)
                    data_dict = SignalResponse.from_domain(latest).model_dump(mode="json")
                    cursor = f"cursor-{latest.sequence}"
                    payload = json.dumps(data_dict)
                    yield f"id: {cursor}\nevent: signal.published\ndata: {payload}\n\n"
                    events_emitted += 1
                    if max_events is not None and events_emitted >= max_events:
                        break
            except Exception as exc:
                logger.warning("Error querying latest signal in SSE stream: %s", exc)

            # Send keepalive ping comment
            yield ": ping\n\n"
            events_emitted += 1
            if max_events is not None and events_emitted >= max_events:
                break

            if await request.is_disconnected():
                break

            # Sleep with disconnect cancellation check
            await asyncio.sleep(heartbeat_interval_seconds)
            if await request.is_disconnected():
                break
    except asyncio.CancelledError:
        raise


@router.get("/signals")
async def stream_signals(
    request: Request,
    market: str = Query("binance:BTCUSDT", description="Standard market identifier"),
    horizon: str = Query("swing_24h", description="Forecast horizon"),
    repo: SignalRepository = Depends(get_signal_repository),
    max_events: int | None = None,
) -> StreamingResponse:
    """Stream real-time signal publications and heartbeats via Server-Sent Events."""
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

    return StreamingResponse(
        signal_event_generator(
            request=request,
            repo=repo,
            market_id=m_id,
            horizon_id=h_id,
            max_events=max_events,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
