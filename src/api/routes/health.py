"""Liveness and readiness health check endpoints."""

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["Health"])


@router.get("/healthz")
@router.get("/health")
@router.get("/health/liveness")
async def health_check() -> dict[str, str]:
    """Kubernetes liveness probe confirming process is responsive."""
    return {"status": "OK"}


@router.get("/readyz")
@router.get("/health/readiness")
@router.get("/health/ready")
async def readiness_check(request: Request, response: Response) -> dict[str, Any]:
    """Kubernetes readiness probe confirming downstream dependencies are operational."""
    db_connected = False
    event_bus_connected = False

    # 1. Active Database Ping (2.0-second timeout)
    repo = getattr(request.app.state, "signal_repository", None)
    if repo is not None:
        try:
            async def _check_db() -> bool:
                if hasattr(repo, "ping"):
                    res = repo.ping()
                    if asyncio.iscoroutine(res):
                        await res
                    return True
                elif hasattr(repo, "_session"):
                    from sqlalchemy import text
                    await repo._session.execute(text("SELECT 1"))
                    return True
                return True

            await asyncio.wait_for(_check_db(), timeout=2.0)
            db_connected = True
        except Exception:
            db_connected = False

    # 2. Active EventBus Ping (2.0-second timeout)
    bus = getattr(request.app.state, "event_bus", None)
    if bus is not None:
        try:
            async def _check_bus() -> bool:
                if hasattr(bus, "ping"):
                    res = bus.ping()
                    if asyncio.iscoroutine(res):
                        await res
                    return True
                elif hasattr(bus, "_nc") and bus._nc is not None:
                    return bool(bus._nc.is_connected)
                return True

            await asyncio.wait_for(_check_bus(), timeout=2.0)
            event_bus_connected = True
        except Exception:
            event_bus_connected = False

    if not db_connected or not event_bus_connected:
        response.status_code = 503
        return {
            "status": "UNHEALTHY",
            "database": "CONNECTED" if db_connected else "DISCONNECTED",
            "event_bus": "CONNECTED" if event_bus_connected else "DISCONNECTED",
        }

    return {
        "status": "READY",
        "database": "CONNECTED",
        "event_bus": "CONNECTED",
    }
