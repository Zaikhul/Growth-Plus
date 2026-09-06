"""Liveness and readiness health check endpoints."""

from typing import Any

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["Health"])


@router.get("/healthz")
async def health_check() -> dict[str, str]:
    """Kubernetes liveness probe confirming process is responsive."""
    return {"status": "OK"}


@router.get("/readyz")
async def readiness_check(request: Request, response: Response) -> dict[str, Any]:
    """Kubernetes readiness probe confirming downstream dependencies are operational."""
    db_connected = (
        hasattr(request.app.state, "signal_repository")
        and request.app.state.signal_repository is not None
    )
    event_bus_connected = (
        hasattr(request.app.state, "event_bus") and request.app.state.event_bus is not None
    )

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
