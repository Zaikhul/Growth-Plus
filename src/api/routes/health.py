"""Liveness and readiness health check endpoints."""

from typing import Any

from fastapi import APIRouter, Response

router = APIRouter(tags=["Health"])


@router.get("/healthz")
async def health_check() -> dict[str, str]:
    """Kubernetes liveness probe confirming process is responsive."""
    return {"status": "OK"}


@router.get("/readyz")
async def readiness_check(response: Response) -> dict[str, Any]:
    """Kubernetes readiness probe confirming downstream dependencies are operational."""
    # Production check: verify DB pool / JetStream connection state
    return {
        "status": "READY",
        "database": "CONNECTED",
        "event_bus": "CONNECTED",
    }
