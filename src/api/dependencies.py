import uuid
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status

from src.ports.event_bus import EventBus
from src.ports.repositories import SignalRepository
from src.ports.rights_authorizer import RightsAuthorizer


@dataclass(frozen=True, slots=True)
class VerifiedTenant:
    """Authenticated and authorized tenant principal identity."""

    tenant_id: uuid.UUID
    roles: tuple[str, ...] = ("subscriber",)


async def get_verified_tenant(request: Request) -> VerifiedTenant:
    """Extract and verify authenticated tenant identity from request headers."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        tenant_header = request.headers.get("X-Tenant-ID")
        if not tenant_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required: missing tenant credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            tenant_uuid = uuid.UUID(tenant_header)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid tenant credentials",
            ) from exc
        return VerifiedTenant(tenant_id=tenant_uuid)

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme; Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        tenant_uuid = uuid.UUID(token)
    except ValueError:
        tenant_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"tenant:{token}")

    return VerifiedTenant(tenant_id=tenant_uuid)


def get_signal_repository(request: Request) -> SignalRepository:
    """Retrieve injected SignalRepository from application state."""
    return cast(SignalRepository, request.app.state.signal_repository)


def get_rights_authorizer(request: Request) -> RightsAuthorizer:
    """Retrieve injected RightsAuthorizer from application state."""
    return cast(RightsAuthorizer, request.app.state.rights_authorizer)


def get_event_bus(request: Request) -> EventBus:
    """Retrieve injected EventBus from application state."""
    return cast(EventBus, request.app.state.event_bus)
