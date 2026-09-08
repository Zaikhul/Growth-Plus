import uuid
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status

from src.ports.event_bus import EventBus
from src.ports.repositories import SignalRepository
from src.ports.rights_authorizer import RightsAuthorizer


from src.api.security import verify_access_token
from src.config.settings import get_settings


@dataclass(frozen=True, slots=True)
class VerifiedTenant:
    """Authenticated and authorized tenant principal identity."""

    tenant_id: uuid.UUID
    roles: tuple[str, ...] = ("subscriber",)


async def get_verified_tenant(request: Request) -> VerifiedTenant:
    """Extract and cryptographically verify authenticated tenant identity from Authorization Bearer token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: missing Bearer credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme; Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()
    jwt_sec = settings.security.jwt_secret
    secret = (
        jwt_sec.get_secret_value()
        if jwt_sec
        else "insecure-development-jwt-secret-replace-in-production-min-32-chars"
    )

    try:
        claims = verify_access_token(token, secret)
        tenant_uuid = uuid.UUID(str(claims.get("tenant_id") or claims.get("sub")))
        roles = tuple(claims.get("roles", ["subscriber"]))
        return VerifiedTenant(tenant_id=tenant_uuid, roles=roles)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_signal_repository(request: Request) -> SignalRepository:
    """Retrieve injected SignalRepository from application state."""
    return cast(SignalRepository, request.app.state.signal_repository)


def get_rights_authorizer(request: Request) -> RightsAuthorizer:
    """Retrieve injected RightsAuthorizer from application state."""
    return cast(RightsAuthorizer, request.app.state.rights_authorizer)


def get_event_bus(request: Request) -> EventBus:
    """Retrieve injected EventBus from application state."""
    return cast(EventBus, request.app.state.event_bus)
