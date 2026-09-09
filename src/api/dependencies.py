import uuid
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status

from src.api.security import verify_access_token
from src.config.settings import get_settings
from src.ports.event_bus import EventBus
from src.ports.repositories import (
    AlertRuleRepository,
    BarRepository,
    ObservationRepository,
    SignalRepository,
    WatchlistRepository,
)
from src.ports.rights_authorizer import RightsAuthorizer


@dataclass(frozen=True, slots=True)
class VerifiedTenant:
    """Authenticated and authorized tenant principal identity."""

    tenant_id: uuid.UUID
    roles: tuple[str, ...] = ("subscriber",)


async def get_verified_tenant(request: Request) -> VerifiedTenant:
    """Extract and cryptographically verify authenticated tenant identity.

    Expects an Authorization header with Bearer credentials.
    """
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
    if not jwt_sec:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication subsystem not configured: missing JWT secret",
        )
    secret = jwt_sec.get_secret_value()

    try:
        claims = verify_access_token(
            token,
            secret,
            expected_issuer="growthplus.ai",
            expected_audience="growthplus-api",
        )
        tenant_uuid = uuid.UUID(str(claims.get("tenant_id") or claims.get("sub")))
        roles = tuple(claims.get("roles", ["subscriber"]))
        return VerifiedTenant(tenant_id=tenant_uuid, roles=roles)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials",
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


def get_watchlist_repository(request: Request) -> WatchlistRepository:
    """Retrieve injected WatchlistRepository from application state."""
    return cast(WatchlistRepository, request.app.state.watchlist_repository)


def get_alert_rule_repository(request: Request) -> AlertRuleRepository:
    """Retrieve injected AlertRuleRepository from application state."""
    return cast(AlertRuleRepository, request.app.state.alert_rule_repository)


def get_bar_repository(request: Request) -> BarRepository:
    """Retrieve injected BarRepository from application state."""
    return cast(BarRepository, request.app.state.bar_repository)


def get_observation_repository(request: Request) -> ObservationRepository:
    """Retrieve injected ObservationRepository from application state."""
    return cast(ObservationRepository, request.app.state.observation_repository)

