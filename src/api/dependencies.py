"""FastAPI dependency injection providers.

Enforces clean hexagonal boundary between API transport and application services.
"""

from typing import cast

from fastapi import Request

from src.ports.event_bus import EventBus
from src.ports.repositories import SignalRepository
from src.ports.rights_authorizer import RightsAuthorizer


def get_signal_repository(request: Request) -> SignalRepository:
    """Retrieve injected SignalRepository from application state."""
    return cast(SignalRepository, request.app.state.signal_repository)


def get_rights_authorizer(request: Request) -> RightsAuthorizer:
    """Retrieve injected RightsAuthorizer from application state."""
    return cast(RightsAuthorizer, request.app.state.rights_authorizer)


def get_event_bus(request: Request) -> EventBus:
    """Retrieve injected EventBus from application state."""
    return cast(EventBus, request.app.state.event_bus)
