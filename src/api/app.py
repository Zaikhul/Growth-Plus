"""FastAPI application factory, lifespan management, and RFC 7807 error handlers.

Enforces PRD Section 4.1, 4.2 & 4.5:
- Clean modular FastAPI application
- Asynchronous repository and event bus lifecycle
- Structured RFC 7807 Problem Details error responses
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.messaging.in_memory import InMemoryEventBus
from src.adapters.persistence.in_memory.repositories import (
    InMemoryBarRepository,
    InMemoryObservationRepository,
    InMemorySignalRepository,
)
from src.adapters.rights.config_authorizer import ConfigRightsAuthorizer
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import (
    alert_rules,
    health,
    notifications,
    signals,
    sources,
    stream,
    watchlists,
)
from src.api.security.cors import registered_origins
from src.config.settings import Settings
from src.domain.errors import (
    InvariantViolationError,
    PointInTimeViolationError,
    RightsViolationError,
    SecurityError,
    StaleDataError,
)
from src.ports.event_bus import EventBus
from src.ports.repositories import BarRepository, ObservationRepository, SignalRepository
from src.ports.rights_authorizer import RightsAuthorizer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize persistence, event bus, and policy engines on startup."""
    from src.config.settings import get_settings

    if not hasattr(app.state, "settings"):
        app.state.settings = get_settings()
    cfg = app.state.settings

    # Initialize default in-memory adapters for service delivery
    # In production, these are wired to PostgreSQL and NATS JetStream
    if not hasattr(app.state, "signal_repository"):
        app.state.signal_repository = InMemorySignalRepository()
    if not hasattr(app.state, "bar_repository"):
        app.state.bar_repository = InMemoryBarRepository()
    if not hasattr(app.state, "observation_repository"):
        app.state.observation_repository = InMemoryObservationRepository()
    if not hasattr(app.state, "event_bus"):
        app.state.event_bus = InMemoryEventBus()
    if not hasattr(app.state, "rights_authorizer"):
        app.state.rights_authorizer = ConfigRightsAuthorizer.from_yaml_file(cfg.app.config_path)

    yield

    # Clean shutdown
    if hasattr(app.state, "event_bus") and hasattr(app.state.event_bus, "close"):
        await app.state.event_bus.close()


def create_app(
    signal_repo: SignalRepository | None = None,
    bar_repo: BarRepository | None = None,
    obs_repo: ObservationRepository | None = None,
    event_bus: EventBus | None = None,
    rights_authorizer: RightsAuthorizer | None = None,
    settings: Settings | None = None,
    enable_rate_limiter: bool | None = None,
    rate_limit_capacity: int | None = None,
    rate_limit_refill_rate: float | None = None,
    allow_ephemeral_adapters: bool | None = None,
) -> FastAPI:
    """Create and configure a production FastAPI application instance."""
    from src.config.settings import get_settings

    cfg = settings or get_settings()

    ephemeral = (
        allow_ephemeral_adapters
        if allow_ephemeral_adapters is not None
        else cfg.app.allow_ephemeral_adapters
    )
    if not ephemeral:
        missing = []
        if signal_repo is None:
            missing.append("signal_repo")
        if bar_repo is None:
            missing.append("bar_repo")
        if obs_repo is None:
            missing.append("obs_repo")
        if event_bus is None:
            missing.append("event_bus")
        if missing:
            msg = (
                "Production mode forbids ephemeral in-memory adapters. Missing: "
                f"{', '.join(missing)}"
            )
            raise InvariantViolationError(msg)

    app = FastAPI(
        title="Growth+ Signal Intelligence API",
        version="1.0.0",
        description="High-integrity calibrated cryptocurrency signal intelligence backend",
        lifespan=lifespan,
    )

    # Dependency injection on application state
    app.state.settings = cfg
    app.state.signal_repository = signal_repo or InMemorySignalRepository()
    app.state.bar_repository = bar_repo or InMemoryBarRepository()
    app.state.observation_repository = obs_repo or InMemoryObservationRepository()
    app.state.event_bus = event_bus or InMemoryEventBus()
    app.state.rights_authorizer = rights_authorizer or ConfigRightsAuthorizer.from_yaml_file(
        cfg.app.config_path
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=registered_origins(cfg),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    use_limiter = (
        enable_rate_limiter if enable_rate_limiter is not None else cfg.app.rate_limit_enabled
    )
    if use_limiter:
        cap = (
            rate_limit_capacity if rate_limit_capacity is not None else cfg.app.rate_limit_capacity
        )
        refill = (
            rate_limit_refill_rate
            if rate_limit_refill_rate is not None
            else cfg.app.rate_limit_refill_rate
        )
        app.add_middleware(
            RateLimitMiddleware,
            capacity=cap,
            refill_rate_per_second=refill,
            max_clients=cfg.app.rate_limit_max_clients,
            trusted_proxy_cidrs=cfg.security.trusted_proxy_cidrs,
        )

    # Routes
    app.include_router(health.router)
    app.include_router(signals.router)
    app.include_router(sources.router)
    app.include_router(notifications.router)
    app.include_router(stream.router)
    app.include_router(alert_rules.router)
    app.include_router(watchlists.router)

    # RFC 7807 Error Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "type": "https://errors.growthplus.ai/http-error",
                "title": exc.detail or "HTTP Error",
                "status": exc.status_code,
                "detail": str(exc.detail),
                "instance": str(request.url.path),
            },
            headers=exc.headers,
        )

    @app.exception_handler(RightsViolationError)
    async def rights_violation_handler(request: Request, exc: RightsViolationError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "type": "https://errors.growthplus.ai/rights-violation",
                "title": "Data Rights Violation",
                "status": 403,
                "detail": exc.message,
                "instance": str(request.url.path),
            },
        )

    @app.exception_handler(SecurityError)
    async def security_error_handler(request: Request, exc: SecurityError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "type": "https://errors.growthplus.ai/security-violation",
                "title": "Security Violation",
                "status": 403,
                "detail": exc.message,
                "instance": str(request.url.path),
            },
        )

    @app.exception_handler(PointInTimeViolationError)
    async def pit_violation_handler(
        request: Request, exc: PointInTimeViolationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "type": "https://errors.growthplus.ai/point-in-time-violation",
                "title": "Point In Time Violation",
                "status": 400,
                "detail": exc.message,
                "instance": str(request.url.path),
            },
        )

    @app.exception_handler(StaleDataError)
    async def stale_data_handler(request: Request, exc: StaleDataError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://errors.growthplus.ai/stale-data",
                "title": "Stale Data",
                "status": 422,
                "detail": exc.message,
                "instance": str(request.url.path),
            },
        )

    @app.exception_handler(InvariantViolationError)
    async def invariant_violation_handler(
        request: Request, exc: InvariantViolationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://errors.growthplus.ai/invariant-violation",
                "title": "Domain Invariant Violation",
                "status": 422,
                "detail": exc.message,
                "instance": str(request.url.path),
            },
        )

    return app
