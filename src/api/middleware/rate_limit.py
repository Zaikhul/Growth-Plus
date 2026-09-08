"""Bounded IP abuse middleware; one event loop owns each middleware instance."""

import json
import math
import os
import time
from collections.abc import Awaitable, Callable, Sequence

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.security.client_limits import BoundedBuckets, client_identity


class TokenBucket:
    """Compatibility wrapper for callers/tests using an individual bucket."""

    def __init__(self, capacity: int, refill_rate_per_second: float) -> None:
        self._store = BoundedBuckets(capacity, refill_rate_per_second, max_clients=1)

    def consume(self, amount: float = 1.0) -> tuple[bool, int, float]:
        if amount != 1:
            raise ValueError("HTTP requests consume one token")
        return self._store.consume("client", time.monotonic())


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        capacity: int | None = None,
        refill_rate_per_second: float | None = None,
        max_clients: int | None = None,
        trusted_proxy_cidrs: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)
        from src.config.settings import get_settings

        cfg = get_settings()
        self._capacity = capacity if capacity is not None else cfg.app.rate_limit_capacity
        refill = (
            refill_rate_per_second
            if refill_rate_per_second is not None
            else cfg.app.rate_limit_refill_rate
        )
        clients = max_clients if max_clients is not None else cfg.app.rate_limit_max_clients

        self._store = BoundedBuckets(self._capacity, refill, clients)

        if trusted_proxy_cidrs is not None:
            self._trusted = tuple(trusted_proxy_cidrs)
        elif "GROWTH_TRUSTED_PROXY_CIDRS_JSON" in os.environ:
            raw = json.loads(os.environ["GROWTH_TRUSTED_PROXY_CIDRS_JSON"])
            if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
                raise ValueError("Trusted proxy networks must be a JSON string array")
            self._trusted = tuple(raw)
        else:
            self._trusted = tuple(cfg.security.trusted_proxy_cidrs)

        # Validate configured networks even before a proxied request arrives.
        import ipaddress

        for cidr in self._trusted:
            ipaddress.ip_network(cidr, strict=True)

    def _get_client_key(self, request: Request) -> str:
        return client_identity(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            self._trusted,
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)
        try:
            key = self._get_client_key(request)
        except ValueError:
            return JSONResponse(status_code=400, content={"code": "INVALID_PROXY_CHAIN"})
        # No await between state reads/writes: atomic within the owning event loop.
        allowed, remaining, retry = self._store.consume(key, time.monotonic())
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"code": "RATE_LIMIT_EXCEEDED"},
                headers={
                    "Retry-After": str(max(1, math.ceil(retry))),
                    "X-RateLimit-Limit": str(self._capacity),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._capacity)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
