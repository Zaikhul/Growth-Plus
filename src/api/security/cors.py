"""Validated registered browser origins; no vendor or model dependencies."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from src.config.settings import Settings


def validate_origin(value: str) -> str:
    """Validate and normalize a single registered origin."""
    url = urlsplit(value)
    scheme = url.scheme.lower()
    host = url.hostname.lower() if url.hostname else ""
    if (
        scheme != "https"
        or not host
        or url.username is not None
        or url.password is not None
        or url.path
        or url.query
        or url.fragment
        or "*" in value
        or any(ord(c) <= 32 for c in value)
    ):
        raise ValueError("CORS requires exact registered HTTPS origins")
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ValueError("Invalid CORS port")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port_suffix = f":{url.port}" if url.port is not None else ""
    return f"{scheme}://{host}{port_suffix}"


def registered_origins(settings: Settings | None = None) -> list[str]:
    """Retrieve and validate registered browser origins from settings or environment."""
    if settings is not None:
        return list(settings.security.cors_origins)

    if "GROWTH_CORS_ORIGINS_JSON" in os.environ:
        raw = json.loads(os.environ["GROWTH_CORS_ORIGINS_JSON"])
        if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
            raise ValueError("GROWTH_CORS_ORIGINS_JSON must be a JSON string array")
        origins: list[str] = []
        for val in raw:
            normalized = validate_origin(val)
            if normalized not in origins:
                origins.append(normalized)
        return origins

    from src.config.settings import get_settings

    return list(get_settings().security.cors_origins)
