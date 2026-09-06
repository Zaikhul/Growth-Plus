"""Validated registered browser origins; no vendor or model dependencies."""

import json
import os
from urllib.parse import urlsplit


def registered_origins() -> list[str]:
    raw = json.loads(os.environ.get("GROWTH_CORS_ORIGINS_JSON", "[]"))
    if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
        raise ValueError("GROWTH_CORS_ORIGINS_JSON must be a JSON string array")
    origins: list[str] = []
    for value in raw:
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
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
        if value not in origins:
            origins.append(value)
    return origins
