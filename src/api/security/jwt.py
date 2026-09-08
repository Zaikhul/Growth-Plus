"""Cryptographic authentication and JWT verification utilities.

Zero-external-dependency implementation of HS256 JWT RFC 7519 standard.
Enforces GAP-02: Cryptographically verified identity and tenant authorization.
"""

import base64
import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from typing import Any


def _b64_url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_url_decode(segment: str) -> bytes:
    padding = "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode((segment + padding).encode("ascii"))


def create_access_token(
    tenant_id: uuid.UUID | str,
    user_id: str | None = None,
    sub: str | None = None,
    roles: list[str] | tuple[str, ...] = ("subscriber",),
    secret: str | None = None,
    expires_in_seconds: int = 3600,
    issuer: str = "growthplus.ai",
    audience: str = "growthplus-api",
) -> str:
    """Issue a cryptographically signed HS256 JWT for a verified tenant."""
    if secret is None:
        try:
            from src.config.settings import get_settings
            jwt_sec = get_settings().security.jwt_secret
            secret = (
                jwt_sec.get_secret_value()
                if jwt_sec
                else "insecure-development-jwt-secret-replace-in-production-min-32-chars"
            )
        except Exception:
            secret = "insecure-development-jwt-secret-replace-in-production-min-32-chars"

    now = int(datetime.now(tz=UTC).timestamp())
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": sub or user_id or str(tenant_id),
        "tenant_id": str(tenant_id),
        "roles": list(roles),
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    header_b64 = _b64_url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64_url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_access_token(
    token: str,
    secret: str,
    expected_issuer: str | None = None,
    expected_audience: str | None = None,
) -> dict[str, Any]:
    """Verify and decode an HS256 JWT token.

    Raises:
        ValueError: If token format, signature, expiration, or claims are invalid.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT token: must contain exactly three segments")

    header_b64, payload_b64, sig_b64 = parts

    # 1. Verify header algorithm
    try:
        header = json.loads(_b64_url_decode(header_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid JWT header encoding") from exc

    if header.get("alg") != "HS256":
        raise ValueError(f"Unsupported algorithm '{header.get('alg')}'; only HS256 is permitted")

    # 2. Cryptographic signature check (constant-time)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64_url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Cryptographic signature verification failed")

    # 3. Decode payload
    try:
        payload = json.loads(_b64_url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid JWT payload encoding") from exc

    # 4. Expiration check
    now = int(datetime.now(tz=UTC).timestamp())
    exp = payload.get("exp")
    if exp is None or not isinstance(exp, (int, float)):
        raise ValueError("Missing or invalid 'exp' expiration claim")
    if now > exp:
        raise ValueError("Token has expired")

    # 5. Issued-at check (allow 60s clock skew)
    iat = payload.get("iat")
    if iat is not None and isinstance(iat, (int, float)):
        if iat > now + 60:
            raise ValueError("Token issued in the future")

    # 6. Issuer and Audience checks
    if expected_issuer and payload.get("iss") != expected_issuer:
        raise ValueError(f"Invalid issuer '{payload.get('iss')}'")

    if expected_audience and payload.get("aud") != expected_audience:
        raise ValueError(f"Invalid audience '{payload.get('aud')}'")

    # 7. Tenant claim check
    tenant_str = payload.get("tenant_id")
    if not tenant_str:
        tenant_str = payload.get("sub")
    if not tenant_str:
        raise ValueError("Missing 'tenant_id' claim in token")

    try:
        uuid.UUID(str(tenant_str))
    except ValueError as exc:
        raise ValueError(f"Invalid UUID in tenant claim: '{tenant_str}'") from exc

    return payload
