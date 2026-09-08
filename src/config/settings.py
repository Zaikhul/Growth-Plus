"""Centralized configuration management for the Growth+ platform.

Leverages Pydantic v2 and python-dotenv to provide strict typing, automatic environment variable
resolution, JSON parsing, secret masking, and fail-fast validation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

# Automatically locate and load .env from project root if present
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.is_file():
    dotenv.load_dotenv(dotenv_path=_ENV_FILE, override=False)


class BaseSettings(BaseModel):
    """Base model for configuration sections with automatic environment variable extraction."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    _env_prefix: str = "GROWTH_"

    def __init__(self, **values: Any) -> None:
        prefix = getattr(self, "_env_prefix", "GROWTH_")
        loaded: dict[str, Any] = {}

        for field_name, field_info in self.__class__.model_fields.items():
            if field_name in values and values[field_name] is not None:
                loaded[field_name] = values[field_name]
                continue

            # Check explicit alias first, then fallback to standard prefix + UPPER(field_name)
            candidates: list[str] = []
            if field_info.alias:
                candidates.append(field_info.alias)
                if not field_info.alias.startswith(prefix):
                    candidates.append(f"{prefix}{field_info.alias.upper()}")
            candidates.append(f"{prefix}{field_name.upper()}")

            val: str | None = None
            for env_var in candidates:
                val = os.getenv(env_var)
                if val is not None:
                    break

            if val is not None:
                cleaned = val.strip()
                # Parse JSON array/dict if string is formatted as JSON
                if (cleaned.startswith("[") and cleaned.endswith("]")) or (
                    cleaned.startswith("{") and cleaned.endswith("}")
                ):
                    try:
                        loaded[field_name] = json.loads(cleaned)
                        continue
                    except json.JSONDecodeError:
                        pass
                loaded[field_name] = cleaned

        super().__init__(**loaded)


# ==============================================================================
# 1. APPLICATION SETTINGS
# ==============================================================================
class AppSettings(BaseSettings):
    """Core application and API server binding configurations."""

    env: Literal["development", "testing", "staging", "production"] = Field(
        default="development",
        alias="GROWTH_ENV",
    )
    debug: bool = Field(default=False, alias="GROWTH_DEBUG")
    host: str = Field(default="0.0.0.0", alias="GROWTH_HOST")
    port: int = Field(default=8000, alias="GROWTH_PORT")
    api_v1_prefix: str = Field(default="/v1", alias="GROWTH_API_V1_PREFIX")
    allow_ephemeral_adapters: bool = Field(
        default=True,
        alias="GROWTH_ALLOW_EPHEMERAL_ADAPTERS",
    )
    config_path: Path = Field(
        default=Path("config/defaults.yaml"),
        alias="GROWTH_CONFIG_PATH",
    )
    rate_limit_enabled: bool = Field(default=True, alias="GROWTH_RATE_LIMIT_ENABLED")
    rate_limit_capacity: int = Field(default=120, alias="GROWTH_RATE_LIMIT_CAPACITY")
    rate_limit_refill_rate: float = Field(default=2.0, alias="GROWTH_RATE_LIMIT_REFILL_RATE")
    rate_limit_max_clients: int = Field(default=4096, alias="GROWTH_RATE_LIMIT_MAX_CLIENTS")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v


# ==============================================================================
# 2. DATABASE SETTINGS
# ==============================================================================
class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pooling parameters."""

    url: str | None = Field(default=None, alias="GROWTH_DATABASE_URL")
    migration_url: str | None = Field(default=None, alias="GROWTH_MIGRATION_DATABASE_URL")
    test_migration_dsn: str | None = Field(default=None, alias="GROWTH_TEST_MIGRATION_DSN")
    pool_size: int = Field(default=10, alias="GROWTH_DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="GROWTH_DATABASE_MAX_OVERFLOW")
    pool_timeout: float = Field(default=30.0, alias="GROWTH_DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=1800, alias="GROWTH_DATABASE_POOL_RECYCLE")


# ==============================================================================
# 3. MESSAGING SETTINGS
# ==============================================================================
class MessagingSettings(BaseSettings):
    """NATS JetStream event streaming parameters."""

    nats_servers: list[str] = Field(
        default_factory=lambda: ["nats://localhost:4222"],
        alias="GROWTH_NATS_SERVERS",
    )
    stream_name: str = Field(default="GROWTH_EVENTS", alias="GROWTH_NATS_STREAM_NAME")
    max_msg_size: int = Field(default=262144, alias="GROWTH_NATS_MAX_MSG_SIZE")
    ack_wait_seconds: int = Field(default=30, alias="GROWTH_NATS_ACK_WAIT_SECONDS")
    max_deliver: int = Field(default=5, alias="GROWTH_NATS_MAX_DELIVER")


# ==============================================================================
# 4. REDIS / CACHE SETTINGS
# ==============================================================================
class CacheSettings(BaseSettings):
    """Cache and ephemeral state parameters."""

    redis_url: str | None = Field(default=None, alias="GROWTH_REDIS_URL")
    default_ttl_seconds: int = Field(default=300, alias="GROWTH_CACHE_DEFAULT_TTL_SECONDS")


# ==============================================================================
# 5. SECURITY & AUTHENTICATION SETTINGS
# ==============================================================================
class SecuritySettings(BaseSettings):
    """Security credentials, CORS origin whitelisting, and proxy rules."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["https://growthplus.ai"],
        alias="GROWTH_CORS_ORIGINS_JSON",
    )
    trusted_proxy_cidrs: list[str] = Field(
        default_factory=list,
        alias="GROWTH_TRUSTED_PROXY_CIDRS_JSON",
    )
    jwt_secret: SecretStr | None = Field(default=None, alias="GROWTH_JWT_SECRET")
    model_verification_secret_key: SecretStr | None = Field(
        default=None,
        alias="GROWTH_MODEL_VERIFICATION_SECRET_KEY",
    )
    webhook_signing_secret: SecretStr | None = Field(
        default=None,
        alias="GROWTH_WEBHOOK_SIGNING_SECRET",
    )

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        """Validate and normalize registered origins."""
        normalized_list: list[str] = []
        for value in origins:
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
                raise ValueError(
                    f"CORS requires exact registered HTTPS origins without wildcards/paths: {value}"
                )
            if url.port is not None and not (1 <= url.port <= 65535):
                raise ValueError(f"Invalid CORS port in origin: {value}")
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            port_suffix = f":{url.port}" if url.port is not None else ""
            normalized = f"{scheme}://{host}{port_suffix}"
            if normalized not in normalized_list:
                normalized_list.append(normalized)
        return normalized_list

    @field_validator("trusted_proxy_cidrs")
    @classmethod
    def validate_trusted_proxy_cidrs(cls, cidrs: list[str]) -> list[str]:
        import ipaddress

        for cidr in cidrs:
            try:
                ipaddress.ip_network(cidr, strict=True)
            except ValueError as exc:
                raise ValueError(f"Invalid trusted proxy CIDR format: {cidr}") from exc
        return cidrs


# ==============================================================================
# 6. EXTERNAL SERVICES & PROVIDERS
# ==============================================================================
class ExternalServicesSettings(BaseSettings):
    """External vendor APIs, timeouts, and user agent parameters."""

    gdelt_endpoint: str = Field(
        default="https://api.gdeltproject.org/api/v2/doc/doc",
        alias="GROWTH_GDELT_BASE_URL",
    )
    gdelt_timeout_seconds: float = Field(
        default=10.0,
        alias="GROWTH_GDELT_REQUEST_TIMEOUT_SECONDS",
    )
    gdelt_connect_timeout_seconds: float = Field(
        default=3.0,
        alias="GROWTH_GDELT_CONNECT_TIMEOUT_SECONDS",
    )
    gdelt_contact_user_agent: str = Field(
        default="growthplus-bot/1.0 (+https://growthplus.ai/bot)",
        alias="GROWTH_GDELT_CONTACT_USER_AGENT",
    )
    bls_endpoint: str = Field(
        default="https://api.bls.gov/publicAPI/v2/timeseries/data/",
        alias="GROWTH_BLS_ENDPOINT",
    )
    bls_api_key: SecretStr | None = Field(default=None, alias="GROWTH_BLS_API_KEY")
    fed_board_release_index: str = Field(
        default="https://www.federalreserve.gov/releases/h15/",
        alias="GROWTH_FED_BOARD_URL",
    )
    webhook_connect_timeout_seconds: float = Field(
        default=3.0,
        alias="GROWTH_WEBHOOK_CONNECT_TIMEOUT_SECONDS",
    )
    webhook_read_timeout_seconds: float = Field(
        default=10.0,
        alias="GROWTH_WEBHOOK_READ_TIMEOUT_SECONDS",
    )
    webhook_max_payload_bytes: int = Field(
        default=262144,
        alias="GROWTH_WEBHOOK_MAX_PAYLOAD_BYTES",
    )
    webhook_max_response_bytes: int = Field(
        default=65536,
        alias="GROWTH_WEBHOOK_MAX_RESPONSE_BYTES",
    )
    webhook_max_redirects: int = Field(
        default=4,
        alias="GROWTH_WEBHOOK_MAX_REDIRECTS",
    )


# ==============================================================================
# 7. ML & QUANTITATIVE SIGNALS
# ==============================================================================
class MLSettings(BaseSettings):
    """Quantitative threshold parameters and inference model hyperparameters."""

    default_horizon: str = Field(default="swing_24h", alias="GROWTH_DEFAULT_HORIZON")
    base_cost_hurdle: float = Field(default=0.0005, alias="GROWTH_BASE_COST_HURDLE")
    horizon_floor_swing: float = Field(default=0.003, alias="GROWTH_HORIZON_FLOOR_SWING")
    horizon_floor_position: float = Field(default=0.012, alias="GROWTH_HORIZON_FLOOR_POSITION")
    ece_gate_threshold: float = Field(default=0.08, alias="GROWTH_ECE_GATE_THRESHOLD")
    bundle_ttl_days: int = Field(default=30, alias="GROWTH_BUNDLE_TTL_DAYS")
    notification_min_confidence: float = Field(
        default=0.60,
        alias="GROWTH_NOTIFICATION_MIN_CONFIDENCE",
    )
    notification_cooldown_seconds: float = Field(
        default=900.0,
        alias="GROWTH_NOTIFICATION_COOLDOWN_SECONDS",
    )


# ==============================================================================
# 8. RESILIENCE & CIRCUIT BREAKERS
# ==============================================================================
class ResilienceSettings(BaseSettings):
    """Resilience policies and circuit breaker timing."""

    circuit_failure_threshold: int = Field(
        default=5,
        alias="GROWTH_CIRCUIT_FAILURE_THRESHOLD",
    )
    circuit_recovery_timeout_seconds: float = Field(
        default=60.0,
        alias="GROWTH_CIRCUIT_RECOVERY_TIMEOUT_SECONDS",
    )
    circuit_probe_timeout_seconds: float = Field(
        default=30.0,
        alias="GROWTH_CIRCUIT_PROBE_TIMEOUT_SECONDS",
    )


# ==============================================================================
# ROOT SETTINGS
# ==============================================================================
class Settings(BaseModel):
    """Root configuration object containing all modular settings sections."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    messaging: MessagingSettings = Field(default_factory=MessagingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    external: ExternalServicesSettings = Field(default_factory=ExternalServicesSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)

    @model_validator(mode="after")
    def validate_production_invariants(self) -> Settings:
        """Enforce strict fail-fast validation for production environments."""
        if self.app.env == "production":
            errors: list[str] = []

            # 1. Ephemeral adapters forbidden in production
            if self.app.allow_ephemeral_adapters:
                errors.append("GROWTH_ALLOW_EPHEMERAL_ADAPTERS must be false in production mode")

            # 2. Database URL mandatory
            if not self.database.url:
                errors.append("GROWTH_DATABASE_URL is required in production mode")

            # 3. JWT secret mandatory with minimum 32 characters
            jwt_sec = self.security.jwt_secret
            if not jwt_sec or len(jwt_sec.get_secret_value()) < 32:
                errors.append(
                    "GROWTH_JWT_SECRET is required and must be at least 32 characters in production"
                )

            # 4. Model verification secret mandatory with minimum 32 characters
            model_key = self.security.model_verification_secret_key
            if not model_key or len(model_key.get_secret_value()) < 32:
                errors.append(
                    "GROWTH_MODEL_VERIFICATION_SECRET_KEY is required and must be at least 32 "
                    "characters in production"
                )

            # 5. Webhook signing secret mandatory
            if not self.security.webhook_signing_secret:
                errors.append("GROWTH_WEBHOOK_SIGNING_SECRET is required in production mode")

            # 6. CORS origins cannot be empty
            if not self.security.cors_origins:
                errors.append(
                    "GROWTH_CORS_ORIGINS_JSON must contain at least one registered origin "
                    "in production"
                )

            if errors:
                raise ValueError(
                    "Production configuration validation failed:\n - " + "\n - ".join(errors)
                )

        return self

    # Top-level convenience properties
    @property
    def env(self) -> str:
        return self.app.env

    @property
    def is_production(self) -> bool:
        return self.app.env == "production"

    @property
    def port(self) -> int:
        return self.app.port

    @property
    def host(self) -> str:
        return self.app.host

    @property
    def database_url(self) -> str | None:
        return self.database.url

    @property
    def cors_origins(self) -> list[str]:
        return self.security.cors_origins

    @property
    def trusted_proxy_cidrs(self) -> list[str]:
        return self.security.trusted_proxy_cidrs

    @property
    def config_path(self) -> Path:
        return self.app.config_path


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings(reload: bool = False) -> Settings:
    """Retrieve the cached Settings instance or construct a freshly reloaded one."""
    if reload:
        _cached_settings.cache_clear()
    return _cached_settings()
