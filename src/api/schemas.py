"""Pydantic v2 transport schemas for REST and SSE APIs.

Enforces PRD Section 4.5 & 4.6:
- Exact JSON representations of Signal, ProbabilityVector, Reasons, and Sources
- CamelCase and snake_case consistency
- ISO 8601 UTC timestamps
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.domain.signals import Signal


class ExplanationFactorSchema(BaseModel):
    """Structured attribution factor."""

    code: str
    direction: str
    attribution_weight: float = 0.0


class CohortReliabilitySchema(BaseModel):
    """95% block-bootstrap accuracy interval for prediction cohort."""

    lower: float
    upper: float
    raw_observations: int
    independent_blocks: int
    method: str = "stationary_block_bootstrap"


class ProbabilityVectorSchema(BaseModel):
    """3-class outcome probabilities summing to 1.0."""

    p_up: float
    p_flat: float
    p_down: float


class SignalResponse(BaseModel):
    """User-facing signal response matching PRD Section 4.6 contract."""

    signal_id: uuid.UUID
    sequence: int
    market_id: str
    horizon: str
    status: str
    mode: str
    cutoff_at: datetime
    issued_at: datetime
    expires_at: datetime
    probabilities: ProbabilityVectorSchema | None = None
    label: str | None = None
    confidence: float | None = None
    confidence_event: str | None = None
    data_quality: float
    outcome_hurdle_log_return: float
    reason_code: str | None = None
    reasons: list[ExplanationFactorSchema] = Field(default_factory=list)
    cohort_reliability: CohortReliabilitySchema | None = None
    model_bundle: str = ""
    is_replay: bool = False

    @classmethod
    def from_domain(cls, signal: Signal) -> "SignalResponse":
        """Map domain Signal entity to API transport schema."""
        probs_schema = None
        if signal.probabilities:
            probs_schema = ProbabilityVectorSchema(
                p_up=signal.probabilities.p_up,
                p_flat=signal.probabilities.p_flat,
                p_down=signal.probabilities.p_down,
            )

        reliability_schema = None
        if signal.cohort_reliability:
            reliability_schema = CohortReliabilitySchema(
                lower=signal.cohort_reliability.lower,
                upper=signal.cohort_reliability.upper,
                raw_observations=signal.cohort_reliability.raw_observations,
                independent_blocks=signal.cohort_reliability.independent_blocks,
                method=signal.cohort_reliability.method,
            )

        reasons = [
            ExplanationFactorSchema(
                code=str(r.code),
                direction=r.direction,
                attribution_weight=r.attribution_weight,
            )
            for r in signal.reasons
        ]

        return cls(
            signal_id=signal.signal_id,
            sequence=signal.sequence,
            market_id=signal.market_id.value,
            horizon=signal.horizon.value,
            status=signal.status.value,
            mode=signal.mode.value,
            cutoff_at=signal.cutoff_at,
            issued_at=signal.issued_at,
            expires_at=signal.expires_at,
            probabilities=probs_schema,
            label=signal.label.value if signal.label else None,
            confidence=signal.confidence,
            confidence_event=signal.confidence_event,
            data_quality=signal.data_quality,
            outcome_hurdle_log_return=signal.outcome_hurdle_log_return,
            reason_code=signal.reason_code.value if signal.reason_code else None,
            reasons=reasons,
            cohort_reliability=reliability_schema,
            model_bundle=signal.model_bundle,
            is_replay=signal.is_replay,
        )


class SignalListResponse(BaseModel):
    """Paginated list of historic signals."""

    items: list[SignalResponse]
    total: int
    limit: int
    offset: int


class SourceDetailSchema(BaseModel):
    """Per-dataset freshness and availability details."""

    source_id: str
    dataset_id: str
    state: str
    freshness_factor: float
    age_seconds: float
    last_update_at: datetime | None = None
    is_hard_stop: bool = False


class SourcesStatusResponse(BaseModel):
    """Operational health across all ingestion feeds."""

    active_mode: str
    overall_quality: float
    sources: list[SourceDetailSchema]
    incidents: list[str] = Field(default_factory=list)


class NotificationSubscribeRequest(BaseModel):
    """Subscription configuration for notification alerts."""

    target: str = Field(..., description="Channel: webhook, telegram, email")
    destination: str = Field(..., description="Endpoint URL, chat_id, or email address")
    markets: list[str] = Field(default_factory=lambda: ["binance:BTCUSDT", "binance:ETHUSDT"])
    horizons: list[str] = Field(default_factory=lambda: ["swing_24h", "scalp_15m"])
    min_conviction: float = Field(default=0.60, ge=0.50, le=1.0)
    cooldown_seconds: int = Field(default=300, ge=30)


class NotificationSubscribeResponse(BaseModel):
    """Confirmation of registered notification preference."""

    subscription_id: uuid.UUID
    target: str
    destination: str
    status: str = "ACTIVE"
    created_at: datetime
