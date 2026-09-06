"""Publication guard for cache failure, source outage and approved model rollback."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CurrentValue[Payload]:
    value: Payload
    cutoff_at: datetime
    expires_at: datetime
    rights_allowed: bool


@dataclass(frozen=True, slots=True)
class ApprovedModel[Payload]:
    predict: Callable[[Payload], Awaitable[Payload]]
    rights_allowed: bool
    compatible: bool
    calibrated: bool


@dataclass(frozen=True, slots=True)
class PublicationResult[Payload]:
    status: str
    reason: str
    value: Payload | None
    rolled_back: bool = False


async def evaluate_current[Payload](
    *,
    clock: Callable[[], datetime],
    authorize_publish: Callable[[], bool],
    read_cache: Callable[[], Awaitable[CurrentValue[Payload] | None]],
    read_authoritative: Callable[[], Awaitable[CurrentValue[Payload] | None]],
    champion: ApprovedModel[Payload],
    fallback: ApprovedModel[Payload] | None,
    commit: Callable[[Payload], Awaitable[None]],
) -> PublicationResult[Payload]:
    """commit must atomically write decision+outbox; publication only follows success."""
    now = clock()
    try:
        candidate = await read_cache()
    except (TimeoutError, ConnectionError) as exc:
        logger.warning("Cache unavailable", extra={"error_type": type(exc).__name__})
        candidate = None
    if candidate is None or not candidate.cutoff_at <= now < candidate.expires_at:
        try:
            candidate = await read_authoritative()
        except (TimeoutError, ConnectionError):
            return PublicationResult("UNAVAILABLE", "SOURCE_OR_STORE_UNAVAILABLE", None)
    if candidate is None or not candidate.cutoff_at <= now < candidate.expires_at:
        return PublicationResult("UNAVAILABLE", "STALE_OR_MISSING_SNAPSHOT", None)
    if not candidate.rights_allowed:
        return PublicationResult("UNAVAILABLE", "RIGHTS_BLOCKED", None)
    choices = (champion,) if fallback is None else (champion, fallback)
    for index, model in enumerate(choices):
        if not (model.rights_allowed and model.compatible and model.calibrated):
            continue
        try:
            prediction = await model.predict(candidate.value)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            logger.warning(
                "Model attempt failed",
                extra={"candidate_index": index, "error_type": type(exc).__name__},
            )
            continue
        # Never catch a failed commit and then return a current prediction.
        if clock() >= candidate.expires_at:
            return PublicationResult("UNAVAILABLE", "EXPIRED_DURING_INFERENCE", None)
        if not authorize_publish():
            return PublicationResult("UNAVAILABLE", "RIGHTS_WITHDRAWN", None)
        await commit(prediction)
        return PublicationResult(
            "CURRENT", "APPROVED_ROLLBACK" if index else "OK", prediction, bool(index)
        )
    return PublicationResult("UNAVAILABLE", "NO_VALID_MODEL", None)
