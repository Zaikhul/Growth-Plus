"""Fixed elapsed-time window for technical warm-up, not a macro YoY join."""

from datetime import datetime, timedelta

from src.domain.time import ensure_utc


def technical_window_start(cutoff: datetime) -> datetime:
    return ensure_utc(cutoff) - timedelta(days=365)
