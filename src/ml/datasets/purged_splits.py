"""UTC expanding OOF splits; target clocks are explicit, not estimated row gaps."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class TargetClock:
    entry_at: datetime
    exit_at: datetime
    available_at: datetime


@dataclass(frozen=True, slots=True)
class Fold:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    cutoff: datetime


def _utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Split timestamps must be timezone-aware UTC")


def expanding_splits(
    cutoffs: Sequence[datetime],
    targets: Sequence[TargetClock],
    *,
    frozen_at: datetime,
    n_splits: int = 5,
) -> tuple[Fold, ...]:
    _utc(frozen_at)
    if n_splits != 5 or len(cutoffs) != len(targets):
        raise ValueError("Five folds and aligned target clocks are required")
    for cutoff, target in zip(cutoffs, targets, strict=True):
        for value in (cutoff, target.entry_at, target.exit_at, target.available_at):
            _utc(value)
        if not cutoff <= target.entry_at < target.exit_at <= target.available_at <= frozen_at:
            raise ValueError("Invalid or unmatured training target")
    times = sorted(set(cutoffs))
    if len(times) < 6:
        raise ValueError("Insufficient history: in-sample fallback is forbidden")
    # Split UTC times, never individual asset rows, into warm-up + five blocks.
    boundaries = [len(times) * i // 6 for i in range(7)]
    result: list[Fold] = []
    for fold in range(1, 6):
        start = times[boundaries[fold]]
        stop = times[boundaries[fold + 1]] if fold < 5 else None
        train = tuple(
            i
            for i, (cutoff, target) in enumerate(zip(cutoffs, targets, strict=True))
            if cutoff < start and target.exit_at < start and target.available_at < start
        )
        validation = tuple(
            i
            for i, cutoff in enumerate(cutoffs)
            if start <= cutoff and (stop is None or cutoff < stop)
        )
        if not train or not validation:
            raise ValueError("Insufficient mature fold history after interval purging")
        result.append(Fold(train, validation, start))
    return tuple(result)


def purge_later_segment(
    targets: Sequence[TargetClock],
    validation_start: datetime,
    validation_end: datetime,
    maximum_horizon: timedelta,
) -> tuple[int, ...]:
    """For outer designs that admit later training: exclude overlap plus embargo."""
    if maximum_horizon <= timedelta(0) or validation_start >= validation_end:
        raise ValueError("Invalid validation/embargo interval")
    return tuple(
        i
        for i, target in enumerate(targets)
        if target.exit_at < validation_start or target.entry_at > validation_end + maximum_horizon
    )
