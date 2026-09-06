"""Aligned, explicit OOF rows: no zero-filled warm-up rows reach fusion."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from src.ml.datasets.purged_splits import Fold, TargetClock, expanding_splits


@dataclass(frozen=True, slots=True)
class OOFResult:
    row_indices: tuple[int, ...]
    probabilities: NDArray[np.float64]
    excluded_rows: tuple[int, ...]
    folds: tuple[Fold, ...]

    def __post_init__(self) -> None:
        values = np.array(self.probabilities, dtype=np.float64, copy=True)
        if values.shape != (len(self.row_indices), 3) or len(set(self.row_indices)) != len(
            self.row_indices
        ):
            raise ValueError("OOF rows/probabilities are misaligned")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)

    def targets_from(self, labels: NDArray[np.int64]) -> NDArray[np.int64]:
        return labels[np.asarray(self.row_indices, dtype=np.int64)]


def plan_oof(
    cutoffs: Sequence[datetime], targets: Sequence[TargetClock], frozen_at: datetime, n_splits: int
) -> tuple[Fold, ...]:
    return expanding_splits(cutoffs, targets, frozen_at=frozen_at, n_splits=n_splits)
