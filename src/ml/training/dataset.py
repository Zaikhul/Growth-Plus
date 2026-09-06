"""Feature extraction, pillar feature matrix construction, and target generation.

Enforces PRD Section 3.7 & 3.8:
- Separation of features by pillar (Technical, Macro, ETF, News)
- Directional target classes: DOWN (0), FLAT (1), UP (2)
- Point-in-time chronological order
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.domain.features import FeatureSnapshot, PillarType
from src.domain.identity import HorizonId
from src.domain.policies.abstention import classify_realized_outcome, compute_economic_hurdle
from src.domain.predictions import OutcomeClass

CLASS_TO_INDEX: dict[OutcomeClass, int] = {
    OutcomeClass.DOWN: 0,
    OutcomeClass.FLAT: 1,
    OutcomeClass.UP: 2,
}

INDEX_TO_CLASS: dict[int, OutcomeClass] = {
    0: OutcomeClass.DOWN,
    1: OutcomeClass.FLAT,
    2: OutcomeClass.UP,
}

PILLAR_FEATURE_PREFIXES: dict[PillarType, str] = {
    PillarType.TECHNICAL: "tech_",
    PillarType.MACRO: "macro_",
    PillarType.ETF: "etf_",
    PillarType.NEWS: "news_",
}


@dataclass(frozen=True, slots=True)
class PillarDataset:
    """Immutable matrix of feature observations and targets for a single pillar expert."""

    pillar: PillarType
    feature_names: tuple[str, ...]
    X: np.ndarray  # Shape: (N, num_features), dtype float64
    y: np.ndarray  # Shape: (N,), dtype int64, values in {0, 1, 2}
    cutoffs: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(f"X length ({len(self.X)}) must match y length ({len(self.y)})")
        if len(self.X) != len(self.cutoffs):
            raise ValueError(
                f"X length ({len(self.X)}) must match cutoffs length ({len(self.cutoffs)})"
            )
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X column dimension ({self.X.shape}) must match "
                f"feature_names count ({len(self.feature_names)})"
            )


def extract_pillar_features(
    snapshot: FeatureSnapshot,
    pillar: PillarType,
    ordered_feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract a numeric 1D array of features for a specific pillar from a snapshot.

    If ordered_feature_names is provided, guarantees exact feature ordering and fills
    missing features with np.nan. Otherwise, discovers features matching the pillar prefix.
    """
    prefix = PILLAR_FEATURE_PREFIXES[pillar]

    if ordered_feature_names is not None:
        names = tuple(ordered_feature_names)
        vals = [snapshot.features.get(k, np.nan) for k in names]
        return np.array(vals, dtype=np.float64), names

    # Discover and sort deterministically
    names = tuple(sorted(k for k in snapshot.features if k.startswith(prefix)))
    vals = [snapshot.features.get(k, np.nan) for k in names]
    return np.array(vals, dtype=np.float64), names


def build_pillar_dataset(
    snapshots: Sequence[FeatureSnapshot],
    outcomes: Sequence[OutcomeClass],
    pillar: PillarType,
    feature_names: Sequence[str] | None = None,
) -> PillarDataset:
    """Build a PillarDataset from paired snapshots and realized outcomes."""
    if len(snapshots) != len(outcomes):
        raise ValueError("snapshots count must match outcomes count")

    if not snapshots:
        return PillarDataset(
            pillar=pillar,
            feature_names=tuple(feature_names or ()),
            X=np.empty((0, len(feature_names or ())), dtype=np.float64),
            y=np.empty((0,), dtype=np.int64),
            cutoffs=(),
        )

    # Determine stable feature names from the first snapshot or explicit schema
    if feature_names is None:
        _, discovered_names = extract_pillar_features(snapshots[0], pillar)
        col_names = discovered_names
    else:
        col_names = tuple(feature_names)

    x_rows: list[np.ndarray] = []
    y_vals: list[int] = []
    cutoffs: list[datetime] = []

    for snap, outcome in zip(snapshots, outcomes, strict=True):
        x_vec, _ = extract_pillar_features(snap, pillar, ordered_feature_names=col_names)
        x_rows.append(x_vec)
        y_vals.append(CLASS_TO_INDEX[outcome])
        cutoffs.append(snap.decision_cutoff)

    X = np.vstack(x_rows) if x_rows else np.empty((0, len(col_names)), dtype=np.float64)
    y = np.array(y_vals, dtype=np.int64)

    return PillarDataset(
        pillar=pillar,
        feature_names=col_names,
        X=X,
        y=y,
        cutoffs=tuple(cutoffs),
    )


def compute_target_outcomes(
    entry_prices: Sequence[float],
    exit_prices: Sequence[float],
    volatilities: Sequence[float],
    horizon: HorizonId,
    cost_hurdle: float = 0.0005,
) -> list[OutcomeClass]:
    """Compute economic hurdle targets for a series of forward entry and exit prices."""
    if len(entry_prices) != len(exit_prices) or len(entry_prices) != len(volatilities):
        raise ValueError("Price and volatility series must have equal lengths")

    outcomes: list[OutcomeClass] = []
    for entry, exit_p, vol in zip(entry_prices, exit_prices, volatilities, strict=True):
        hurdle = compute_economic_hurdle(
            horizon=horizon,
            cost_hurdle_log_return=cost_hurdle,
            trailing_volatility_estimate=vol,
        )
        outcomes.append(classify_realized_outcome(entry, exit_p, hurdle))
    return outcomes
