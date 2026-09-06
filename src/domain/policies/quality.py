"""Quality metrics and system quality score calculation.

Enforces Section 3.10:
- Component quality factor: q_m = validity_m * completeness_m * freshness_m
- System quality score: Q = sum_m (b_{m,h} * q_m) computed across all 4 full-mode priors
"""

from collections.abc import Mapping
from dataclasses import dataclass

from src.domain.errors import InvariantViolationError
from src.domain.features import PillarType
from src.domain.identity import HorizonId

FULL_MODE_PRIORS: Mapping[HorizonId, Mapping[PillarType, float]] = {
    HorizonId.SCALP_15M: {
        PillarType.MACRO: 0.05,
        PillarType.ETF: 0.05,
        PillarType.NEWS: 0.15,
        PillarType.TECHNICAL: 0.75,
    },
    HorizonId.SWING_24H: {
        PillarType.MACRO: 0.20,
        PillarType.ETF: 0.25,
        PillarType.NEWS: 0.20,
        PillarType.TECHNICAL: 0.35,
    },
    HorizonId.POSITION_30D: {
        PillarType.MACRO: 0.30,
        PillarType.ETF: 0.30,
        PillarType.NEWS: 0.15,
        PillarType.TECHNICAL: 0.25,
    },
}


@dataclass(frozen=True, slots=True)
class PillarQuality:
    """Component quality factors for a pillar."""

    validity: float  # 0.0 if semantic, rights, or integrity error, else 1.0
    completeness: float  # [0.0, 1.0]
    freshness: float  # [0.0, 1.0]

    def __post_init__(self) -> None:
        for name, val in [
            ("validity", self.validity),
            ("completeness", self.completeness),
            ("freshness", self.freshness),
        ]:
            if val < 0.0 or val > 1.0:
                raise InvariantViolationError(f"PillarQuality {name} ({val}) must be in [0.0, 1.0]")

    @property
    def score(self) -> float:
        """q_m = validity * completeness * freshness."""
        return self.validity * self.completeness * self.freshness


def compute_system_quality(
    horizon: HorizonId,
    pillar_qualities: Mapping[PillarType, PillarQuality],
) -> float:
    """Compute system quality score Q over all four full-mode priors.

    Missing pillars have a quality factor of 0.0, making missing pillars
    immediately visible in the overall quality score.
    """
    priors = FULL_MODE_PRIORS[horizon]
    total_q = 0.0
    for pillar, prior_weight in priors.items():
        q_m = pillar_qualities[pillar].score if pillar in pillar_qualities else 0.0
        total_q += prior_weight * q_m
    return min(1.0, max(0.0, total_q))
