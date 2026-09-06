"""Classification policy and label determination for Growth+.

Enforces the 6-level mutually exclusive Label Priority Table from PRD Section 3.9:
1. Strong Buy: p_up >= 0.75, margin >= 0.40, Q >= 0.90, FULL mode, >= 2 agreeing pillars,
   rel_lower >= 0.60, no event block
2. Strong Sell: p_down >= 0.75, margin >= 0.40, Q >= 0.90, FULL mode, >= 2 agreeing pillars,
   rel_lower >= 0.60, no event block
3. Buy: p_up >= 0.60, margin >= 0.20, Q >= 0.75, approved mask
4. Sell: p_down >= 0.60, margin >= 0.20, Q >= 0.75, approved mask
5. Neutral: p_flat >= 0.50, p_flat largest class, Q >= 0.75, approved mask
6. Abstain: all remaining cases -> status=ABSTAIN, reason=LOW_CONVICTION
"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.domain.features import SourceCoverageMode
from src.domain.predictions import OutcomeClass, PillarPrediction, ProbabilityVector
from src.domain.signals import SignalLabel, SignalReasonCode, SignalStatus


@dataclass(frozen=True, slots=True)
class ClassificationContext:
    """Input context required to evaluate the classification policy."""

    probabilities: ProbabilityVector
    quality_score: float
    mode: SourceCoverageMode
    pillar_predictions: Sequence[PillarPrediction]
    cohort_reliability_lower_95: float | None = None
    has_event_block: bool = False
    is_cohort_calibrated: bool = True


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Evaluated label, confidence, and decision status."""

    status: SignalStatus
    label: SignalLabel | None
    confidence: float | None
    confidence_event: str | None
    reason_code: SignalReasonCode | None = None


def count_agreeing_pillars(
    pillar_predictions: Sequence[PillarPrediction],
    direction: OutcomeClass,
) -> int:
    """Count pillars whose standalone probability exceeds flat and opposite direction."""
    return sum(1 for p in pillar_predictions if p.agrees_with_direction(direction))


def _check_strong_conditions(
    prob_target: float,
    prob_opposite: float,
    ctx: ClassificationContext,
    target_class: OutcomeClass,
) -> bool:
    """Validate all necessary criteria for Strong Buy / Strong Sell."""
    if ctx.mode != SourceCoverageMode.FULL:
        return False
    if ctx.has_event_block:
        return False
    if not ctx.is_cohort_calibrated:
        return False
    if prob_target < 0.75:
        return False
    if (prob_target - prob_opposite) < 0.40:
        return False
    if ctx.quality_score < 0.90:
        return False
    if ctx.cohort_reliability_lower_95 is None or ctx.cohort_reliability_lower_95 < 0.60:
        return False
    if count_agreeing_pillars(ctx.pillar_predictions, target_class) < 2:
        return False
    return True


def evaluate_classification(ctx: ClassificationContext) -> ClassificationResult:
    """Evaluate calibrated probabilities against the priority ladder."""
    p_up = ctx.probabilities.p_up
    p_down = ctx.probabilities.p_down
    p_flat = ctx.probabilities.p_flat

    # Priority 1: Strong Buy
    if _check_strong_conditions(p_up, p_down, ctx, OutcomeClass.UP):
        return ClassificationResult(
            status=SignalStatus.READY,
            label=SignalLabel.STRONG_BUY,
            confidence=p_up,
            confidence_event="UP_ABOVE_FROZEN_HURDLE",
        )

    # Priority 2: Strong Sell
    if _check_strong_conditions(p_down, p_up, ctx, OutcomeClass.DOWN):
        return ClassificationResult(
            status=SignalStatus.READY,
            label=SignalLabel.STRONG_SELL,
            confidence=p_down,
            confidence_event="DOWN_BELOW_FROZEN_HURDLE",
        )

    # Base quality and calibration gate for standard labels
    if ctx.quality_score >= 0.75 and ctx.is_cohort_calibrated:
        # Priority 3: Buy
        if p_up >= 0.60 and (p_up - p_down) >= 0.20:
            return ClassificationResult(
                status=SignalStatus.READY,
                label=SignalLabel.BUY,
                confidence=p_up,
                confidence_event="UP_ABOVE_FROZEN_HURDLE",
            )

        # Priority 4: Sell
        if p_down >= 0.60 and (p_down - p_up) >= 0.20:
            return ClassificationResult(
                status=SignalStatus.READY,
                label=SignalLabel.SELL,
                confidence=p_down,
                confidence_event="DOWN_BELOW_FROZEN_HURDLE",
            )

        # Priority 5: Neutral
        if p_flat >= 0.50 and ctx.probabilities.largest_class == OutcomeClass.FLAT:
            return ClassificationResult(
                status=SignalStatus.READY,
                label=SignalLabel.NEUTRAL,
                confidence=p_flat,
                confidence_event="FLAT_WITHIN_FROZEN_HURDLE",
            )

    # Priority 6: Abstain under low conviction or unmet gates
    return ClassificationResult(
        status=SignalStatus.ABSTAIN,
        label=None,
        confidence=None,
        confidence_event=None,
        reason_code=SignalReasonCode.LOW_CONVICTION,
    )
