"""Deterministic log-odds decomposition and explanation engine.

Enforces PRD Section 3.11:
- Store per-pillar contributions to selected-class vs opposite-class log odds
  before and after temperature scaling.
- Invariant: sum(contributions) + intercept_diff reproduces reported log-odds
  difference within 1e-6.
- Deterministic reason codes and explanation factors mapping supporting and opposing pillars.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.domain.features import PillarType
from src.domain.predictions import OutcomeClass, PillarPrediction, ProbabilityVector
from src.domain.signals import SignalExplanationFactor, SignalReasonCode

# Mapping of pillar and direction to standard signal reason codes
PILLAR_REASON_MAP: dict[tuple[PillarType, str], SignalReasonCode] = {
    (PillarType.TECHNICAL, "UP"): SignalReasonCode.TECH_MOMENTUM_SUPPORT,
    (PillarType.TECHNICAL, "DOWN"): SignalReasonCode.TECH_TREND_SUPPORT,
    (PillarType.ETF, "UP"): SignalReasonCode.ETF_RECENT_FLOW_SUPPORT,
    (PillarType.ETF, "DOWN"): SignalReasonCode.ETF_OUTFLOW_PRESSURE,
    (PillarType.MACRO, "UP"): SignalReasonCode.MACRO_INFLATION_EASING,
    (PillarType.MACRO, "DOWN"): SignalReasonCode.MACRO_RATE_PRESSURE,
    (PillarType.NEWS, "UP"): SignalReasonCode.NEWS_SENTIMENT_BULLISH,
    (PillarType.NEWS, "DOWN"): SignalReasonCode.NEWS_REGULATORY_HEADWIND,
}


@dataclass(frozen=True, slots=True)
class PillarLogOddsAttribution:
    """Exact numerical contribution of an individual pillar to log-odds."""

    pillar: PillarType
    weight: float
    raw_log_odds_diff: float  # log(p_{m, selected}) - log(p_{m, opposite})
    contribution_pre_temp: float  # w_m * raw_log_odds_diff
    contribution_post_temp: float  # contribution_pre_temp / T


@dataclass(frozen=True, slots=True)
class LogOddsExplanation:
    """Complete mathematical decomposition of ensemble log odds."""

    selected_class: OutcomeClass
    opposite_class: OutcomeClass
    intercept_diff_pre_temp: float
    intercept_diff_post_temp: float
    pillar_attributions: tuple[PillarLogOddsAttribution, ...]
    total_log_odds_diff_pre_temp: float
    total_log_odds_diff_post_temp: float
    numerical_reconstruction_error: float
    temperature: float

    def verify_reconstruction(self, tolerance: float = 1e-6) -> bool:
        """Verify the PRD Section 3.11 invariant: sum(contribs) + intercept_diff == total_diff."""
        return self.numerical_reconstruction_error <= tolerance


class ExplanationEngine:
    """Decomposes late-fusion probability vectors into deterministic log-odds attributions."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tolerance = tolerance

    def decompose_log_odds(
        self,
        ensemble_probs: ProbabilityVector,
        pillar_predictions: Mapping[PillarType, PillarPrediction],
        operational_weights: Mapping[PillarType, float],
        intercepts: Sequence[float],
        temperature: float,
    ) -> LogOddsExplanation:
        """Decompose selected-class vs opposite-class log-odds into exact per-pillar
        contributions.
        """
        # Determine selected class and opposite class
        selected = ensemble_probs.largest_class
        if selected == OutcomeClass.UP:
            opposite = OutcomeClass.DOWN
            k_sel, k_opp = 2, 0
        elif selected == OutcomeClass.DOWN:
            opposite = OutcomeClass.UP
            k_sel, k_opp = 0, 2
        else:  # FLAT
            # Compare FLAT against whichever directional probability is higher
            if ensemble_probs.p_up >= ensemble_probs.p_down:
                opposite = OutcomeClass.UP
                k_sel, k_opp = 1, 2
            else:
                opposite = OutcomeClass.DOWN
                k_sel, k_opp = 1, 0

        # Intercept difference
        a_sel = float(intercepts[k_sel])
        a_opp = float(intercepts[k_opp])
        intercept_diff_pre = a_sel - a_opp
        intercept_diff_post = intercept_diff_pre / temperature

        attributions: list[PillarLogOddsAttribution] = []
        sum_contribs_pre = 0.0

        for pillar, pred in pillar_predictions.items():
            w = operational_weights.get(pillar, 0.0)
            probs = [pred.probabilities.p_down, pred.probabilities.p_flat, pred.probabilities.p_up]
            p_sel = max(1e-6, probs[k_sel])
            p_opp = max(1e-6, probs[k_opp])

            raw_diff = math.log(p_sel) - math.log(p_opp)
            contrib_pre = w * raw_diff
            contrib_post = contrib_pre / temperature

            attributions.append(
                PillarLogOddsAttribution(
                    pillar=pillar,
                    weight=w,
                    raw_log_odds_diff=raw_diff,
                    contribution_pre_temp=contrib_pre,
                    contribution_post_temp=contrib_post,
                )
            )
            sum_contribs_pre += contrib_pre

        reconstructed_pre = intercept_diff_pre + sum_contribs_pre

        # Reported post-temp log odds from the ensemble probabilities
        ens_probs = [ensemble_probs.p_down, ensemble_probs.p_flat, ensemble_probs.p_up]
        log_sel = math.log(max(1e-6, ens_probs[k_sel]))
        log_opp = math.log(max(1e-6, ens_probs[k_opp]))
        reported_post = log_sel - log_opp
        reported_pre = reported_post * temperature

        recon_err = abs(reconstructed_pre - reported_pre)

        return LogOddsExplanation(
            selected_class=selected,
            opposite_class=opposite,
            intercept_diff_pre_temp=intercept_diff_pre,
            intercept_diff_post_temp=intercept_diff_post,
            pillar_attributions=tuple(attributions),
            total_log_odds_diff_pre_temp=reported_pre,
            total_log_odds_diff_post_temp=reported_post,
            numerical_reconstruction_error=recon_err,
            temperature=temperature,
        )

    def generate_signal_factors(
        self,
        explanation: LogOddsExplanation,
        max_factors: int = 4,
    ) -> tuple[SignalExplanationFactor, ...]:
        """Generate structured explanation factors identifying supporting and opposing forces."""
        # Sort attributions by post-temp contribution magnitude
        sorted_attribs = sorted(
            explanation.pillar_attributions,
            key=lambda a: abs(a.contribution_post_temp),
            reverse=True,
        )

        factors: list[SignalExplanationFactor] = []
        has_opposing = False

        for attr in sorted_attribs[:max_factors]:
            is_supporting = attr.contribution_post_temp >= 0.0
            if is_supporting:
                direction = explanation.selected_class.value
            else:
                direction = explanation.opposite_class.value
                has_opposing = True

            code = PILLAR_REASON_MAP.get(
                (attr.pillar, direction),
                f"{attr.pillar.value.upper()}_{direction}",
            )

            factors.append(
                SignalExplanationFactor(
                    code=code,
                    direction=direction,
                    attribution_weight=attr.contribution_post_temp,
                    pillar=attr.pillar.value if hasattr(attr.pillar, "value") else str(attr.pillar),
                )
            )

        # If no opposing factor in top items, try to find one if present
        if not has_opposing:
            opposing = [a for a in sorted_attribs if a.contribution_post_temp < 0.0]
            if opposing:
                opp = opposing[0]
                direction = explanation.opposite_class.value
                code = PILLAR_REASON_MAP.get(
                    (opp.pillar, direction),
                    f"{opp.pillar.value.upper()}_{direction}",
                )
                if len(factors) >= max_factors:
                    factors.pop()
                factors.append(
                    SignalExplanationFactor(
                        code=code,
                        direction=direction,
                        attribution_weight=opp.contribution_post_temp,
                        pillar=opp.pillar.value if hasattr(opp.pillar, "value") else str(opp.pillar),
                    )
                )

        return tuple(factors)
