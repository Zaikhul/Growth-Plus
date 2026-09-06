"""Abstention rules, economic hurdle calculation, and outcome thresholds.

Enforces Section 3.8:
- Frozen symmetric hurdle: theta_{h,t} = max(c_{h,t}, k_h * sigma_hat_{h,t}, b_h)
- Parameters:
    scalp:    k=0.25, b=0.001
    swing:    k=0.35, b=0.003
    position: k=0.50, b=0.015
- Directional outcome classes:
    UP:   r_h > theta
    DOWN: r_h < -theta
    FLAT: otherwise
"""

import math
from dataclasses import dataclass

from src.domain.identity import HorizonId
from src.domain.predictions import OutcomeClass


@dataclass(frozen=True, slots=True)
class HurdleParameters:
    """Parameters for computing the economic hurdle theta."""

    volatility_multiplier_k: float
    baseline_hurdle_b: float


HURDLE_REGISTRY: dict[HorizonId, HurdleParameters] = {
    HorizonId.SCALP_15M: HurdleParameters(volatility_multiplier_k=0.25, baseline_hurdle_b=0.001),
    HorizonId.SWING_24H: HurdleParameters(volatility_multiplier_k=0.35, baseline_hurdle_b=0.003),
    HorizonId.POSITION_30D: HurdleParameters(volatility_multiplier_k=0.50, baseline_hurdle_b=0.015),
}


def compute_economic_hurdle(
    horizon: HorizonId,
    cost_hurdle_log_return: float,
    trailing_volatility_estimate: float,
) -> float:
    """Compute frozen symmetric hurdle theta_{h,t} = max(c_{h,t}, k_h * sigma_hat_{h,t}, b_h).

    c: registered round-trip cost hurdle (log-return units)
    sigma_hat: trailing-only horizon-volatility estimate
    """
    params = HURDLE_REGISTRY[horizon]
    vol_component = params.volatility_multiplier_k * max(0.0, trailing_volatility_estimate)
    theta = max(cost_hurdle_log_return, vol_component, params.baseline_hurdle_b)
    return theta


def classify_realized_outcome(
    entry_price_usd: float,
    exit_price_usd: float,
    hurdle_theta: float,
) -> OutcomeClass:
    """Classify realized market outcome based on log return r_h = ln(P_exit / P_entry).

    UP if r_h > theta
    DOWN if r_h < -theta
    FLAT otherwise
    """
    if entry_price_usd <= 0.0 or exit_price_usd <= 0.0:
        return OutcomeClass.FLAT

    log_return = math.log(exit_price_usd / entry_price_usd)
    if log_return > hurdle_theta:
        return OutcomeClass.UP
    if log_return < -hurdle_theta:
        return OutcomeClass.DOWN
    return OutcomeClass.FLAT
