"""Constrained Log-Opinion Pooling Fusion Engine.

Enforces PRD Section 3.10:
- Baseline horizon priors:
    Scalp 15m:    Macro 0.05, ETF 0.05, News 0.15, Technical 0.75
    Swing 24h:    Macro 0.20, ETF 0.25, News 0.20, Technical 0.35
    Position 30d: Macro 0.30, ETF 0.30, News 0.15, Technical 0.25
- Learned weights beta_{m,h} constrained to [0.5 * b_{m,h}, 1.5 * b_{m,h}] with sum(beta) = 1.0.
- Quality-weighted operational weights:
    w_{m,h,t} = (beta_{m,h} * q_{m,t}) / sum_j(beta_{j,h} * q_{j,t})
- Late fusion logits and temperature softmax:
    z_k = a_{h,k} + sum_m w_{m,h,t} * log(max(p_{m,k}, 1e-6))
    p_k = softmax(z / T_h)_k
- Approved degraded mode masks: FULL, CORE_NO_ETF, TECH_MACRO
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize

from src.domain.errors import InvariantViolationError
from src.domain.features import PillarType, SourceCoverageMode
from src.domain.identity import HorizonId
from src.domain.predictions import ProbabilityVector

logger = logging.getLogger(__name__)


class DegradedCoverageError(InvariantViolationError):
    """Raised when the active mask cannot be served without synthesizing evidence."""

# Baseline product priors from PRD Section 3.10
BASELINE_PRIORS: dict[HorizonId, dict[PillarType, float]] = {
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

# Permitted active pillars per source coverage mode
MODE_ACTIVE_PILLARS: dict[SourceCoverageMode, tuple[PillarType, ...]] = {
    SourceCoverageMode.FULL: (
        PillarType.TECHNICAL,
        PillarType.MACRO,
        PillarType.ETF,
        PillarType.NEWS,
    ),
    SourceCoverageMode.CORE_NO_ETF: (
        PillarType.TECHNICAL,
        PillarType.MACRO,
        PillarType.NEWS,
    ),
    SourceCoverageMode.TECH_MACRO: (
        PillarType.TECHNICAL,
        PillarType.MACRO,
    ),
    SourceCoverageMode.RESEARCH: (
        PillarType.TECHNICAL,
        PillarType.MACRO,
        PillarType.ETF,
        PillarType.NEWS,
    ),
}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return cast(np.ndarray, exp_vals / np.sum(exp_vals, axis=-1, keepdims=True))


class LogOpinionPoolFusion:
    """Implements constrained log-opinion pooling with quality modulation
    and temperature scaling.
    """

    def __init__(
        self,
        horizon: HorizonId,
        mode: SourceCoverageMode = SourceCoverageMode.FULL,
        learned_weights: Mapping[PillarType, float] | None = None,
        intercepts: Sequence[float] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._horizon = horizon
        self._mode = mode
        self._temperature = max(0.01, float(temperature))
        default_intercepts = [0.0, 0.0, 0.0]
        self._intercepts = np.array(
            intercepts if intercepts is not None else default_intercepts,
            dtype=np.float64,
        )

        active_pillars = MODE_ACTIVE_PILLARS.get(mode, MODE_ACTIVE_PILLARS[SourceCoverageMode.FULL])
        base_priors = BASELINE_PRIORS[horizon]

        # Filter base priors for active pillars in this mode and normalize
        active_base = {p: base_priors[p] for p in active_pillars}
        active_sum = sum(active_base.values())
        norm_base = {p: v / active_sum for p, v in active_base.items()}

        self._fitted = learned_weights is not None
        if learned_weights is not None:
            # Validate bounds: beta in [0.5 * b, 1.5 * b]
            weights_dict: dict[PillarType, float] = {}
            for p in active_pillars:
                val = learned_weights.get(p, norm_base[p])
                b = norm_base[p]
                # Allow tiny numerical tolerance of 1e-4 on bounds
                if not (0.5 * b - 1e-4 <= val <= 1.5 * b + 1e-4):
                    raise InvariantViolationError(
                        f"Weight for {p} ({val:.4f}) outside permitted bound "
                        f"[{0.5 * b:.4f}, {1.5 * b:.4f}]"
                    )
                weights_dict[p] = float(val)

            # Re-normalize to exact sum 1.0 if not already normalized
            total = sum(weights_dict.values())
            if abs(total - 1.0) > 1e-12 and total > 0.0:
                self._weights = {p: v / total for p, v in weights_dict.items()}
            else:
                self._weights = weights_dict
        else:
            self._weights = norm_base

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def horizon(self) -> HorizonId:
        return self._horizon

    @property
    def mode(self) -> SourceCoverageMode:
        return self._mode

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Temperature must be positive")
        self._temperature = max(0.01, float(value))

    @property
    def weights(self) -> dict[PillarType, float]:
        return dict(self._weights)

    @property
    def intercepts(self) -> np.ndarray:
        return self._intercepts.copy()

    def compute_operational_weights(
        self,
        quality_factors: Mapping[PillarType, float],
    ) -> dict[PillarType, float]:
        """Compute operational weights w_{m,h,t} modulated by quality factor q_{m,t} in [0, 1].

        w_{m,h,t} = (beta_{m,h} * q_{m,t}) / sum_j(beta_{j,h} * q_{j,t})
        """
        raw_weights: dict[PillarType, float] = {}
        for pillar, beta in self._weights.items():
            q = max(0.0, min(1.0, quality_factors.get(pillar, 0.0)))
            raw_weights[pillar] = beta * q

        total = sum(raw_weights.values())
        if total <= 1e-9:
            raise DegradedCoverageError(
                f"Total operational weight across active pillars is {total:.2e} <= 1e-9; "
                "cannot serve without synthesizing evidence"
            )

        return {p: w / total for p, w in raw_weights.items()}

    def compute_fusion_logits(
        self,
        pillar_probs: Mapping[PillarType, ProbabilityVector],
        quality_factors: Mapping[PillarType, float],
    ) -> tuple[np.ndarray, dict[PillarType, float]]:
        """Compute pre-temperature fusion logits z and operational weights w.

        z_k = a_{h,k} + sum_m w_{m,h,t} * log(max(p_{m,k}, 1e-6))
        Classes: 0 -> DOWN, 1 -> FLAT, 2 -> UP
        """
        op_weights = self.compute_operational_weights(quality_factors)
        z = self._intercepts.copy()

        for pillar, w in op_weights.items():
            if w <= 1e-9:
                continue
            if pillar not in pillar_probs:
                raise DegradedCoverageError(
                    f"Required active pillar '{pillar.value}' with weight {w:.4f} "
                    "is missing from pillar probabilities"
                )
            p_vec = pillar_probs[pillar]
            probs_arr = np.array([p_vec.p_down, p_vec.p_flat, p_vec.p_up], dtype=np.float64)
            log_p = np.log(np.maximum(probs_arr, 1e-6))
            z += w * log_p

        return z, op_weights

    def fuse(
        self,
        pillar_probs: Mapping[PillarType, ProbabilityVector],
        quality_factors: Mapping[PillarType, float],
    ) -> ProbabilityVector:
        """Execute full late-fusion pipeline and return calibrated ProbabilityVector."""
        z, _ = self.compute_fusion_logits(pillar_probs, quality_factors)
        # Scale by temperature T_h
        scaled_z = z / self._temperature
        probs = softmax(scaled_z)

        p_down = float(probs[0])
        p_flat = float(probs[1])
        p_up = float(probs[2])

        # Numerical adjustment to ensure strict sum to 1.0 within 1e-5
        total = p_down + p_flat + p_up
        if total > 0:
            p_down /= total
            p_flat /= total
            p_up = 1.0 - p_down - p_flat

        return ProbabilityVector(p_up=p_up, p_flat=p_flat, p_down=p_down)

    def fit_weights(
        self,
        oof_pillar_probs: Mapping[PillarType, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Fit regularized fusion weights on out-of-fold predictions subject to PRD constraints."""
        active_pillars = list(self._weights.keys())
        base_weights = np.array([self._weights[p] for p in active_pillars], dtype=np.float64)

        # Bounds: [0.5 * b_i, 1.5 * b_i]
        bounds = [(0.5 * b, 1.5 * b) for b in base_weights]

        def objective(weights: np.ndarray) -> float:
            w_norm = weights / np.sum(weights)
            # Log pooling cross-entropy
            # Shape of oof_pillar_probs[p]: (N, 3)
            log_p_weighted = np.zeros_like(oof_pillar_probs[active_pillars[0]])
            for i, p in enumerate(active_pillars):
                p_arr = np.maximum(oof_pillar_probs[p], 1e-6)
                log_p_weighted += w_norm[i] * np.log(p_arr)

            # Softmax
            shifted = log_p_weighted - np.max(log_p_weighted, axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            p_fusion = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

            # Cross-entropy loss + L2 regularization toward prior
            n_samples = len(y_true)
            rows = np.arange(n_samples)
            log_likelihood = np.log(np.maximum(p_fusion[rows, y_true], 1e-12))
            nll = -np.mean(log_likelihood)
            l2_reg = 0.5 * np.sum((weights - base_weights) ** 2)
            return float(nll + l2_reg)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        res = minimize(
            objective,
            x0=base_weights,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        if res.success:
            fitted_w = res.x / np.sum(res.x)
            self._weights = {p: float(fitted_w[i]) for i, p in enumerate(active_pillars)}
            self._fitted = True
        else:
            self._fitted = False
            logger.warning(
                "Fusion weight optimization failed (%s). Retaining baseline weights.",
                res.message,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self._horizon.value,
            "mode": self._mode.value,
            "learned_weights": {k.value: v for k, v in self._weights.items()},
            "intercepts": self._intercepts.tolist(),
            "temperature": self._temperature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogOpinionPoolFusion":
        weights = {PillarType(k): float(v) for k, v in data["learned_weights"].items()}
        return cls(
            horizon=HorizonId(data["horizon"]),
            mode=SourceCoverageMode(data["mode"]),
            learned_weights=weights,
            intercepts=data.get("intercepts"),
            temperature=float(data.get("temperature", 1.0)),
        )
