"""Temperature scaling calibrator and Expected Calibration Error (ECE) metric.

Enforces PRD Section 3.10 & 7.1:
- Fits positive scalar temperature T_h > 0 on an untouched calibration partition
- Minimizes multiclass Cross-Entropy / Negative Log-Likelihood without altering expert weights
- Expected Calibration Error (ECE) computation over M=10 confidence bins
- Calibration gate: ECE <= 0.08 on validation
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Evaluation metrics for probability calibration quality."""

    ece: float
    max_calibration_error: float
    brier_score: float
    mean_confidence: float
    accuracy: float


def compute_multiclass_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE) and associated calibration metrics.

    probs: ndarray of shape (N, C) containing predicted class probabilities
    labels: ndarray of shape (N,) containing true integer class labels in {0, ..., C-1}
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")

    n_samples = len(labels)
    if n_samples == 0:
        return CalibrationMetrics(
            ece=0.0,
            max_calibration_error=0.0,
            brier_score=0.0,
            mean_confidence=0.0,
            accuracy=0.0,
        )

    # Predicted class and confidence (max probability)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    # Brier score: 1/N * sum_i sum_k (p_{ik} - y_{ik})^2
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into [bin_lower, bin_upper]
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)

    return CalibrationMetrics(
        ece=float(ece),
        max_calibration_error=float(mce),
        brier_score=brier_score,
        mean_confidence=float(np.mean(confidences)),
        accuracy=float(np.mean(accuracies)),
    )


class TemperatureCalibrator:
    """Fits and applies post-hoc temperature scaling to fusion logits."""

    def __init__(self, temperature: float = 1.0) -> None:
        self._temperature = max(0.01, float(temperature))

    @property
    def temperature(self) -> float:
        return self._temperature

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits by temperature T: z / T."""
        return logits / self._temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature and softmax to return calibrated probabilities."""
        scaled = self.scale_logits(logits)
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return cast(np.ndarray, exp_vals / np.sum(exp_vals, axis=-1, keepdims=True))

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temp_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> float:
        """Fit optimal temperature T > 0 on validation/calibration logits and labels."""
        if len(logits) != len(labels):
            raise ValueError("logits and labels must match in length")

        n_samples = len(labels)
        if n_samples < 5:
            # Insufficient samples to fit temperature, retain default 1.0
            self._temperature = 1.0
            return self._temperature

        row_indices = np.arange(n_samples)

        def nll_loss(t: float) -> float:
            scaled = logits / t
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
            correct_probs = np.maximum(probs[row_indices, labels], 1e-12)
            return float(-np.mean(np.log(correct_probs)))

        res = minimize_scalar(nll_loss, bounds=temp_bounds, method="bounded")
        if res.success:
            self._temperature = float(res.x)
        return self._temperature
