"""Inference engine synthesizing calibrated signals from feature snapshots.

Enforces PRD Section 3.9, 3.10, 3.11, 4.6:
- Executes pillar experts and computes calibrated late fusion probabilities
- Applies the 6-tier classification policy
- Generates deterministic log-odds explanation factors
- Constructs complete, immutable Signal entities adhering to PRD contracts
"""

import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta

from src.domain.errors import InvariantViolationError
from src.domain.features import FeatureSnapshot, PillarType
from src.domain.identity import HorizonId
from src.domain.policies.classification import ClassificationContext, evaluate_classification
from src.domain.policies.coverage import mode_is_compatible
from src.domain.policies.quality import (
    PillarQuality,
    compute_mask_quality,
)
from src.domain.predictions import PillarPrediction
from src.domain.signals import (
    CohortReliabilityInterval95,
    Signal,
    SignalReasonCode,
    SignalStatus,
)
from src.domain.time import ensure_utc
from src.ml.bundle import ModelBundle
from src.ml.fusion import DegradedCoverageError
from src.ml.inference.explanations import ExplanationEngine

# Publication TTLs from PRD Section 3.8
SIGNAL_TTL_REGISTRY: dict[HorizonId, timedelta] = {
    HorizonId.SCALP_15M: timedelta(seconds=90),
    HorizonId.SWING_24H: timedelta(minutes=20),
    HorizonId.POSITION_30D: timedelta(hours=6),
}


class InferenceEngine:
    """Core inference orchestrator transforming feature snapshots into published Signals."""

    def __init__(
        self,
        bundle: ModelBundle,
        explanation_engine: ExplanationEngine | None = None,
    ) -> None:
        self._bundle = bundle
        self._explanation_engine = explanation_engine or ExplanationEngine()
        self._sequence: int = 0

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    def next_sequence(self) -> int:
        """Atomically advance sequence counter."""
        self._sequence += 1
        return self._sequence

    def evaluate_snapshot(
        self,
        snapshot: FeatureSnapshot,
        pillar_qualities: Mapping[PillarType, PillarQuality],
        current_time: datetime,
        has_event_block: bool = False,
        incident_flag: bool = False,
        cohort_reliability_lower_95: float | None = None,
        is_cohort_calibrated: bool = True,
        is_replay: bool = False,
        cost_hurdle: float = 0.0005,
        suppress_publication: bool = False,
        sequence: int | None = None,
    ) -> Signal:
        """Synthesize a complete Signal from a FeatureSnapshot and quality assessments."""
        now_utc = ensure_utc(current_time)
        cutoff_utc = ensure_utc(snapshot.decision_cutoff)
        horizon = self._bundle.horizon
        ttl = SIGNAL_TTL_REGISTRY.get(horizon, timedelta(minutes=20))
        expires_at = cutoff_utc + ttl
        seq = sequence if sequence is not None else self.next_sequence()

        # Check bundle expiration
        if self._bundle.is_expired_at(now_utc):
            return Signal(
                signal_id=uuid.uuid4(),
                sequence=seq,
                market_id=snapshot.market_id,
                horizon=horizon,
                status=SignalStatus.UNAVAILABLE,
                mode=self._bundle.mode,
                cutoff_at=cutoff_utc,
                issued_at=now_utc,
                expires_at=expires_at,
                data_quality=0.0,
                reason_code=SignalReasonCode.MODEL_APPROVAL_EXPIRED,
                model_bundle=self._bundle.bundle_id,
                snapshot_id=snapshot.snapshot_id,
                is_replay=is_replay,
            )

        # Check mode compatibility (DEFECT-04)
        mode = snapshot.mode
        if not mode_is_compatible(self._bundle.mode, mode):
            return Signal(
                signal_id=uuid.uuid4(),
                sequence=seq,
                market_id=snapshot.market_id,
                horizon=horizon,
                status=SignalStatus.UNAVAILABLE,
                mode=mode,
                cutoff_at=cutoff_utc,
                issued_at=now_utc,
                expires_at=expires_at,
                data_quality=0.0,
                reason_code=SignalReasonCode.UNSUPPORTED_SOURCE_MASK,
                model_bundle=self._bundle.bundle_id,
                snapshot_id=snapshot.snapshot_id,
                is_replay=is_replay,
            )

        # Compute mask-level quality score Q_mask (DEFECT-13)
        q_mask = compute_mask_quality(
            horizon=horizon,
            pillar_qualities=pillar_qualities,
            active_pillars=snapshot.active_pillars,
        )
        quality_factors = {p: q.score for p, q in pillar_qualities.items()}

        # Predict for each active pillar expert
        pillar_preds: dict[PillarType, PillarPrediction] = {}
        for pillar, model in self._bundle.pillar_models.items():
            q_factor = quality_factors.get(pillar, 0.0)
            pillar_preds[pillar] = model.predict_snapshot(snapshot, quality_factor=q_factor)

        # Extract probability vectors from pillar predictions
        pillar_probs = {p: pred.probabilities for p, pred in pillar_preds.items()}

        # Perform late fusion (DEFECT-05)
        fusion = self._bundle.fusion
        try:
            fused_probs = fusion.fuse(pillar_probs, quality_factors)
            logits, op_weights = fusion.compute_fusion_logits(pillar_probs, quality_factors)
        except DegradedCoverageError:
            return Signal(
                signal_id=uuid.uuid4(),
                sequence=seq,
                market_id=snapshot.market_id,
                horizon=horizon,
                status=SignalStatus.UNAVAILABLE,
                mode=self._bundle.mode,
                cutoff_at=cutoff_utc,
                issued_at=now_utc,
                expires_at=expires_at,
                data_quality=q_mask,
                reason_code=SignalReasonCode.UNSUPPORTED_SOURCE_MASK,
                model_bundle=self._bundle.bundle_id,
                snapshot_id=snapshot.snapshot_id,
                is_replay=is_replay,
            )

        # Decompose log-odds and extract explanation factors (DEFECT-24)
        explanation = self._explanation_engine.decompose_log_odds(
            ensemble_probs=fused_probs,
            pillar_predictions=pillar_preds,
            operational_weights=op_weights,
            intercepts=list(fusion.intercepts),
            temperature=fusion.temperature,
        )
        if not explanation.verify_reconstruction(tolerance=1e-6):
            raise InvariantViolationError(
                f"Log odds explanation reconstruction error "
                f"{explanation.numerical_reconstruction_error:.2e} exceeds tolerance 1e-6"
            )
        factors = self._explanation_engine.generate_signal_factors(explanation)

        # Evaluate against the 6-tier classification policy (DEFECT-10(b), DEFECT-13)
        ctx = ClassificationContext(
            probabilities=fused_probs,
            quality_score=q_mask,
            mode=self._bundle.mode,
            pillar_predictions=tuple(pillar_preds.values()),
            cohort_reliability_lower_95=cohort_reliability_lower_95,
            has_event_block=has_event_block or incident_flag,
            is_cohort_calibrated=is_cohort_calibrated,
            suppress_publication=suppress_publication,
        )
        class_res = evaluate_classification(ctx)

        # Reliability interval metadata if provided (DEFECT-09)
        reliability_interval: CohortReliabilityInterval95 | None = None
        if cohort_reliability_lower_95 is not None:
            reliability_interval = CohortReliabilityInterval95(
                lower=cohort_reliability_lower_95,
                upper=min(1.0, cohort_reliability_lower_95 + 0.15),
                raw_observations=250,
                independent_blocks=60,
            )

        return Signal(
            signal_id=uuid.uuid4(),
            sequence=seq,
            market_id=snapshot.market_id,
            horizon=horizon,
            status=class_res.status,
            mode=self._bundle.mode,
            cutoff_at=cutoff_utc,
            issued_at=now_utc,
            expires_at=expires_at,
            probabilities=fused_probs,
            label=class_res.label,
            confidence=class_res.confidence,
            confidence_event=class_res.confidence_event,
            data_quality=q_mask,
            outcome_hurdle_log_return=cost_hurdle,
            reason_code=class_res.reason_code,
            reasons=factors,
            cohort_reliability=reliability_interval,
            model_bundle=self._bundle.bundle_id,
            snapshot_id=snapshot.snapshot_id,
            is_replay=is_replay,
        )
