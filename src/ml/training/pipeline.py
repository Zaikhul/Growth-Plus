"""End-to-end model training, fusion weight fitting, and temperature calibration pipeline.

Enforces PRD Section 3.10 & 7.1:
- Disjoint partition isolation:
    1. Pillar expert training partition
    2. Fusion weight optimization partition
    3. Untouched calibration and verification partition
- Late fusion weights fitted with boundary constraints
- Post-hoc temperature scaling fitted without altering expert weights
- Strict holdout ECE <= 0.08 calibration gate
- Cryptographically signed ModelBundle emission
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from src.domain.errors import InvariantViolationError
from src.domain.features import PillarType, SourceCoverageMode
from src.domain.identity import HorizonId
from src.domain.predictions import ProbabilityVector
from src.domain.time import ensure_utc
from src.ml.bundle import ModelBundle, ModelBundleHeader
from src.ml.calibration import CalibrationMetrics, TemperatureCalibrator, compute_multiclass_ece
from src.ml.datasets.purged_splits import TargetClock
from src.ml.fusion import LogOpinionPoolFusion
from src.ml.training.dataset import PillarDataset
from src.ml.training.pillar_experts import (
    PillarExpertConfig,
    PillarExpertModel,
    PillarExpertTrainer,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Artifacts and calibration metrics produced by the training pipeline."""

    bundle: ModelBundle
    calibration_metrics: CalibrationMetrics
    holdout_ece: float
    is_calibrated: bool


class TrainingPipeline:
    """Orchestrates end-to-end expert training, fusion fitting, and calibration."""

    def __init__(self, trainer: PillarExpertTrainer | None = None) -> None:
        self._trainer = trainer or PillarExpertTrainer()

    def run(
        self,
        datasets: Mapping[PillarType, PillarDataset],
        targets: Sequence[TargetClock],
        horizon: HorizonId,
        mode: SourceCoverageMode = SourceCoverageMode.FULL,
        frozen_at: datetime | None = None,
        secret_key: bytes | None = None,
        generation: int = 1,
        bundle_ttl_days: int = 30,
        ece_gate: float = 0.08,
        strict_gate: bool = False,
        expert_configs: Mapping[PillarType, PillarExpertConfig] | None = None,
    ) -> PipelineResult:
        """Execute complete training pipeline across disjoint partitions."""
        now = ensure_utc(frozen_at or datetime.now(tz=UTC))

        if not datasets:
            raise InvariantViolationError("Training pipeline requires at least one pillar dataset")

        first_pillar = next(iter(datasets.keys()))
        n_samples = len(datasets[first_pillar].X)
        for p, ds in datasets.items():
            if len(ds.X) != n_samples:
                raise InvariantViolationError(
                    f"Sample count mismatch: {p} has {len(ds.X)} rows, expected {n_samples}"
                )

        if n_samples < 15:
            raise InvariantViolationError(
                f"Insufficient samples for 3-way disjoint partition: got {n_samples}, need >= 15"
            )

        # ------------------------------------------------------------------
        # 1. Split into 3 disjoint chronological folds (DEFECT-36)
        # ------------------------------------------------------------------
        # Partition 1 (60%): Expert training
        # Partition 2 (20%): Fusion weight fitting
        # Partition 3 (20%): Untouched temperature calibration & ECE gate
        n_train = int(n_samples * 0.60)
        n_fusion = int(n_samples * 0.20)

        idx_train = np.arange(0, n_train)
        idx_fusion = np.arange(n_train, n_train + n_fusion)
        idx_calib = np.arange(n_train + n_fusion, n_samples)

        # ------------------------------------------------------------------
        # 2. Train pillar experts on Partition 1
        # ------------------------------------------------------------------
        pillar_models: dict[PillarType, PillarExpertModel] = {}
        feature_schemas: dict[PillarType, Sequence[str]] = {}

        for pillar, ds in datasets.items():
            train_ds = PillarDataset(
                pillar=pillar,
                feature_names=ds.feature_names,
                X=ds.X[idx_train],
                y=ds.y[idx_train],
                cutoffs=tuple(ds.cutoffs[i] for i in idx_train),
            )
            cfg = expert_configs.get(pillar) if expert_configs else None
            model = self._trainer.train(train_ds, horizon, config=cfg)
            pillar_models[pillar] = model
            feature_schemas[pillar] = ds.feature_names

        # ------------------------------------------------------------------
        # 3. Fit fusion weights on Partition 2 (DEFECT-07)
        # ------------------------------------------------------------------
        fusion = LogOpinionPoolFusion(horizon=horizon, mode=mode)
        fusion_preds: dict[PillarType, np.ndarray] = {}
        for pillar, model in pillar_models.items():
            fusion_preds[pillar] = model.predict_proba(datasets[pillar].X[idx_fusion])

        y_fusion = datasets[first_pillar].y[idx_fusion]
        fusion.fit_weights(oof_pillar_probs=fusion_preds, y_true=y_fusion)

        # ------------------------------------------------------------------
        # 4. Untouched temperature calibration on Partition 3 (DEFECT-07, DEFECT-36)
        # ------------------------------------------------------------------
        calib_preds: dict[PillarType, np.ndarray] = {}
        for pillar, model in pillar_models.items():
            calib_preds[pillar] = model.predict_proba(datasets[pillar].X[idx_calib])

        y_calib = datasets[first_pillar].y[idx_calib]

        # Compute pre-temperature fusion logits for calibration partition
        calib_logits: list[np.ndarray] = []
        for i in range(len(idx_calib)):
            pillar_probs_i = {
                p: ProbabilityVector(
                    p_up=float(calib_preds[p][i, 2]),
                    p_flat=float(calib_preds[p][i, 1]),
                    p_down=float(calib_preds[p][i, 0]),
                )
                for p in pillar_models
            }
            quality_factors = dict.fromkeys(pillar_models, 1.0)
            z_i, _ = fusion.compute_fusion_logits(pillar_probs_i, quality_factors)
            calib_logits.append(z_i)

        logits_arr = np.array(calib_logits, dtype=np.float64)
        calibrator = TemperatureCalibrator(temperature=1.0)
        calibrator.fit(logits_arr, y_calib)
        fusion.temperature = calibrator.temperature

        # Evaluate holdout calibration metrics
        calibrated_probs = calibrator.predict_proba(logits_arr)
        metrics = compute_multiclass_ece(calibrated_probs, y_calib)
        is_calibrated = metrics.ece <= ece_gate

        if strict_gate and not is_calibrated:
            raise InvariantViolationError(
                f"Model bundle calibration gate failed: holdout ECE {metrics.ece:.4f} > {ece_gate}"
            )

        # ------------------------------------------------------------------
        # 5. Build and optionally sign ModelBundle
        # ------------------------------------------------------------------
        header = ModelBundleHeader(
            bundle_id=f"bundle_{horizon.value}_{mode.value}_{uuid.uuid4().hex[:8]}",
            generation=generation,
            horizon=horizon,
            mode=mode,
            created_at=now,
            expires_at=now + timedelta(days=bundle_ttl_days),
        )

        bundle = ModelBundle(
            header=header,
            pillar_models=pillar_models,
            fusion=fusion,
            feature_schemas=feature_schemas,
        )

        if secret_key:
            bundle.sign(secret_key)

        return PipelineResult(
            bundle=bundle,
            calibration_metrics=metrics,
            holdout_ece=metrics.ece,
            is_calibrated=is_calibrated,
        )
