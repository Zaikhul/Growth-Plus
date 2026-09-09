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

from src.domain.errors import EmptyPartitionError, InvariantViolationError
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
        strict_gate: bool = True,
        expert_configs: Mapping[PillarType, PillarExpertConfig] | None = None,
    ) -> PipelineResult:
        """Execute complete training pipeline across disjoint partitions."""
        now = ensure_utc(frozen_at or datetime.now(tz=UTC))

        if not datasets:
            raise InvariantViolationError("Training pipeline requires at least one pillar dataset")

        first_pillar = next(iter(datasets.keys()))
        n_samples = len(datasets[first_pillar].X)
        cutoffs = datasets[first_pillar].cutoffs
        for p, ds in datasets.items():
            if len(ds.X) != n_samples:
                raise InvariantViolationError(
                    f"Sample count mismatch: {p} has {len(ds.X)} rows, expected {n_samples}"
                )
            if len(ds.cutoffs) != n_samples or ds.cutoffs != cutoffs:
                raise InvariantViolationError(
                    f"Cutoffs mismatch for pillar {p}: cutoffs must be strictly aligned"
                )

        if not targets:
            raise InvariantViolationError("Training pipeline requires non-empty targets")
        if len(targets) != n_samples:
            raise InvariantViolationError(
                f"Targets count mismatch: got {len(targets)} targets for {n_samples} samples"
            )

        # Enforce chronological ordering (DEFECT-02 / T02)
        for i in range(len(cutoffs) - 1):
            if cutoffs[i] > cutoffs[i + 1]:
                raise InvariantViolationError(
                    "Temporal order violation: training cutoffs must be strictly non-decreasing"
                )

        # Enforce target clocks validity and UTC
        for cutoff, target in zip(cutoffs, targets, strict=True):
            for instant in (cutoff, target.entry_at, target.exit_at, target.available_at):
                if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
                    raise InvariantViolationError("Split timestamps must be timezone-aware UTC")
            if not (cutoff <= target.entry_at < target.exit_at <= target.available_at <= now):
                msg = (
                    f"Invalid or unmatured training target clock: cutoff={cutoff}, "
                    f"entry={target.entry_at}, exit={target.exit_at}, "
                    f"available={target.available_at}, frozen_at={now}"
                )
                raise InvariantViolationError(msg)

        if n_samples < 15:
            raise InvariantViolationError(
                f"Insufficient samples for 4-way purged partition: got {n_samples}, need >= 15"
            )

        # ------------------------------------------------------------------
        # 1. Split into disjoint chronological folds with target interval purging
        # ------------------------------------------------------------------
        # Partition 1 (~50%): Expert training (idx_train)
        # Partition 2 (~20%): Fusion weight fitting (idx_fusion)
        # Partition 3 (~15%): Temperature calibration fitting (idx_calib)
        # Partition 4 (~15%): Untouched holdout evaluation & ECE gate (idx_eval)
        n_train = int(n_samples * 0.50)
        n_fusion = int(n_samples * 0.20)
        n_calib = int(n_samples * 0.15)

        idx_train_raw = np.arange(0, n_train)
        idx_fusion_raw = np.arange(n_train, n_train + n_fusion)
        idx_calib_raw = np.arange(n_train + n_fusion, n_train + n_fusion + n_calib)
        idx_eval = np.arange(n_train + n_fusion + n_calib, n_samples)

        fusion_start = cutoffs[n_train]
        calib_start = cutoffs[n_train + n_fusion]
        eval_start = cutoffs[n_train + n_fusion + n_calib]

        # Purge boundaries: target exit and availability must precede next era start
        purged_train = [
            int(i)
            for i in idx_train_raw
            if (
                targets[int(i)].exit_at < fusion_start
                and targets[int(i)].available_at < fusion_start
            )
        ]
        if not purged_train:
            raise EmptyPartitionError(
                "Purged expert training partition (Partition 1) contains zero valid samples"
            )
        idx_train = np.array(purged_train)

        purged_fusion = [
            int(i)
            for i in idx_fusion_raw
            if (
                targets[int(i)].exit_at < calib_start and targets[int(i)].available_at < calib_start
            )
        ]
        if not purged_fusion:
            raise EmptyPartitionError(
                "Purged fusion weight fitting partition (Partition 2) contains zero valid samples"
            )
        idx_fusion = np.array(purged_fusion)

        purged_calib = [
            int(i)
            for i in idx_calib_raw
            if (targets[int(i)].exit_at < eval_start and targets[int(i)].available_at < eval_start)
        ]
        if not purged_calib:
            raise EmptyPartitionError(
                "Purged calibration fitting partition (Partition 3) contains zero valid samples"
            )
        idx_calib = np.array(purged_calib)

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
        # 4. Temperature calibration fitting on Partition 3 (DEFECT-07, DEFECT-36)
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

        # ------------------------------------------------------------------
        # 5. Evaluate holdout calibration metrics strictly on isolated Partition 4 (T02)
        # ------------------------------------------------------------------
        eval_preds: dict[PillarType, np.ndarray] = {}
        for pillar, model in pillar_models.items():
            eval_preds[pillar] = model.predict_proba(datasets[pillar].X[idx_eval])

        y_eval = datasets[first_pillar].y[idx_eval]

        eval_logits: list[np.ndarray] = []
        for i in range(len(idx_eval)):
            pillar_probs_i = {
                p: ProbabilityVector(
                    p_up=float(eval_preds[p][i, 2]),
                    p_flat=float(eval_preds[p][i, 1]),
                    p_down=float(eval_preds[p][i, 0]),
                )
                for p in pillar_models
            }
            quality_factors = dict.fromkeys(pillar_models, 1.0)
            z_i, _ = fusion.compute_fusion_logits(pillar_probs_i, quality_factors)
            eval_logits.append(z_i)

        eval_logits_arr = np.array(eval_logits, dtype=np.float64)
        calibrated_eval_probs = calibrator.predict_proba(eval_logits_arr)
        metrics = compute_multiclass_ece(calibrated_eval_probs, y_eval)
        is_calibrated = metrics.ece <= ece_gate

        if strict_gate and not is_calibrated:
            raise InvariantViolationError(
                f"Model bundle calibration gate failed: holdout ECE {metrics.ece:.4f} > {ece_gate}"
            )

        # ------------------------------------------------------------------
        # 6. Build and optionally sign ModelBundle
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
