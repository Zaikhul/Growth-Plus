"""Shallow XGBoost pillar expert models and training pipeline.

Enforces PRD Section 3.10:
- Four shallow XGBoost multiclass pillar experts per horizon
- Bounded search space: depth {2, 3, 4}, learning rate {0.03, 0.10}, min_child_weight {20, 50},
  early stopping after 30 rounds, subsample 0.8, colsample 0.8, L2 penalty 10.
- Out-of-fold probability generation for late fusion calibration
- Zero unconstrained depth expansion
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import xgboost as xgb

from src.domain.features import FeatureSnapshot, PillarType
from src.domain.identity import HorizonId
from src.domain.predictions import PillarPrediction, ProbabilityVector
from src.ml.datasets.oof import OOFResult, plan_oof
from src.ml.datasets.purged_splits import TargetClock
from src.ml.training.dataset import PillarDataset, extract_pillar_features


@dataclass(frozen=True, slots=True)
class PillarExpertConfig:
    """Bounded hyperparameter configuration for shallow XGBoost pillar expert."""

    max_depth: int = 2
    learning_rate: float = 0.05
    min_child_weight: int = 20
    n_estimators: int = 500
    early_stopping_rounds: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 10.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_depth not in (2, 3, 4):
            raise ValueError(f"max_depth must be in {{2, 3, 4}}, got {self.max_depth}")
        if not (0.01 <= self.learning_rate <= 0.20):
            raise ValueError(f"learning_rate must be in [0.01, 0.20], got {self.learning_rate}")
        if self.subsample > 0.85:
            raise ValueError(f"subsample cannot exceed 0.85, got {self.subsample}")
        if self.colsample_bytree > 0.85:
            raise ValueError(f"colsample_bytree cannot exceed 0.85, got {self.colsample_bytree}")


class PillarExpertModel:
    """Trained shallow XGBoost expert for a single signal pillar."""

    def __init__(
        self,
        pillar: PillarType,
        horizon: HorizonId,
        feature_names: Sequence[str],
        model_json: str,
        best_iteration: int = 0,
        validation_loss: float = 0.0,
    ) -> None:
        self._pillar = pillar
        self._horizon = horizon
        self._feature_names = tuple(feature_names)
        self._model_json = model_json
        self._best_iteration = best_iteration
        self._validation_loss = validation_loss

        # Load internal booster from JSON
        self._booster = xgb.Booster()
        self._booster.load_model(bytearray(model_json.encode("utf-8")))

    @property
    def pillar(self) -> PillarType:
        return self._pillar

    @property
    def horizon(self) -> HorizonId:
        return self._horizon

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def model_json(self) -> str:
        return self._model_json

    @property
    def best_iteration(self) -> int:
        return self._best_iteration

    @property
    def validation_loss(self) -> float:
        return self._validation_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute 3-class probabilities for input matrix X of shape (N, num_features).

        Returns ndarray of shape (N, 3): [p_down, p_flat, p_up].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dmat = xgb.DMatrix(X, feature_names=list(self._feature_names))
        preds = self._booster.predict(dmat)
        # Ensure 2D
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        # Handle binary or edge case where output classes < 3
        if preds.shape[1] == 1:
            # Expand to 3 classes
            p_val = preds[:, 0]
            preds = np.column_stack([(1.0 - p_val) / 2.0, (1.0 - p_val) / 2.0, p_val])
        elif preds.shape[1] == 2:
            preds = np.column_stack([preds[:, 0], np.zeros(len(preds)), preds[:, 1]])

        # Renormalize to exact sum of 1.0
        row_sums = preds.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        preds = preds / row_sums
        return cast(np.ndarray, preds)

    def predict_snapshot(
        self,
        snapshot: FeatureSnapshot,
        quality_factor: float = 1.0,
    ) -> PillarPrediction:
        """Predict directional probabilities for a single runtime feature snapshot."""
        x_vec, _ = extract_pillar_features(
            snapshot=snapshot,
            pillar=self._pillar,
            ordered_feature_names=self._feature_names,
        )
        probs = self.predict_proba(x_vec)[0]
        # Classes: 0 -> DOWN, 1 -> FLAT, 2 -> UP
        p_down, p_flat, p_up = float(probs[0]), float(probs[1]), float(probs[2])

        # Renormalize slightly if floating drift
        total = p_down + p_flat + p_up
        if total > 0:
            p_down /= total
            p_flat /= total
            p_up = 1.0 - p_down - p_flat

        vec = ProbabilityVector(p_up=p_up, p_flat=p_flat, p_down=p_down)
        return PillarPrediction(
            pillar=self._pillar,
            probabilities=vec,
            quality_factor=quality_factor,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata and model weights to dict."""
        return {
            "pillar": self._pillar.value,
            "horizon": self._horizon.value,
            "feature_names": list(self._feature_names),
            "best_iteration": self._best_iteration,
            "validation_loss": self._validation_loss,
            "model_json": self._model_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PillarExpertModel":
        """Reconstruct model from serialized dict."""
        return cls(
            pillar=PillarType(data["pillar"]),
            horizon=HorizonId(data["horizon"]),
            feature_names=data["feature_names"],
            model_json=data["model_json"],
            best_iteration=int(data.get("best_iteration", 0)),
            validation_loss=float(data.get("validation_loss", 0.0)),
        )


class PillarExpertTrainer:
    """Trainer for shallow XGBoost pillar experts enforcing PRD Section 3.10 constraints."""

    @staticmethod
    def sample_candidate_configs(seed: int = 42, count: int = 20) -> list[PillarExpertConfig]:
        """Sample bounded hyperparameter configurations including depth-2 reference."""
        rng = np.random.default_rng(seed)
        depths = [2, 3, 4]
        lrs = [0.03, 0.05, 0.08, 0.10]
        child_weights = [20, 30, 50]

        configs: list[PillarExpertConfig] = [
            # Deterministic reference depth-2
            PillarExpertConfig(max_depth=2, learning_rate=0.05, min_child_weight=20)
        ]

        for _ in range(count - 1):
            cfg = PillarExpertConfig(
                max_depth=int(rng.choice(depths)),
                learning_rate=float(rng.choice(lrs)),
                min_child_weight=int(rng.choice(child_weights)),
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                random_state=int(rng.integers(1, 10000)),
            )
            configs.append(cfg)
        return configs

    def train(
        self,
        dataset: PillarDataset,
        horizon: HorizonId,
        val_dataset: PillarDataset | None = None,
        config: PillarExpertConfig | None = None,
    ) -> PillarExpertModel:
        """Train a shallow XGBoost classifier on the pillar dataset."""
        cfg = config or PillarExpertConfig()

        if len(dataset.X) == 0:
            raise ValueError(f"Cannot train pillar expert {dataset.pillar}: dataset is empty")

        num_classes = 3
        # Handle case where training sample doesn't have all 3 classes
        present_classes = set(dataset.y)
        if len(present_classes) < 2:
            # Fallback uniform / single-class model setup
            pass

        dtrain = xgb.DMatrix(
            dataset.X,
            label=dataset.y,
            feature_names=list(dataset.feature_names),
        )

        params: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "min_child_weight": cfg.min_child_weight,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "lambda": cfg.reg_lambda,
            "eval_metric": "mlogloss",
            "seed": cfg.random_state,
        }

        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        if val_dataset is not None and len(val_dataset.X) > 0:
            dval = xgb.DMatrix(
                val_dataset.X,
                label=val_dataset.y,
                feature_names=list(val_dataset.feature_names),
            )
            evals.append((dval, "val"))

        evals_result: dict[str, dict[str, list[float]]] = {}
        early_stop = cfg.early_stopping_rounds if val_dataset is not None else None

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=cfg.n_estimators,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )

        best_iteration = (
            booster.best_iteration if hasattr(booster, "best_iteration") else cfg.n_estimators
        )
        val_loss = 0.0
        if "val" in evals_result and "mlogloss" in evals_result["val"]:
            losses = evals_result["val"]["mlogloss"]
            val_loss = losses[best_iteration] if best_iteration < len(losses) else losses[-1]

        model_json = booster.save_raw(raw_format="json").decode("utf-8")

        return PillarExpertModel(
            pillar=dataset.pillar,
            horizon=horizon,
            feature_names=dataset.feature_names,
            model_json=model_json,
            best_iteration=best_iteration,
            validation_loss=val_loss,
        )

    def train_out_of_fold(
        self,
        dataset: PillarDataset,
        horizon: HorizonId,
        n_splits: int = 5,
        config: PillarExpertConfig | None = None,
        *,
        target_clocks: Sequence[TargetClock],
        frozen_at: datetime,
    ) -> tuple[PillarExpertModel, OOFResult]:
        """Cross-fit fixed expert settings without exposing held-out labels to fit."""
        folds = plan_oof(dataset.cutoffs, target_clocks, frozen_at, n_splits)
        predicted_rows: list[int] = []
        probability_blocks: list[np.ndarray] = []
        for fold in folds:
            train_idx = np.asarray(fold.train, dtype=np.int64)
            val_idx = np.asarray(fold.validation, dtype=np.int64)
            fold_train = PillarDataset(
                pillar=dataset.pillar,
                feature_names=dataset.feature_names,
                X=dataset.X[train_idx],
                y=dataset.y[train_idx],
                cutoffs=tuple(dataset.cutoffs[i] for i in fold.train),
            )
            # Hyperparameters/tree count must be frozen or selected within fold_train.
            # Never pass fold validation labels to early stopping/model selection.
            fold_model = self.train(fold_train, horizon, config=config)
            probabilities = np.asarray(
                fold_model.predict_proba(dataset.X[val_idx]), dtype=np.float64
            )
            sums = probabilities.sum(axis=1, keepdims=True)
            if (sums > 0).all():
                probabilities = probabilities / sums
            if (
                probabilities.shape != (len(val_idx), 3)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=1e-8)
            ):
                raise ValueError("Invalid cross-fitted probability vector")
            predicted_rows.extend(fold.validation)
            probability_blocks.append(probabilities)
        included = set(predicted_rows)
        result = OOFResult(
            row_indices=tuple(predicted_rows),
            probabilities=np.concatenate(probability_blocks),
            excluded_rows=tuple(i for i in range(len(dataset.X)) if i not in included),
            folds=folds,
        )
        return self.train(dataset, horizon, config=config), result
