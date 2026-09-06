"""Feature calculation and quantitative signal synthesis worker.

Enforces PRD Section 4.2 & 4.3:
- Runs strictly at closed-bar watermarks and material event cutoffs
- Executes Point-in-Time As-Of Join Engine preventing future data leakage
- Evaluates InferenceEngine and ClassificationPolicy
- Transactionally persists Signal and stages OutboxRecord
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from src.domain.features import PillarType
from src.domain.identity import HorizonId, MarketId
from src.domain.policies.quality import PillarQuality
from src.domain.signals import Signal
from src.features.asof_join import PointInTimeAsOfEngine
from src.ml.inference.engine import InferenceEngine
from src.ports.repositories import OutboxRepository, SignalRepository


class FeatureSignalWorker:
    """Orchestrator worker calculating PIT features and synthesizing validated Signals."""

    def __init__(
        self,
        market_id: MarketId,
        horizon_id: HorizonId,
        asof_engine: PointInTimeAsOfEngine,
        inference_engine: InferenceEngine,
        signal_repo: SignalRepository,
        outbox_repo: OutboxRepository | None = None,
    ) -> None:
        self._market_id = market_id
        self._horizon_id = horizon_id
        self._asof_engine = asof_engine
        self._inference_engine = inference_engine
        self._signal_repo = signal_repo
        self._outbox_repo = outbox_repo

    async def evaluate_decision_slot(
        self,
        cutoff_at: datetime,
        current_time: datetime | None = None,
        pillar_qualities: Mapping[PillarType, PillarQuality] | None = None,
        has_event_block: bool = False,
        cohort_reliability_lower_95: float | None = 0.65,
    ) -> Signal:
        """Execute complete pipeline from PIT feature extraction to transactional signal commit."""
        now = current_time or datetime.now(tz=UTC)

        # 1. Point-in-Time Feature Extraction (available_at <= cutoff_at)
        snapshot = await self._asof_engine.build_snapshot(
            market_id=self._market_id,
            horizon=self._horizon_id,
            decision_cutoff=cutoff_at,
        )

        # 2. Default high-quality assessment if none provided
        qualities = pillar_qualities or {
            p: PillarQuality(validity=1.0, completeness=1.0, freshness=1.0) for p in PillarType
        }

        # 3. Model Inference and Policy Evaluation
        signal = self._inference_engine.evaluate_snapshot(
            snapshot=snapshot,
            pillar_qualities=qualities,
            current_time=now,
            has_event_block=has_event_block,
            cohort_reliability_lower_95=cohort_reliability_lower_95,
        )

        # 4. Commit published signal to append-only ledger
        await self._signal_repo.append_signal(signal)

        # 5. Stage transactional outbox event
        if self._outbox_repo is not None:
            await self._outbox_repo.append_outbox(
                event_id=uuid.uuid4(),
                event_type="growth.signal.committed.v1",
                dedupe_key=f"{self._market_id.value}-{signal.sequence}",
                payload={
                    "signal_id": str(signal.signal_id),
                    "sequence": signal.sequence,
                    "market_id": signal.market_id.value,
                    "horizon": signal.horizon.value,
                    "status": signal.status.value,
                    "label": signal.label.value if signal.label else None,
                    "confidence": signal.confidence,
                    "issued_at": signal.issued_at.isoformat(),
                    "expires_at": signal.expires_at.isoformat(),
                },
            )

        return signal
