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
from src.domain.macro import MacroReleaseCalendarEvent
from src.domain.policies.event_windows import evaluate_event_window
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
        cohort_reliability_lower_95: float | None = None,
        macro_events: list[MacroReleaseCalendarEvent] | None = None,
    ) -> Signal:
        """Execute complete pipeline from PIT feature extraction to transactional signal commit."""
        now = current_time or datetime.now(tz=UTC)

        # 1. Point-in-Time Feature Extraction (available_at <= cutoff_at)
        snapshot, measured_qualities = await self._asof_engine.build_snapshot(
            market_id=self._market_id,
            horizon=self._horizon_id,
            decision_cutoff=cutoff_at,
        )

        # 2. Use measured qualities from AsofJoinEngine if not overridden (DEFECT-11, 28)
        qualities = pillar_qualities if pillar_qualities is not None else measured_qualities

        # 3. Macro Event Window Evaluation (DEFECT-10(a))
        suppress_publication = False
        effective_has_event_block = has_event_block
        if macro_events:
            ev_eval = evaluate_event_window(
                horizon=self._horizon_id,
                decision_time=cutoff_at,
                events=macro_events,
            )
            if ev_eval.is_in_blackout:
                effective_has_event_block = True
            if ev_eval.suppress_publication:
                suppress_publication = True

        # 4. Model Inference and Policy Evaluation
        signal = self._inference_engine.evaluate_snapshot(
            snapshot=snapshot,
            pillar_qualities=qualities,
            current_time=now,
            has_event_block=effective_has_event_block,
            cohort_reliability_lower_95=cohort_reliability_lower_95,
            suppress_publication=suppress_publication,
        )

        # 5. Commit published signal to append-only ledger
        await self._signal_repo.append_signal(signal)

        # 6. Stage transactional outbox event with deterministic dedupe_key (DEFECT-01)
        if self._outbox_repo is not None:
            dedupe_key = (
                f"{self._market_id.value}:{self._horizon_id.value}:"
                f"{cutoff_at.isoformat()}:{snapshot.digest}"
            )
            await self._outbox_repo.append_outbox(
                event_id=uuid.uuid4(),
                event_type="growth.signal.committed.v1",
                dedupe_key=dedupe_key,
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
