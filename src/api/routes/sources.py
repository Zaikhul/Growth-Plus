"""Operational source status and feed freshness route."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import (
    get_bar_repository,
    get_observation_repository,
    get_rights_authorizer,
)
from src.api.schemas import SourceDetailSchema, SourcesStatusResponse
from src.domain.identity import AssetId, MarketId
from src.domain.rights import DataOperation
from src.ports.repositories import BarRepository, ObservationRepository
from src.ports.rights_authorizer import RightsAuthorizer

router = APIRouter(prefix="/v1/sources", tags=["Sources"])


@router.get("/status", response_model=SourcesStatusResponse)
async def get_sources_status(
    authorizer: RightsAuthorizer = Depends(get_rights_authorizer),
    bar_repo: BarRepository = Depends(get_bar_repository),
    obs_repo: ObservationRepository = Depends(get_observation_repository),
) -> SourcesStatusResponse:
    """Retrieve operational status, freshness factors, and active coverage modes."""
    now = datetime.now(tz=UTC)
    tracked_feeds = [
        ("binance", "spot_trades", DataOperation.COLLECT),
        ("fed_board", "h15_yields", DataOperation.COLLECT),
        ("bls", "cpi", DataOperation.COMPUTE_FEATURES),
        ("farside", "etf_flows", DataOperation.COMPUTE_FEATURES),
    ]

    sources: list[SourceDetailSchema] = []
    incidents: list[str] = []

    for source_id, dataset_id, operation in tracked_feeds:
        decision = authorizer.authorize(source_id, dataset_id, operation, eval_time=now)
        if not decision.allowed:
            state = "DISABLED" if "DISABLED" in decision.reason_code else "RIGHTS_BLOCKED"
            incidents.append(f"{source_id}:{dataset_id} unavailable: {decision.reason_code}")
            sources.append(
                SourceDetailSchema(
                    source_id=source_id,
                    dataset_id=dataset_id,
                    state=state,
                    freshness_factor=0.0,
                    age_seconds=0.0,
                    last_update_at=None,
                    is_hard_stop=True,
                )
            )
            continue

        # Telemetry resolution from actual repositories
        last_update: datetime | None = None
        age_seconds: float = 60.0
        freshness_factor: float = 1.0
        state = "HEALTHY"

        if source_id == "binance" and dataset_id == "spot_trades":
            bar = await bar_repo.get_latest_closed_bar(
                MarketId("binance:BTCUSDT"), as_of_time=now
            )
            if bar is not None:
                last_update = bar.available_for_decision_at
                age_seconds = max(0.0, (now - last_update).total_seconds())
                freshness_factor = (
                    1.0 if age_seconds <= 120.0 else max(0.0, 1.0 - (age_seconds - 120.0) / 3600.0)
                )
                state = "HEALTHY" if freshness_factor > 0.5 else "DEGRADED"
            else:
                last_update = now
        elif source_id == "farside" and dataset_id == "etf_flows":
            etf = await obs_repo.get_latest_etf_flow(AssetId("BTC"), as_of_time=now)
            if etf is not None:
                last_update = etf[1].time_envelope.available_at
                age_seconds = max(0.0, (now - last_update).total_seconds())
                freshness_factor = (
                    1.0
                    if age_seconds <= 86400.0
                    else max(0.0, 1.0 - (age_seconds - 86400.0) / 86400.0)
                )
                state = "HEALTHY" if freshness_factor > 0.5 else "DEGRADED"
            else:
                last_update = now
                age_seconds = 3600.0
        else:
            macro_items = await obs_repo.get_macro_observations(dataset_id, as_of_time=now)
            if macro_items:
                last_update = macro_items[-1][1].time_envelope.available_at
                age_seconds = max(0.0, (now - last_update).total_seconds())
                freshness_factor = (
                    1.0
                    if age_seconds <= 7 * 86400.0
                    else max(0.0, 1.0 - (age_seconds - 7 * 86400.0) / (7 * 86400.0))
                )
                state = "HEALTHY" if freshness_factor > 0.5 else "DEGRADED"
            else:
                last_update = now
                age_seconds = 3600.0

        sources.append(
            SourceDetailSchema(
                source_id=source_id,
                dataset_id=dataset_id,
                state=state,
                freshness_factor=round(freshness_factor, 4),
                age_seconds=round(age_seconds, 1),
                last_update_at=last_update,
                is_hard_stop=False,
            )
        )

    active_mode = "FULL" if not incidents else "TECH_MACRO"
    overall_quality = 0.98 if not incidents else 0.75

    return SourcesStatusResponse(
        active_mode=active_mode,
        overall_quality=overall_quality,
        sources=sources,
        incidents=incidents,
    )
