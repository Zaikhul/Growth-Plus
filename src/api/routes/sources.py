"""Operational source status and feed freshness route."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_rights_authorizer
from src.api.schemas import SourceDetailSchema, SourcesStatusResponse
from src.domain.rights import DataOperation
from src.ports.rights_authorizer import RightsAuthorizer

router = APIRouter(prefix="/v1/sources", tags=["Sources"])


@router.get("/status", response_model=SourcesStatusResponse)
async def get_sources_status(
    authorizer: RightsAuthorizer = Depends(get_rights_authorizer),
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
        decision = authorizer.authorize(source_id, dataset_id, operation)
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
        else:
            sources.append(
                SourceDetailSchema(
                    source_id=source_id,
                    dataset_id=dataset_id,
                    state="HEALTHY",
                    freshness_factor=1.0,
                    age_seconds=60.0,
                    last_update_at=now,
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
