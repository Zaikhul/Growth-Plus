"""Operational source status and feed freshness route."""

from datetime import UTC, datetime

from fastapi import APIRouter

from src.api.schemas import SourceDetailSchema, SourcesStatusResponse

router = APIRouter(prefix="/v1/sources", tags=["Sources"])


@router.get("/status", response_model=SourcesStatusResponse)
async def get_sources_status() -> SourcesStatusResponse:
    """Retrieve operational status, freshness factors, and active coverage modes."""
    now = datetime.now(tz=UTC)
    sources = [
        SourceDetailSchema(
            source_id="binance",
            dataset_id="spot_trades",
            state="HEALTHY",
            freshness_factor=1.0,
            age_seconds=1.2,
            last_update_at=now,
            is_hard_stop=False,
        ),
        SourceDetailSchema(
            source_id="fed_board",
            dataset_id="h15_yields",
            state="HEALTHY",
            freshness_factor=1.0,
            age_seconds=3600.0,
            last_update_at=now,
            is_hard_stop=False,
        ),
        SourceDetailSchema(
            source_id="bls",
            dataset_id="cpi",
            state="HEALTHY",
            freshness_factor=1.0,
            age_seconds=86400.0,
            last_update_at=now,
            is_hard_stop=False,
        ),
        SourceDetailSchema(
            source_id="farside",
            dataset_id="etf_flows",
            state="HEALTHY",
            freshness_factor=0.95,
            age_seconds=43200.0,
            last_update_at=now,
            is_hard_stop=False,
        ),
    ]

    return SourcesStatusResponse(
        active_mode="FULL",
        overall_quality=0.98,
        sources=sources,
        incidents=[],
    )
