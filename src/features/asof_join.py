"""Point-in-Time As-Of Join Engine.

Enforces PRD Section 3.1 & 3.7:
- All joined observations strictly enforce available_at <= decision_cutoff
- Technical bars enforce available_for_decision_at (close + 2s) <= decision_cutoff
- Scalar limit <= 192 features
- Active pillar mask resolution: FULL, CORE_NO_ETF, TECH_MACRO
- Immutable FeatureSnapshot construction with input UUID lineage
"""

import uuid
from datetime import datetime

from src.domain.features import FeatureSnapshot, PillarType
from src.domain.identity import AssetId, HorizonId, MarketId
from src.domain.macro import MacroSeriesId
from src.domain.policies.coverage import resolve_coverage_mode
from src.domain.policies.freshness import (
    VENUE_TRADE_HARD_STOP_SECONDS,
    calculate_freshness_decay,
    is_market_tape_stale,
)
from src.domain.policies.quality import PillarQuality
from src.domain.rights import DataOperation
from src.domain.time import PointInTimeCutoff, ensure_utc
from src.features.lookback import technical_window_start
from src.features.news import news_features
from src.features.technical.indicators import TechnicalIndicatorsEngine
from src.ports.news import NewsReader
from src.ports.repositories import BarRepository, ObservationRepository
from src.ports.rights_authorizer import RightsAuthorizer


class PointInTimeAsOfEngine:
    """Deterministic Point-in-Time as-of join engine."""

    def __init__(
        self,
        bar_repo: BarRepository,
        obs_repo: ObservationRepository,
        authorizer: RightsAuthorizer,
        indicators_engine: TechnicalIndicatorsEngine | None = None,
        feature_set_version: str = "features_1.0.0",
        *,
        news_reader: NewsReader | None = None,
    ) -> None:
        self._bar_repo = bar_repo
        self._obs_repo = obs_repo
        self._authorizer = authorizer
        self._indicators = indicators_engine or TechnicalIndicatorsEngine()
        self._version = feature_set_version
        self._news_reader: NewsReader | None = news_reader

    async def build_snapshot(
        self,
        market_id: MarketId,
        horizon: HorizonId,
        decision_cutoff: datetime,
    ) -> tuple[FeatureSnapshot, dict[PillarType, PillarQuality]]:
        """Construct an immutable feature snapshot and measured qualities as of cutoff."""
        cutoff_utc = ensure_utc(decision_cutoff)
        pit = PointInTimeCutoff(cutoff_at=cutoff_utc)

        lineage: list[uuid.UUID] = []
        scalars: dict[str, float] = {}
        active_pillars: list[PillarType] = []
        qualities: dict[PillarType, PillarQuality] = {}

        # ----------------------------------------------------------------------
        # 1. Technical Pillar (Closed bars only)
        # ----------------------------------------------------------------------
        # Window of bars up to cutoff
        bars = await self._bar_repo.get_bars(
            market_id=market_id,
            start_time=technical_window_start(cutoff_utc),  # Generous start
            end_time=cutoff_utc,
        )
        # Filter strictly closed bars eligible at cutoff
        eligible_bars = [
            b for b in bars if b.is_closed and b.available_for_decision_at <= cutoff_utc
        ]

        if eligible_bars:
            tech_feats = self._indicators.compute_features(eligible_bars)
            scalars.update(tech_feats)
            active_pillars.append(PillarType.TECHNICAL)
            latest_bar = eligible_bars[-1]
            is_stale = is_market_tape_stale(horizon, latest_bar.bar_close_at, cutoff_utc)
            decay = calculate_freshness_decay(
                last_update_at=latest_bar.bar_close_at,
                as_of_time=cutoff_utc,
                expected_cadence_seconds=60.0,
                hard_stop_seconds=VENUE_TRADE_HARD_STOP_SECONDS[horizon],
            )
            completeness = min(1.0, len(eligible_bars) / 60.0)
            qualities[PillarType.TECHNICAL] = PillarQuality(
                validity=0.0 if is_stale else 1.0,
                completeness=completeness,
                freshness=decay.freshness_factor,
            )
        else:
            qualities[PillarType.TECHNICAL] = PillarQuality(
                validity=1.0, completeness=0.0, freshness=0.0
            )

        # ----------------------------------------------------------------------
        # 2. Macroeconomic Pillar (available_at <= cutoff)
        # ----------------------------------------------------------------------
        # Check entitlement first
        auth_macro = self._authorizer.authorize("bls", "cpi", DataOperation.COMPUTE_FEATURES)
        macro_obs_found = False
        if auth_macro.allowed:
            cpi_obs = await self._obs_repo.get_macro_observations(
                series_id=str(MacroSeriesId.US_CPI_HEADLINE_SA),
                as_of_time=cutoff_utc,
            )
            if cpi_obs:
                latest_cpi, latest_cpi_env = max(
                    cpi_obs,
                    key=lambda pair: (pair[1].time_envelope.available_at, pair[0].revision_seq),
                )
                pit.assert_available(latest_cpi_env.time_envelope.available_at)
                lineage.append(latest_cpi_env.record_id)
                scalars["macro_cpi_level"] = latest_cpi.value
                if (
                    "mom_pct" in latest_cpi_env.payload
                    and latest_cpi_env.payload["mom_pct"] is not None
                ):
                    scalars["macro_cpi_mom_pct"] = float(latest_cpi_env.payload["mom_pct"])
                macro_obs_found = True

            effr_obs = await self._obs_repo.get_macro_observations(
                series_id=str(MacroSeriesId.US_EFFR),
                as_of_time=cutoff_utc,
            )
            if effr_obs:
                latest_effr, latest_effr_env = max(
                    effr_obs,
                    key=lambda pair: (pair[1].time_envelope.available_at, pair[0].revision_seq),
                )
                pit.assert_available(latest_effr_env.time_envelope.available_at)
                lineage.append(latest_effr_env.record_id)
                scalars["macro_effr_rate"] = latest_effr.value
                macro_obs_found = True

            if macro_obs_found:
                active_pillars.append(PillarType.MACRO)
                qualities[PillarType.MACRO] = PillarQuality(
                    validity=1.0, completeness=1.0, freshness=1.0
                )
            else:
                qualities[PillarType.MACRO] = PillarQuality(
                    validity=1.0, completeness=0.0, freshness=0.0
                )
        else:
            qualities[PillarType.MACRO] = PillarQuality(
                validity=0.0, completeness=0.0, freshness=0.0
            )

        # ----------------------------------------------------------------------
        # 3. ETF Flow Pillar (available_at <= cutoff)
        # ----------------------------------------------------------------------
        auth_etf = self._authorizer.authorize(
            "farside", "us_spot_etf_flows", DataOperation.COMPUTE_FEATURES
        )
        asset = AssetId.BTC if "BTC" in str(market_id) else AssetId.ETH
        if auth_etf.allowed:
            etf_record = await self._obs_repo.get_latest_etf_flow(
                asset_id=asset,
                as_of_time=cutoff_utc,
            )
            if etf_record:
                latest_etf, latest_etf_env = etf_record
                pit.assert_available(latest_etf_env.time_envelope.available_at)
                lineage.append(latest_etf_env.record_id)
                scalars["etf_net_flow_usd"] = latest_etf.flow_usd
                scalars["etf_coverage_ratio"] = latest_etf.coverage_ratio
                active_pillars.append(PillarType.ETF)
                qualities[PillarType.ETF] = PillarQuality(
                    validity=1.0,
                    completeness=max(0.0, min(1.0, latest_etf.coverage_ratio)),
                    freshness=1.0,
                )
            else:
                qualities[PillarType.ETF] = PillarQuality(
                    validity=1.0, completeness=0.0, freshness=0.0
                )
        else:
            qualities[PillarType.ETF] = PillarQuality(validity=0.0, completeness=0.0, freshness=0.0)

        # ----------------------------------------------------------------------
        # 4. Mode Determination (PRD Section 3.13)
        # ----------------------------------------------------------------------
        if self._news_reader is not None:
            window = await self._news_reader.read(str(asset), cutoff_utc)
            contribution = news_features(window, cutoff_utc, str(horizon))
            if contribution is not None:
                news_scalars, news_lineage = contribution
                scalars.update(news_scalars)
                lineage.extend(uuid.UUID(record_id) for record_id in news_lineage)
                active_pillars.append(PillarType.NEWS)
                qualities[PillarType.NEWS] = PillarQuality(
                    validity=1.0, completeness=1.0, freshness=1.0
                )
            else:
                qualities[PillarType.NEWS] = PillarQuality(
                    validity=1.0, completeness=0.0, freshness=0.0
                )
        else:
            qualities[PillarType.NEWS] = PillarQuality(
                validity=1.0, completeness=0.0, freshness=0.0
            )

        mode = resolve_coverage_mode(active_pillars)

        now_computed = ensure_utc(datetime.now(cutoff_utc.tzinfo))
        if now_computed < cutoff_utc:
            now_computed = cutoff_utc

        snapshot = FeatureSnapshot(
            snapshot_id=uuid.uuid4(),
            market_id=market_id,
            horizon=horizon,
            cutoff_at=cutoff_utc,
            computed_at=now_computed,
            feature_set_version=self._version,
            scalars=scalars,
            active_pillars=tuple(active_pillars),
            mode=mode,
            lineage_record_ids=tuple(lineage),
        )
        return snapshot, qualities
