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
    ) -> FeatureSnapshot:
        """Construct an immutable feature snapshot as of decision_cutoff."""
        cutoff_utc = ensure_utc(decision_cutoff)
        pit = PointInTimeCutoff(cutoff_at=cutoff_utc)

        lineage: list[uuid.UUID] = []
        scalars: dict[str, float] = {}
        active_pillars: list[PillarType] = []

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
        eligible_bars = [b for b in bars if b.available_for_decision_at <= cutoff_utc]

        if eligible_bars:
            tech_feats = self._indicators.compute_features(eligible_bars)
            scalars.update(tech_feats)
            active_pillars.append(PillarType.TECHNICAL)

        # ----------------------------------------------------------------------
        # 2. Macroeconomic Pillar (available_at <= cutoff)
        # ----------------------------------------------------------------------
        # Check entitlement first
        auth_macro = self._authorizer.authorize("bls", "cpi", DataOperation.COMPUTE_FEATURES)
        if auth_macro.allowed:
            cpi_obs = await self._obs_repo.get_macro_observations(
                series_id=str(MacroSeriesId.US_CPI_HEADLINE_SA),
                as_of_time=cutoff_utc,
            )
            if cpi_obs:
                latest_cpi, latest_cpi_env = cpi_obs[-1]
                pit.assert_available(latest_cpi_env.time_envelope.available_at)
                lineage.append(latest_cpi_env.record_id)
                scalars["macro_cpi_level"] = latest_cpi.value
                if (
                    "mom_pct" in latest_cpi_env.payload
                    and latest_cpi_env.payload["mom_pct"] is not None
                ):
                    scalars["macro_cpi_mom_pct"] = float(latest_cpi_env.payload["mom_pct"])

            effr_obs = await self._obs_repo.get_macro_observations(
                series_id=str(MacroSeriesId.US_EFFR),
                as_of_time=cutoff_utc,
            )
            if effr_obs:
                latest_effr, latest_effr_env = effr_obs[-1]
                pit.assert_available(latest_effr_env.time_envelope.available_at)
                lineage.append(latest_effr_env.record_id)
                scalars["macro_effr_rate"] = latest_effr.value

            if cpi_obs or effr_obs:
                active_pillars.append(PillarType.MACRO)

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

        mode = resolve_coverage_mode(active_pillars)

        now_computed = ensure_utc(datetime.now(cutoff_utc.tzinfo))
        if now_computed < cutoff_utc:
            now_computed = cutoff_utc

        return FeatureSnapshot(
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
