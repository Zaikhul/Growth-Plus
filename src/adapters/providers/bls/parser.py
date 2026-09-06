"""BLS API v2 CPI timeseries parser.

Enforces PRD Section 3.3:
- Excludes annual-average 'M13' records from monthly calculations
- Maps BLS series to canonical MacroSeriesId
- Calculates month-on-month and year-on-year percent changes
- Wraps parsed data into MacroObservation and ObservationEnvelope
"""

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from src.domain.identity import DatasetId, SourceId
from src.domain.macro import MacroObservation, MacroSeriesId
from src.domain.observations import ObservationEnvelope
from src.domain.time import TimeEnvelope, ensure_utc

BLS_SERIES_MAPPING: dict[str, tuple[MacroSeriesId, bool]] = {
    "CUSR0000SA0": (MacroSeriesId.US_CPI_HEADLINE_SA, True),
    "CUUR0000SA0": (MacroSeriesId.US_CPI_HEADLINE_NSA, False),
    "CUSR0000SA0L1E": (MacroSeriesId.US_CPI_CORE_SA, True),
    "CUUR0000SA0L1E": (MacroSeriesId.US_CPI_CORE_NSA, False),
}


class BLSCPIParser:
    """Parser for BLS API v2 timeseries JSON responses."""

    def __init__(self, parser_version: str = "1.0.0") -> None:
        self._parser_version = parser_version

    def parse_payload(
        self,
        raw_json_str: str,
        acquired_at: datetime,
    ) -> list[tuple[MacroObservation, ObservationEnvelope]]:
        """Parse BLS response JSON into normalized macro observations."""
        data: Mapping[str, Any] = json.loads(raw_json_str)
        results = data.get("Results", {})
        series_list = results.get("series", [])

        acq_utc = ensure_utc(acquired_at)
        raw_digest = hashlib.sha256(raw_json_str.encode("utf-8")).hexdigest()

        output: list[tuple[MacroObservation, ObservationEnvelope]] = []

        for s in series_list:
            bls_id = s.get("seriesID")
            if bls_id not in BLS_SERIES_MAPPING:
                continue

            canonical_series, is_sa = BLS_SERIES_MAPPING[bls_id]
            data_items = s.get("data", [])

            # Filter out M13 (annual average) records (PRD Section 3.3)
            monthly_items = [item for item in data_items if item.get("period") != "M13"]

            # Sort chronologically (year, period)
            monthly_items.sort(key=lambda item: (item.get("year", ""), item.get("period", "")))

            for i, item in enumerate(monthly_items):
                val_str = item.get("value", "")
                if not val_str:
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    continue

                year = item.get("year", "")
                period = item.get("period", "")
                ref_period = f"{year}-{period}"

                # Calculate changes
                prior_val = (
                    float(monthly_items[i - 1]["value"])
                    if i > 0 and monthly_items[i - 1].get("value")
                    else None
                )
                prior_year_val = (
                    float(monthly_items[i - 12]["value"])
                    if i >= 12 and monthly_items[i - 12].get("value")
                    else None
                )

                obs = MacroObservation(
                    series_id=canonical_series,
                    source_id=SourceId("bls"),
                    reference_period=ref_period,
                    value=val,
                    unit="INDEX",
                    is_seasonally_adjusted=is_sa,
                )

                canonical_dict = {
                    "series_id": str(canonical_series),
                    "reference_period": ref_period,
                    "value": val,
                    "unit": "INDEX",
                    "mom_pct": obs.percent_change(prior_val),
                    "yoy_pct": obs.percent_change(prior_year_val),
                }
                canonical_digest = hashlib.sha256(
                    json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
                ).hexdigest()

                env = ObservationEnvelope(
                    record_id=uuid.uuid4(),
                    source_id=SourceId("bls"),
                    dataset_id=DatasetId("cpi"),
                    source_record_key=f"{bls_id}:{ref_period}",
                    schema_version="1.0.0",
                    parser_version=self._parser_version,
                    time_envelope=TimeEnvelope(
                        event_time=acq_utc,
                        reference_period=ref_period,
                        first_seen_at=acq_utc,
                        available_at=acq_utc,
                        published_at=acq_utc,
                    ),
                    raw_digest=raw_digest,
                    canonical_digest=canonical_digest,
                    rights_policy_id="policy_bls_default",
                    rights_version="1.0.0",
                    payload=canonical_dict,
                )
                output.append((obs, env))

        return output
