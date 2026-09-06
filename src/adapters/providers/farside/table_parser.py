"""Farside Investors daily ETF table HTML parser.

Enforces PRD Section 3.4:
- Constituent fund reconciliation against reported total
- Status lifecycle: RECONCILED, PARTIAL, DISPUTED
- Safe HTML parsing using lxml
"""

import hashlib
import json
import re
import uuid
from datetime import UTC, date, datetime

from lxml import html

from src.domain.flows import EtfFlowObservation, EtfProductStatus
from src.domain.identity import AssetId, DatasetId, SourceId
from src.domain.observations import ObservationEnvelope
from src.domain.time import TimeEnvelope, ensure_utc


class FarsideTableParser:
    """Parser for Farside Investors daily ETF net flow tables."""

    def __init__(self, asset_id: AssetId = AssetId.BTC, parser_version: str = "1.0.0") -> None:
        self._asset_id = asset_id
        self._parser_version = parser_version

    def parse_html_str(
        self,
        html_str: str,
        acquired_at: datetime,
    ) -> list[tuple[EtfFlowObservation, ObservationEnvelope]]:
        """Parse HTML string and reconcile daily flows."""
        tree = html.fromstring(html_str)
        table = tree.xpath("//table")
        if not table:
            return []

        rows = table[0].xpath(".//tr")
        if len(rows) < 2:
            return []

        # Parse headers
        header_cells = rows[0].xpath(".//th | .//td")
        headers = [c.text_content().strip() for c in header_cells]

        # Identify constituent fund columns vs Total
        fund_columns: list[tuple[int, str]] = []
        total_col_idx: int | None = None

        for idx, h in enumerate(headers):
            if idx == 0 or not h:
                continue
            if h.lower() == "total":
                total_col_idx = idx
            else:
                fund_columns.append((idx, h))

        acq_utc = ensure_utc(acquired_at)
        raw_digest = hashlib.sha256(html_str.encode("utf-8")).hexdigest()
        output: list[tuple[EtfFlowObservation, ObservationEnvelope]] = []

        # Parse body rows
        for row in rows[1:]:
            cells = row.xpath(".//td")
            if not cells:
                continue

            date_str = cells[0].text_content().strip()
            session_date = self._parse_session_date(date_str)
            if session_date is None:
                continue

            constituent_sum = 0.0
            covered_funds_count = 0
            has_nan = False

            for col_idx, _ticker in fund_columns:
                if col_idx < len(cells):
                    val_text = cells[col_idx].text_content().strip().replace(",", "")
                    try:
                        val_m = float(val_text)
                        constituent_sum += val_m
                        covered_funds_count += 1
                    except ValueError:
                        has_nan = True

            reported_total_m: float | None = None
            if total_col_idx is not None and total_col_idx < len(cells):
                total_text = cells[total_col_idx].text_content().strip().replace(",", "")
                try:
                    reported_total_m = float(total_text)
                except ValueError:
                    reported_total_m = None

            # Reconciliation logic
            if reported_total_m is None or has_nan:
                status = EtfProductStatus.PARTIAL
                flow_usd = constituent_sum * 1_000_000.0
            elif abs(constituent_sum - reported_total_m) < 0.2:
                status = EtfProductStatus.RECONCILED
                flow_usd = reported_total_m * 1_000_000.0
            else:
                status = EtfProductStatus.DISPUTED
                flow_usd = reported_total_m * 1_000_000.0

            obs = EtfFlowObservation(
                asset_id=self._asset_id,
                fund_id="TOTAL",
                ticker_at_time="TOTAL",
                issuer="Aggregate",
                jurisdiction="US",
                session_date=session_date,
                flow_usd=flow_usd,
                status=status,
                covered_funds=covered_funds_count,
                expected_funds=len(fund_columns),
            )

            canonical_dict = {
                "asset_id": str(self._asset_id),
                "fund_id": "TOTAL",
                "session_date": session_date.isoformat(),
                "flow_usd": flow_usd,
                "status": str(status),
                "covered_funds": covered_funds_count,
                "expected_funds": len(fund_columns),
            }
            canonical_digest = hashlib.sha256(
                json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
            ).hexdigest()

            event_time = datetime(
                session_date.year,
                session_date.month,
                session_date.day,
                21,
                0,
                0,
                tzinfo=UTC,
            )
            env = ObservationEnvelope(
                record_id=uuid.uuid4(),
                source_id=SourceId("farside"),
                dataset_id=DatasetId("us_spot_etf_flows"),
                source_record_key=f"farside:{self._asset_id}:{session_date}",
                schema_version="1.0.0",
                parser_version=self._parser_version,
                time_envelope=TimeEnvelope(
                    event_time=event_time,
                    reference_period=str(session_date),
                    first_seen_at=acq_utc,
                    available_at=acq_utc,
                    published_at=acq_utc,
                ),
                raw_digest=raw_digest,
                canonical_digest=canonical_digest,
                rights_policy_id="policy_farside_default",
                rights_version="1.0.0",
                payload=canonical_dict,
            )
            output.append((obs, env))

        return output

    def _parse_session_date(self, date_str: str) -> date | None:
        """Parse dates like '04 Sep 2026' or '2026-09-04'."""
        if not date_str:
            return None
        # Try ISO format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            try:
                return date.fromisoformat(date_str)
            except ValueError:
                pass
        # Try '04 Sep 2026'
        try:
            dt = datetime.strptime(date_str, "%d %b %Y")
            return dt.date()
        except ValueError:
            pass
        return None
