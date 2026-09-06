"""Federal Reserve Board H.15 selected interest rates XML feed parser.

Enforces PRD Section 3.3 & Section 6.1:
- Safe XML parsing with lxml.etree (resolve_entities=False, no_network=True)
- Extracts EFFR, 2Y and 10Y constant maturity Treasury yields
- Computes 10Y - 2Y slope in basis points
"""

import hashlib
import json
import uuid
from datetime import datetime

from lxml import etree

from src.domain.identity import DatasetId, SourceId
from src.domain.macro import MacroObservation, MacroSeriesId
from src.domain.observations import ObservationEnvelope
from src.domain.time import TimeEnvelope, ensure_utc

FED_SERIES_MAP: dict[str, MacroSeriesId] = {
    "RIFSPFF_N.B": MacroSeriesId.US_EFFR,
    "RIFLGFCY02_N.B": MacroSeriesId.US_UST_2Y,
    "RIFLGFCY10_N.B": MacroSeriesId.US_UST_10Y,
}


class FedH15Parser:
    """Namespace-aware safe XML parser for Federal Reserve Board H.15 releases."""

    def __init__(self, parser_version: str = "1.0.0") -> None:
        self._parser_version = parser_version

    def parse_xml_bytes(
        self,
        xml_bytes: bytes,
        acquired_at: datetime,
    ) -> list[tuple[MacroObservation, ObservationEnvelope]]:
        """Parse XML bytes using hardened parser settings."""
        # Hardened parser rejecting external entities (XXE defense)
        xml_parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            recover=False,
        )
        root = etree.fromstring(xml_bytes, parser=xml_parser)

        acq_utc = ensure_utc(acquired_at)
        raw_digest = hashlib.sha256(xml_bytes).hexdigest()

        # Find all Series elements regardless of namespace prefix
        series_nodes = root.xpath(
            "//*[local-name()='Series']",
        )
        output: list[tuple[MacroObservation, ObservationEnvelope]] = []

        for s_node in series_nodes:
            s_id = s_node.get("id")
            if s_id not in FED_SERIES_MAP:
                continue

            canonical_series = FED_SERIES_MAP[s_id]
            obs_nodes = s_node.xpath(".//*[local-name()='Observation']")

            for o_node in obs_nodes:
                obs_date = o_node.get("date")
                val_str = o_node.get("value")
                if not obs_date or not val_str:
                    continue

                try:
                    val = float(val_str)
                except ValueError:
                    continue

                obs = MacroObservation(
                    series_id=canonical_series,
                    source_id=SourceId("fed_board"),
                    reference_period=obs_date,
                    value=val,
                    unit="PERCENT",
                )

                canonical_dict = {
                    "series_id": str(canonical_series),
                    "reference_period": obs_date,
                    "value": val,
                    "unit": "PERCENT",
                }
                canonical_digest = hashlib.sha256(
                    json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
                ).hexdigest()

                env = ObservationEnvelope(
                    record_id=uuid.uuid4(),
                    source_id=SourceId("fed_board"),
                    dataset_id=DatasetId("h15"),
                    source_record_key=f"{s_id}:{obs_date}",
                    schema_version="1.0.0",
                    parser_version=self._parser_version,
                    time_envelope=TimeEnvelope(
                        event_time=acq_utc,
                        reference_period=obs_date,
                        first_seen_at=acq_utc,
                        available_at=acq_utc,
                        published_at=acq_utc,
                    ),
                    raw_digest=raw_digest,
                    canonical_digest=canonical_digest,
                    rights_policy_id="policy_fed_board_default",
                    rights_version="1.0.0",
                    payload=canonical_dict,
                )
                output.append((obs, env))

        return output
