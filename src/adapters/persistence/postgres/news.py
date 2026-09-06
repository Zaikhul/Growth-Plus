"""Append-only news versions and filter-before-rank PIT reconstruction."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from pydantic import TypeAdapter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.contracts.news_evidence import NewsEvidence, NewsWindow
from src.ports.news import NewsRights


class PostgresNewsStore:
    def __init__(
        self, engine: AsyncEngine, asset: str, rights: NewsRights, approved_model_version: str
    ) -> None:
        if asset not in ("BTC", "ETH"):
            raise ValueError("Unsupported news asset")
        self.engine, self.asset, self.rights = engine, asset, rights
        self.approved_model_version = approved_model_version

    async def append(
        self,
        records: Sequence[NewsEvidence],
        checked_at: datetime,
        coverage_capped: bool,
        model_version: str,
    ) -> None:
        self.rights.assert_retention()
        async with self.engine.begin() as connection:
            for item in records:
                if item.asset != self.asset:
                    raise ValueError("Wrong asset in news store")
                payload = asdict(item)
                for key in ("published_at", "source_observed_at", "first_seen_at", "available_at"):
                    value = payload[key]
                    payload[key] = value.isoformat() if value is not None else None
                await connection.execute(
                    text(
                        "INSERT INTO news.items(record_id,asset,source_key,revision,"
                        "available_at,payload) "
                        "VALUES (:id,:asset,:key,:revision,:available,CAST(:payload AS jsonb)) "
                        "ON CONFLICT (record_id) DO NOTHING"
                    ),
                    {
                        "id": item.record_id,
                        "asset": item.asset,
                        "key": item.source_key,
                        "revision": item.revision,
                        "available": item.available_at,
                        "payload": json.dumps(payload),
                    },
                )
            await connection.execute(
                text(
                    "INSERT INTO news.polls VALUES (:asset,:at,:capped,:model) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "asset": self.asset,
                    "at": checked_at,
                    "capped": coverage_capped,
                    "model": model_version,
                },
            )

    async def read(self, asset: str, cutoff: datetime) -> NewsWindow:
        self.rights.assert_feature_use()
        if asset != self.asset:
            raise ValueError("Wrong news asset")
        async with self.engine.connect() as connection:
            poll = (
                (
                    await connection.execute(
                        text(
                            "SELECT * FROM news.polls WHERE asset=:asset AND checked_at<=:cutoff "
                            "ORDER BY checked_at DESC LIMIT 1"
                        ),
                        {"asset": asset, "cutoff": cutoff},
                    )
                )
                .mappings()
                .first()
            )
            rows = (
                (
                    await connection.execute(
                        text(
                            "WITH eligible AS (SELECT * FROM news.items WHERE asset=:asset "
                            "AND available_at<=:cutoff), ranked AS (SELECT payload, "
                            "row_number() OVER (PARTITION BY source_key ORDER BY revision DESC, "
                            "available_at DESC,record_id DESC) AS rank FROM eligible) "
                            "SELECT payload FROM ranked WHERE rank=1"
                        ),
                        {"asset": asset, "cutoff": cutoff},
                    )
                )
                .scalars()
                .all()
            )
        records = []
        adapter = TypeAdapter(NewsEvidence)
        for raw in rows:
            records.append(adapter.validate_json(json.dumps(raw), strict=True))
        self.rights.assert_feature_use()
        return NewsWindow(
            tuple(records),
            poll["checked_at"] if poll else cutoff,
            bool(poll["coverage_capped"]) if poll else True,
            bool(
                poll
                and poll["model_version"] == self.approved_model_version
                and self.approved_model_version
            ),
            True,
        )
