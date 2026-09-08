"""One shared PostgreSQL lock serializes all GDELT collectors and backfills."""

import asyncio
from datetime import datetime

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresBudgetedGdeltTransport:
    def __init__(
        self,
        engine: AsyncEngine,
        contact_user_agent: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
    ) -> None:
        from src.config.settings import get_settings

        cfg = get_settings().external
        agent = contact_user_agent or cfg.gdelt_contact_user_agent
        if not agent or "\n" in agent:
            raise ValueError("A monitored crawler contact is required")
        self.engine, self.user_agent = engine, agent
        self.endpoint = endpoint or cfg.gdelt_endpoint
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else cfg.gdelt_timeout_seconds
        )
        self.connect_timeout_seconds = (
            connect_timeout_seconds
            if connect_timeout_seconds is not None
            else cfg.gdelt_connect_timeout_seconds
        )

    async def fetch(self, asset: str, start: datetime, end: datetime) -> bytes:
        queries = {"BTC": '(bitcoin OR "BTC crypto")', "ETH": '(ethereum OR "ETH crypto")'}
        query = queries[asset]
        async with self.engine.begin() as connection:
            await connection.execute(text("SELECT pg_advisory_xact_lock(78204622)"))
            # Minimum five seconds between request starts, even across processes.
            await asyncio.sleep(5)
            timeout = httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, trust_env=False
            ) as client:
                async with client.stream(
                    "GET",
                    self.endpoint,
                    params={
                        "query": query,
                        "mode": "ArtList",
                        "format": "json",
                        "maxrecords": 250,
                        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                    },
                    headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"},
                ) as response:
                    response.raise_for_status()
                    if "application/json" not in response.headers.get("content-type", "").lower():
                        raise ValueError("Unexpected GDELT MIME type")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 10 * 1024 * 1024:
                            raise ValueError("GDELT response exceeds limit")
                    return bytes(body)
