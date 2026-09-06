"""Strict metadata parser. Linked articles are never downloaded."""

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

from src.domain.contracts.news_evidence import NewsEvidence
from src.ports.news import NewsAnnotator


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
        raise ValueError("Invalid article URL")
    discarded = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
    }
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in discarded
        ]
    )
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path or "/", query, ""))


def parse_articles(
    body: bytes,
    *,
    asset: str,
    first_seen_at: datetime,
    available_at: datetime,
    rights_version: str,
    annotator: NewsAnnotator,
) -> tuple[tuple[NewsEvidence, ...], bool]:
    if len(body) > 10 * 1024 * 1024:
        raise ValueError("GDELT response exceeds 10 MiB")
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
        raise ValueError("GDELT schema drift")
    articles = payload["articles"]
    if len(articles) > 250:
        raise ValueError("GDELT ArtList limit violated")
    model_version = annotator.approved_model_version
    if not model_version:
        raise ValueError("Unapproved news model cannot create active news features")
    result: list[NewsEvidence] = []
    for raw in articles:
        if not isinstance(raw, dict):
            raise ValueError("Invalid GDELT article")
        for key in ("url", "title", "seendate", "domain"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"Missing GDELT {key}")
        url = canonical_url(raw["url"])
        # GDELT seendate is a source observation clock, not a publisher release time.
        observed = datetime.strptime(raw["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if observed > first_seen_at:
            raise ValueError("Future GDELT source timestamp")
        cleaned = "".join(
            c
            for c in unicodedata.normalize("NFC", raw["title"])
            if not unicodedata.category(c).startswith("C") or c in "\n\t"
        )
        title = cleaned[:8000]
        language = raw.get("language", "UNKNOWN")
        if not isinstance(language, str):
            raise ValueError("Invalid language metadata")
        if language == "English":
            event, sentiment, relevance, status = annotator.annotate(title, asset, url, observed)
        else:
            event = hashlib.sha256(url.encode()).hexdigest()
            sentiment, relevance, status = 0.0, 0.0, None

        digest = hashlib.sha256(
            (asset + "\n" + url + "\n" + title + "\n" + observed.isoformat()).encode()
        ).hexdigest()
        result.append(
            NewsEvidence(
                str(uuid5(NAMESPACE_URL, digest)),
                url,
                int(observed.timestamp() * 1_000_000),
                asset,
                raw["domain"].lower(),
                url,
                title,
                len(cleaned) > 8000,
                None,
                observed,
                first_seen_at,
                available_at,
                event,
                sentiment,
                relevance,
                status,
                rights_version,
                model_version,
                language,
            )
        )
    return tuple(result), len(articles) == 250


class GdeltParser:
    parse = staticmethod(parse_articles)
