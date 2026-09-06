"""Pure PIT feature construction. No network, database or model fitting."""

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta

from src.domain.contracts.news_evidence import NewsEvidence, NewsWindow


def news_features(
    window: NewsWindow, cutoff: datetime, horizon: str
) -> tuple[dict[str, float], tuple[str, ...]] | None:
    if (
        not window.rights_allowed
        or not window.model_approved
        or window.coverage_capped
        or window.checked_at > cutoff
        or cutoff - window.checked_at >= timedelta(minutes=60)
    ):
        return None
    settings = {
        "scalp_15m": (15 * 60, 60 * 60, 30 * 60),
        "swing_24h": (60 * 60, 24 * 60 * 60, 6 * 60 * 60),
        "position_30d": (86400, 7 * 86400, 48 * 60 * 60),
    }
    short, long, half_life = settings[horizon]
    # Availability filtering precedes selecting a revision, then event deduplication.
    versions: dict[str, NewsEvidence] = {}
    for item in window.records:
        if item.available_at <= cutoff:
            rev_key = (item.revision, item.available_at, item.record_id)
            previous = versions.get(item.source_key)
            if previous is None or rev_key > (
                previous.revision,
                previous.available_at,
                previous.record_id,
            ):
                versions[item.source_key] = item
    events: dict[str, NewsEvidence] = {}
    for item in sorted(versions.values(), key=lambda x: (x.available_at, x.record_id)):
        age = (cutoff - (item.published_at or item.source_observed_at)).total_seconds()
        if 0 <= age <= long:
            events[item.event_id] = item
    # Suppress near-identical syndications within 48 hours; a correction/denial is
    # retained as a distinct state rather than erased by textual similarity.
    retained: dict[str, NewsEvidence] = {}
    shingles: dict[str, set[tuple[str, ...]]] = {}
    postings: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key, item in sorted(events.items(), key=lambda row: (row[1].available_at, row[0])):
        tokens = re.findall(r"\w+", item.title.lower())
        parts = {tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))}
        candidates = set()
        for part in parts:
            candidates.update(postings[part])
            if len(candidates) > 512:
                return None  # Explicit unavailable feature block, never inflated diversity.
        duplicate = False
        for prior_key in candidates:
            prior = retained[prior_key]
            within = (
                abs((item.source_observed_at - prior.source_observed_at).total_seconds())
                <= 48 * 3600
            )
            union = parts | shingles[prior_key]
            if (
                within
                and item.status == prior.status
                and union
                and len(parts & shingles[prior_key]) / len(union) >= 0.85
            ):
                duplicate = True
                break
        if not duplicate:
            retained[key], shingles[key] = item, parts
            for part in parts:
                postings[part].add(key)
    events = retained
    publishers: dict[str, list[tuple[float, float]]] = defaultdict(list)
    short_count = 0
    for item in events.values():
        age = (cutoff - (item.published_at or item.source_observed_at)).total_seconds()
        weight = item.relevance * math.exp(-math.log(2) * age / half_life)
        if weight > 0:
            publishers[item.publisher].append((item.sentiment, weight))
        short_count += age <= short
    # Equal publisher priors capped at 0.30. Unallocated mass remains explicit.
    share = min(0.30, 1 / len(publishers)) if publishers else 0.0
    sentiment = 0.0
    for values in publishers.values():
        total = sum(weight for _, weight in values)
        sentiment += share * sum(score * weight for score, weight in values) / total
    count = len(events)
    features = {
        "news_non_english_count": float(
            sum(item.language != "English" for item in events.values())
        ),
        "news_sentiment": sentiment,
        "news_event_count": float(count),
        "news_short_event_count": float(short_count),
        "news_publisher_count": float(len(publishers)),
        "news_concentrated": float(len(publishers) < 4),
        "news_weight_coverage": share * len(publishers),
        "news_rumor_share": sum(x.status == "RUMOR" for x in events.values()) / count
        if count
        else 0.0,
        "news_correction_share": sum(x.status == "CORRECTED" for x in events.values()) / count
        if count
        else 0.0,
        "news_poll_age_seconds": (cutoff - window.checked_at).total_seconds(),
    }
    return features, tuple(item.record_id for item in events.values())
