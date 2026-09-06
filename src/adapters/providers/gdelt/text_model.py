"""Signed JSON TF-IDF/logistic heads. Training approval is required, never invented."""

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from src.domain.contracts.news_evidence import NewsStatus


@dataclass(frozen=True, slots=True)
class LinearHead:
    classes: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]

    def probabilities(self, vector: tuple[float, ...]) -> tuple[float, ...]:
        logits = tuple(
            b + sum(w * x for w, x in zip(row, vector, strict=True))
            for row, b in zip(self.weights, self.intercepts, strict=True)
        )
        shift = max(logits)
        terms = tuple(math.exp(value - shift) for value in logits)
        total = sum(terms)
        return tuple(value / total for value in terms)


class JsonTfidfNewsAnnotator:
    def __init__(self, artifact: bytes, signature_hex: str, verification_key: bytes) -> None:
        if not verification_key or len(artifact) > 25 * 1024 * 1024:
            raise ValueError("Invalid text artifact input")
        expected = hmac.new(verification_key, artifact, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature_hex):
            raise ValueError("Text artifact signature invalid")
        payload = json.loads(artifact)
        if not isinstance(payload, dict) or payload.get("schema") != "growth-text-tfidf-v1":
            raise ValueError("Unsupported text artifact")
        for metric in ("sentiment_macro_f1", "relevance_macro_f1", "alpha"):
            value = payload.get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Invalid signed evaluation metric")
        if (
            payload.get("annotated_items", 0) < 2000
            or payload.get("holdout_items", 0) < 400
            or payload.get("sentiment_macro_f1", 0) < 0.75
            or payload.get("relevance_macro_f1", 0) < 0.85
            or payload.get("alpha", 0) < 0.67
            or not payload.get("independent_approval_id")
        ):
            raise ValueError("Text model has not passed PRD approval gates")
        vocabulary = payload.get("vocabulary")
        idf = payload.get("idf")
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or len(vocabulary) > 20000
            or any(not isinstance(token, str) or not token for token in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or not isinstance(idf, list)
            or len(idf) != len(vocabulary)
        ):
            raise ValueError("Invalid frozen TF-IDF vocabulary")
        self.vocabulary = tuple(vocabulary)
        self.idf = tuple(float(value) for value in idf)
        if any(not math.isfinite(value) or value <= 0 for value in self.idf):
            raise ValueError("Invalid IDF values")
        self.heads = {}
        expected_classes = {
            "sentiment": ("NEGATIVE", "NEUTRAL", "POSITIVE"),
            "BTC": ("IRRELEVANT", "RELEVANT"),
            "ETH": ("IRRELEVANT", "RELEVANT"),
            "status": ("RUMOR", "ANNOUNCED", "EFFECTIVE", "DENIED", "CORRECTED"),
        }
        for name, classes in expected_classes.items():
            raw = payload["heads"][name]
            if tuple(raw["classes"]) != classes:
                raise ValueError("Text head class order mismatch")
            head = LinearHead(
                classes,
                tuple(tuple(float(x) for x in row) for row in raw["weights"]),
                tuple(float(x) for x in raw["intercepts"]),
            )
            if (
                len(head.weights) != len(classes)
                or len(head.intercepts) != len(classes)
                or any(len(row) != len(self.vocabulary) for row in head.weights)
                or any(not math.isfinite(x) for row in head.weights for x in row)
                or any(not math.isfinite(x) for x in head.intercepts)
            ):
                raise ValueError("Invalid linear text head")
            self.heads[name] = head
        self._version = hashlib.sha256(artifact).hexdigest()

    @property
    def approved_model_version(self) -> str:
        return self._version

    def annotate(
        self, title: str, asset: str, url: str, published_at: datetime
    ) -> tuple[str, float, float, NewsStatus]:
        if len(title) > 8000 or asset not in ("BTC", "ETH"):
            raise ValueError("Invalid text inference input")
        tokens = re.findall(r"\b\w\w+\b", title.lower())
        bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:], strict=False)]
        counts = Counter(tokens + bigrams)
        raw = tuple(
            (1 + math.log(counts[token])) * idf if counts[token] else 0.0
            for token, idf in zip(self.vocabulary, self.idf, strict=True)
        )
        norm = math.sqrt(sum(value * value for value in raw))
        vector = tuple(value / norm if norm else 0.0 for value in raw)
        sentiment = self.heads["sentiment"].probabilities(vector)
        relevance = self.heads[asset].probabilities(vector)[1]
        status_probabilities = self.heads["status"].probabilities(vector)
        statuses: tuple[NewsStatus, ...] = (
            "RUMOR",
            "ANNOUNCED",
            "EFFECTIVE",
            "DENIED",
            "CORRECTED",
        )
        status = statuses[max(range(5), key=status_probabilities.__getitem__)]
        # No inferred shared originating event without evidence; exact URL is a safe initial ID.
        event_id = hashlib.sha256(url.encode()).hexdigest()
        return event_id, sentiment[2] - sentiment[0], relevance, status
