"""Bounded IP abuse guard; authenticated account/tenant quotas remain independent."""

import ipaddress
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class Bucket:
    tokens: float
    updated_at: float


class BoundedBuckets:
    def __init__(self, capacity: int, refill: float, max_clients: int = 4096) -> None:
        if capacity < 1 or not math.isfinite(refill) or refill <= 0 or max_clients < 1:
            raise ValueError("Invalid bucket policy")
        self.capacity, self.refill, self.max_clients = capacity, refill, max_clients
        self.entries: OrderedDict[str, Bucket] = OrderedDict()

    def consume(self, key: str, now: float) -> tuple[bool, int, float]:
        # Buckets refill completely before eviction, so churn cannot forgive a debt.
        full_after = self.capacity / self.refill
        while self.entries:
            _, oldest = next(iter(self.entries.items()))
            if now - oldest.updated_at < full_after:
                break
            self.entries.popitem(last=False)
        bucket = self.entries.pop(key, None)
        if bucket is None:
            if len(self.entries) >= self.max_clients:
                return False, 0, full_after
            bucket = Bucket(float(self.capacity), now)
        bucket.tokens = min(
            self.capacity, bucket.tokens + max(0, now - bucket.updated_at) * self.refill
        )
        bucket.updated_at = now
        allowed = bucket.tokens >= 1
        if allowed:
            bucket.tokens -= 1
        self.entries[key] = bucket
        return allowed, int(bucket.tokens), 0.0 if allowed else (1 - bucket.tokens) / self.refill


def client_identity(peer: str | None, forwarded: str | None, trusted_cidrs: Sequence[str]) -> str:
    if peer is None:
        return "unknown-peer"
    try:
        direct = ipaddress.ip_address(peer)
    except ValueError:
        return "unknown-peer"  # Shared fail-restrictive bucket, never an attacker key.
    trusted = tuple(ipaddress.ip_network(cidr, strict=True) for cidr in trusted_cidrs)
    if forwarded is None or not any(direct in network for network in trusted):
        return str(direct)
    if len(forwarded) > 4096:
        raise ValueError("Forwarded address chain too long")
    parts = forwarded.split(",")
    if len(parts) > 16:
        raise ValueError("Too many forwarding hops")
    chain = [ipaddress.ip_address(value.strip()) for value in parts] + [direct]
    while len(chain) > 1 and any(chain[-1] in network for network in trusted):
        chain.pop()
    return str(chain[-1])
