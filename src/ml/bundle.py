"""Signed Model Bundle specification, packaging, and serialization.

Enforces PRD Section 4.1 & 4.3:
- Immutable container for pillar models, fusion weights, calibration temperature,
  and feature schemas
- Cryptographic HMAC-SHA256 signature guaranteeing authenticity and tamper-detection
- Monotonically increasing generation sequence
"""

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.domain.errors import InvariantViolationError
from src.domain.features import PillarType, SourceCoverageMode
from src.domain.identity import HorizonId
from src.domain.time import ensure_utc
from src.ml.fusion import LogOpinionPoolFusion
from src.ml.training.pillar_experts import PillarExpertModel


@dataclass(frozen=True, slots=True)
class ModelBundleHeader:
    """Metadata header for a signed model bundle."""

    bundle_id: str
    generation: int
    horizon: HorizonId
    mode: SourceCoverageMode
    created_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", ensure_utc(self.created_at))
        object.__setattr__(self, "expires_at", ensure_utc(self.expires_at))
        if self.expires_at <= self.created_at:
            raise InvariantViolationError("expires_at must be strictly greater than created_at")


class ModelBundle:
    """Immutable, validated, and signed ensemble model package."""

    def __init__(
        self,
        header: ModelBundleHeader,
        pillar_models: Mapping[PillarType, PillarExpertModel],
        fusion: LogOpinionPoolFusion,
        feature_schemas: Mapping[PillarType, Sequence[str]],
        signature: str = "",
    ) -> None:
        self._header = header
        self._pillar_models = dict(pillar_models)
        self._fusion = fusion
        self._feature_schemas = {p: tuple(names) for p, names in feature_schemas.items()}
        self._signature = signature

    @property
    def header(self) -> ModelBundleHeader:
        return self._header

    @property
    def bundle_id(self) -> str:
        return self._header.bundle_id

    @property
    def generation(self) -> int:
        return self._header.generation

    @property
    def horizon(self) -> HorizonId:
        return self._header.horizon

    @property
    def mode(self) -> SourceCoverageMode:
        return self._header.mode

    @property
    def pillar_models(self) -> dict[PillarType, PillarExpertModel]:
        return dict(self._pillar_models)

    @property
    def fusion(self) -> LogOpinionPoolFusion:
        return self._fusion

    @property
    def feature_schemas(self) -> dict[PillarType, tuple[str, ...]]:
        return dict(self._feature_schemas)

    @property
    def signature(self) -> str:
        return self._signature

    def is_expired_at(self, current_time: datetime) -> bool:
        """Check if model approval/validity has expired."""
        return ensure_utc(current_time) >= self._header.expires_at

    def canonical_payload(self) -> str:
        """Construct canonical deterministic JSON string for hashing and signing."""
        data = {
            "bundle_id": self._header.bundle_id,
            "generation": self._header.generation,
            "horizon": self._header.horizon.value,
            "mode": self._header.mode.value,
            "created_at": self._header.created_at.isoformat(),
            "expires_at": self._header.expires_at.isoformat(),
            "fusion": self._fusion.to_dict(),
            "feature_schemas": {p.value: list(names) for p, names in self._feature_schemas.items()},
            "pillar_models": {p.value: model.to_dict() for p, model in self._pillar_models.items()},
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def sign(self, secret_key: bytes) -> str:
        """Compute HMAC-SHA256 signature and set on the bundle."""
        payload = self.canonical_payload().encode("utf-8")
        sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
        self._signature = sig
        return sig

    def verify_signature(self, secret_key: bytes) -> bool:
        """Verify HMAC-SHA256 signature against current bundle payload."""
        if not self._signature:
            return False
        payload = self.canonical_payload().encode("utf-8")
        expected_sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self._signature, expected_sig)

    def to_json(self) -> str:
        """Serialize complete signed bundle to JSON string."""
        data = {
            "bundle_id": self._header.bundle_id,
            "generation": self._header.generation,
            "horizon": self._header.horizon.value,
            "mode": self._header.mode.value,
            "created_at": self._header.created_at.isoformat(),
            "expires_at": self._header.expires_at.isoformat(),
            "fusion": self._fusion.to_dict(),
            "feature_schemas": {p.value: list(names) for p, names in self._feature_schemas.items()},
            "pillar_models": {p.value: model.to_dict() for p, model in self._pillar_models.items()},
            "signature": self._signature,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelBundle":
        """Deserialize and construct ModelBundle from JSON string."""
        data = json.loads(json_str)
        header = ModelBundleHeader(
            bundle_id=data["bundle_id"],
            generation=int(data["generation"]),
            horizon=HorizonId(data["horizon"]),
            mode=SourceCoverageMode(data["mode"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )
        fusion = LogOpinionPoolFusion.from_dict(data["fusion"])
        feature_schemas = {PillarType(k): tuple(v) for k, v in data["feature_schemas"].items()}
        pillar_models = {
            PillarType(k): PillarExpertModel.from_dict(v) for k, v in data["pillar_models"].items()
        }
        return cls(
            header=header,
            pillar_models=pillar_models,
            fusion=fusion,
            feature_schemas=feature_schemas,
            signature=data.get("signature", ""),
        )
