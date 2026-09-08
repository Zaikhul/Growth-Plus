"""Cryptographic signature verifier and model bundle admission gate.

Enforces PRD Section 4.1 & 4.3:
- Cryptographic HMAC-SHA256 signature verification preventing model tampering
- Expiration check against evaluation time
- Schema compatibility check against incoming feature snapshots
- Monotonically increasing generation progression
"""

from datetime import datetime

from src.domain.errors import InvariantViolationError, SecurityError
from src.domain.features import FeatureSnapshot
from src.domain.time import ensure_utc
from src.ml.bundle import ModelBundle


class SignedModelVerifier:
    """Verifies authenticity, integrity, expiration, and compatibility of ModelBundles."""

    def __init__(self, secret_key: bytes | None = None) -> None:
        if secret_key is None:
            from src.config.settings import get_settings

            key_setting = get_settings().security.model_verification_secret_key
            if key_setting is not None:
                secret_key = key_setting.get_secret_value().encode("utf-8")
        if not secret_key:
            raise SecurityError("Model verification secret key cannot be empty")
        self._secret_key = secret_key

    def verify_and_admit(
        self,
        bundle: ModelBundle,
        current_time: datetime,
        current_generation: int = -1,
    ) -> ModelBundle:
        """Perform strict admission checks on a ModelBundle.

        Raises SecurityError on signature failure or tampering.
        Raises InvariantViolationError on expiration or generation regression.
        """
        # 1. Cryptographic signature check
        if not bundle.verify_signature(self._secret_key):
            raise SecurityError(
                f"Model bundle '{bundle.bundle_id}' signature verification failed: "
                "invalid or tampered"
            )

        # 2. Expiration check
        eval_time = ensure_utc(current_time)
        if bundle.is_expired_at(eval_time):
            raise InvariantViolationError(
                f"Model bundle '{bundle.bundle_id}' expired at "
                f"{bundle.header.expires_at} (eval time: {eval_time})"
            )

        # 3. Generation monotonicity check
        if bundle.generation <= current_generation:
            raise InvariantViolationError(
                f"Model bundle generation {bundle.generation} must be strictly "
                f"greater than active generation {current_generation}"
            )

        return bundle

    def verify_snapshot_compatibility(
        self,
        bundle: ModelBundle,
        snapshot: FeatureSnapshot,
    ) -> bool:
        """Verify that incoming snapshot contains the required feature columns for the bundle."""
        snap_features = snapshot.features
        for _pillar, schema_cols in bundle.feature_schemas.items():
            for col in schema_cols:
                if col not in snap_features:
                    return False
        return True
