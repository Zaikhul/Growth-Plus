"""Notification rule registration and verification endpoints."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from src.adapters.notifications.webhook import validate_destination
from src.api.dependencies import VerifiedTenant, get_verified_tenant
from src.api.schemas import (
    NotificationSubscribeRequest,
    NotificationSubscribeResponse,
    NotificationVerifyRequest,
    NotificationVerifyResponse,
)

router = APIRouter(prefix="/v1/notifications", tags=["Notifications"])


@dataclass
class SubscriptionRecord:
    subscription_id: uuid.UUID
    tenant_id: uuid.UUID
    target: str
    destination: str
    status: str
    hashed_challenge_token: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3


# In-memory storage for subscription registrations and challenges
SUBSCRIPTION_REGISTRY: dict[uuid.UUID, SubscriptionRecord] = {}

# Out-of-band delivery log (e.g. email/webhook simulated outbox)
OUT_OF_BAND_DISPATCH_LOG: dict[uuid.UUID, str] = {}


def get_dispatched_challenge(subscription_id: str | uuid.UUID) -> str | None:
    """Retrieve simulated out-of-band challenge token for testing."""
    if isinstance(subscription_id, uuid.UUID):
        sid = subscription_id
    else:
        sid = uuid.UUID(str(subscription_id))
    return OUT_OF_BAND_DISPATCH_LOG.get(sid)


@router.post(
    "/subscribe",
    response_model=NotificationSubscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def subscribe_notifications(
    payload: NotificationSubscribeRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> NotificationSubscribeResponse:
    """Register a new notification preference requiring challenge verification.

    The challenge token is dispatched out-of-band and NEVER returned in the response body.
    """
    now = datetime.now(tz=UTC)
    if payload.target == "webhook":
        try:
            validate_destination(payload.destination)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid webhook destination") from exc
    elif payload.target == "email":
        if "@" not in payload.destination or "." not in payload.destination.split("@")[-1]:
            raise HTTPException(status_code=422, detail="Invalid email destination")
    elif payload.target == "telegram":
        if not payload.destination.strip():
            raise HTTPException(status_code=422, detail="Invalid telegram destination")

    subscription_id = uuid.uuid4()
    raw_challenge_token = secrets.token_hex(16)
    hashed_token = hashlib.sha256(raw_challenge_token.encode("utf-8")).hexdigest()

    record = SubscriptionRecord(
        subscription_id=subscription_id,
        tenant_id=tenant.tenant_id,
        target=payload.target,
        destination=payload.destination,
        status="PENDING_VERIFICATION",
        hashed_challenge_token=hashed_token,
        created_at=now,
        expires_at=now + timedelta(hours=24),
        attempts=0,
        max_attempts=3,
    )
    SUBSCRIPTION_REGISTRY[subscription_id] = record
    OUT_OF_BAND_DISPATCH_LOG[subscription_id] = raw_challenge_token

    return NotificationSubscribeResponse(
        subscription_id=subscription_id,
        target=payload.target,
        destination=payload.destination,
        status="PENDING_VERIFICATION",
        challenge_token=None,  # Never leaked in registration response body
        created_at=now,
    )


@router.post(
    "/verify",
    response_model=NotificationVerifyResponse,
    status_code=status.HTTP_200_OK,
)
async def verify_notification_subscription(
    payload: NotificationVerifyRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> NotificationVerifyResponse:
    """Verify challenge token and transition subscription to ACTIVE state."""
    record = SUBSCRIPTION_REGISTRY.get(payload.subscription_id)
    if record is None or record.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found for authenticated tenant",
        )

    now = datetime.now(tz=UTC)
    if now >= record.expires_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Challenge verification token has expired",
        )

    if record.attempts >= record.max_attempts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum challenge verification attempts exceeded",
        )

    record.attempts += 1
    submitted_hash = hashlib.sha256(payload.challenge_token.encode("utf-8")).hexdigest()

    if not secrets.compare_digest(record.hashed_challenge_token, submitted_hash):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid challenge verification token",
        )

    record.status = "ACTIVE"
    OUT_OF_BAND_DISPATCH_LOG.pop(payload.subscription_id, None)

    return NotificationVerifyResponse(
        subscription_id=record.subscription_id,
        status="ACTIVE",
        verified_at=now,
    )
