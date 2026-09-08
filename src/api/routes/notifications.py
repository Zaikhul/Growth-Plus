"""Notification rule registration and verification endpoints."""

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
    challenge_token: str
    created_at: datetime
    expires_at: datetime


# In-memory storage for subscription registrations and challenges
SUBSCRIPTION_REGISTRY: dict[uuid.UUID, SubscriptionRecord] = {}


@router.post(
    "/subscribe",
    response_model=NotificationSubscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def subscribe_notifications(
    payload: NotificationSubscribeRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> NotificationSubscribeResponse:
    """Register a new notification preference requiring challenge verification."""
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
    challenge_token = secrets.token_hex(16)
    record = SubscriptionRecord(
        subscription_id=subscription_id,
        tenant_id=tenant.tenant_id,
        target=payload.target,
        destination=payload.destination,
        status="PENDING_VERIFICATION",
        challenge_token=challenge_token,
        created_at=now,
        expires_at=now + timedelta(hours=24),
    )
    SUBSCRIPTION_REGISTRY[subscription_id] = record

    return NotificationSubscribeResponse(
        subscription_id=subscription_id,
        target=payload.target,
        destination=payload.destination,
        status="PENDING_VERIFICATION",
        challenge_token=challenge_token,
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

    if record.challenge_token != payload.challenge_token:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid challenge verification token",
        )

    record.status = "ACTIVE"
    return NotificationVerifyResponse(
        subscription_id=record.subscription_id,
        status="ACTIVE",
        verified_at=datetime.now(tz=UTC),
    )
