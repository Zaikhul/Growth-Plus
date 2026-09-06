"""Notification rule registration endpoint."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from src.adapters.notifications.webhook import validate_destination
from src.api.schemas import NotificationSubscribeRequest, NotificationSubscribeResponse

router = APIRouter(prefix="/v1/notifications", tags=["Notifications"])


@router.post(
    "/subscribe",
    response_model=NotificationSubscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def subscribe_notifications(
    payload: NotificationSubscribeRequest,
) -> NotificationSubscribeResponse:
    """Register or update a user's notification preference."""
    if payload.target == "webhook":
        try:
            validate_destination(payload.destination)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid webhook destination") from exc
    return NotificationSubscribeResponse(
        subscription_id=uuid.uuid4(),
        target=payload.target,
        destination=payload.destination,
        status="PENDING_VERIFICATION",
        created_at=datetime.now(tz=UTC),
    )
