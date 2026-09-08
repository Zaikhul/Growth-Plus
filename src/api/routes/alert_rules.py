"""Alert rules and notification preferences endpoints."""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.adapters.notifications.webhook import validate_destination
from src.api.dependencies import VerifiedTenant, get_verified_tenant

router = APIRouter(prefix="/v1/alert-rules", tags=["Alert Rules"])


class AlertRuleDTO(BaseModel):
    id: str
    name: str
    markets: list[str]
    horizons: list[str]
    labels: list[str]
    min_confidence: float = 0.60
    cooldown_seconds: int = 900
    target_channels: list[str] = Field(default_factory=lambda: ["webhook"])
    destination: str
    destination_status: str = "PENDING_VERIFICATION"
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class AlertRuleCreateRequest(BaseModel):
    id: str | None = None
    name: str = Field(..., min_length=1, max_length=80)
    markets: list[str] = Field(..., min_length=1, max_length=20)
    horizons: list[str] = Field(..., min_length=1, max_length=3)
    labels: list[str] = Field(default_factory=lambda: ["Strong Buy", "Buy", "Sell", "Strong Sell"])
    min_confidence: float = Field(default=0.60, ge=0.50, le=1.0)
    cooldown_seconds: int = Field(default=900, ge=900)
    target_channels: list[str] = Field(default_factory=lambda: ["webhook"])
    destination: str = Field(..., min_length=8)
    destination_status: str = "PENDING_VERIFICATION"
    is_active: bool = True
    consent_accepted: bool = True


class AlertRulePatchRequest(BaseModel):
    is_active: bool | None = None
    min_confidence: float | None = None


# In-memory storage for alert rules
ALERT_RULE_STORE: dict[str, dict[str, Any]] = {}


@router.get("", response_model=list[AlertRuleDTO])
async def list_alert_rules(
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> list[AlertRuleDTO]:
    """Retrieve all alert rules for the authenticated tenant."""
    tenant_str = str(tenant.tenant_id)
    items = [
        AlertRuleDTO(**data)
        for data in ALERT_RULE_STORE.values()
        if data.get("tenant_id") == tenant_str
    ]
    if not items:
        # Default baseline rule
        now = datetime.now(tz=UTC)
        default_rule = AlertRuleDTO(
            id=f"rule-btc-swing-{tenant_str[:8]}",
            name="BTC Swing High Conviction",
            markets=["binance:BTCUSDT"],
            horizons=["swing_24h"],
            labels=["Strong Buy", "Strong Sell"],
            min_confidence=0.65,
            cooldown_seconds=900,
            target_channels=["webhook"],
            destination="https://api.systematic-quant.internal/hooks/growth-plus",
            destination_status="VERIFIED",
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        return [default_rule]
    return items


@router.post("", response_model=AlertRuleDTO, status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    payload: AlertRuleCreateRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> AlertRuleDTO:
    """Create a new alert rule for the authenticated tenant."""
    if not payload.consent_accepted:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="You must acknowledge regulatory disclosure terms to create an alert.",
        )

    # Validate destination if webhook
    if "webhook" in payload.target_channels:
        try:
            validate_destination(payload.destination)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid webhook destination: {exc}",
            ) from exc

    rule_id = payload.id or f"rule-{uuid.uuid4()}"
    now = datetime.now(tz=UTC)
    record = {
        "id": rule_id,
        "tenant_id": str(tenant.tenant_id),
        "name": payload.name,
        "markets": payload.markets,
        "horizons": payload.horizons,
        "labels": payload.labels,
        "min_confidence": payload.min_confidence,
        "cooldown_seconds": payload.cooldown_seconds,
        "target_channels": payload.target_channels,
        "destination": payload.destination,
        "destination_status": payload.destination_status,
        "is_active": payload.is_active,
        "created_at": now,
        "updated_at": now,
    }
    ALERT_RULE_STORE[rule_id] = record
    return AlertRuleDTO(**record)


@router.patch("/{rule_id}", response_model=AlertRuleDTO)
async def patch_alert_rule(
    rule_id: str,
    payload: AlertRulePatchRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> AlertRuleDTO:
    """Toggle or update an alert rule for the authenticated tenant."""
    record = ALERT_RULE_STORE.get(rule_id)
    if record is None or record.get("tenant_id") != str(tenant.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert rule '{rule_id}' not found",
        )
    if payload.is_active is not None:
        record["is_active"] = payload.is_active
    if payload.min_confidence is not None:
        record["min_confidence"] = payload.min_confidence
    record["updated_at"] = datetime.now(tz=UTC)
    return AlertRuleDTO(**record)


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert_rule(
    rule_id: str,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
) -> None:
    """Delete an alert rule for the authenticated tenant."""
    record = ALERT_RULE_STORE.get(rule_id)
    if record is None or record.get("tenant_id") != str(tenant.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert rule '{rule_id}' not found",
        )
    del ALERT_RULE_STORE[rule_id]
