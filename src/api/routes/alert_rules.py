"""Alert rules and notification preferences endpoints."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.adapters.notifications.webhook import validate_destination
from src.api.dependencies import (
    VerifiedTenant,
    get_alert_rule_repository,
    get_verified_tenant,
)
from src.ports.repositories import AlertRuleRepository, TenantAlertRule

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
    labels: list[str] = Field(
        default_factory=lambda: ["Strong Buy", "Buy", "Sell", "Strong Sell"]
    )
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


@router.get("", response_model=list[AlertRuleDTO])
async def list_alert_rules(
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: AlertRuleRepository = Depends(get_alert_rule_repository),
) -> list[AlertRuleDTO]:
    """Retrieve all alert rules for the authenticated tenant."""
    items = await repo.list_by_tenant(tenant.tenant_id)
    if not items:
        # Default baseline rule
        now = datetime.now(tz=UTC)
        default_id = uuid.uuid5(tenant.tenant_id, "default-btc-swing-rule")
        default_rule = TenantAlertRule(
            id=default_id,
            tenant_id=tenant.tenant_id,
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
            consent_at=now,
            version=1,
            created_at=now,
            updated_at=now,
        )
        await repo.save(default_rule)
        return [
            AlertRuleDTO(
                id=str(default_rule.id),
                name=default_rule.name,
                markets=list(default_rule.markets),
                horizons=list(default_rule.horizons),
                labels=list(default_rule.labels),
                min_confidence=default_rule.min_confidence,
                cooldown_seconds=default_rule.cooldown_seconds,
                target_channels=list(default_rule.target_channels),
                destination=default_rule.destination,
                destination_status=default_rule.destination_status,
                is_active=default_rule.is_active,
                created_at=now,
                updated_at=now,
            )
        ]

    return [
        AlertRuleDTO(
            id=str(item.id),
            name=item.name,
            markets=list(item.markets),
            horizons=list(item.horizons),
            labels=list(item.labels),
            min_confidence=item.min_confidence,
            cooldown_seconds=item.cooldown_seconds,
            target_channels=list(item.target_channels),
            destination=item.destination,
            destination_status=item.destination_status,
            is_active=item.is_active,
            created_at=item.created_at or datetime.now(tz=UTC),
            updated_at=item.updated_at or datetime.now(tz=UTC),
        )
        for item in items
    ]


@router.post("", response_model=AlertRuleDTO, status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    payload: AlertRuleCreateRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: AlertRuleRepository = Depends(get_alert_rule_repository),
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

    rule_id: uuid.UUID
    if payload.id:
        try:
            rule_id = uuid.UUID(payload.id)
        except ValueError:
            rule_id = uuid.uuid5(tenant.tenant_id, payload.id)
    else:
        rule_id = uuid.uuid4()

    now = datetime.now(tz=UTC)
    entity = TenantAlertRule(
        id=rule_id,
        tenant_id=tenant.tenant_id,
        name=payload.name,
        markets=payload.markets,
        horizons=payload.horizons,
        labels=payload.labels,
        min_confidence=payload.min_confidence,
        cooldown_seconds=payload.cooldown_seconds,
        target_channels=payload.target_channels,
        destination=payload.destination,
        destination_status=payload.destination_status,
        is_active=payload.is_active,
        consent_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )
    saved = await repo.save(entity)

    return AlertRuleDTO(
        id=str(saved.id),
        name=saved.name,
        markets=list(saved.markets),
        horizons=list(saved.horizons),
        labels=list(saved.labels),
        min_confidence=saved.min_confidence,
        cooldown_seconds=saved.cooldown_seconds,
        target_channels=list(saved.target_channels),
        destination=saved.destination,
        destination_status=saved.destination_status,
        is_active=saved.is_active,
        created_at=saved.created_at or now,
        updated_at=saved.updated_at or now,
    )


@router.patch("/{rule_id}", response_model=AlertRuleDTO)
async def patch_alert_rule(
    rule_id: str,
    payload: AlertRulePatchRequest,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: AlertRuleRepository = Depends(get_alert_rule_repository),
) -> AlertRuleDTO:
    """Toggle or update an alert rule for the authenticated tenant."""
    try:
        rule_uuid = uuid.UUID(rule_id)
    except ValueError:
        rule_uuid = uuid.uuid5(tenant.tenant_id, rule_id)

    existing = await repo.get_by_id(tenant.tenant_id, rule_uuid)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert rule '{rule_id}' not found",
        )

    new_active = payload.is_active if payload.is_active is not None else existing.is_active
    new_confidence = (
        payload.min_confidence if payload.min_confidence is not None else existing.min_confidence
    )

    updated_entity = TenantAlertRule(
        id=existing.id,
        tenant_id=existing.tenant_id,
        name=existing.name,
        markets=existing.markets,
        horizons=existing.horizons,
        labels=existing.labels,
        min_confidence=new_confidence,
        cooldown_seconds=existing.cooldown_seconds,
        target_channels=existing.target_channels,
        destination=existing.destination,
        destination_status=existing.destination_status,
        is_active=new_active,
        consent_at=existing.consent_at,
        version=existing.version,
        created_at=existing.created_at,
        updated_at=datetime.now(tz=UTC),
    )
    saved = await repo.save(updated_entity, expected_version=existing.version)

    return AlertRuleDTO(
        id=str(saved.id),
        name=saved.name,
        markets=list(saved.markets),
        horizons=list(saved.horizons),
        labels=list(saved.labels),
        min_confidence=saved.min_confidence,
        cooldown_seconds=saved.cooldown_seconds,
        target_channels=list(saved.target_channels),
        destination=saved.destination,
        destination_status=saved.destination_status,
        is_active=saved.is_active,
        created_at=saved.created_at or datetime.now(tz=UTC),
        updated_at=saved.updated_at or datetime.now(tz=UTC),
    )


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert_rule(
    rule_id: str,
    tenant: VerifiedTenant = Depends(get_verified_tenant),
    repo: AlertRuleRepository = Depends(get_alert_rule_repository),
) -> None:
    """Delete an alert rule for the authenticated tenant."""
    try:
        rule_uuid = uuid.UUID(rule_id)
    except ValueError:
        rule_uuid = uuid.uuid5(tenant.tenant_id, rule_id)

    deleted = await repo.delete(tenant.tenant_id, rule_uuid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert rule '{rule_id}' not found",
        )
