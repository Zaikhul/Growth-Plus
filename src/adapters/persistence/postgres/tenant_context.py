"""Tenant context may only be supplied after transport authentication/authorization."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


@asynccontextmanager
async def tenant_transaction(
    engine: AsyncEngine, verified_tenant_id: UUID
) -> AsyncIterator[AsyncConnection]:
    async with engine.connect() as connection:
        async with connection.begin():
            role = (
                await connection.execute(
                    text("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user")
                )
            ).one()
            if role.rolsuper or role.rolbypassrls:
                raise PermissionError("Unsafe application database role")
            owner_count = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid=c.relnamespace "
                        "WHERE n.nspname='tenant' "
                        "AND pg_has_role(current_user,c.relowner,'MEMBER')"
                    )
                )
            ).scalar_one()
            if owner_count:
                raise PermissionError("Application role must not own tenant tables")
            await connection.execute(
                text("SELECT set_config('app.tenant_id',:tenant,true)"),
                {"tenant": str(verified_tenant_id)},
            )
            yield connection
        # Transaction-local settings have expired before connection returns to pool.
