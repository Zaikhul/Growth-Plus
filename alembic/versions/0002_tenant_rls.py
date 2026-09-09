"""0002: Tenant RLS

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-09
"""

from pathlib import Path
from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "migrations" / "0002_tenant_rls.sql"
    if sql_path.is_file():
        sql = sql_path.read_text(encoding="utf-8")
        op.execute(sql)


def downgrade() -> None:
    op.execute("DROP SCHEMA IF EXISTS tenant CASCADE;")
