"""0001: Partition maintenance

Revision ID: 0001
Revises: None
Create Date: 2026-09-09
"""

from pathlib import Path
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "migrations" / "0001_partition_maintenance.sql"
    if sql_path.is_file():
        sql = sql_path.read_text(encoding="utf-8")
        op.execute(sql)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS ops.ensure_partitions(timestamptz, timestamptz);")
