"""0005: Lossless persistence

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-09
"""

from pathlib import Path
from alembic import op

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "migrations" / "0005_lossless_persistence.sql"
    if sql_path.is_file():
        sql = sql_path.read_text(encoding="utf-8")
        op.execute(sql)


def downgrade() -> None:
    op.execute("ALTER TABLE features.snapshots DROP COLUMN IF EXISTS lineage_record_ids;")
