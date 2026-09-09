"""0004: Claude Opus remediations

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-09
"""

from pathlib import Path
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "migrations" / "0004_claude_opus_remediations.sql"
    if sql_path.is_file():
        sql = sql_path.read_text(encoding="utf-8")
        op.execute(sql)


def downgrade() -> None:
    op.execute("ALTER TABLE signals.signals DROP COLUMN IF EXISTS reason_code;")
