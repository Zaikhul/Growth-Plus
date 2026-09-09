"""0003: News items and polls

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-09
"""

from pathlib import Path
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "migrations" / "0003_news.sql"
    if sql_path.is_file():
        sql = sql_path.read_text(encoding="utf-8")
        op.execute(sql)


def downgrade() -> None:
    op.execute("DROP SCHEMA IF EXISTS news CASCADE;")
