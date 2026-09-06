CREATE SCHEMA IF NOT EXISTS news;
CREATE TABLE IF NOT EXISTS news.items (
    record_id uuid PRIMARY KEY,
    asset text NOT NULL CHECK (asset IN ('BTC','ETH')),
    source_key text NOT NULL,
    revision bigint NOT NULL,
    available_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    UNIQUE(asset,source_key,revision,record_id)
);
CREATE INDEX IF NOT EXISTS news_asof ON news.items(asset,available_at,source_key,revision);
CREATE TABLE IF NOT EXISTS news.polls (
    asset text NOT NULL,
    checked_at timestamptz NOT NULL,
    coverage_capped boolean NOT NULL,
    model_version text NOT NULL,
    PRIMARY KEY(asset,checked_at)
);
