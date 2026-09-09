-- Growth+ Lossless Persistence Migration (GP-007)
-- Enforces schema additions for macro observation envelopes, ETF flows, and snapshot lineage

-- 1. Macro observation envelope additions
ALTER TABLE macro.observations
    ADD COLUMN IF NOT EXISTS is_seasonally_adjusted boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS source_record_key varchar(128) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS schema_version varchar(32) NOT NULL DEFAULT 'macro_v1',
    ADD COLUMN IF NOT EXISTS parser_version varchar(32) NOT NULL DEFAULT 'parser_v1',
    ADD COLUMN IF NOT EXISTS rights_policy_id varchar(64) NOT NULL DEFAULT 'rights_public',
    ADD COLUMN IF NOT EXISTS rights_version varchar(32) NOT NULL DEFAULT 'v1';

-- 2. ETF observation envelope and product additions
ALTER TABLE flows.etf_observations
    ADD COLUMN IF NOT EXISTS ticker_at_time varchar(32) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS issuer varchar(64) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS jurisdiction varchar(16) NOT NULL DEFAULT 'US',
    ADD COLUMN IF NOT EXISTS is_seed_or_conversion boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS raw_digest varchar(64) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS canonical_digest varchar(64) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_id varchar(64) NOT NULL DEFAULT 'farside',
    ADD COLUMN IF NOT EXISTS source_record_key varchar(128) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS schema_version varchar(32) NOT NULL DEFAULT 'flows_v1',
    ADD COLUMN IF NOT EXISTS parser_version varchar(32) NOT NULL DEFAULT 'parser_v1',
    ADD COLUMN IF NOT EXISTS rights_policy_id varchar(64) NOT NULL DEFAULT 'rights_public',
    ADD COLUMN IF NOT EXISTS rights_version varchar(32) NOT NULL DEFAULT 'v1';

-- 3. Feature snapshot lineage IDs
ALTER TABLE features.snapshots
    ADD COLUMN IF NOT EXISTS lineage_record_ids jsonb NOT NULL DEFAULT '[]'::jsonb;
