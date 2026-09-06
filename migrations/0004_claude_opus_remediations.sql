-- Growth+ Claude Opus Audit Remediations Migration
-- Enforces schema corrections for DEFECT-01, DEFECT-02, DEFECT-40

-- 1. Macro observation revisions & UUID lineage
ALTER TABLE macro.observations
    ALTER COLUMN supersedes_id TYPE uuid USING supersedes_id::uuid;

ALTER TABLE macro.observations
    ADD CONSTRAINT uq_macro_series_revision UNIQUE (series_id, reference_period, revision_seq);

-- 2. ETF observation revision sequencing
ALTER TABLE flows.etf_observations
    ADD COLUMN IF NOT EXISTS revision_seq integer NOT NULL DEFAULT 0;

-- 3. Signal current pointer cutoff tracking for monotonic advance
ALTER TABLE signals.current
    ADD COLUMN IF NOT EXISTS cutoff_at timestamptz NOT NULL DEFAULT '1970-01-01 00:00:00+00';

-- 4. Signal reason code persistence
ALTER TABLE signals.signals
    ADD COLUMN IF NOT EXISTS reason_code varchar(64);

-- 5. Outbox dedupe key width expansion (supports market:horizon:cutoff:digest)
ALTER TABLE ops.outbox
    ALTER COLUMN dedupe_key TYPE varchar(256);
