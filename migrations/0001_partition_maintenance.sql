-- Execute as schema owner after Base.metadata.create_all(). UTC, half-open bounds.
CREATE SCHEMA IF NOT EXISTS ops;
CREATE OR REPLACE FUNCTION ops.ensure_partitions(p_start timestamptz, p_end timestamptz)
RETURNS void LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE
    parent record;
    boundary timestamptz;
    next_boundary timestamptz;
    suffix text;
BEGIN
    IF p_start IS NULL OR p_end IS NULL OR p_start >= p_end
       OR p_end - p_start > interval '10 years' THEN
        RAISE EXCEPTION 'invalid partition provisioning interval';
    END IF;
    PERFORM pg_advisory_xact_lock(78204621);
    FOR parent IN
        SELECT * FROM (VALUES
            ('market', 'trades', 'day'),
            ('market', 'quotes', 'day'),
            ('market', 'bars', 'month'),
            ('features', 'snapshots', 'month'),
            ('signals', 'signals', 'month')
        ) AS targets(schema_name, table_name, cadence)
    LOOP
        -- quotes is not defined in the supplied models; provision it once present.
        IF to_regclass(format('%I.%I', parent.schema_name, parent.table_name)) IS NULL THEN
            IF parent.table_name = 'quotes' THEN CONTINUE; END IF;
            RAISE EXCEPTION 'required parent missing: %.%', parent.schema_name, parent.table_name;
        END IF;
        -- Provision DEFAULT partition to catch any out-of-range rows safely
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.%I PARTITION OF %I.%I DEFAULT',
            parent.schema_name, parent.table_name || '_default',
            parent.schema_name, parent.table_name);
        boundary := date_trunc(parent.cadence, p_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
        WHILE boundary < p_end LOOP
            next_boundary := ((boundary AT TIME ZONE 'UTC') +
                CASE parent.cadence WHEN 'day' THEN interval '1 day'
                ELSE interval '1 month' END) AT TIME ZONE 'UTC';
            suffix := to_char(boundary AT TIME ZONE 'UTC',
                CASE parent.cadence WHEN 'day' THEN 'YYYYMMDD' ELSE 'YYYYMM' END);
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.%I PARTITION OF %I.%I FOR VALUES FROM (%L) TO (%L)',
                parent.schema_name, parent.table_name || '_' || suffix,
                parent.schema_name, parent.table_name, boundary, next_boundary);
            IF NOT EXISTS (
                SELECT 1 FROM pg_inherits
                WHERE inhparent = to_regclass(format('%I.%I', parent.schema_name, parent.table_name))
                AND inhrelid = to_regclass(format('%I.%I', parent.schema_name,
                                                 parent.table_name || '_' || suffix))
            ) THEN
                RAISE EXCEPTION 'partition name occupied by unrelated relation';
            END IF;
            boundary := next_boundary;
        END LOOP;
    END LOOP;
END $$;
REVOKE ALL ON FUNCTION ops.ensure_partitions(timestamptz,timestamptz) FROM PUBLIC;
-- Run daily as the migration/partition owner; backfills provision their own range.
SELECT ops.ensure_partitions(now() - interval '7 days', now() + interval '62 days');
