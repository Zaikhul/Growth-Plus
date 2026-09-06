-- Execute as the migration owner. These roles contain no embedded login secret.
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'growth_app') THEN
        CREATE ROLE growth_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    END IF;
END $$;
ALTER ROLE growth_app NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
CREATE SCHEMA IF NOT EXISTS tenant;
REVOKE CREATE ON SCHEMA tenant FROM PUBLIC, growth_app;
GRANT USAGE ON SCHEMA tenant TO growth_app;

CREATE TABLE IF NOT EXISTS tenant.watchlists (
    id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    name text NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
    markets text[] NOT NULL CHECK (cardinality(markets) <= 20),
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0)
);
CREATE TABLE IF NOT EXISTS tenant.alert_rules (
    id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    specification jsonb NOT NULL CHECK (jsonb_typeof(specification) = 'object'),
    consent_at timestamptz NOT NULL,
    enabled boolean NOT NULL DEFAULT false,
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0)
);
CREATE TABLE IF NOT EXISTS tenant.notification_deliveries (
    id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    rule_id uuid NOT NULL,
    signal_id uuid NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING','SENT','EXPIRED','FAILED')),
    created_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL CHECK (expires_at > created_at)
);
CREATE UNIQUE INDEX IF NOT EXISTS alert_rule_tenant_key ON tenant.alert_rules (tenant_id,id);
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_constraint WHERE conname = 'delivery_rule_tenant_fk'
                   AND conrelid = 'tenant.notification_deliveries'::regclass) THEN
        ALTER TABLE tenant.notification_deliveries ADD CONSTRAINT delivery_rule_tenant_fk
            FOREIGN KEY (tenant_id, rule_id) REFERENCES tenant.alert_rules(tenant_id,id);
    END IF;
END $$;
DO $$
DECLARE item record;
BEGIN
    FOR item IN SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='tenant' AND c.relkind IN ('r','p') LOOP
        IF NOT EXISTS (SELECT FROM pg_attribute
                       WHERE attrelid=format('tenant.%I',item.relname)::regclass
                         AND attname='tenant_id' AND atttypid='uuid'::regtype
                         AND attnotnull AND NOT attisdropped) THEN
            RAISE EXCEPTION 'tenant table % requires nonnull uuid tenant_id', item.relname;
        END IF;
        EXECUTE format('ALTER TABLE tenant.%I ENABLE ROW LEVEL SECURITY',item.relname);
        EXECUTE format('ALTER TABLE tenant.%I FORCE ROW LEVEL SECURITY',item.relname);
        EXECUTE format('DROP POLICY IF EXISTS tenant_allow ON tenant.%I',item.relname);
        EXECUTE format('DROP POLICY IF EXISTS tenant_fence ON tenant.%I',item.relname);
        EXECUTE format(
            'CREATE POLICY tenant_allow ON tenant.%I FOR ALL TO growth_app USING '
            '(tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK '
            '(tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)', item.relname);
        -- Restrictive policy also fences any other permissive policy on the table.
        EXECUTE format(
            'CREATE POLICY tenant_fence ON tenant.%I AS RESTRICTIVE FOR ALL TO growth_app USING '
            '(tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK '
            '(tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)', item.relname);
        EXECUTE format('REVOKE ALL ON tenant.%I FROM PUBLIC',item.relname);
        EXECUTE format('GRANT SELECT,INSERT,UPDATE,DELETE ON tenant.%I TO growth_app',item.relname);
    END LOOP;
END $$;
