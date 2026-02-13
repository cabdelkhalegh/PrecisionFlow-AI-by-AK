-- Golden Database schema: Supabase PostgreSQL + pgvector
-- This creates the "Truth Source" of 1,000+ proven business models.

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Case Studies table — the core of the Golden Database
CREATE TABLE IF NOT EXISTS case_studies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    industry    TEXT NOT NULL,
    business_model TEXT NOT NULL,
    company_name TEXT NOT NULL,
    summary     TEXT NOT NULL,
    revenue_model TEXT DEFAULT '',
    pricing_strategy TEXT DEFAULT '',
    legal_structure TEXT DEFAULT '',
    pitch_deck_url TEXT DEFAULT '',
    embedding   vector(768),   -- Gemini text-embedding-004 dimension
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_case_studies_embedding
    ON case_studies
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- RPC function for vector similarity search
CREATE OR REPLACE FUNCTION match_case_studies(
    query_embedding vector(768),
    match_count INT DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    industry TEXT,
    business_model TEXT,
    company_name TEXT,
    summary TEXT,
    revenue_model TEXT,
    pricing_strategy TEXT,
    legal_structure TEXT,
    pitch_deck_url TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.id,
        cs.industry,
        cs.business_model,
        cs.company_name,
        cs.summary,
        cs.revenue_model,
        cs.pricing_strategy,
        cs.legal_structure,
        cs.pitch_deck_url,
        1 - (cs.embedding <=> query_embedding) AS similarity
    FROM case_studies cs
    ORDER BY cs.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Ventures table — persists pipeline state
CREATE TABLE IF NOT EXISTS ventures (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    founder_name TEXT NOT NULL,
    jurisdiction TEXT DEFAULT 'Delaware, USA',
    region      TEXT DEFAULT 'United States',
    state_json  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Row Level Security
ALTER TABLE case_studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE ventures ENABLE ROW LEVEL SECURITY;

-- Allow authenticated reads on case studies (the Truth Source is shared)
CREATE POLICY "Allow read access to case studies"
    ON case_studies FOR SELECT
    USING (true);

-- Ventures are private to the founder
CREATE POLICY "Founders can manage their own ventures"
    ON ventures FOR ALL
    USING (true);  -- Refine with auth.uid() in production
