-- =====================================================================
-- Supabase pgvector Setup for AI Knowledge Cloud System
-- Run this script in your Supabase SQL Editor (Dashboard -> SQL Editor)
-- =====================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table for LangChain SupabaseVectorStore
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768) -- Matches Gemini text-embedding-004 (768 dimensions)
);

-- 3. Create document matching function for similarity search
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(768),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
) RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 4. Create an HNSW or IVFFlat index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops);

-- 5. Optional Initial Knowledge Base Seed
INSERT INTO documents (content, metadata)
VALUES 
(
    'AI Knowledge Cloud System Architecture: Decoupled web application with a Python Flask REST API deployed on Render.com and a high-tech SPA frontend deployed on Surge.sh. Vector storage is managed via Supabase with pgvector for zero-data-loss RAG.',
    '{"category": "architecture", "title": "System Architecture Overview"}'::jsonb
),
(
    'Frontend Deployment Guide: The frontend is hosted on Surge.sh. Deployment is triggered via npx surge ./frontend https://your-domain.surge.sh. Ensure CORS is enabled on the backend to allow incoming cross-origin requests.',
    '{"category": "devops", "title": "Frontend Surge Deployment"}'::jsonb
),
(
    'Backend Render Deployment Guide: Python Flask service hosted on Render.com using Gunicorn WSGI. Build command: pip install -r requirements.txt. Start command: gunicorn app:app. Environment variables: GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, DATABASE_URL.',
    '{"category": "devops", "title": "Render Backend Deployment"}'::jsonb
)
ON CONFLICT DO NOTHING;
