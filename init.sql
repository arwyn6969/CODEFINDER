-- Initialize Ancient Text Analyzer Database
-- This script sets up the initial database structure

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create initial indexes for full-text search
-- Additional tables will be created by SQLAlchemy models