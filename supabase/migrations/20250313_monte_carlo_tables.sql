-- Monte Carlo simulation tables migration
-- This script creates the necessary tables for storing Monte Carlo simulations and scenarios

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Monte Carlo Simulations Table
CREATE TABLE IF NOT EXISTS monte_carlo_simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    request JSONB NOT NULL,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_monte_carlo_simulations_user_id ON monte_carlo_simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_monte_carlo_simulations_created_at ON monte_carlo_simulations(created_at DESC);

-- Scenario Definitions Table for reusable scenarios
CREATE TABLE IF NOT EXISTS monte_carlo_scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    asset_class VARCHAR(50) NOT NULL,
    variables JSONB NOT NULL,
    correlation_matrix JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_monte_carlo_scenarios_user_id ON monte_carlo_scenarios(user_id);
CREATE INDEX IF NOT EXISTS idx_monte_carlo_scenarios_created_at ON monte_carlo_scenarios(created_at DESC);

-- Row-Level Security (RLS) Policies
-- These policies ensure users can only access their own data

-- RLS for monte_carlo_simulations
ALTER TABLE monte_carlo_simulations ENABLE ROW LEVEL SECURITY;

-- Policy for selecting simulations (users can only see their own)
CREATE POLICY select_own_simulations 
    ON monte_carlo_simulations 
    FOR SELECT 
    USING (auth.uid() = user_id);

-- Policy for inserting simulations (users can only insert their own)
CREATE POLICY insert_own_simulations 
    ON monte_carlo_simulations 
    FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

-- Policy for updating simulations (users can only update their own)
CREATE POLICY update_own_simulations 
    ON monte_carlo_simulations 
    FOR UPDATE 
    USING (auth.uid() = user_id);

-- Policy for deleting simulations (users can only delete their own)
CREATE POLICY delete_own_simulations 
    ON monte_carlo_simulations 
    FOR DELETE 
    USING (auth.uid() = user_id);

-- RLS for monte_carlo_scenarios
ALTER TABLE monte_carlo_scenarios ENABLE ROW LEVEL SECURITY;

-- Policy for selecting scenarios (users can only see their own)
CREATE POLICY select_own_scenarios 
    ON monte_carlo_scenarios 
    FOR SELECT 
    USING (auth.uid() = user_id);

-- Policy for inserting scenarios (users can only insert their own)
CREATE POLICY insert_own_scenarios 
    ON monte_carlo_scenarios 
    FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

-- Policy for updating scenarios (users can only update their own)
CREATE POLICY update_own_scenarios 
    ON monte_carlo_scenarios 
    FOR UPDATE 
    USING (auth.uid() = user_id);

-- Policy for deleting scenarios (users can only delete their own)
CREATE POLICY delete_own_scenarios 
    ON monte_carlo_scenarios 
    FOR DELETE 
    USING (auth.uid() = user_id);

-- Add admin policies to allow administrators to access all records
-- First, ensure we have a is_admin column in the users table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_schema = 'auth' 
        AND table_name = 'users' 
        AND column_name = 'is_admin'
    ) THEN
        ALTER TABLE auth.users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
    END IF;
END$$;

-- Admin policies for monte_carlo_simulations
CREATE POLICY admin_access_simulations 
    ON monte_carlo_simulations 
    FOR ALL 
    USING (
        EXISTS (
            SELECT 1 FROM auth.users 
            WHERE id = auth.uid() AND is_admin = TRUE
        )
    );

-- Admin policies for monte_carlo_scenarios
CREATE POLICY admin_access_scenarios 
    ON monte_carlo_scenarios 
    FOR ALL 
    USING (
        EXISTS (
            SELECT 1 FROM auth.users 
            WHERE id = auth.uid() AND is_admin = TRUE
        )
    );

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update the updated_at column on update
CREATE TRIGGER set_timestamp_monte_carlo_simulations
BEFORE UPDATE ON monte_carlo_simulations
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER set_timestamp_monte_carlo_scenarios
BEFORE UPDATE ON monte_carlo_scenarios
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
