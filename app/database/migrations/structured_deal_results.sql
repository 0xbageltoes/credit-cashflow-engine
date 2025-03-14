-- Migration script to safely add structured_deal_results table
-- Can be run in Supabase web dashboard

-- Only create the table if it doesn't already exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'structured_deal_results'
    ) THEN
        -- Create structured_deal_results table for storing absbox structured deal analysis
        CREATE TABLE public.structured_deal_results (
            id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
            user_id uuid REFERENCES auth.users NOT NULL,
            deal_name text NOT NULL,
            execution_time numeric NOT NULL,
            bond_cashflows jsonb NOT NULL,
            pool_cashflows jsonb NOT NULL,
            pool_statistics jsonb NOT NULL,
            metrics jsonb NOT NULL,
            status text NOT NULL,
            error text,
            error_type text,
            created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
            updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
        );

        -- Create index if it doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND tablename = 'structured_deal_results' 
            AND indexname = 'structured_deal_results_user_id_idx'
        ) THEN
            CREATE INDEX structured_deal_results_user_id_idx ON public.structured_deal_results(user_id);
        END IF;

        -- Enable Row Level Security
        ALTER TABLE public.structured_deal_results ENABLE ROW LEVEL SECURITY;

        -- Create RLS policies if they don't exist
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE schemaname = 'public' 
            AND tablename = 'structured_deal_results' 
            AND policyname = 'Users can only see their own structured deal results'
        ) THEN
            CREATE POLICY "Users can only see their own structured deal results"
                ON structured_deal_results FOR SELECT
                USING (auth.uid() = user_id);
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE schemaname = 'public' 
            AND tablename = 'structured_deal_results' 
            AND policyname = 'Users can only insert their own structured deal results'
        ) THEN
            CREATE POLICY "Users can only insert their own structured deal results"
                ON structured_deal_results FOR INSERT
                WITH CHECK (auth.uid() = user_id);
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE schemaname = 'public' 
            AND tablename = 'structured_deal_results' 
            AND policyname = 'Users can only update their own structured deal results'
        ) THEN
            CREATE POLICY "Users can only update their own structured deal results"
                ON structured_deal_results FOR UPDATE
                USING (auth.uid() = user_id);
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE schemaname = 'public' 
            AND tablename = 'structured_deal_results' 
            AND policyname = 'Users can only delete their own structured deal results'
        ) THEN
            CREATE POLICY "Users can only delete their own structured deal results"
                ON structured_deal_results FOR DELETE
                USING (auth.uid() = user_id);
        END IF;

        RAISE NOTICE 'Created structured_deal_results table with RLS policies and indexes';
    ELSE
        -- Table already exists, check if we need to add the updated_at column
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'structured_deal_results' 
            AND column_name = 'updated_at'
        ) THEN
            ALTER TABLE public.structured_deal_results 
            ADD COLUMN updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL;
            
            RAISE NOTICE 'Added updated_at column to existing structured_deal_results table';
        ELSE
            RAISE NOTICE 'structured_deal_results table already exists with all required columns';
        END IF;
    END IF;
END
$$;
