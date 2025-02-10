-- Drop existing policies if they exist
drop policy if exists "Users can view their own scenarios" on public.cashflow_scenarios;
drop policy if exists "Users can insert their own scenarios" on public.cashflow_scenarios;
drop policy if exists "Users can update their own scenarios" on public.cashflow_scenarios;
drop policy if exists "Users can delete their own scenarios" on public.cashflow_scenarios;

drop policy if exists "Users can view their own forecasts" on public.forecast_runs;
drop policy if exists "Users can insert their own forecasts" on public.forecast_runs;
drop policy if exists "Users can update their own forecasts" on public.forecast_runs;
drop policy if exists "Users can delete their own forecasts" on public.forecast_runs;

-- Create cashflow_scenarios table
create table if not exists public.cashflow_scenarios (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade,
    name text not null,
    description text,
    forecast_request jsonb not null,
    tags text[] default array[]::text[],
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Create forecast_runs table
create table if not exists public.forecast_runs (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade,
    scenario_name text not null,
    total_principal numeric not null,
    total_interest numeric not null,
    npv numeric not null,
    irr numeric not null,
    duration numeric not null,
    convexity numeric not null,
    monte_carlo_results jsonb,
    created_at timestamptz default now()
);

-- Create forecast_projections table
create table if not exists public.forecast_projections (
    id uuid default gen_random_uuid() primary key,
    forecast_id uuid references public.forecast_runs(id) on delete cascade,
    user_id uuid references auth.users(id) on delete cascade,
    period integer not null,
    date text not null,
    principal numeric not null,
    interest numeric not null,
    total_payment numeric not null,
    remaining_balance numeric not null,
    created_at timestamptz default now()
);

-- Create indexes for better query performance
CREATE INDEX idx_scenarios_user_id ON public.cashflow_scenarios(user_id);
CREATE INDEX idx_scenarios_created_at ON public.cashflow_scenarios(created_at);
CREATE INDEX idx_forecast_runs_user_id ON public.forecast_runs(user_id);
CREATE INDEX idx_forecast_runs_created_at ON public.forecast_runs(created_at);

-- Add composite indexes for common query patterns
CREATE INDEX idx_scenarios_user_created ON public.cashflow_scenarios(user_id, created_at DESC);
CREATE INDEX idx_forecast_user_created ON public.forecast_runs(user_id, created_at DESC);

-- Enable RLS
alter table public.cashflow_scenarios enable row level security;
alter table public.forecast_runs enable row level security;
alter table public.forecast_projections enable row level security;

-- Create RLS policies
create policy "Users can view their own scenarios"
    on public.cashflow_scenarios
    for select
    using (auth.uid() = user_id);

create policy "Users can insert their own scenarios"
    on public.cashflow_scenarios
    for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own scenarios"
    on public.cashflow_scenarios
    for update
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users can delete their own scenarios"
    on public.cashflow_scenarios
    for delete
    using (auth.uid() = user_id);

create policy "Users can view their own forecasts"
    on public.forecast_runs
    for select
    using (auth.uid() = user_id);

create policy "Users can insert their own forecasts"
    on public.forecast_runs
    for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own forecasts"
    on public.forecast_runs
    for update
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users can delete their own forecasts"
    on public.forecast_runs
    for delete
    using (auth.uid() = user_id);

create policy "Users can view their own forecast projections"
    on public.forecast_projections
    for select
    using (auth.uid() = user_id);

create policy "Users can insert their own forecast projections"
    on public.forecast_projections
    for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own forecast projections"
    on public.forecast_projections
    for update
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users can delete their own forecast projections"
    on public.forecast_projections
    for delete
    using (auth.uid() = user_id);

-- Create updated_at trigger function
create or replace function public.handle_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

-- Create trigger for updated_at
drop trigger if exists handle_updated_at on public.cashflow_scenarios;
create trigger handle_updated_at
    before update on public.cashflow_scenarios
    for each row
    execute function public.handle_updated_at();
