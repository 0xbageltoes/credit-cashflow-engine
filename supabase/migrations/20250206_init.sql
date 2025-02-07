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

-- Enable RLS
alter table public.cashflow_scenarios enable row level security;

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

-- Create forecast_runs table
create table if not exists public.forecast_runs (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade,
    request jsonb not null,
    projections jsonb not null,
    created_at timestamptz default now()
);

-- Enable RLS
alter table public.forecast_runs enable row level security;

-- Create RLS policies
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
