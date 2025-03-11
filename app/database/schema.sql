-- Enable Row Level Security
alter table public.loans enable row level security;
alter table public.cashflow_projections enable row level security;
alter table public.monte_carlo_results enable row level security;
alter table public.audit_log enable row level security;

-- Create tables
create table public.loans (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    principal numeric not null,
    interest_rate numeric not null,
    term_months integer not null,
    start_date date not null,
    prepayment_assumption numeric,
    rate_type text check (rate_type in ('fixed', 'hybrid')),
    balloon_payment numeric,
    interest_only_periods integer,
    rate_spread numeric,
    rate_cap numeric,
    rate_floor numeric,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.cashflow_projections (
    id uuid default uuid_generate_v4() primary key,
    loan_id uuid references public.loans not null,
    user_id uuid references auth.users not null,
    period integer not null,
    date date not null,
    principal numeric not null,
    interest numeric not null,
    total_payment numeric not null,
    remaining_balance numeric not null,
    is_interest_only boolean not null,
    is_balloon boolean not null,
    rate numeric not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.monte_carlo_results (
    id uuid default uuid_generate_v4() primary key,
    loan_id uuid references public.loans not null,
    user_id uuid references auth.users not null,
    npv_distribution jsonb not null,
    confidence_intervals jsonb not null,
    var_metrics jsonb not null,
    sensitivity_analysis jsonb not null,
    stress_test_results jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.audit_log (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    action text not null,
    entity_type text not null,
    entity_id uuid not null,
    changes jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create forecast_runs table to store forecast metadata and results
create table public.forecast_runs (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    scenario_name text not null,
    total_principal numeric not null,
    total_interest numeric not null,
    npv numeric not null,
    irr numeric not null,
    duration numeric not null,
    convexity numeric not null,
    yield_value numeric,
    spread_value numeric,
    macaulay_duration numeric,
    modified_duration numeric,
    discount_margin numeric,
    debt_service_coverage numeric,
    weighted_average_life numeric,
    monte_carlo_results jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create cashflow_scenarios table to store scenario definitions
create table public.cashflow_scenarios (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    name text not null,
    description text,
    forecast_config jsonb not null,
    tags text[],
    is_template boolean default false,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create forecast_projections table to store projected cashflows
create table public.forecast_projections (
    id uuid default uuid_generate_v4() primary key,
    forecast_id uuid references public.forecast_runs not null,
    user_id uuid references auth.users not null,
    period integer not null,
    date date not null,
    principal numeric not null,
    interest numeric not null,
    total_payment numeric not null,
    remaining_balance numeric not null,
    default_amount numeric,
    prepayment_amount numeric,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create enhanced_analytics_results table for storing absbox enhanced analytics
create table public.enhanced_analytics_results (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    loan_id uuid references public.loans not null,
    npv numeric not null,
    irr numeric not null,
    yield_value numeric not null,
    duration numeric not null,
    macaulay_duration numeric not null,
    convexity numeric not null,
    discount_margin numeric,
    z_spread numeric,
    e_spread numeric,
    weighted_average_life numeric not null,
    debt_service_coverage numeric,
    interest_coverage_ratio numeric,
    sensitivity_metrics jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create structured_deal_results table for storing absbox structured deal analysis
create table public.structured_deal_results (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users not null,
    deal_name text not null,
    execution_time numeric not null,
    bond_cashflows jsonb not null,
    pool_cashflows jsonb not null,
    pool_statistics jsonb not null,
    metrics jsonb not null,
    status text not null,
    error text,
    error_type text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes
create index loans_user_id_idx on public.loans(user_id);
create index cashflow_projections_loan_id_idx on public.cashflow_projections(loan_id);
create index monte_carlo_results_loan_id_idx on public.monte_carlo_results(loan_id);
create index audit_log_user_id_idx on public.audit_log(user_id);
create index audit_log_entity_idx on public.audit_log(entity_type, entity_id);

-- Create indexes for new tables
create index forecast_runs_user_id_idx on public.forecast_runs(user_id);
create index cashflow_scenarios_user_id_idx on public.cashflow_scenarios(user_id);
create index forecast_projections_forecast_id_idx on public.forecast_projections(forecast_id);
create index forecast_projections_user_id_idx on public.forecast_projections(user_id);

-- Create indexes for enhanced analytics tables
create index enhanced_analytics_results_user_id_idx on public.enhanced_analytics_results(user_id);
create index enhanced_analytics_results_loan_id_idx on public.enhanced_analytics_results(loan_id);
create index structured_deal_results_user_id_idx on public.structured_deal_results(user_id);

-- Row Level Security policies
create policy "Users can only see their own loans"
    on loans for select
    using (auth.uid() = user_id);

create policy "Users can only insert their own loans"
    on loans for insert
    with check (auth.uid() = user_id);

create policy "Users can only update their own loans"
    on loans for update
    using (auth.uid() = user_id);

create policy "Users can only delete their own loans"
    on loans for delete
    using (auth.uid() = user_id);

-- Similar policies for other tables
create policy "Users can only see their own projections"
    on cashflow_projections for select
    using (auth.uid() = user_id);

create policy "Users can only see their own monte carlo results"
    on monte_carlo_results for select
    using (auth.uid() = user_id);

create policy "Users can only see their own audit logs"
    on audit_log for select
    using (auth.uid() = user_id);

-- Add RLS policies for new tables
create policy "Users can only see their own forecast runs"
    on forecast_runs for select
    using (auth.uid() = user_id);

create policy "Users can only see their own scenarios"
    on cashflow_scenarios for select
    using (auth.uid() = user_id);

create policy "Users can only see their own forecast projections"
    on forecast_projections for select
    using (auth.uid() = user_id);

create policy "Users can only update their own forecast runs"
    on forecast_runs for update
    using (auth.uid() = user_id);

create policy "Users can only update their own scenarios"
    on cashflow_scenarios for update
    using (auth.uid() = user_id);

create policy "Users can only insert their own forecast runs"
    on forecast_runs for insert
    with check (auth.uid() = user_id);

create policy "Users can only insert their own scenarios"
    on cashflow_scenarios for insert
    with check (auth.uid() = user_id);

create policy "Users can only insert their own forecast projections"
    on forecast_projections for insert
    with check (auth.uid() = user_id);

create policy "Users can only delete their own forecast runs"
    on forecast_runs for delete
    using (auth.uid() = user_id);

create policy "Users can only delete their own scenarios"
    on cashflow_scenarios for delete
    using (auth.uid() = user_id);

create policy "Users can only delete their own forecast projections"
    on forecast_projections for delete
    using (auth.uid() = user_id);

-- RLS policies for enhanced analytics tables
create policy "Users can only see their own enhanced analytics"
    on enhanced_analytics_results for select
    using (auth.uid() = user_id);

create policy "Users can only insert their own enhanced analytics"
    on enhanced_analytics_results for insert
    with check (auth.uid() = user_id);

create policy "Users can only update their own enhanced analytics"
    on enhanced_analytics_results for update
    using (auth.uid() = user_id);

create policy "Users can only delete their own enhanced analytics"
    on enhanced_analytics_results for delete
    using (auth.uid() = user_id);

create policy "Users can only see their own deal results"
    on structured_deal_results for select
    using (auth.uid() = user_id);

create policy "Users can only insert their own deal results"
    on structured_deal_results for insert
    with check (auth.uid() = user_id);

create policy "Users can only update their own deal results"
    on structured_deal_results for update
    using (auth.uid() = user_id);

create policy "Users can only delete their own deal results"
    on structured_deal_results for delete
    using (auth.uid() = user_id);

-- Enable RLS on new tables
alter table public.enhanced_analytics_results enable row level security;
alter table public.structured_deal_results enable row level security;

-- Functions
create or replace function public.handle_updated_at()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

-- Triggers
create trigger handle_loans_updated_at
    before update on public.loans
    for each row
    execute function public.handle_updated_at();

create trigger handle_enhanced_analytics_updated_at
    before update on public.enhanced_analytics_results
    for each row
    execute function public.handle_updated_at();
