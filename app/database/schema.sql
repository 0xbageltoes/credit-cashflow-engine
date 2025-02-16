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

-- Create indexes
create index loans_user_id_idx on public.loans(user_id);
create index cashflow_projections_loan_id_idx on public.cashflow_projections(loan_id);
create index monte_carlo_results_loan_id_idx on public.monte_carlo_results(loan_id);
create index audit_log_user_id_idx on public.audit_log(user_id);
create index audit_log_entity_idx on public.audit_log(entity_type, entity_id);

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
