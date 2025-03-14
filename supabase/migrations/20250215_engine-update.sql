-- Create loans table
create table if not exists public.loans (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade,
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
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Create monte_carlo_results table
create table if not exists public.monte_carlo_results (
    id uuid default gen_random_uuid() primary key,
    loan_id uuid references public.loans(id) on delete cascade,
    user_id uuid references auth.users(id) on delete cascade,
    npv_distribution jsonb not null,
    confidence_intervals jsonb not null,
    var_metrics jsonb not null,
    sensitivity_analysis jsonb not null,
    stress_test_results jsonb,
    created_at timestamptz default now()
);

-- Create audit_log table
create table if not exists public.audit_log (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade,
    action text not null,
    entity_type text not null,
    entity_id uuid not null,
    changes jsonb,
    created_at timestamptz default now()
);

-- Create indexes
create index if not exists idx_loans_user_id on public.loans(user_id);
create index if not exists idx_loans_created_at on public.loans(created_at);
create index if not exists idx_monte_carlo_loan_id on public.monte_carlo_results(loan_id);
create index if not exists idx_monte_carlo_user_id on public.monte_carlo_results(user_id);
create index if not exists idx_audit_user_id on public.audit_log(user_id);
create index if not exists idx_audit_entity on public.audit_log(entity_type, entity_id);

-- Enable RLS
alter table public.loans enable row level security;
alter table public.monte_carlo_results enable row level security;
alter table public.audit_log enable row level security;

-- Create RLS policies for loans
create policy "Users can view their own loans"
    on public.loans
    for select
    using (auth.uid() = user_id);

create policy "Users can insert their own loans"
    on public.loans
    for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own loans"
    on public.loans
    for update
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users can delete their own loans"
    on public.loans
    for delete
    using (auth.uid() = user_id);

-- Create RLS policies for monte_carlo_results
create policy "Users can view their own monte carlo results"
    on public.monte_carlo_results
    for select
    using (auth.uid() = user_id);

create policy "Users can insert their own monte carlo results"
    on public.monte_carlo_results
    for insert
    with check (auth.uid() = user_id);

-- Create RLS policies for audit_log
create policy "Users can view their own audit logs"
    on public.audit_log
    for select
    using (auth.uid() = user_id);

create policy "Users can insert audit logs"
    on public.audit_log
    for insert
    with check (auth.uid() = user_id);

-- Create trigger for loans updated_at
drop trigger if exists handle_loans_updated_at on public.loans;
create trigger handle_loans_updated_at
    before update on public.loans
    for each row
    execute function public.handle_updated_at();