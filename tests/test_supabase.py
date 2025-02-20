"""Tests for Supabase database integration"""
import pytest
from uuid import uuid4
from datetime import datetime
import pandas as pd
from app.database.supabase import SupabaseClient
from app.models.cashflow import (
    LoanData,
    CashflowForecastRequest,
    CashflowForecastResponse,
    CashflowProjection,
    MonteCarloResults
)
from tests.test_data import (
    get_sample_loan,
    get_monte_carlo_config,
    get_economic_factors
)

@pytest.fixture
def supabase_client():
    """Create a Supabase client for testing"""
    client = SupabaseClient()
    
    # Sign in with existing user - this is synchronous
    response = client.client.auth.sign_in_with_password({
        "email": "aleanos@icloud.com",
        "password": "Password1"
    })
    
    # Get user ID from session
    user_id = response.user.id
    client.user_id = user_id
    
    return client

@pytest.fixture
def test_loan():
    """Create a test loan"""
    return get_sample_loan()

def test_loan_crud(supabase_client, test_loan):
    """Test loan CRUD operations"""
    # Create loan
    created = supabase_client.create_loan(supabase_client.user_id, test_loan)
    assert created["id"] is not None
    assert created["principal"] == test_loan.principal
    assert created["interest_rate"] == test_loan.interest_rate
    
    # Get loan
    loan_id = created["id"]
    retrieved = supabase_client.get_loan(supabase_client.user_id, loan_id)
    assert retrieved is not None
    assert retrieved["id"] == loan_id
    assert retrieved["principal"] == test_loan.principal
    
    # Update loan
    updated_loan = test_loan.model_copy()
    updated_loan.principal = 200000.0
    updated = supabase_client.update_loan(supabase_client.user_id, loan_id, updated_loan)
    assert updated["principal"] == 200000.0
    
    # List loans
    loans = supabase_client.list_loans(supabase_client.user_id)
    assert len(loans) > 0
    assert any(loan["id"] == loan_id for loan in loans)
    
    # Delete loan
    supabase_client.delete_loan(supabase_client.user_id, loan_id)
    deleted = supabase_client.get_loan(supabase_client.user_id, loan_id)
    assert deleted is None

def test_cashflow_projections(supabase_client, test_loan):
    """Test saving and retrieving cashflow projections"""
    loan_id = str(uuid4())
    
    # Create sample projections
    projections = [
        CashflowProjection(
            period=i + 1,
            date=(datetime.now() + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
            principal=1000.0,
            interest=50.0,
            total_payment=1050.0,
            remaining_balance=99000.0 - (i * 1000.0),
            is_interest_only=False,
            is_balloon=False,
            rate=0.05
        )
        for i in range(12)
    ]
    
    # Create sample Monte Carlo results
    monte_carlo_results = MonteCarloResults(
        npv_distribution=[100000.0 + i * 1000.0 for i in range(10)],
        default_scenarios=[{"period": i, "probability": 0.02} for i in range(5)],
        prepayment_scenarios=[{"period": i, "probability": 0.05} for i in range(5)],
        rate_scenarios=[{"period": i, "rate": 0.05} for i in range(5)],
        confidence_intervals={
            "npv": {
                "95": [95000.0, 105000.0],
                "99": [93000.0, 107000.0]
            }
        },
        var_metrics={
            "var_95": 5000.0,
            "var_99": 7000.0,
            "expected_shortfall": 6000.0
        },
        sensitivity_analysis={
            "rate_sensitivity": -0.8
        }
    )
    
    # Create response object
    response = CashflowForecastResponse(
        projections=projections,
        monte_carlo_results=monte_carlo_results,
        summary_metrics={
            "total_principal": 100000.0,
            "total_interest": 5000.0,
            "total_payments": 105000.0,
            "npv": 100000.0
        },
        computation_time=1.5
    )
    
    # Save projections
    supabase_client.save_cashflow_projections(supabase_client.user_id, loan_id, response)
    
    # Retrieve projections
    saved_projections = supabase_client.get_cashflow_projections(supabase_client.user_id, loan_id)
    assert len(saved_projections) == len(projections)
    assert saved_projections[0]["principal"] == projections[0].principal
    assert saved_projections[0]["interest"] == projections[0].interest
    assert saved_projections[0]["total_payment"] == projections[0].total_payment
    assert saved_projections[0]["remaining_balance"] == projections[0].remaining_balance
    
    # Retrieve Monte Carlo results
    monte_carlo = supabase_client.get_monte_carlo_results(supabase_client.user_id, loan_id)
    assert monte_carlo is not None
    assert monte_carlo["npv_distribution"] == monte_carlo_results.npv_distribution
    assert monte_carlo["confidence_intervals"] == monte_carlo_results.confidence_intervals
