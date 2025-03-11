"""Tests for the absbox integration in the cashflow engine."""
import pytest
import os
import json
from datetime import date
from fastapi.testclient import TestClient

from app.main import app
from app.models.cashflow import (
    CashflowForecastRequest,
    LoanData
)
from app.services.absbox_service import AbsBoxService

client = TestClient(app)

@pytest.fixture
def mock_auth_header():
    """Mock authorization header for testing."""
    return {"Authorization": "Bearer test_token"}

@pytest.fixture
def test_loan_data():
    """Sample loan data for testing."""
    return LoanData(
        loan_id="test-loan-001",
        principal=100000.0,
        interest_rate=4.5,
        term_months=360,
        origination_date=date(2023, 1, 1),
        loan_type="fixed",
        payment_frequency="monthly",
        amortization_type="level_payment"
    )

@pytest.fixture
def test_forecast_request(test_loan_data):
    """Sample forecast request for testing."""
    return CashflowForecastRequest(
        loan_data=test_loan_data,
        discount_rate=0.05,
        run_monte_carlo=False,
        include_analytics=True
    )

def test_absbox_service_initialization():
    """Test that the AbsBoxService can be initialized."""
    service = AbsBoxService()
    assert service is not None

def test_generate_amortization_schedule(test_loan_data):
    """Test that amortization schedule can be generated with absbox."""
    service = AbsBoxService()
    schedule = service.generate_amortization_schedule(test_loan_data)
    
    # Check basic structure
    assert len(schedule) > 0
    assert "period" in schedule[0]
    assert "date" in schedule[0]
    assert "principal" in schedule[0]
    assert "interest" in schedule[0]
    assert "total_payment" in schedule[0]
    assert "remaining_balance" in schedule[0]

def test_calculate_analytics_metrics(test_loan_data):
    """Test that analytics metrics can be calculated with absbox."""
    service = AbsBoxService()
    schedule = service.generate_amortization_schedule(test_loan_data)
    
    # Calculate analytics metrics
    metrics = service.calculate_analytics_metrics(
        schedule, 
        discount_rate=0.05
    )
    
    # Check that required metrics exist
    assert metrics is not None
    assert "npv" in metrics
    assert "irr" in metrics
    assert "duration" in metrics
    assert "convexity" in metrics
    
    # Basic validation
    assert metrics["npv"] != 0
    assert -0.1 <= metrics["irr"] <= 0.2  # Reasonable IRR range
    assert metrics["duration"] > 0
    assert metrics["convexity"] >= 0

@pytest.mark.asyncio
async def test_generate_cashflow_forecast(test_forecast_request):
    """Test that a full cashflow forecast can be generated."""
    service = AbsBoxService()
    forecast = service.generate_cashflow_forecast(test_forecast_request)
    
    # Check structure
    assert forecast.loan_id == test_forecast_request.loan_data.loan_id
    assert len(forecast.projections) > 0
    assert "total_principal" in forecast.summary_metrics
    assert "total_interest" in forecast.summary_metrics
    assert "npv" in forecast.summary_metrics
    assert "irr" in forecast.summary_metrics
    assert "duration" in forecast.summary_metrics
    assert "convexity" in forecast.summary_metrics

@pytest.mark.skip("Endpoint testing requires auth setup")
def test_forecast_endpoint(mock_auth_header, test_forecast_request):
    """Test the forecast endpoint with absbox integration."""
    response = client.post(
        "/api/v1/forecasts/",
        headers=mock_auth_header,
        json=test_forecast_request.model_dump()
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "loan_id" in data
    assert "projections" in data
    assert "summary_metrics" in data
    
    metrics = data["summary_metrics"]
    assert "npv" in metrics
    assert "irr" in metrics
    assert "duration" in metrics
    assert "convexity" in metrics
