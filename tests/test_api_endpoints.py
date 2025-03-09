"""
Tests for the API endpoints of the Credit Cashflow Engine
"""

import pytest
from fastapi.testclient import TestClient
import datetime
import uuid
from unittest.mock import patch, MagicMock
from fastapi import Depends

# Import the app
from app.main import app
from app.core.auth import get_current_user


# Mock the authentication dependency
@pytest.fixture(autouse=True)
def mock_auth():
    """Mock the authentication process to bypass JWT validation"""
    # Create a mock user that will be returned by the dependency
    mock_user = {
        "id": "00000000-0000-0000-0000-000000000000",
        "email": "test@example.com",
        "role": "user"
    }
    
    # Override the get_current_user dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    yield
    
    # Reset dependency overrides after test
    app.dependency_overrides = {}


@pytest.fixture
def client():
    """Create a test client for the API"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    """Mocked authentication headers"""
    return {"Authorization": "Bearer test_token"}


def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime" in data


def test_readiness_endpoint(client):
    """Test the readiness endpoint"""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


@patch("app.services.cashflow.CashflowService.calculate_loan_cashflow")
def test_cashflow_calculate_endpoint(mock_calculate, client, auth_headers):
    """Test the cashflow calculation endpoint with mocked service"""
    # Set up mock return value
    mock_calculate.return_value = {
        "cashflows": [
            {"period": 1, "date": "2025-02-01", "payment": 536.82, "principal": 120.15, "interest": 416.67, "balance": 99879.85},
            {"period": 2, "date": "2025-03-01", "payment": 536.82, "principal": 120.65, "interest": 416.17, "balance": 99759.20}
        ],
        "summary": {
            "total_interest": 93255.78,
            "total_payments": 193255.78,
            "loan_amount": 100000.00
        }
    }

    # Prepare the request - updated to match LoanData model
    loan_data = {
        "principal": 100000,
        "interest_rate": 0.05,  # Changed from rate
        "term_months": 360,     # Changed from term
        "start_date": "2025-01-01"
    }

    # Make the request
    response = client.post("/api/v1/cashflow/calculate", json=loan_data, headers=auth_headers)

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "cashflows" in data
    assert "summary" in data
    assert len(data["cashflows"]) == 2


@patch("app.services.cashflow.CashflowService.calculate_batch")
def test_cashflow_batch_endpoint(mock_batch, client, auth_headers):
    """Test the batch calculation endpoint with mocked service"""
    # Set up mock return value
    mock_batch.return_value = {
        "results": [
            {
                "id": "loan-1",
                "summary": {
                    "total_interest": 93255.78,
                    "total_payments": 193255.78,
                    "loan_amount": 100000.00
                }
            },
            {
                "id": "loan-2",
                "summary": {
                    "total_interest": 52500.12,
                    "total_payments": 152500.12,
                    "loan_amount": 100000.00
                }
            }
        ],
        "execution_time": 0.125
    }

    # Prepare the request - updated to match BatchLoanRequest model
    batch_data = {
        "loans": [
            {
                "principal": 100000,
                "interest_rate": 0.05,  # Changed from rate
                "term_months": 360,     # Changed from term
                "start_date": "2025-01-01"
            },
            {
                "principal": 100000,
                "interest_rate": 0.035,  # Changed from rate
                "term_months": 240,      # Changed from term
                "start_date": "2025-02-01"
            }
        ],
        "parallel": True
    }

    # Make the request
    response = client.post("/api/v1/cashflow/calculate-batch", json=batch_data, headers=auth_headers)

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "execution_time" in data
    assert len(data["results"]) == 2


@patch("app.services.analytics.AnalyticsService.get_api_metrics")
def test_metrics_endpoint(mock_metrics, client, auth_headers):
    """Test the metrics endpoint with mocked analytics service"""
    # Set up mock return value
    mock_metrics.return_value = {
        "total_requests": 1250,
        "average_response_time": 0.125,
        "error_rate": 0.01,
        "requests_by_endpoint": {
            "/api/v1/cashflow/calculate": 850,
            "/api/v1/cashflow/calculate-batch": 400
        }
    }

    # Make the request
    response = client.get("/api/v1/api-metrics", headers=auth_headers)  # Changed from /metrics to /api-metrics

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "average_response_time" in data
    assert "error_rate" in data
    assert "requests_by_endpoint" in data


def test_invalid_auth(client):
    """Test that endpoints require authentication"""
    # Override the dependency for this test to ensure it requires auth
    app.dependency_overrides = {}
    
    # Try to access an endpoint without auth headers
    response = client.post("/api/v1/cashflow/calculate", 
                         json={"principal": 100000, "interest_rate": 0.05, "term_months": 360, "start_date": "2025-01-01"})
    
    assert response.status_code in [401, 403]  # Either is acceptable
    
    # Restore the dependency override for other tests
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "00000000-0000-0000-0000-000000000000",
        "email": "test@example.com",
        "role": "user"
    }
