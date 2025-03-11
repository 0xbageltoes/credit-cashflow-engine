"""
Tests for the structured products API endpoints
"""
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import json

from fastapi.testclient import TestClient
from app.models.structured_products import (
    StructuredDealRequest,
    StructuredDealResponse,
    LoanPoolConfig,
    LoanConfig,
    WaterfallConfig,
    AccountConfig,
    BondConfig,
    WaterfallAction,
    ScenarioConfig,
    DefaultCurveConfig
)
from app.main import app
from app.services.absbox_service import AbsBoxService

client = TestClient(app)

# Test data
def get_sample_deal_payload():
    """Get a sample deal payload for the API"""
    today = date.today()
    start_date = today - timedelta(days=30)
    
    return {
        "deal_name": "API Test Deal",
        "pool": {
            "pool_name": "Test Pool",
            "loans": [
                {
                    "balance": 200000.0,
                    "rate": 0.05,
                    "term": 360,
                    "start_date": start_date.isoformat(),
                    "rate_type": "fixed"
                },
                {
                    "balance": 100000.0,
                    "rate": 0.06,
                    "term": 240,
                    "start_date": start_date.isoformat(),
                    "rate_type": "fixed"
                }
            ]
        },
        "waterfall": {
            "start_date": today.isoformat(),
            "accounts": [
                {
                    "name": "ReserveFund",
                    "initial_balance": 5000.0
                }
            ],
            "bonds": [
                {
                    "name": "ClassA",
                    "balance": 250000.0,
                    "rate": 0.04
                },
                {
                    "name": "ClassB",
                    "balance": 50000.0,
                    "rate": 0.06
                }
            ],
            "actions": [
                {
                    "source": "CollectedInterest",
                    "target": "ClassA",
                    "amount": "Interest"
                },
                {
                    "source": "CollectedInterest",
                    "target": "ClassB",
                    "amount": "Interest"
                },
                {
                    "source": "CollectedPrincipal",
                    "target": "ClassA",
                    "amount": "OutstandingPrincipal"
                },
                {
                    "source": "CollectedPrincipal",
                    "target": "ClassB",
                    "amount": "OutstandingPrincipal"
                }
            ]
        },
        "scenario": {
            "name": "Base Scenario",
            "default_curve": {
                "vector": [0.01, 0.02, 0.02, 0.015, 0.01]
            }
        }
    }

def get_sample_scenarios_payload():
    """Get sample scenarios for the API"""
    return [
        {
            "name": "Base Case",
            "default_curve": {
                "vector": [0.01, 0.015, 0.02, 0.015, 0.01]
            }
        },
        {
            "name": "Stressed Case",
            "default_curve": {
                "vector": [0.03, 0.05, 0.06, 0.04, 0.03]
            }
        },
        {
            "name": "Recovery Case",
            "default_curve": {
                "vector": [0.02, 0.01, 0.005, 0.0, 0.0]
            }
        }
    ]

# Mock the JWT authentication for testing
@pytest.fixture(autouse=True)
def mock_auth():
    """Mock the authentication for testing"""
    with patch("app.api.deps.get_current_user") as mock:
        mock.return_value = {"user_id": "test-user", "role": "admin"}
        yield mock

# Mock the AbsBox service for testing
@pytest.fixture
def mock_absbox_service():
    """Mock the AbsBox service for testing"""
    with patch("app.services.absbox_service.AbsBoxService") as mock:
        service_instance = mock.return_value
        
        # Mock the analyze_deal method
        service_instance.analyze_deal.return_value = StructuredDealResponse(
            deal_name="Test Deal",
            execution_time=0.5,
            bond_cashflows=[
                {"date": "2023-01-01", "ClassA": 1000.0, "ClassB": 500.0},
                {"date": "2023-02-01", "ClassA": 1000.0, "ClassB": 500.0}
            ],
            pool_cashflows=[
                {"date": "2023-01-01", "principal": 1200.0, "interest": 300.0},
                {"date": "2023-02-01", "principal": 1200.0, "interest": 290.0}
            ],
            pool_statistics={"totalPrincipal": 2400.0, "totalInterest": 590.0},
            metrics={"bond_metrics": {}, "pool_metrics": {}},
            status="success"
        )
        
        # Mock the run_scenario_analysis method
        service_instance.run_scenario_analysis.return_value = [
            {
                "scenario_name": "Base Case",
                "bond_metrics": {"ClassA": {"irr": 0.04}, "ClassB": {"irr": 0.06}},
                "pool_metrics": {"default_rate": 0.015},
                "execution_time": 0.5,
                "status": "success"
            },
            {
                "scenario_name": "Stressed Case",
                "bond_metrics": {"ClassA": {"irr": 0.035}, "ClassB": {"irr": 0.05}},
                "pool_metrics": {"default_rate": 0.04},
                "execution_time": 0.5,
                "status": "success"
            }
        ]
        
        # Mock the health_check method
        service_instance.health_check.return_value = {
            "absbox_version": "0.9.3",
            "engine_type": "local",
            "local_engine_status": "healthy"
        }
        
        yield service_instance

# Tests
def test_health_check_endpoint(mock_absbox_service):
    """Test the health check endpoint"""
    # Make request
    response = client.get("/api/v1/structured-products/health")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "absbox_version" in data
    assert "engine_type" in data
    
    # Verify service was called
    assert mock_absbox_service.health_check.called

def test_analyze_deal_endpoint(mock_absbox_service):
    """Test the analyze deal endpoint"""
    # Prepare payload
    payload = get_sample_deal_payload()
    
    # Make request
    response = client.post(
        "/api/v1/structured-products/deals/analyze",
        json=payload
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["deal_name"] == "Test Deal"
    assert data["status"] == "success"
    assert len(data["bond_cashflows"]) == 2
    assert len(data["pool_cashflows"]) == 2
    
    # Verify service was called with correct parameters
    mock_absbox_service.analyze_deal.assert_called_once()
    args, _ = mock_absbox_service.analyze_deal.call_args
    assert args[0].deal_name == "API Test Deal"

def test_run_scenarios_endpoint(mock_absbox_service):
    """Test the run scenarios endpoint"""
    # Prepare payload
    deal_payload = get_sample_deal_payload()
    scenarios_payload = get_sample_scenarios_payload()
    
    # Make request
    response = client.post(
        "/api/v1/structured-products/deals/scenarios",
        json={
            **deal_payload,
            "scenarios": scenarios_payload
        }
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2  # Number of scenarios in the mock
    assert data[0]["scenario_name"] == "Base Case"
    assert data[1]["scenario_name"] == "Stressed Case"
    
    # Verify service was called with correct parameters
    mock_absbox_service.run_scenario_analysis.assert_called_once()
    args, _ = mock_absbox_service.run_scenario_analysis.call_args
    assert args[0].deal_name == "API Test Deal"
    assert len(args[1]) == 3  # Number of scenarios in the payload

def test_get_metrics_endpoint():
    """Test the metrics endpoint"""
    # Make request
    response = client.get("/api/v1/structured-products/metrics")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "calculation_metrics" in data
