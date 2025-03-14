"""
Tests for the enhanced analytics API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import datetime
import uuid
import json
from unittest.mock import patch, MagicMock
from fastapi import Depends

# Import the app and dependencies
from app.main import app
from app.models.analytics import EnhancedAnalyticsRequest, EnhancedAnalyticsResult, RiskMetrics, SensitivityAnalysis, AnalyticsResponse
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.api.deps import get_current_user

# Custom JSON encoder for handling date objects in tests
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

# Setup test environment variables
print("Setting up test environment variables...")
import os
os.environ["ENVIRONMENT"] = "test"
os.environ["REDIS_URL"] = "redis://:AS8mAAIjcDFmMjJhZTIzY2ZiYmY0MTJkYmQzZDQ1MWYwMWQyYzI0MXAxMA@easy-macaw-12070.upstash.io:6379"

# Create test client with authentication bypass
@pytest.fixture
def client():
    """Create a test client with auth dependency overridden"""
    # Create a mock user that will be returned by the dependency
    mock_user = {
        "id": "00000000-0000-0000-0000-000000000000",
        "email": "test@example.com", 
        "role": "user"
    }
    
    # Override the dependency with our dummy function that always returns the mock user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Create test client
    test_client = TestClient(app)
    
    yield test_client
    
    # Clean up dependency overrides after test
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

# No need for auth_headers fixture since we're bypassing auth entirely
@pytest.fixture
def auth_headers():
    """No longer needed but kept for compatibility with existing tests"""
    return {"Authorization": "Bearer dummy_token"}

@pytest.fixture
def sample_enhanced_analytics_request():
    """Sample request data for enhanced analytics"""
    return EnhancedAnalyticsRequest(
        principal=100000.0,
        interest_rate=0.05,
        term_months=360,
        start_date=datetime.date(2023, 1, 1),
        prepayment_rate=0.03,
        default_rate=0.01,
        recovery_rate=0.6,
        discount_rate=0.04,
        rate_type="fixed",
        payment_frequency="monthly"
    )


@pytest.fixture
def sample_enhanced_analytics_result():
    """Sample result data for enhanced analytics"""
    return EnhancedAnalyticsResult(
        npv=98500.0,
        irr=0.053,
        yield_value=0.052,
        duration=4.25,
        macaulay_duration=4.50,
        convexity=0.35,
        discount_margin=0.02,
        z_spread=0.015,
        e_spread=0.018,
        weighted_average_life=5.2,
        debt_service_coverage=1.25,
        interest_coverage_ratio=2.1,
        sensitivity_metrics={
            "rate_up_1pct": -4.25,
            "rate_down_1pct": 4.50
        }
    )


@pytest.fixture
def sample_risk_metrics():
    """Sample risk metrics data"""
    return RiskMetrics(
        var_95=5000.0,
        var_99=7500.0,
        expected_shortfall=8000.0,
        average_loss=3000.0,
        max_loss=10000.0,
        probability_of_default=0.02,
        loss_given_default=0.4,
        stress_loss=25000.0,
        volatility=0.15,
        stress_test_results={
            "rate_shock_up_2pct": -8.5,
            "rate_shock_down_2pct": 9.0,
            "default_doubled": -5.0
        },
        confidence_intervals={
            "95": {"lower": 95000.0, "upper": 105000.0},
            "99": {"lower": 92000.0, "upper": 108000.0}
        }
    )


@pytest.fixture
def sample_sensitivity_analysis():
    """Sample sensitivity analysis data"""
    return SensitivityAnalysis(
        rate_sensitivity={
            "up_1pct": -5.0,
            "down_1pct": 5.5
        },
        prepayment_sensitivity={
            "up_50pct": -2.0,
            "down_50pct": 1.8
        },
        default_sensitivity={
            "up_50pct": -3.0,
            "down_50pct": 2.5
        },
        recovery_sensitivity={
            "up_20pct": 1.0,
            "down_20pct": -1.2
        }
    )


@patch.object(AbsBoxServiceEnhanced, "calculate_enhanced_metrics")
def test_enhanced_analytics_endpoint(mock_calculate_metrics, client, auth_headers, sample_enhanced_analytics_request, sample_enhanced_analytics_result):
    """Test the enhanced analytics endpoint"""
    # Configure the mock
    mock_calculate_metrics.return_value = sample_enhanced_analytics_result.model_dump()
    
    # Make request to the endpoint
    response = client.post(
        "/api/v1/enhanced-analytics/enhanced-analytics/",
        headers=auth_headers,
        json=json.loads(json.dumps(sample_enhanced_analytics_request.model_dump(), cls=DateEncoder))
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["metrics"]["npv"] == sample_enhanced_analytics_result.npv
    assert data["metrics"]["yield_value"] == sample_enhanced_analytics_result.yield_value
    assert data["cache_hit"] is False


@patch.object(AbsBoxServiceEnhanced, "calculate_risk_metrics")
def test_risk_metrics_endpoint(mock_calculate_risk, client, auth_headers, sample_enhanced_analytics_request, sample_risk_metrics):
    """Test the risk metrics endpoint"""
    # Configure the mock
    mock_calculate_risk.return_value = sample_risk_metrics.model_dump()
    
    # Make request to the endpoint
    response = client.post(
        "/api/v1/enhanced-analytics/risk-metrics/",
        headers=auth_headers,
        json=json.loads(json.dumps(sample_enhanced_analytics_request.model_dump(), cls=DateEncoder))
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["metrics"]["var_95"] == sample_risk_metrics.var_95
    assert data["metrics"]["expected_shortfall"] == sample_risk_metrics.expected_shortfall
    assert data["cache_hit"] is False


@patch.object(AbsBoxServiceEnhanced, "calculate_sensitivity")
def test_sensitivity_analysis_endpoint(mock_calculate_sensitivity, client, auth_headers, sample_enhanced_analytics_request, sample_sensitivity_analysis):
    """Test the sensitivity analysis endpoint"""
    # Configure the mock
    mock_calculate_sensitivity.return_value = sample_sensitivity_analysis.model_dump()
    
    # Make request to the endpoint
    response = client.post(
        "/api/v1/enhanced-analytics/sensitivity-analysis/",
        headers=auth_headers,
        json=json.loads(json.dumps(sample_enhanced_analytics_request.model_dump(), cls=DateEncoder))
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["metrics"]["rate_sensitivity"]["up_1pct"] == sample_sensitivity_analysis.rate_sensitivity["up_1pct"]
    assert data["metrics"]["default_sensitivity"]["up_50pct"] == sample_sensitivity_analysis.default_sensitivity["up_50pct"]
    assert data["cache_hit"] is False


@patch.object(AbsBoxServiceEnhanced, "get_from_cache")
@patch.object(AbsBoxServiceEnhanced, "calculate_enhanced_metrics")
def test_enhanced_analytics_cache_hit(mock_calculate_metrics, mock_get_from_cache, client, auth_headers, sample_enhanced_analytics_request, sample_enhanced_analytics_result):
    """Test the enhanced analytics endpoint with cache hit"""
    # Configure the mocks
    mock_get_from_cache.return_value = sample_enhanced_analytics_result.model_dump()
    
    # Make request to the endpoint
    response = client.post(
        "/api/v1/enhanced-analytics/enhanced-analytics/",
        headers=auth_headers,
        json=json.loads(json.dumps(sample_enhanced_analytics_request.model_dump(), cls=DateEncoder))
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["metrics"]["npv"] == sample_enhanced_analytics_result.npv
    assert data["cache_hit"] is True
    
    # Verify that calculate_enhanced_metrics was not called
    mock_calculate_metrics.assert_not_called()


@patch.object(AbsBoxServiceEnhanced, "calculate_enhanced_metrics")
def test_batch_analytics_endpoint(mock_calculate_metrics, client, auth_headers, sample_enhanced_analytics_request, sample_enhanced_analytics_result):
    """Test the batch analytics endpoint"""
    # Configure the mock
    mock_calculate_metrics.return_value = sample_enhanced_analytics_result.model_dump()
    
    # Create a batch request with 3 items
    batch_requests = json.loads(json.dumps([sample_enhanced_analytics_request.model_dump() for _ in range(3)], cls=DateEncoder))
    
    # Make request to the endpoint
    response = client.post(
        "/api/v1/enhanced-analytics/batch-analytics/",
        headers=auth_headers,
        json=batch_requests
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # The response is a list of AnalyticsResponse objects, not a dict with a "results" key
    assert isinstance(data, list)
    assert len(data) == 3
    
    # Verify each result contains expected fields
    for item in data:
        assert "status" in item
        assert "execution_time" in item
        assert "metrics" in item
        assert item["status"] == "success"
