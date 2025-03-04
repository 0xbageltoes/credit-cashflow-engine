import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from app.core.redis_cache import RedisCache
from app.models.cashflow import (
    LoanData,
    CashflowForecastRequest,
    BatchForecastRequest
)
from datetime import datetime
import json
from app.core.config import settings
from tests.conftest import TEST_USER_ID
from app.core.auth import get_current_user

@pytest.fixture
def client(cashflow_service):
    """Create test client with mocked services"""
    app.dependency_overrides = {}  # Reset any existing overrides
    
    # Override the get_cashflow_service dependency
    def get_test_cashflow_service():
        return cashflow_service
    
    # Override the get_current_user dependency
    async def override_get_current_user():
        return {"id": TEST_USER_ID, "role": "authenticated"}
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[settings.get_cashflow_service] = get_test_cashflow_service
    
    client = TestClient(app)
    # Add mock auth header
    client.headers["Authorization"] = f"Bearer test-token"
    return client

@pytest.fixture
def redis_cache():
    return RedisCache()

@pytest.fixture
def sample_loan_request():
    return {
        "loan": {
            "principal": 100000,
            "rate": 0.05,
            "term": 360,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "interest_only_period": 0,
            "balloon_payment": False
        },
        "monte_carlo_config": {
            "num_simulations": 1000,
            "volatility": 0.15,
            "rate_shock": 0.01
        }
    }

@pytest.fixture
def batch_request():
    return {
        "loans": [
            {
                "principal": 100000,
                "rate": 0.05,
                "term": 360,
                "start_date": datetime.now().strftime("%Y-%m-%d"),
                "interest_only_period": 0,
                "balloon_payment": False
            }
            for _ in range(3)
        ],
        "monte_carlo_config": {
            "num_simulations": 1000,
            "volatility": 0.15,
            "rate_shock": 0.01
        }
    }

@pytest.fixture
def cashflow_service():
    service = AsyncMock()
    service.generate_forecast.return_value = {
        "projections": [],
        "metrics": {
            "npv": 1000.0,
            "irr": 0.08
        }
    }
    return service

class TestAPIEndpoints:
    """Test API endpoints functionality"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_single_forecast(self, client, sample_loan_request):
        """Test single forecast endpoint"""
        response = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response.status_code == 200
        data = response.json()
        assert "projections" in data
        assert "metrics" in data

    def test_batch_forecast(self, client, batch_request):
        """Test batch forecast endpoint"""
        response = client.post(
            "/api/v1/forecast/batch",
            json=batch_request
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(batch_request["loans"])

    def test_async_forecast(self, client, sample_loan_request):
        """Test async forecast endpoint"""
        response = client.post(
            "/api/v1/forecast/async",
            json=sample_loan_request
        )
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

    def test_stress_test(self, client, sample_loan_request):
        """Test stress test endpoint"""
        stress_scenarios = {
            "scenarios": [
                {
                    "name": "High Interest",
                    "rate_shock": 0.02,
                    "default_multiplier": 1.5
                },
                {
                    "name": "Recession",
                    "rate_shock": 0.03,
                    "default_multiplier": 2.0
                }
            ]
        }
        
        request_data = {**sample_loan_request, "stress_scenarios": stress_scenarios}
        response = client.post("/api/v1/forecast/stress", json=request_data)
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

class TestCaching:
    """Test Redis caching functionality"""

    def test_cache_forecast(self, client, redis_cache, sample_loan_request):
        """Test forecast caching"""
        # First request should miss cache
        response1 = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response1.status_code == 200

        # Second request should hit cache
        response2 = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response2.status_code == 200
        assert response1.json() == response2.json()

    def test_cache_invalidation(self, client, redis_cache, sample_loan_request):
        """Test cache invalidation"""
        # Make initial request
        response1 = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response1.status_code == 200

        # Invalidate cache
        key = f"forecast:{hash(json.dumps(sample_loan_request, sort_keys=True))}"
        redis_cache.delete(key)

        # Next request should miss cache
        response2 = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response2.status_code == 200

    def test_cache_compression(self, client, redis_cache, sample_loan_request):
        """Test JSON compression for large payloads"""
        # Create large payload
        sample_loan_request["monte_carlo_config"]["num_simulations"] = 10000

        response = client.post(
            "/api/v1/forecast",
            json=sample_loan_request
        )
        assert response.status_code == 200

def test_forecast_endpoint(client, sample_loan_request):
    """Test forecast endpoint"""
    response = client.post("/api/v1/forecast", json=sample_loan_request)
    assert response.status_code == 200
    data = response.json()
    assert "projections" in data
    assert "metrics" in data
    assert "npv" in data["metrics"]
    assert "irr" in data["metrics"]

def test_batch_forecast(client, batch_request):
    """Test batch forecast endpoint"""
    response = client.post("/api/v1/forecast/batch", json=batch_request)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(batch_request["loans"])
    for result in data:
        assert "projections" in result
        assert "metrics" in result

def test_async_forecast(client, sample_loan_request):
    """Test async forecast endpoint"""
    response = client.post("/api/v1/forecast/async", json=sample_loan_request)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data

def test_stress_test(client, sample_loan_request):
    """Test stress test endpoint"""
    stress_scenarios = {
        "scenarios": [
            {
                "name": "High Interest",
                "rate_shock": 0.02,
                "default_multiplier": 1.5
            },
            {
                "name": "Recession",
                "rate_shock": 0.03,
                "default_multiplier": 2.0
            }
        ]
    }
    
    request_data = {**sample_loan_request, "stress_scenarios": stress_scenarios}
    response = client.post("/api/v1/forecast/stress", json=request_data)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
