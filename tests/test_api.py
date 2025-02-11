import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.redis_cache import RedisCache
from app.models.cashflow import (
    LoanData,
    CashflowForecastRequest,
    BatchForecastRequest
)
from datetime import datetime
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def redis_cache():
    return RedisCache()

@pytest.fixture
def sample_loan_request():
    """Create a sample loan request"""
    return {
        "loans": [{
            "principal": 100000,
            "interest_rate": 0.05,
            "term_months": 360,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "prepayment_assumption": 0.02
        }],
        "discount_rate": 0.05,
        "run_monte_carlo": True
    }

@pytest.fixture
def batch_request():
    """Create a batch request with multiple loans"""
    return {
        "forecasts": [
            {
                "loans": [{
                    "principal": 100000 + i * 50000,
                    "interest_rate": 0.05,
                    "term_months": 360,
                    "start_date": datetime.now().strftime("%Y-%m-%d"),
                    "prepayment_assumption": 0.02
                }],
                "discount_rate": 0.05,
                "run_monte_carlo": True
            }
            for i in range(3)
        ],
        "parallel": True,
        "chunk_size": 2
    }

class TestAPIEndpoints:
    """Test API endpoints functionality"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_single_forecast(self, client, sample_loan_request):
        """Test single forecast endpoint"""
        response = client.post("/api/v1/forecast", json=sample_loan_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "projections" in data
        assert "summary_metrics" in data
        assert len(data["projections"]) == sample_loan_request["loans"][0]["term_months"]

    async def test_batch_forecast(self, client, batch_request):
        """Test batch forecast endpoint"""
        response = client.post("/api/v1/forecast/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "forecasts" in data
        assert len(data["forecasts"]) == len(batch_request["forecasts"])
        assert "batch_id" in data

    async def test_async_forecast(self, client, sample_loan_request):
        """Test async forecast endpoint"""
        # Start async forecast
        response = client.post("/api/v1/forecast/async", json=sample_loan_request)
        assert response.status_code == 202
        
        task_id = response.json()["task_id"]
        assert task_id is not None
        
        # Check status
        status_response = client.get(f"/api/v1/forecast/status/{task_id}")
        assert status_response.status_code == 200
        assert "status" in status_response.json()

    async def test_stress_test(self, client, sample_loan_request):
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
                    "default_multiplier": 2.0,
                    "prepay_multiplier": 0.5
                }
            ]
        }
        
        request_data = {**sample_loan_request, "stress_scenarios": stress_scenarios}
        response = client.post("/api/v1/forecast/stress-test", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "base_case" in data
        assert "stress_results" in data
        assert len(data["stress_results"]) == len(stress_scenarios["scenarios"])

class TestCaching:
    """Test Redis caching functionality"""

    async def test_cache_forecast(self, client, redis_cache, sample_loan_request):
        """Test forecast caching"""
        # Make initial request
        response = client.post("/api/v1/forecast", json=sample_loan_request)
        forecast_id = response.json()["scenario_id"]
        
        # Verify cache exists
        cache_key = f"forecast:{forecast_id}"
        cached_data = await redis_cache.get(cache_key)
        assert cached_data is not None
        
        # Verify cached data matches response
        assert json.loads(cached_data)["summary_metrics"] == response.json()["summary_metrics"]

    async def test_cache_invalidation(self, client, redis_cache, sample_loan_request):
        """Test cache invalidation"""
        # Make initial request
        response = client.post("/api/v1/forecast", json=sample_loan_request)
        forecast_id = response.json()["scenario_id"]
        
        # Verify cache exists
        cache_key = f"forecast:{forecast_id}"
        assert await redis_cache.get(cache_key) is not None
        
        # Invalidate cache
        await redis_cache.delete(cache_key)
        assert await redis_cache.get(cache_key) is None

    async def test_cache_compression(self, client, redis_cache, sample_loan_request):
        """Test JSON compression for large payloads"""
        # Create a large payload
        sample_loan_request["loans"] *= 10  # Multiply loans to create larger payload
        
        # Make request
        response = client.post("/api/v1/forecast", json=sample_loan_request)
        forecast_id = response.json()["scenario_id"]
        
        # Verify compressed cache
        cache_key = f"forecast:{forecast_id}"
        raw_cached_data = await redis_cache.get_raw(cache_key)
        assert raw_cached_data is not None
        assert len(raw_cached_data) < len(json.dumps(response.json()))
