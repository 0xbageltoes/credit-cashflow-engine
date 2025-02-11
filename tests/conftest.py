"""Test configuration and fixtures"""
import pytest
import os
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock
import numpy as np

# Load environment variables from .env.test file
load_dotenv(".env.test")

# Set test environment variables if not present
if not os.getenv("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = "https://test.supabase.co"
if not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_key"
if not os.getenv("REDIS_URL"):
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
if not os.getenv("CELERY_BROKER_URL"):
    os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/1"
if not os.getenv("CELERY_RESULT_BACKEND"):
    os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/2"

@pytest.fixture(autouse=True)
def mock_supabase(monkeypatch):
    """Mock Supabase client for testing"""
    class MockSupabase:
        def __init__(self, *args, **kwargs):
            pass
            
        def table(self, *args, **kwargs):
            return self
            
        def insert(self, *args, **kwargs):
            return {"data": [{"id": "test_id"}]}
            
        def execute(self, *args, **kwargs):
            return {"data": [{"id": "test_id"}]}

        def upsert(self, *args, **kwargs):
            return {"data": [{"id": "test_id"}]}

        def select(self, *args, **kwargs):
            return self

        def eq(self, *args, **kwargs):
            return self

        def single(self, *args, **kwargs):
            return {"data": {"id": "test_id"}}
            
    monkeypatch.setattr("supabase.create_client", lambda *args, **kwargs: MockSupabase())

@pytest.fixture(autouse=True)
def mock_redis_cache(monkeypatch):
    """Mock Redis cache for testing"""
    class MockRedisCache:
        def __init__(self):
            self.data = {}

        async def get(self, key):
            return self.data.get(key)

        async def set(self, key, value, *args, **kwargs):
            self.data[key] = value

        async def delete(self, key):
            self.data.pop(key, None)

        async def get_raw(self, key):
            return self.data.get(key)

    monkeypatch.setattr("app.core.redis_cache.RedisCache", MockRedisCache)

@pytest.fixture(autouse=True)
def mock_analytics_service(monkeypatch):
    """Mock AnalyticsService for testing"""
    class MockAnalyticsService:
        def __init__(self, *args, **kwargs):
            pass

        async def analyze_cashflows(self, cashflows, discount_rate=0.05, run_monte_carlo=True):
            from app.services.analytics import AnalyticsResult
            return AnalyticsResult(
                npv=1000.0,
                irr=0.08,
                duration=5.0,
                convexity=25.0,
                monte_carlo_results={
                    "mean": np.array([100.0] * len(cashflows)).tolist(),
                    "std": np.array([10.0] * len(cashflows)).tolist(),
                    "percentile_5": np.array([80.0] * len(cashflows)).tolist(),
                    "percentile_95": np.array([120.0] * len(cashflows)).tolist()
                }
            )

    monkeypatch.setattr("app.services.analytics.AnalyticsService", MockAnalyticsService)

@pytest.fixture(autouse=True)
def mock_economic_factors(monkeypatch):
    """Mock economic factors for testing"""
    async def mock_get_economic_factors(*args, **kwargs):
        return {
            "market_rate": 0.045,
            "inflation_rate": 0.02,
            "unemployment_rate": 0.05,
            "gdp_growth": 0.025,
            "house_price_appreciation": 0.03,
            "month": 6
        }

    monkeypatch.setattr(
        "app.services.cashflow.CashflowService._get_economic_factors",
        mock_get_economic_factors
    )
