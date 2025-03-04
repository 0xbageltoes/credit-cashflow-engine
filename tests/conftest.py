import os
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv
from app.core.config import settings
from app.services.cashflow import CashflowService
from app.core.redis_cache import RedisCache
from app.services.analytics import AnalyticsService
from supabase import create_client, Client
import uuid

# Load environment variables from .env file
load_dotenv()

# Test constants
TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

@pytest.fixture(scope="session")
def supabase_client():
    """Create real Supabase client for tests"""
    client = create_client(
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_SERVICE_ROLE_KEY,
    )
    
    # Set up test data
    test_scenario_id = str(uuid.uuid4())
    client.table("cashflow_scenarios").upsert({
        "id": test_scenario_id,
        "user_id": TEST_USER_ID,
        "name": "Test Scenario",
        "description": "Test scenario for unit tests",
        "forecast_request": {
            "loans": [{
                "principal": 100000,
                "rate": 0.05,
                "term": 360,
                "start_date": "2025-01-01"
            }]
        },
        "tags": ["test"]
    }).execute()
    
    yield client
    
    # Clean up test data
    client.table("cashflow_scenarios").delete().eq("user_id", TEST_USER_ID).execute()

@pytest.fixture
def mock_redis():
    """Mock Redis client for tests"""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    return mock

@pytest.fixture
def mock_analytics():
    """Mock analytics service"""
    mock = MagicMock()
    mock.analyze_cashflows.return_value = {
        "npv": 1000.0,
        "irr": 0.08,
        "duration": 5.0,
        "convexity": 25.0,
        "monte_carlo_results": {
            "mean": [100.0],
            "std": [10.0],
            "percentile_5": [80.0],
            "percentile_95": [120.0]
        }
    }
    return mock

@pytest.fixture
def cashflow_service(supabase_client, mock_redis, mock_analytics):
    """Create CashflowService with real Supabase and mocked Redis/Analytics"""
    return CashflowService(
        supabase_client=supabase_client,
        redis_cache=mock_redis,
        analytics_service=mock_analytics
    )

@pytest.fixture
def sample_loan_request():
    """Create a sample loan request"""
    return {
        "loan": {
            "principal": 100000,
            "rate": 0.05,
            "term": 360,
            "start_date": "2025-01-01",
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
    """Create a batch request with multiple loans"""
    return {
        "loans": [
            {
                "principal": 100000,
                "rate": 0.05,
                "term": 360,
                "start_date": "2025-01-01",
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
