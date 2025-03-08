import os
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv
import sys

# Load environment variables from .env.test
load_dotenv(".env.test")

# Add a check at the very beginning to see if we're in a test environment
# and set required environment variables before any imports happen
def setup_test_environment():
    """Set up environment variables for testing."""
    # Only run this setup once
    if os.environ.get("TEST_ENV_SETUP_DONE"):
        return
        
    print("Setting up test environment variables...")
    required_vars = {
        "ENV": "testing",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "dummy_key",
        "SUPABASE_SERVICE_ROLE_KEY": "dummy_service_role_key",
        "SUPABASE_JWT_SECRET": "dummy_jwt_secret",
        "NEXT_PUBLIC_SUPABASE_URL": "https://example.supabase.co",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY": "dummy_anon_key",
        "UPSTASH_REDIS_HOST": "localhost",
        "UPSTASH_REDIS_PORT": "6379",
        "UPSTASH_REDIS_PASSWORD": "dummy_password",
        "CORS_ORIGINS": '["http://localhost:3000", "https://example.com"]'
    }
    
    # Set any missing environment variables
    for key, value in required_vars.items():
        if not os.environ.get(key):
            os.environ[key] = value
    
    # Mark as done so we don't set them again
    os.environ["TEST_ENV_SETUP_DONE"] = "1"

# Run the environment setup before importing any app modules
setup_test_environment()

# Now we can safely import app modules
from app.core.config import settings
from app.services.cashflow import CashflowService
from app.core.redis_cache import RedisCache
from app.services.analytics import AnalyticsService
from supabase import create_client, Client
import uuid

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

@pytest.fixture(scope="function")
def redis_cache():
    """Fixture for Redis cache for tests."""
    # Create a Redis cache instance
    redis_instance = RedisCache()
    print(f"Redis URL from settings: {settings.REDIS_URL}")
    
    try:
        # Clear the cache before tests
        redis_instance.clear()
        yield redis_instance
    finally:
        # Clean up after tests
        try:
            redis_instance.clear()
        except Exception as e:
            print(f"Error clearing cache: {e}")

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
