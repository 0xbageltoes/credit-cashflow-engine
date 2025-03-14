import os
import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import pathlib
import uuid
import time
import json

# Import our test utilities - these will have already initialized the environment
from app.core.test_utils import (
    create_test_jwt_token,
    create_test_tokens,
    create_test_request_with_token, 
    create_mock_supabase_service,
    create_mock_redis_service
)

# Now we can safely import app modules
from app.core.config import settings
from app.services.cashflow import CashflowService
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
from app.services.analytics import AnalyticsService

# Test constants
TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

@pytest.fixture(scope="session")
def mock_supabase_service():
    """Create a mock Supabase service for testing"""
    return create_mock_supabase_service()

@pytest.fixture(scope="session")
def mock_redis_service():
    """Create a mock Redis service for testing"""
    return create_mock_redis_service()

@pytest.fixture(scope="function")
def redis_service():
    """Fixture for Redis service for tests with proper error handling"""
    # Create a Redis instance with production-ready error handling
    redis_instance = RedisService()
    
    try:
        # Clear test keys before tests
        yield redis_instance
    finally:
        # Clean up after tests with proper error handling
        try:
            # Only clear test-specific keys for isolation
            redis_instance.delete_pattern("test:*")
        except Exception as e:
            print(f"Error clearing test keys: {e}")

@pytest.fixture
def mock_redis():
    """Mock Redis client for tests"""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    # Add proper error handling for production readiness
    mock.get.side_effect = lambda key, default=None: default
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
def cashflow_service(mock_supabase_service, mock_redis, mock_analytics):
    """Create CashflowService with mocked dependencies"""
    return CashflowService(
        supabase_service=mock_supabase_service,
        redis_service=mock_redis,
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

@pytest.fixture
def jwt_payload():
    """Create a valid JWT payload for testing"""
    return {
        "sub": TEST_USER_ID,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "jti": str(uuid.uuid4()),
        "user_roles": ["user"],
        "email": "test@example.com"
    }

@pytest.fixture
def valid_access_token():
    """Create a valid access token for testing"""
    token, _ = create_test_jwt_token(
        user_id=TEST_USER_ID,
        email="test@example.com",
        roles=["user"]
    )
    return token

@pytest.fixture
def admin_access_token():
    """Create a valid admin access token for testing"""
    token, _ = create_test_jwt_token(
        user_id="admin-user-id",
        email="admin@example.com",
        roles=["admin", "user"]
    )
    return token

@pytest.fixture
def expired_token():
    """Create an expired token for testing"""
    token, _ = create_test_jwt_token(
        user_id=TEST_USER_ID,
        email="test@example.com",
        roles=["user"],
        expires_in_seconds=-3600  # Expired 1 hour ago
    )
    return token

@pytest.fixture
def mock_request_with_token(valid_access_token):
    """Create a mock request with a valid token"""
    return create_test_request_with_token(valid_access_token)
