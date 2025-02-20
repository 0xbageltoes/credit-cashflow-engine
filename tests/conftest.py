import os
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv
from app.core.config import settings

# Load environment variables from .env.test file
load_dotenv(".env.test")

# Set test environment variables if not set
if not settings.NEXT_PUBLIC_SUPABASE_URL:
    os.environ["NEXT_PUBLIC_SUPABASE_URL"] = "http://localhost:54321"
    os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"] = "test-key"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-key"
    os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret"

# Set Redis test environment variables
if not settings.UPSTASH_REDIS_HOST:
    os.environ["UPSTASH_REDIS_HOST"] = "localhost"
    os.environ["UPSTASH_REDIS_PORT"] = "6379"
    os.environ["UPSTASH_REDIS_PASSWORD"] = ""

class MockRedis:
    def __init__(self):
        self.data = {}
        self.ttl = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttl[key] = ttl
        return True

    def set(self, key, value):
        self.data[key] = value
        return True

    def delete(self, key):
        if key in self.data:
            del self.data[key]
            if key in self.ttl:
                del self.ttl[key]
            return True
        return False

    def incr(self, key):
        if key not in self.data:
            self.data[key] = 1
        else:
            self.data[key] = int(self.data[key]) + 1
        return self.data[key]

    def expire(self, key, seconds):
        if key in self.data:
            self.ttl[key] = seconds
            return True
        return False

@pytest.fixture
def mock_redis():
    return MockRedis()

@pytest.fixture(autouse=True)
def mock_redis_client(monkeypatch, mock_redis):
    """Automatically mock Redis client for all tests"""
    def mock_from_url(*args, **kwargs):
        return mock_redis

    monkeypatch.setattr("redis.Redis.from_url", mock_from_url)
    return mock_redis

@pytest.fixture(autouse=True)
def mock_supabase(monkeypatch):
    """Mock Supabase client for tests"""
    mock = MagicMock()
    
    def mock_create_client(*args, **kwargs):
        return mock

    # Mock the create_client function directly
    import app.core.supabase
    monkeypatch.setattr(app.core.supabase, "create_client", mock_create_client)
    return mock

@pytest.fixture
def mock_analytics():
    """Mock analytics service"""
    return MagicMock()
