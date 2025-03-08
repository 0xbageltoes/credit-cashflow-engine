import pytest
import json
import zlib
from app.core.redis_cache import RedisCache
from app.core.cache import RateLimiter, cache_response
from app.core.config import settings

@pytest.fixture
def redis_cache():
    return RedisCache()

@pytest.fixture
def rate_limiter():
    return RateLimiter()

@pytest.mark.asyncio
class TestRedisIntegration:
    """Integration tests for Redis functionality"""

    async def test_basic_operations(self, redis_cache):
        """Test basic Redis operations (set, get, delete)"""
        # Test set and get
        test_data = {"key": "value", "number": 42}
        success = await redis_cache.set_async("test_key", test_data)
        assert success is True

        result = await redis_cache.get_async("test_key")
        assert result == test_data

        # Test delete
        success = await redis_cache.delete_async("test_key")
        assert success is True
        
        result = await redis_cache.get_async("test_key")
        assert result is None

    async def test_compression(self, redis_cache):
        """Test compression for large data"""
        # Create large data
        large_data = {
            "data": "x" * 10000,  # Create string that will be compressed well
            "numbers": list(range(1000))
        }

        # Set with compression
        success = await redis_cache.set_async("large_key", large_data, compress=True)
        assert success is True

        # Verify we can retrieve the data
        result = await redis_cache.get_async("large_key")
        assert result["data"] == "x" * 10000
        assert result["numbers"] == list(range(1000))
        
        # Clean up
        await redis_cache.delete_async("large_key")

    async def test_task_status(self, redis_cache):
        """Test task status operations"""
        task_id = "test_task_123"
        status = {
            "status": "running",
            "progress": 50,
            "message": "Processing data"
        }

        # Set task status
        await redis_cache.set_task_status(task_id, status)
        
        # Get task status
        result = await redis_cache.get_task_status(task_id)
        assert result == status
        
        # Clean up
        await redis_cache.delete_async(f"task_status:{task_id}")

    async def test_forecast_results(self, redis_cache):
        """Test forecast result operations"""
        task_id = "forecast_123"
        forecast_data = {
            "cashflows": [100, 200, 300],
            "metrics": {"irr": 0.15, "npv": 1000},
            "scenarios": ["base", "stress"]
        }

        # Set forecast result
        await redis_cache.set_forecast_result(task_id, forecast_data)
        
        # Get forecast result
        result = await redis_cache.get_forecast_result(task_id)
        assert result == forecast_data
        
        # Clean up
        await redis_cache.delete_async(f"forecast_result:{task_id}")

    def test_rate_limiter(self, rate_limiter):
        """Test rate limiting functionality"""
        user_id = "test_user"
        max_requests = 5
        window_seconds = 1
        
        # Should not be rate limited initially
        assert not rate_limiter.is_rate_limited(user_id, max_requests, window_seconds)
        
        # Make several requests
        for _ in range(max_requests):
            rate_limiter.is_rate_limited(user_id, max_requests, window_seconds)
        
        # Should be rate limited after max_requests
        assert rate_limiter.is_rate_limited(user_id, max_requests, window_seconds)

    async def test_cache_decorator(self, redis_cache):
        """Test cache decorator functionality"""
        call_count = 0
        
        @cache_response(ttl=10)
        async def test_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return {"sum": arg1 + arg2}
        
        # First call should execute the function
        result1 = await test_function(1, 2)
        assert result1 == {"sum": 3}
        assert call_count == 1
        
        # Second call should use cached result
        result2 = await test_function(1, 2)
        assert result2 == {"sum": 3}
        assert call_count == 1  # Call count shouldn't increase
        
        # Different arguments should execute the function again
        result3 = await test_function(2, 3)
        assert result3 == {"sum": 5}
        assert call_count == 2
