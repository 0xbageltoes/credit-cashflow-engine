import pytest
import json
import uuid
import time
from unittest.mock import patch, MagicMock

from app.services.redis_service import RedisService, RedisConfig
from redis.exceptions import RedisError

@pytest.fixture
def redis_service():
    """Create a Redis service for testing"""
    config = RedisConfig(
        host=None,  # Will use settings values
        port=None,
        password=None,
        db=0,
        socket_timeout=2,
        socket_connect_timeout=2,
        retry_on_timeout=True,
        health_check_interval=5,
        max_connections=5
    )
    return RedisService(config)

@pytest.mark.asyncio
class TestRedisAuthService:
    """Test the Redis authentication service with JWT integration"""
    
    async def test_token_storage_and_retrieval(self, redis_service):
        """Test storing and retrieving JWT tokens"""
        # Create a test JWT payload
        user_id = str(uuid.uuid4())
        jwt_id = str(uuid.uuid4())
        
        test_jwt_payload = {
            "sub": user_id,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "jti": jwt_id,
            "user_roles": ["user"],
            "email": "test@example.com"
        }
        
        # Set key with TTL
        jwt_key = f"jwt_payload:{jwt_id}"
        ttl = 3600
        
        # Store the token
        success = await redis_service.set(jwt_key, json.dumps(test_jwt_payload), ttl)
        assert success is True
        
        # Retrieve the token
        stored_payload = await redis_service.get(jwt_key)
        assert stored_payload is not None
        parsed_payload = json.loads(stored_payload)
        
        # Verify payload data
        assert parsed_payload["sub"] == user_id
        assert parsed_payload["jti"] == jwt_id
        
        # Verify TTL
        ttl_remaining = await redis_service.get_ttl(jwt_key)
        assert ttl_remaining is not None
        assert 0 < ttl_remaining <= 3600
        
        # Clean up
        success = await redis_service.delete(jwt_key)
        assert success is True

    async def test_pattern_key_operations(self, redis_service):
        """Test pattern-based key operations"""
        # Create several test JWT payloads
        user_id = str(uuid.uuid4())
        prefix = "jwt_payload:"
        
        payloads = []
        keys = []
        
        # Create 3 tokens for the same user
        for i in range(3):
            jwt_id = str(uuid.uuid4())
            key = f"{prefix}{jwt_id}"
            keys.append(key)
            
            payload = {
                "sub": user_id,
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "jti": jwt_id,
                "user_roles": ["user"],
                "email": "test@example.com"
            }
            payloads.append(payload)
            
            # Store the token
            success = await redis_service.set(key, json.dumps(payload), 3600)
            assert success is True
        
        # Test keys pattern matching
        matched_keys = await redis_service.keys(f"{prefix}*")
        assert len(matched_keys) >= 3
        for key in keys:
            assert key in matched_keys
        
        # Test delete pattern
        deleted_count = await redis_service.delete_pattern(f"{prefix}*")
        assert deleted_count >= 3
        
        # Verify keys are gone
        for key in keys:
            exists = await redis_service.exists(key)
            assert exists is False

    async def test_invalidate_user_tokens(self, redis_service):
        """Test invalidating all tokens for a specific user"""
        # Create tokens for two different users
        user1_id = str(uuid.uuid4())
        user2_id = str(uuid.uuid4())
        prefix = "jwt_payload:"
        
        # Create 3 tokens for user1
        for i in range(3):
            jwt_id = str(uuid.uuid4())
            payload = {
                "sub": user1_id,
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "jti": jwt_id,
                "user_roles": ["user"],
                "email": "user1@example.com"
            }
            await redis_service.set(f"{prefix}{jwt_id}", json.dumps(payload), 3600)
        
        # Create 2 tokens for user2
        for i in range(2):
            jwt_id = str(uuid.uuid4())
            payload = {
                "sub": user2_id,
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "jti": jwt_id,
                "user_roles": ["user"],
                "email": "user2@example.com"
            }
            await redis_service.set(f"{prefix}{jwt_id}", json.dumps(payload), 3600)
        
        # Invalidate tokens for user1
        invalidated_count = await redis_service.invalidate_user_tokens(user1_id)
        assert invalidated_count == 3
        
        # Check that user2's tokens still exist
        keys = await redis_service.keys(f"{prefix}*")
        assert len(keys) == 2
        
        # Verify remaining tokens belong to user2
        for key in keys:
            stored_payload = await redis_service.get(key)
            parsed_payload = json.loads(stored_payload)
            assert parsed_payload["sub"] == user2_id
        
        # Clean up
        await redis_service.delete_pattern(f"{prefix}*")

    async def test_increment_operations(self, redis_service):
        """Test counter increment operations for rate limiting"""
        # Test key
        key = f"rate_limit:test:{uuid.uuid4()}"
        
        # Test increments
        value1 = await redis_service.increment(key, 1, ttl=10)
        assert value1 == 1
        
        value2 = await redis_service.increment(key, 1)
        assert value2 == 2
        
        value3 = await redis_service.increment(key, 3)
        assert value3 == 5
        
        # Verify TTL is set
        ttl = await redis_service.get_ttl(key)
        assert ttl is not None
        assert ttl <= 10
        
        # Clean up
        await redis_service.delete(key)

    async def test_error_handling(self, redis_service):
        """Test error handling and graceful fallbacks"""
        # Mock Redis error
        with patch.object(redis_service._async_client, 'get', side_effect=RedisError("Test error")):
            # Test get with Redis error
            result = await redis_service.get("any_key", default="fallback")
            assert result == "fallback"
        
        # Test when Redis connection is unavailable
        with patch.object(redis_service, '_connection_error', True):
            # Test operations with connection error
            assert await redis_service.get("any_key", default="fallback") == "fallback"
            assert await redis_service.set("any_key", "value") is False
            assert await redis_service.delete("any_key") is False
            assert await redis_service.exists("any_key") is False
            assert await redis_service.keys("pattern:*") == []
            assert await redis_service.delete_pattern("pattern:*") == 0
            assert await redis_service.get_ttl("any_key") is None
            assert await redis_service.expire("any_key", 60) is False
            assert await redis_service.increment("any_key", 1) is None
            assert await redis_service.invalidate_user_tokens("user_id") == 0

    async def test_initialize_from_config(self, redis_service):
        """Test initializing Redis from config"""
        # Create a mock Redis instance
        mock_async_client = MagicMock()
        mock_sync_client = MagicMock()
        
        # Mock Redis clients and ping
        with patch('redis.asyncio.Redis', return_value=mock_async_client), \
             patch('redis.Redis', return_value=mock_sync_client), \
             patch.object(mock_async_client, 'ping', return_value=True):
            
            # Test successful initialization
            await redis_service.initialize_from_config()
            assert redis_service._connection_error is False
            
        # Test failed health check
        with patch('redis.asyncio.Redis', return_value=mock_async_client), \
             patch('redis.Redis', return_value=mock_sync_client), \
             patch.object(mock_async_client, 'ping', return_value=False):
            
            # Force reset connection error flag
            redis_service._connection_error = False
            
            # Test initialization with failed health check
            await redis_service.initialize_from_config()
            assert redis_service._connection_error is True

    async def test_sync_operations(self, redis_service):
        """Test synchronous Redis operations"""
        key = f"sync_test:{uuid.uuid4()}"
        value = {"test": "data"}
        
        # Test synchronous set
        success = redis_service.set_sync(key, json.dumps(value), ttl=10)
        assert success is True
        
        # Test synchronous get
        stored_value = redis_service.get_sync(key)
        assert stored_value is not None
        assert json.loads(stored_value) == value
        
        # Test synchronous exists
        exists = redis_service.exists_sync(key)
        assert exists is True
        
        # Test synchronous TTL
        ttl = redis_service.get_ttl_sync(key)
        assert ttl is not None
        assert ttl <= 10
        
        # Test synchronous delete
        success = redis_service.delete_sync(key)
        assert success is True
        
        # Verify key is gone
        exists = redis_service.exists_sync(key)
        assert exists is False

if __name__ == "__main__":
    pytest.main()
