import redis
import json
from typing import Any, Optional
from functools import wraps
from datetime import datetime
from app.core.config import settings

class RedisManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
        return cls._instance

    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        return self.client.get(key)

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return self.client.set(key, value, ex=ttl or settings.CACHE_TTL)

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return bool(self.client.delete(key))

    def increment(self, key: str) -> int:
        """Increment counter"""
        return self.client.incr(key)

    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        return self.client.expire(key, ttl)

class RateLimiter:
    def __init__(self):
        self.redis = RedisManager()

    def is_rate_limited(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        key = f"rate_limit:{user_id}"
        count = self.redis.get(key)
        
        if count is None:
            # First request
            self.redis.set(key, 1, settings.RATE_LIMIT_WINDOW)
            return False
            
        count = int(count)
        if count >= settings.RATE_LIMIT_REQUESTS:
            return True
            
        self.redis.increment(key)
        return False

def cache_response(ttl: Optional[int] = None):
    """Decorator to cache API responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = RedisManager()
            
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, json.dumps(result), ttl)
            
            return result
        return wrapper
    return decorator
