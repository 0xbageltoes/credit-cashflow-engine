import json
from typing import Any, Optional
from redis import Redis
from functools import wraps
from app.core.config import settings
import ssl
from urllib.parse import urlparse

class RedisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance.client = Redis.from_url(
                settings.REDIS_URL_WITH_SSL,
                decode_responses=True,
                ssl=True,
                ssl_cert_reqs=None
            )
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Error getting from cache: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            serialized = json.dumps(value)
            if ttl:
                return bool(self.client.setex(key, ttl, serialized))
            return bool(self.client.set(key, serialized))
        except Exception as e:
            print(f"Error setting cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting from cache: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            print(f"Error checking cache: {str(e)}")
            return False

class RateLimiter:
    def __init__(self):
        self.cache = RedisCache()

    def is_rate_limited(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        key = f"rate_limit:{user_id}"
        count = self.cache.get(key) or 0
        
        if count >= settings.RATE_LIMIT_REQUESTS:
            return True
        
        self.cache.set(key, count + 1)
        if count == 0:
            self.cache.client.expire(key, settings.RATE_LIMIT_WINDOW)
        
        return False

def cache_response(ttl: int = 3600):
    """Decorator to cache function responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # If not in cache, execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
