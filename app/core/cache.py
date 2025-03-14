import json
from typing import Any, Optional
from redis import Redis
from functools import wraps
from app.core.config import settings
import sqlite3
import json
import time
import functools
from typing import Any, Optional
from datetime import datetime, timedelta

class RedisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance.client = Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
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

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            return bool(self.client.setex(key, ttl, json.dumps(value)))
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

def get_redis_client() -> Redis:
    """
    Get a Redis client instance.
    
    Returns:
        Redis: A configured Redis client
    """
    try:
        return Redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        print(f"Error creating Redis client: {str(e)}")
        return None

class SQLiteCache:
    """SQLite-based cache implementation for local development and testing"""
    
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expiry INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get value and check expiry
        cursor.execute(
            "SELECT value, expiry FROM cache WHERE key = ?",
            (key,)
        )
        result = cursor.fetchone()
        
        if result:
            value, expiry = result
            if expiry is None or expiry > time.time():
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                # Remove expired key
                self.delete(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL in seconds"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert value to JSON if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            expiry = time.time() + ttl if ttl else None
            
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, expiry)
                VALUES (?, ?, ?)
                """,
                (key, value, expiry)
            )
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
            
        finally:
            conn.close()
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
            
        finally:
            conn.close()
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM cache")
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
            
        finally:
            conn.close()

class RateLimiter:
    def __init__(self):
        self.cache = RedisCache()
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_WINDOW
        # Using in-memory dict for testing
        self.test_counters = {}

    def is_rate_limited(self, user_id: str, max_requests: int = None, window_seconds: int = None) -> bool:
        """Check if user has exceeded rate limit (legacy method)"""
        max_requests = max_requests or self.max_requests
        window_seconds = window_seconds or self.window_seconds
        
        remaining, _ = self.check_rate_limit(user_id, max_requests, window_seconds)
        return remaining <= 0
    
    def check_rate_limit(self, user_id: str, max_requests: int = None, window_seconds: int = None) -> tuple[int, int]:
        """
        Check rate limit for a user
        
        Returns:
            tuple[int, int]: (remaining requests, reset time in unix seconds)
        """
        max_requests = max_requests or self.max_requests
        window_seconds = window_seconds or self.window_seconds
        
        # For test environment, use in-memory tracking
        if settings.ENVIRONMENT == "test":
            now = int(time.time())
            window_key = f"{now // window_seconds}"
            
            # Initialize counter if not exists
            if user_id not in self.test_counters:
                self.test_counters[user_id] = {}
            
            if window_key not in self.test_counters[user_id]:
                self.test_counters[user_id][window_key] = 0
                
            # Increment counter
            self.test_counters[user_id][window_key] += 1
            current_count = self.test_counters[user_id][window_key]
            
            # Calculate remaining requests and reset time
            remaining = max(0, max_requests - current_count)
            reset_time = ((now // window_seconds) + 1) * window_seconds
            
            return remaining, reset_time
        
        # Create a unique key for this user and rate limit window
        key = f"rate_limit:{user_id}"
        now = int(time.time())
        window_key = f"{key}:{now // window_seconds}"
        
        try:
            # Get current pipeline for atomic operations
            pipeline = self.cache.client.pipeline()
            
            # Increment counter for current window
            pipeline.incr(window_key)
            
            # Set expiration if not already set
            pipeline.expire(window_key, window_seconds)
            
            # Get the current count
            result = pipeline.execute()
            current_count = result[0]
            
            # Calculate remaining requests and reset time
            remaining = max(0, max_requests - current_count)
            reset_time = ((now // window_seconds) + 1) * window_seconds
            
            return remaining, reset_time
            
        except Exception as e:
            import logging
            logging.error(f"Error checking rate limit: {str(e)}")
            
            # On error, allow the request to pass but with a warning
            return 1, now + window_seconds

def cache_response(ttl: int = 3600):
    """Decorator to cache function responses"""
    def decorator(func):
        cache = RedisCache()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # If not in cache, execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator
