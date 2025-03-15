"""Hierarchical Cache Service Module

This module provides a multi-level caching implementation with memory and Redis layers:
- In-memory LRU cache for fastest access to frequently used data
- Redis persistent cache as the second layer
- Comprehensive error handling and fallback mechanisms
- Configurable TTLs for each cache layer
- Cache invalidation for both layers
- Deterministic cache key generation
"""

import logging
import hashlib
import json
import time
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Callable, TypeVar, Tuple
import asyncio
import inspect
from functools import wraps
from urllib.parse import urlparse

# Redis imports
import redis
import redis.asyncio

# Project imports
from app.core.exceptions import CacheError, ApplicationError

# Type variable for the cached decorator
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)


class HierarchicalCacheService:
    """Multi-level caching service (memory → Redis)"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        prefix: str = "app:",
        redis_ttl: int = 3600,  # 1 hour default for Redis
        memory_ttl: int = 300,  # 5 minutes default for memory
        memory_max_items: int = 1000,
        logger: Optional[logging.Logger] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        redis_ssl: Optional[bool] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        health_check_interval: int = 30
    ):
        """Initialize hierarchical cache
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all cache entries
            redis_ttl: Default TTL for Redis cache
            memory_ttl: Default TTL for memory cache
            memory_max_items: Maximum items in memory cache
            logger: Optional logger
            redis_host: Redis host (if not using URL)
            redis_port: Redis port (if not using URL)
            redis_password: Redis password (if not using URL)
            redis_ssl: Force SSL/TLS connection (for Upstash)
            socket_timeout: Redis socket timeout
            socket_connect_timeout: Redis socket connect timeout
            health_check_interval: Health check interval
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Try to get Redis configuration from environment if not provided
        redis_url = redis_url or os.environ.get("REDIS_URL")
        redis_host = redis_host or os.environ.get("UPSTASH_REDIS_HOST")
        redis_port = redis_port or int(os.environ.get("UPSTASH_REDIS_PORT", "6379"))
        redis_password = redis_password or os.environ.get("UPSTASH_REDIS_PASSWORD")
        
        # Determine if we should use SSL (required for Upstash)
        # If redis_url starts with rediss:// or explicitly set
        if redis_ssl is None:
            redis_ssl = redis_url is not None and redis_url.startswith("rediss://")
        
        # Initialize Redis client with error handling
        try:
            if redis_url:
                # Use URL if provided (preferred method)
                self.redis = redis.asyncio.from_url(
                    redis_url,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    health_check_interval=health_check_interval,
                    decode_responses=True  # Always decode Redis responses to strings
                )
            elif redis_host and redis_password:
                # Construct connection using separate parameters
                self.redis = redis.asyncio.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    ssl=redis_ssl,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    health_check_interval=health_check_interval,
                    decode_responses=True  # Always decode Redis responses to strings
                )
            else:
                self.logger.warning("No Redis connection details provided. Running in memory-only mode.")
                self.redis = None
                self._redis_available = False
            
            if self.redis is not None:
                self._redis_available = True
                self.logger.info(f"Redis connection initialized to: "
                               f"{redis_host or (urlparse(redis_url).hostname if redis_url else 'unknown')}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis client: {str(e)}")
            self.redis = None
            self._redis_available = False
        
        self.prefix = prefix
        self.redis_ttl = redis_ttl
        self.memory_ttl = memory_ttl
        self.memory_max_items = memory_max_items
        
        # Initialize memory cache
        self.memory_cache = {}
        self.expiry_times = {}
        self.access_times = {}  # For LRU eviction
        
        # Stats tracking
        self._memory_hits = 0
        self._memory_misses = 0
        self._redis_hits = 0
        self._redis_misses = 0
        self._errors = 0
        
        self.logger.info(f"Initialized HierarchicalCacheService with prefix '{prefix}'")
        
        # Verify Redis connection with ping in background
        if self._redis_available and self.redis is not None:
            asyncio.create_task(self._verify_redis_connection())
    
    async def _verify_redis_connection(self):
        """Verify Redis connection with ping in background"""
        try:
            result = await self.redis.ping()
            if result:
                self.logger.info("Redis connection verified successfully")
            else:
                self.logger.warning("Redis ping returned False")
                self._redis_available = False
        except Exception as e:
            self.logger.warning(f"Redis connection verification failed: {str(e)}")
            self._redis_available = False
    
    async def get(self, key: str) -> Any:
        """Get a value from cache (memory first, then Redis)"""
        prefixed_key = f"{self.prefix}{key}"
        
        # Check memory cache first
        if prefixed_key in self.memory_cache:
            # Check if expired
            if datetime.now() < self.expiry_times.get(prefixed_key, datetime.min):
                # Update access time for LRU
                self.access_times[prefixed_key] = datetime.now()
                self._memory_hits += 1
                return self.memory_cache[prefixed_key]
            
            # Expired, remove from memory cache
            self._remove_from_memory(prefixed_key)
        
        self._memory_misses += 1
        
        # Try Redis if not in memory or expired
        if not self._redis_available or self.redis is None:
            return None
        
        try:
            value = await self.redis.get(prefixed_key)
            if value is not None:
                # Deserialize value
                try:
                    deserialized = json.loads(value)
                    
                    # Update memory cache
                    self._add_to_memory(prefixed_key, deserialized)
                    
                    self._redis_hits += 1
                    return deserialized
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to deserialize value for key {key}")
                    self._errors += 1
                    return None
        except Exception as e:
            self.logger.warning(f"Redis get error for key {key}: {str(e)}")
            self._errors += 1
        
        self._redis_misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        redis_ttl: Optional[int] = None,
        memory_ttl: Optional[int] = None
    ) -> bool:
        """Set a value in both memory and Redis cache"""
        prefixed_key = f"{self.prefix}{key}"
        
        # Add to memory cache
        self._add_to_memory(
            prefixed_key,
            value,
            memory_ttl or self.memory_ttl
        )
        
        # Set in Redis if available
        if not self._redis_available or self.redis is None:
            return False
        
        try:
            serialized = json.dumps(value)
            ttl = redis_ttl if redis_ttl is not None else self.redis_ttl
            
            await self.redis.set(prefixed_key, serialized, ex=ttl)
            return True
        except Exception as e:
            self.logger.warning(f"Redis set error for key {key}: {str(e)}")
            self._errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from both memory and Redis cache"""
        prefixed_key = f"{self.prefix}{key}"
        
        # Remove from memory
        memory_removed = self._remove_from_memory(prefixed_key)
        
        # Remove from Redis if available
        if not self._redis_available or self.redis is None:
            return memory_removed
        
        try:
            result = await self.redis.delete(prefixed_key)
            return result > 0 or memory_removed
        except Exception as e:
            self.logger.warning(f"Redis delete error for key {key}: {str(e)}")
            self._errors += 1
            return memory_removed
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern
        
        Args:
            pattern: Pattern to match (will be prefixed)
            
        Returns:
            Number of keys invalidated
        """
        prefixed_pattern = f"{self.prefix}{pattern}"
        count = 0
        
        # Remove matching keys from memory
        memory_keys_to_remove = []
        for k in list(self.memory_cache.keys()):
            if prefixed_pattern in k:
                memory_keys_to_remove.append(k)
                count += 1
        
        # Actually remove the keys (separate loop to avoid modifying during iteration)
        for k in memory_keys_to_remove:
            self._remove_from_memory(k)
        
        # Remove matching keys from Redis if available
        if not self._redis_available or self.redis is None:
            return count
        
        try:
            # Find matching keys
            keys = await self.redis.keys(f"{prefixed_pattern}*")
            
            if keys:
                # Delete keys
                num_deleted = await self.redis.delete(*keys)
                count += num_deleted
        except Exception as e:
            self.logger.warning(f"Redis pattern invalidation error: {str(e)}")
            self._errors += 1
        
        return count
    
    def _add_to_memory(
        self,
        key: str,
        value: Any,
        ttl: int
    ):
        """Add an item to memory cache with LRU eviction
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        # Check if at capacity and need to evict
        if (
            key not in self.memory_cache and
            len(self.memory_cache) >= self.memory_max_items
        ):
            # Use LRU eviction policy
            if self.access_times:
                # Find least recently used key
                lru_key = min(self.access_times, key=self.access_times.get)
                self._remove_from_memory(lru_key)
        
        # Add to memory cache
        self.memory_cache[key] = value
        self.expiry_times[key] = datetime.now() + timedelta(seconds=ttl)
        self.access_times[key] = datetime.now()
    
    def _remove_from_memory(self, key: str) -> bool:
        """Remove an item from memory cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if item was removed
        """
        was_present = key in self.memory_cache
        
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if key in self.expiry_times:
            del self.expiry_times[key]
        
        if key in self.access_times:
            del self.access_times[key]
            
        return was_present
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dict with cache statistics
        """
        memory_total = self._memory_hits + self._memory_misses
        redis_total = self._redis_hits + self._redis_misses
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "memory_max_items": self.memory_max_items,
            "memory_hit_rate": (self._memory_hits / memory_total) if memory_total > 0 else 0,
            "memory_hits": self._memory_hits,
            "memory_misses": self._memory_misses,
            "redis_hit_rate": (self._redis_hits / redis_total) if redis_total > 0 else 0,
            "redis_hits": self._redis_hits,
            "redis_misses": self._redis_misses,
            "redis_available": self._redis_available,
            "errors": self._errors
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check
        
        Returns:
            Dict with health status
        """
        status = {
            "memory_cache": {
                "status": "ok",
                "size": len(self.memory_cache),
                "capacity": self.memory_max_items
            },
            "redis_cache": {
                "status": "unknown",
                "connected": self._redis_available
            }
        }
        
        if self._redis_available and self.redis is not None:
            try:
                # Try simple ping operation
                result = await self.redis.ping()
                status["redis_cache"]["status"] = "ok" if result else "error"
            except Exception as e:
                status["redis_cache"]["status"] = "error"
                status["redis_cache"]["error"] = str(e)
                self._redis_available = False
        else:
            status["redis_cache"]["status"] = "unavailable"
        
        return status
    
    async def close(self):
        """Close Redis connection"""
        if self.redis is not None:
            try:
                await self.redis.close()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing Redis connection: {str(e)}")


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Deterministic cache key string
    """
    try:
        # Handle non-serializable objects in args and kwargs
        def make_serializable(obj):
            if hasattr(obj, 'model_dump'):
                # For Pydantic v2 models
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                # For Pydantic v1 models
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                # Regular objects with __dict__
                return str(obj.__dict__)
            # Use string representation as fallback
            return str(obj)
            
        # Process args
        serializable_args = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                serializable_args.append(arg)
            else:
                serializable_args.append(make_serializable(arg))
        
        # Process kwargs
        serializable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                serializable_kwargs[k] = v
            else:
                serializable_kwargs[k] = make_serializable(v)
        
        # Convert to JSON strings with sorted keys
        args_str = json.dumps(serializable_args, sort_keys=True)
        kwargs_str = json.dumps(serializable_kwargs, sort_keys=True)
        
        # Create a hash of the combined string
        combined = f"{args_str}:{kwargs_str}".encode()
        return hashlib.md5(combined).hexdigest()
        
    except Exception as e:
        logger.warning(f"Error generating cache key: {e}")
        # Return a fallback key that won't match any existing cache
        return f"no_cache_{time.time()}"


def cached(
    cache_service: HierarchicalCacheService,
    key_prefix: str,
    redis_ttl: Optional[int] = None,
    memory_ttl: Optional[int] = None
):
    """Decorator to cache function results
    
    Args:
        cache_service: Cache service
        key_prefix: Prefix for cache keys
        redis_ttl: Optional Redis TTL override
        memory_ttl: Optional memory TTL override
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if cache service is initialized
            if cache_service is None:
                # No caching available, just call the function
                return await func(*args, **kwargs)
            
            # Extract function signature for better cache keys
            try:
                signature = inspect.signature(func)
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Get function arguments as dictionary
                arg_dict = dict(bound_args.arguments)
                
                # Generate cache key
                key = f"{key_prefix}:{func.__name__}:{generate_cache_key(**arg_dict)}"
                
                # Try to get from cache
                cached_value = await cache_service.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await cache_service.set(
                    key, 
                    result, 
                    redis_ttl=redis_ttl,
                    memory_ttl=memory_ttl
                )
                
                return result
                
            except Exception as e:
                # Log error but continue with function execution
                logger.warning(f"Cache error in decorator: {str(e)}")
                # Call original function as fallback
                return await func(*args, **kwargs)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if cache service is initialized
            if cache_service is None:
                # No caching available, just call the function
                return func(*args, **kwargs)
                
            # For synchronous functions, create an event loop and run the async version
            async def run_async():
                return await async_wrapper(*args, **kwargs)
            
            try:
                # Check if we're in an event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new loop for this operation
                    new_loop = asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(run_async())
                    finally:
                        new_loop.close()
                else:
                    # Use the existing loop
                    return loop.run_until_complete(run_async())
            except RuntimeError:
                # No event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(run_async())
                finally:
                    loop.close()
        
        # Check if the wrapped function is a coroutine function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Factory function to create a preconfigured HierarchicalCacheService
def create_cache_service(
    prefix: str = "app:",
    redis_ttl: int = 3600,
    memory_ttl: int = 300,
    memory_max_items: int = 1000
) -> HierarchicalCacheService:
    """Create a hierarchical cache service with configuration from environment
    
    Args:
        prefix: Key prefix for all cache entries
        redis_ttl: Default TTL for Redis cache
        memory_ttl: Default TTL for memory cache
        memory_max_items: Maximum items in memory cache
        
    Returns:
        Configured HierarchicalCacheService instance
    """
    # Get Redis configuration from environment
    redis_url = os.environ.get("REDIS_URL")
    redis_host = os.environ.get("UPSTASH_REDIS_HOST")
    redis_port = int(os.environ.get("UPSTASH_REDIS_PORT", "6379"))
    redis_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
    
    # Default to Upstash SSL configuration
    redis_ssl = True if redis_url and redis_url.startswith("rediss://") else None
    
    # Create and return the cache service
    return HierarchicalCacheService(
        redis_url=redis_url,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        redis_ssl=redis_ssl,
        prefix=prefix,
        redis_ttl=redis_ttl,
        memory_ttl=memory_ttl,
        memory_max_items=memory_max_items
    )


# Best practices documentation
"""
HIERARCHICAL CACHING BEST PRACTICES

1. Key Generation
   - Use consistent key generation for all cache operations
   - Include all parameters that affect the output in the cache key
   - Use the generate_cache_key function for consistent hashing

2. TTL Settings
   - Set appropriate TTLs based on data volatility
   - Use shorter TTLs for memory cache than Redis
   - Consider data access patterns when setting TTLs

3. Cache Invalidation
   - Invalidate caches when source data changes
   - Use invalidate_pattern for bulk invalidation
   - Consider implementing cache versioning for major data changes

4. Error Handling
   - System always falls back to original function on cache errors
   - Monitor error rates through the health_check method
   - Cache errors should never cause application failures

5. Memory Cache Sizing
   - Size the memory cache based on request patterns
   - Monitor hit rates to optimize memory cache size
   - Adjust memory_max_items based on memory constraints

6. Performance Considerations
   - Memory cache provides microsecond access times
   - Redis provides millisecond access times
   - Avoid caching very large objects in memory
   - Consider compression for large objects in Redis

7. Cache Stampedes
   - Implement background refresh for frequently accessed items
   - Consider staggered TTLs to prevent mass expirations
   - For high-traffic keys, consider probabilistic early expiration

8. Monitoring
   - Monitor memory and Redis hit rates separately
   - Track error rates and types
   - Monitor memory cache size and eviction rates

9. Upstash Redis Specifics
   - Always use SSL/TLS for Upstash connections (rediss:// protocol)
   - Configure proper retry mechanisms for serverless environments
   - Set appropriate timeouts for network latency
   - Use connection pooling efficiently in serverless contexts
"""
