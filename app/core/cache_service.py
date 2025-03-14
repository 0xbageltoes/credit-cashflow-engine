"""
Unified Cache Service

This module provides a comprehensive caching system with:
- Hierarchical caching (memory → Redis)
- Error handling with graceful fallbacks
- Support for complex object serialization
- Configurable TTL settings
- Proper connection handling
"""
import json
import logging
import asyncio
import pickle
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from datetime import datetime, timedelta
import hashlib
import inspect
from functools import wraps

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from app.core.exceptions import CacheError, ApplicationError

T = TypeVar('T')

class RedisConfig:
    """Configuration options for Redis connections
    
    This class contains all configuration options for Redis connections,
    including timeouts, retry settings, and connection limits.
    """
    
    def __init__(
        self,
        url: str,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 1.0,
        retry_on_timeout: bool = True,
        max_connections: int = 10,
        health_check_interval: int = 30,
        max_retries: int = 3,
        decode_responses: bool = False,  # We handle this ourselves for better control
    ):
        """Initialize Redis configuration
        
        Args:
            url: Redis connection URL (redis://host:port/db)
            socket_timeout: Timeout for socket operations in seconds
            socket_connect_timeout: Timeout for socket connections in seconds
            retry_on_timeout: Whether to retry operations on timeout
            max_connections: Maximum number of connections in the pool
            health_check_interval: Interval for connection health checks
            max_retries: Maximum number of retries for operations
            decode_responses: Whether Redis should decode responses
        """
        self.url = url
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.decode_responses = decode_responses
    
    def get_connection_pool(self) -> redis.ConnectionPool:
        """Create a Redis connection pool from configuration
        
        Returns:
            ConnectionPool: Configured Redis connection pool
        """
        return redis.ConnectionPool.from_url(
            self.url,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
            retry_on_timeout=self.retry_on_timeout,
            max_connections=self.max_connections,
            health_check_interval=self.health_check_interval,
            decode_responses=self.decode_responses,
        )


class CacheService:
    """Unified caching service with hierarchical caching
    
    This service provides a comprehensive caching solution that:
    - Uses memory as first-level cache for performance
    - Falls back to Redis for distributed caching
    - Handles serialization of complex objects
    - Provides graceful fallbacks when Redis is unavailable
    - Implements proper error handling and logging
    """
    
    def __init__(
        self,
        redis_config: Optional[RedisConfig] = None,
        memory_ttl: int = 300,  # 5 minutes default
        default_ttl: int = 3600,  # 1 hour default for Redis
        serializer: Optional[Callable[[Any], bytes]] = None,
        deserializer: Optional[Callable[[bytes], Any]] = None,
        logger: Optional[logging.Logger] = None,
        max_memory_items: int = 1000,  # Maximum items in memory cache
    ):
        """Initialize the cache service
        
        Args:
            redis_config: Redis configuration options
            memory_ttl: Default TTL for memory cache in seconds
            default_ttl: Default TTL for Redis cache in seconds
            serializer: Custom serializer function for complex objects
            deserializer: Custom deserializer function for complex objects
            logger: Logger instance
            max_memory_items: Maximum items to store in memory cache
        """
        self.logger = logger or logging.getLogger(__name__)
        self.memory_ttl = memory_ttl
        self.default_ttl = default_ttl
        self.redis_available = False
        self.redis_client = None
        self.max_memory_items = max_memory_items
        
        # Custom serialization functions or use defaults
        self.serializer = serializer or self._default_serializer
        self.deserializer = deserializer or self._default_deserializer
        
        # In-memory cache
        self.memory_cache: Dict[str, Any] = {}
        self.expiry_times: Dict[str, datetime] = {}
        self.access_times: Dict[str, datetime] = {}  # For LRU eviction
        
        # Initialize Redis client if config provided
        if redis_config:
            self._initialize_redis_client(redis_config)
    
    def _initialize_redis_client(self, config: RedisConfig) -> None:
        """Initialize the Redis client
        
        Args:
            config: Redis configuration
        """
        try:
            # Create connection pool
            pool = config.get_connection_pool()
            
            # Create client with pool
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Mark as potentially available (will be verified on first use)
            self.redis_available = True
            
            self.logger.info(f"Redis client initialized with URL: {config.url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {str(e)}")
            self.redis_available = False
    
    async def _verify_redis_connection(self) -> bool:
        """Verify Redis connection is working
        
        Returns:
            bool: True if Redis is available, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            if not self.redis_available:
                self.logger.info("Redis connection restored")
                self.redis_available = True
            return True
        except Exception as e:
            if self.redis_available:
                self.logger.warning(f"Redis connection lost: {str(e)}")
                self.redis_available = False
            return False
    
    def _default_serializer(self, value: Any) -> bytes:
        """Default serialization using pickle with fallback to JSON
        
        Args:
            value: Value to serialize
            
        Returns:
            bytes: Serialized value
            
        Raises:
            CacheError: If serialization fails
        """
        try:
            # Try pickle first for most accurate serialization
            return pickle.dumps(value)
        except Exception as pickle_error:
            try:
                # Fall back to JSON for simpler objects
                return json.dumps(value).encode('utf-8')
            except Exception as json_error:
                raise CacheError(
                    f"Failed to serialize cache value: {str(json_error)}",
                    context={"value_type": type(value).__name__},
                    cause=json_error
                )
    
    def _default_deserializer(self, data: bytes) -> Any:
        """Default deserialization using pickle with fallback to JSON
        
        Args:
            data: Serialized data to deserialize
            
        Returns:
            Any: Deserialized value
            
        Raises:
            CacheError: If deserialization fails
        """
        if not data:
            return None
        
        # Try pickle first
        try:
            return pickle.loads(data)
        except Exception as pickle_error:
            # Fall back to JSON
            try:
                return json.loads(data.decode('utf-8'))
            except Exception as json_error:
                raise CacheError(
                    f"Failed to deserialize cache value: {str(json_error)}",
                    cause=json_error
                )
    
    def _add_to_memory_cache(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Add a value to memory cache with LRU eviction if needed
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
        """
        # Evict items if cache is full
        if len(self.memory_cache) >= self.max_memory_items and key not in self.memory_cache:
            self._evict_lru_items(1)  # Evict at least one item
        
        # Add to cache
        self.memory_cache[key] = value
        
        # Set expiry time
        ttl_seconds = ttl if ttl is not None else self.memory_ttl
        self.expiry_times[key] = datetime.now() + timedelta(seconds=ttl_seconds)
        
        # Update access time for LRU
        self.access_times[key] = datetime.now()
    
    def _evict_lru_items(self, count: int = 1) -> None:
        """Evict least recently used items from memory cache
        
        Args:
            count: Number of items to evict
        """
        if not self.access_times:
            return
        
        # Sort by access time
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        # Evict oldest accessed items
        for key in sorted_keys[:count]:
            self._remove_from_memory_cache(key)
    
    def _remove_from_memory_cache(self, key: str) -> None:
        """Remove an item from memory cache
        
        Args:
            key: Cache key to remove
        """
        self.memory_cache.pop(key, None)
        self.expiry_times.pop(key, None)
        self.access_times.pop(key, None)
    
    def _update_access_time(self, key: str) -> None:
        """Update access time for a cache key
        
        Args:
            key: Cache key
        """
        if key in self.memory_cache:
            self.access_times[key] = datetime.now()
    
    def _check_memory_expiry(self) -> None:
        """Check and remove expired items from memory cache"""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self.expiry_times.items()
            if now > expiry
        ]
        
        for key in expired_keys:
            self._remove_from_memory_cache(key)
    
    async def get(self, key: str) -> Any:
        """Get a value from cache (memory first, then Redis)
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
            
        Raises:
            CacheError: If a deserialization error occurs
        """
        # Clean expired items occasionally
        self._check_memory_expiry()
        
        # Check memory cache first
        if key in self.memory_cache:
            # Update access time for LRU
            self._update_access_time(key)
            
            # Check if expired
            if datetime.now() < self.expiry_times.get(key, datetime.min):
                self.logger.debug(f"Memory cache hit: {key}")
                return self.memory_cache[key]
            
            # Remove expired item
            self._remove_from_memory_cache(key)
        
        # Try Redis if available
        if self.redis_client and await self._verify_redis_connection():
            try:
                value_bytes = await self.redis_client.get(key)
                if value_bytes:
                    # Deserialize the value
                    try:
                        value = self.deserializer(value_bytes)
                        
                        # Add to memory cache
                        self._add_to_memory_cache(key, value)
                        
                        self.logger.debug(f"Redis cache hit: {key}")
                        return value
                    except ApplicationError:
                        # Re-raise application errors directly
                        raise
                    except Exception as e:
                        self.logger.error(f"Failed to deserialize Redis value: {str(e)}")
                        # Try to delete the corrupted value
                        try:
                            await self.redis_client.delete(key)
                        except Exception:
                            pass
            except Exception as e:
                self.logger.error(f"Redis get error: {str(e)}")
                self.redis_available = False
        
        self.logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        memory_ttl: Optional[int] = None
    ) -> bool:
        """Set a value in both memory and Redis cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL for Redis in seconds (None uses default)
            memory_ttl: TTL for memory cache in seconds (None uses default)
            
        Returns:
            bool: True if set in Redis, False otherwise (always sets in memory)
            
        Raises:
            CacheError: If a serialization error occurs
        """
        # Set in memory cache
        self._add_to_memory_cache(key, value, memory_ttl)
        
        # Set in Redis if available
        if self.redis_client and await self._verify_redis_connection():
            try:
                # Serialize the value
                value_bytes = self.serializer(value)
                
                # Set in Redis with TTL
                redis_ttl = ttl if ttl is not None else self.default_ttl
                await self.redis_client.set(key, value_bytes, ex=redis_ttl)
                
                self.logger.debug(f"Set value in Redis: {key}")
                return True
            except ApplicationError:
                # Re-raise application errors directly
                raise
            except Exception as e:
                self.logger.error(f"Redis set error: {str(e)}")
                self.redis_available = False
        
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from both memory and Redis cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if deleted from Redis, False otherwise (always deletes from memory)
        """
        # Remove from memory cache
        self._remove_from_memory_cache(key)
        
        # Remove from Redis if available
        if self.redis_client and await self._verify_redis_connection():
            try:
                await self.redis_client.delete(key)
                self.logger.debug(f"Deleted from Redis: {key}")
                return True
            except Exception as e:
                self.logger.error(f"Redis delete error: {str(e)}")
                self.redis_available = False
        
        return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries by pattern
        
        Args:
            pattern: Pattern to match keys
            
        Returns:
            int: Number of keys invalidated
        """
        count = 0
        
        # Invalidate from memory cache
        memory_keys = [k for k in self.memory_cache.keys() if pattern in k]
        for key in memory_keys:
            self._remove_from_memory_cache(key)
            count += 1
        
        # Invalidate from Redis if available
        if self.redis_client and await self._verify_redis_connection():
            try:
                # Find matching keys
                cursor = b'0'
                redis_keys = []
                
                # Use scan for efficient pattern matching
                while cursor:
                    cursor, keys = await self.redis_client.scan(
                        cursor=cursor, 
                        match=f"*{pattern}*",
                        count=100
                    )
                    redis_keys.extend(keys)
                    
                    # Break if we've completed the scan
                    if cursor == b'0':
                        break
                
                # Delete found keys
                if redis_keys:
                    deleted = await self.redis_client.delete(*redis_keys)
                    count += deleted
                    self.logger.debug(f"Invalidated {deleted} keys from Redis with pattern: {pattern}")
            
            except Exception as e:
                self.logger.error(f"Redis pattern invalidation error: {str(e)}")
                self.redis_available = False
        
        return count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache service health
        
        Returns:
            Dict with health status information
        """
        redis_status = "unknown"
        redis_error = None
        
        if self.redis_client:
            try:
                if await self._verify_redis_connection():
                    # Get some stats
                    info = await self.redis_client.info()
                    redis_status = "available"
                    redis_info = {
                        "version": info.get("redis_version", "unknown"),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "connected_clients": info.get("connected_clients", "unknown"),
                    }
                else:
                    redis_status = "unavailable"
                    redis_info = {}
            except Exception as e:
                redis_status = "error"
                redis_error = str(e)
                redis_info = {}
        else:
            redis_status = "disabled"
            redis_info = {}
        
        return {
            "memory_cache": {
                "items": len(self.memory_cache),
                "max_items": self.max_memory_items,
                "usage_percent": len(self.memory_cache) / self.max_memory_items * 100 if self.max_memory_items > 0 else 0,
            },
            "redis": {
                "status": redis_status,
                "error": redis_error,
                "info": redis_info if redis_status == "available" else {},
            },
            "timestamp": datetime.now().isoformat(),
        }


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a consistent cache key from args/kwargs
    
    This function creates a stable, consistent hash for caching based on
    the input arguments, ensuring the same inputs always produce the same key.
    
    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        str: Cache key string
    """
    # Start with calling function name if available
    frame = inspect.currentframe().f_back
    if frame:
        # Get caller's qualified name if possible
        module = frame.f_globals.get('__name__', '')
        function = frame.f_code.co_name
        prefix = f"{module}.{function}"
    else:
        prefix = "cache"
    
    # Combine all args and kwargs
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    
    # Create a hash of the combined string
    key_str = ";".join(key_parts)
    hashed = hashlib.md5(key_str.encode()).hexdigest()
    
    return f"{prefix}:{hashed}"


def cached(
    ttl: Optional[int] = None,
    memory_ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    cache_service_param: Optional[str] = None
):
    """Decorator for caching function results
    
    This decorator can be used with both async and sync functions to
    cache their results using the CacheService.
    
    Args:
        ttl: TTL for Redis cache in seconds
        memory_ttl: TTL for memory cache in seconds
        key_prefix: Optional prefix for cache keys
        cache_service_param: Name of parameter that contains the cache service
            If None, tries to find a parameter named 'cache' or 'cache_service'
            
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Find cache service in parameters
            cache_service = None
            
            if cache_service_param:
                # Explicitly specified parameter name
                if cache_service_param in kwargs:
                    cache_service = kwargs[cache_service_param]
                else:
                    # Try to find in positional args using signature
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    
                    # Find index of cache_service_param
                    try:
                        idx = param_names.index(cache_service_param)
                        if idx < len(args):
                            cache_service = args[idx]
                    except ValueError:
                        pass
            else:
                # Try common parameter names
                for param_name in ['cache', 'cache_service']:
                    if param_name in kwargs:
                        cache_service = kwargs[param_name]
                        break
            
            if not cache_service or not isinstance(cache_service, CacheService):
                # No cache service found, just call the function
                return await func(*args, **kwargs)
            
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            
            # Create signature-based args that exclude the cache service
            sig = inspect.signature(func)
            
            # Create filtered kwargs without the cache service
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if v is not cache_service:
                    filtered_kwargs[k] = v
            
            # Create filtered args without the cache service
            filtered_args = []
            param_names = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names) and arg is not cache_service:
                    filtered_args.append(arg)
            
            # Generate key using filtered args/kwargs
            key = f"{prefix}:{generate_cache_key(*filtered_args, **filtered_kwargs)}"
            
            # Try to get from cache
            cached_result = await cache_service.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call the function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_service.set(key, result, ttl=ttl, memory_ttl=memory_ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we still need async for cache operations
            # Create an event loop and run the async wrapper
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
