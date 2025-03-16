"""
Rate limiting module for the credit-cashflow-engine API

This module provides rate limiting functionality for API endpoints to protect
against abuse and ensure fair resource allocation among users in production.
It uses an in-memory backend for development and Redis for production.
"""

from fastapi import Request, HTTPException, status
from typing import Optional, Callable, Dict, Any, Union, List
import time
import hashlib
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from starlette.requests import Request
import asyncio
import logging
from functools import wraps
import os

from app.core.config import settings
from app.services.redis_service import RedisService
from app.services.unified_redis_service import UnifiedRedisService
from app.core.cache_service import CacheService, get_cache

# Configure logging
logger = logging.getLogger(__name__)

class CustomRateLimiter:
    """
    Custom rate limiter for the API that supports both in-memory and Redis backends.
    
    Features:
    - Configurable rate limits per endpoint
    - Per-user and per-IP limiting options
    - Graceful degradation to in-memory when Redis isn't available
    - Flexible key generation strategies
    """
    
    def __init__(self):
        """Initialize the rate limiter with no backend initially"""
        self._initialized = False
        self._backend = "memory"  # Default to in-memory for safety
        self._in_memory_store: Dict[str, Dict[str, float]] = {}
        self._redis_service = None
        self._unified_redis_service = None
        self._cache_service = None
        self._default_limit = f"{settings.RATE_LIMIT_REQUESTS}/hour"
        
        # When running tests or in development mode, use simpler rate limiting
        if settings.ENVIRONMENT in ["test", "development"]:
            self._default_limit = "1000/minute"  # Much higher limit for testing
        
        # Log setup info
        logger.info(f"Rate limiter initialized for environment: {settings.ENVIRONMENT}")
        logger.info(f"Default rate limit set to: {self._default_limit}")
    
    async def init(self):
        """Initialize the rate limiter with the appropriate backend"""
        if self._initialized:
            logger.debug("Rate limiter already initialized, skipping")
            return
        
        logger.info("Initializing rate limiter...")
        
        # Track initialization attempts for services
        services_attempted = []
        
        try:
            # First try to use UnifiedRedisService which has better production capabilities
            try:
                logger.info("Attempting to initialize rate limiter with UnifiedRedisService")
                services_attempted.append("unified_redis")
                self._unified_redis_service = UnifiedRedisService()
                
                # Test connection
                if await self._unified_redis_service.ping():
                    self._backend = "unified_redis"
                    logger.info("Rate limiter initialized with UnifiedRedisService backend")
                else:
                    logger.warning("UnifiedRedisService ping returned False, trying next backend")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedRedisService for rate limiting: {str(e)}")
            
            # If UnifiedRedisService fails, try the CacheService
            if self._backend == "memory" and "cache_service" not in services_attempted:
                try:
                    logger.info("Attempting to initialize rate limiter with CacheService")
                    services_attempted.append("cache_service")
                    self._cache_service = get_cache()
                    
                    # Test the connection
                    result = await self._cache_service.health_check()
                    if result.get("status") == "ok":
                        self._backend = "cache_service"
                        logger.info("Rate limiter initialized with CacheService backend")
                    else:
                        logger.warning("CacheService health check failed, trying next backend")
                except Exception as e:
                    logger.warning(f"Failed to initialize CacheService for rate limiting: {str(e)}")
            
            # Finally, try the direct RedisService if available
            if self._backend == "memory" and "redis" not in services_attempted:
                try:
                    logger.info("Attempting to initialize rate limiter with RedisService")
                    services_attempted.append("redis")
                    self._redis_service = RedisService()
                    
                    # Test connectivity
                    result = await self._redis_service.ping()
                    if result:
                        self._backend = "redis"
                        logger.info("Rate limiter initialized with RedisService backend")
                    else:
                        logger.warning("RedisService ping returned False, falling back to memory backend")
                except Exception as e:
                    logger.warning(f"Failed to initialize RedisService for rate limiting: {str(e)}")
            
            # Print diagnostic info when using Upstash in production
            if self._backend != "memory" and settings.ENVIRONMENT == "production":
                if "UPSTASH_REDIS_HOST" in os.environ:
                    logger.info("Using Upstash Redis for rate limiting in production")
                else:
                    logger.info(f"Using {self._backend} for rate limiting in production")
                    
        except Exception as e:
            logger.error(f"Unexpected error during rate limiter initialization: {str(e)}")
            self._backend = "memory"
                
        # Always ensure we have in-memory store as a fallback
        if self._backend == "memory":
            logger.info("Rate limiter initialized with in-memory backend")
            if settings.ENVIRONMENT == "production":
                logger.warning(
                    "WARNING: Using in-memory rate limiting in production environment. "
                    "This is not recommended for distributed deployments."
                )
            
        self._initialized = True
        logger.info(f"Rate limiter initialization complete. Using {self._backend} backend.")
    
    def limit(
        self,
        limit_value: str = None,
        key_func: Callable[[Request], str] = None,
    ):
        """
        Rate limiting decorator for FastAPI endpoints
        
        Args:
            limit_value: Rate limit string in format "number/time_period"
            key_func: Function to generate the key from request
        """
        limit_value = limit_value or self._default_limit
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request object
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    for _, value in kwargs.items():
                        if isinstance(value, Request):
                            request = value
                            break
                
                if not request:
                    # Can't find request object, skip rate limiting
                    logger.warning("Rate limiting skipped: No request object found")
                    return await func(*args, **kwargs)
                
                # Generate key from request
                if key_func:
                    try:
                        key = key_func(request)
                    except Exception as e:
                        logger.error(f"Error generating rate limit key: {str(e)}")
                        return await func(*args, **kwargs)
                else:
                    # Default key is IP address or user ID if authenticated
                    key = self._default_key_func(request)
                
                rate_limit_exceeded = await self._check_rate_limit(key, limit_value)
                
                if rate_limit_exceeded:
                    retry_after, limit = self._parse_limit(limit_value)
                    logger.warning(f"Rate limit exceeded for key: {key}, limit: {limit_value}")
                    
                    # Return a standardized 429 error with helpful headers
                    retry_seconds = int(retry_after)
                    reset_time = int(time.time()) + retry_seconds
                    
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "message": "Rate limit exceeded. Too many requests.",
                            "retry_after_seconds": retry_seconds,
                            "limit": limit,
                            "reset_timestamp": reset_time,
                        },
                        headers={
                            "Retry-After": str(retry_seconds),
                            "X-Rate-Limit-Limit": str(limit),
                            "X-Rate-Limit-Reset": str(reset_time),
                            "X-Rate-Limit-Policy": limit_value,
                        },
                    )
                
                # Execute the original function
                return await func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def _default_key_func(self, request: Request) -> str:
        """
        Default function to generate a key from a request
        
        Args:
            request: FastAPI request object
            
        Returns:
            str: Key for rate limiting
        """
        client_ip = request.client.host if request.client else "unknown"
        
        # Try to get user ID from JWT token
        user = getattr(request, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"
        
        # Try to get from authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            # Hash the token for privacy
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            return f"token:{token_hash}"
        
        # Fall back to IP address
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, key: str, limit_value: str) -> bool:
        """
        Check if a request exceeds the rate limit
        
        Args:
            key: Rate limiting key
            limit_value: Rate limit in format "number/time_period"
            
        Returns:
            bool: True if rate limit exceeded, False otherwise
        """
        if not self._initialized:
            try:
                await self.init()
            except Exception as e:
                logger.error(f"Failed to initialize rate limiter: {str(e)}")
                return False
        
        # Parse limit value
        window, limit = self._parse_limit(limit_value)
        
        # Add namespace to key
        namespaced_key = f"ratelimit:{key}:{limit}:{window}"
        
        # Check rate limit based on backend
        try:
            if self._backend == "unified_redis" and self._unified_redis_service:
                return await self._check_rate_limit_unified_redis(namespaced_key, window, limit)
            elif self._backend == "cache_service" and self._cache_service:
                return await self._check_rate_limit_cache_service(namespaced_key, window, limit)
            elif self._backend == "redis" and self._redis_service:
                return await self._check_rate_limit_redis(namespaced_key, window, limit)
            else:
                return self._check_rate_limit_memory(namespaced_key, window, limit)
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            # In case of error, fail open to allow the request
            return False
    
    async def _check_rate_limit_unified_redis(self, key: str, window: int, limit: int) -> bool:
        """
        Check rate limit using UnifiedRedisService
        
        Args:
            key: Rate limiting key
            window: Time window in seconds
            limit: Maximum number of requests
            
        Returns:
            bool: True if rate limit exceeded, False otherwise
        """
        try:
            now = int(time.time())
            
            # Use the rolling window algorithm with sorted sets
            # 1. Remove entries older than the window
            # 2. Add current timestamp
            # 3. Count entries in window
            
            # Get the raw Redis client for pipeline operations
            redis_client = self._unified_redis_service._async_client
            pipeline = redis_client.pipeline()
            
            # Remove entries older than the window
            pipeline.zremrangebyscore(key, 0, now - window)
            
            # Add current request with timestamp as score
            pipeline.zadd(key, {str(now): now})
            
            # Count remaining items
            pipeline.zcard(key)
            
            # Set expiry to auto-cleanup
            pipeline.expire(key, window)
            
            # Execute all commands
            results = await pipeline.execute()
            count = results[2]
            
            # Check if over limit
            return count > limit
            
        except Exception as e:
            logger.error(f"Error checking rate limit with UnifiedRedisService: {str(e)}")
            # In production, we want to fail open rather than block legitimate traffic
            return False
    
    async def _check_rate_limit_cache_service(self, key: str, window: int, limit: int) -> bool:
        """
        Check rate limit using CacheService
        
        Args:
            key: Rate limiting key
            window: Time window in seconds
            limit: Maximum number of requests
            
        Returns:
            bool: True if rate limit exceeded, False otherwise
        """
        try:
            redis_client = getattr(self._cache_service, "_redis_async", None)
            if not redis_client:
                logger.warning("CacheService doesn't have _redis_async attribute, falling back to memory")
                return self._check_rate_limit_memory(key, window, limit)
                
            pipeline = redis_client.pipeline()
            now = int(time.time())
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window)
            # Add current request
            pipeline.zadd(key, {str(now): now})
            # Count requests in window
            pipeline.zcard(key)
            # Set expiry
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            count = results[2]
            
            return count > limit
        except Exception as e:
            logger.error(f"Error checking rate limit with CacheService: {str(e)}")
            return False  # Fail open on errors
    
    async def _check_rate_limit_redis(self, key: str, window: int, limit: int) -> bool:
        """
        Check rate limit using RedisService
        
        Args:
            key: Rate limiting key
            window: Time window in seconds
            limit: Maximum number of requests
            
        Returns:
            bool: True if rate limit exceeded, False otherwise
        """
        try:
            # Make sure we have the async client available
            redis_client = getattr(self._redis_service, "_redis_async", None)
            if not redis_client:
                logger.warning("RedisService doesn't have _redis_async attribute, falling back to memory")
                return self._check_rate_limit_memory(key, window, limit)
                
            # Use Redis sorted set for sliding window
            now = int(time.time())
            
            # Execute pipeline of commands
            pipeline = redis_client.pipeline()
            pipeline.zremrangebyscore(key, 0, now - window)  # Remove old entries
            pipeline.zadd(key, {str(now): now})  # Add current request
            pipeline.zcard(key)  # Count requests in window
            pipeline.expire(key, window)  # Set expiry
            
            results = await pipeline.execute()
            count = results[2]
            
            return count > limit
        except Exception as e:
            logger.error(f"Error checking rate limit with RedisService: {str(e)}")
            return False  # Fail open on errors
    
    def _check_rate_limit_memory(self, key: str, window: int, limit: int) -> bool:
        """
        Check rate limit using in-memory store
        
        Args:
            key: Rate limiting key
            window: Time window in seconds
            limit: Maximum number of requests
            
        Returns:
            bool: True if rate limit exceeded, False otherwise
        """
        now = time.time()
        
        # Initialize if key doesn't exist
        if key not in self._in_memory_store:
            self._in_memory_store[key] = {}
        
        # Clean up old timestamps
        self._in_memory_store[key] = {
            ts: timestamp for ts, timestamp in self._in_memory_store[key].items()
            if timestamp > now - window
        }
        
        # Add current request
        timestamp = str(now)
        self._in_memory_store[key][timestamp] = now
        
        # Check if limit exceeded
        return len(self._in_memory_store[key]) > limit
    
    def _parse_limit(self, limit_value: str) -> tuple:
        """
        Parse a limit value string
        
        Args:
            limit_value: Limit value in format "number/time_period"
            
        Returns:
            tuple: (window_seconds, max_requests)
        """
        parts = limit_value.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid rate limit format: {limit_value}")
            # Default to 100 requests per hour
            return 3600, 100
        
        try:
            max_requests = int(parts[0])
            time_unit = parts[1].lower()
            
            if time_unit.endswith("s"):
                time_unit = time_unit[:-1]
            
            if time_unit == "second":
                window = 1
            elif time_unit == "minute":
                window = 60
            elif time_unit == "hour":
                window = 3600
            elif time_unit == "day":
                window = 86400
            else:
                logger.error(f"Unknown time unit: {time_unit}")
                window = 3600  # Default to hour
            
            return window, max_requests
        except Exception as e:
            logger.error(f"Error parsing rate limit: {str(e)}")
            return 3600, 100  # Default to 100 per hour


# Create a single instance of the rate limiter
limiter = CustomRateLimiter()

# Function to initialize the rate limiter (called during app startup)
async def init_rate_limiter():
    """Initialize the rate limiter during application startup"""
    try:
        await limiter.init()
    except Exception as e:
        # Make sure to catch and log any errors during initialization
        # This prevents app startup from failing if rate limiting can't be initialized
        logger.error(f"Failed to initialize rate limiter: {str(e)}")
        logger.warning("Rate limiting will use in-memory fallback")
