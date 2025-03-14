"""
Rate Limiting Middleware

This module provides a robust rate limiting middleware for FastAPI applications
using Redis as a backend. It supports:
- Per-endpoint rate limiting
- Per-user rate limiting
- Differentiated rate limits based on user roles
- Sliding window algorithm for accurate rate limiting
- Graceful fallback when Redis is unavailable
"""
import time
import logging
from typing import Callable, Dict, Optional, Tuple, Union, List, Set

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis

from app.core.config import settings
from app.core.auth import get_current_user_id

# Setup logging
logger = logging.getLogger(__name__)

class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, retry_after: int):
        """
        Initialize the exception
        
        Args:
            retry_after: Seconds until the client can retry
        """
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )

class RateLimiter:
    """Rate limiter using Redis sliding window algorithm"""
    
    def __init__(
        self, 
        redis_client: redis.Redis,
        key_prefix: str = "rate_limit:",
        fallback_allowed: bool = True
    ):
        """
        Initialize the rate limiter
        
        Args:
            redis_client: Redis client
            key_prefix: Prefix for Redis keys
            fallback_allowed: Whether to allow requests when Redis is unavailable
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.fallback_allowed = fallback_allowed
        self._redis_available = True
    
    def _get_redis_key(self, key: str) -> str:
        """
        Get the full Redis key
        
        Args:
            key: Base key
            
        Returns:
            Full Redis key
        """
        return f"{self.key_prefix}{key}"
    
    def is_rate_limited(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> Tuple[bool, int]:
        """
        Check if a key is rate limited using sliding window algorithm
        
        Args:
            key: The rate limit key (e.g., "ip:{ip}" or "user:{user_id}")
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_limited, retry_after)
        """
        # Skip rate limiting if Redis is unavailable and fallback is allowed
        if not self._redis_available and self.fallback_allowed:
            return False, 0
        
        current_time = int(time.time())
        full_key = self._get_redis_key(key)
        
        try:
            # Start pipeline for atomic operations
            pipeline = self.redis.pipeline()
            
            # Remove old entries outside the current window
            window_start = current_time - window_seconds
            pipeline.zremrangebyscore(full_key, 0, window_start)
            
            # Count current entries in the window
            pipeline.zcard(full_key)
            
            # Add current request with current timestamp score
            pipeline.zadd(full_key, {str(current_time): current_time})
            
            # Set expiration on the key to automatically clean up
            pipeline.expire(full_key, window_seconds)
            
            # Execute pipeline
            _, current_count, _, _ = pipeline.execute()
            
            # Check if rate limit exceeded
            if current_count > max_requests:
                # Get the oldest timestamp in the window
                oldest = self.redis.zrange(full_key, 0, 0, withscores=True)
                
                if oldest:
                    # Calculate retry after time based on the oldest request
                    oldest_timestamp = int(oldest[0][1])
                    retry_after = oldest_timestamp + window_seconds - current_time
                    return True, max(1, retry_after)
                
                # Fallback retry after time
                return True, window_seconds
            
            # Reset Redis availability flag if we got here
            self._redis_available = True
            return False, 0
        
        except redis.RedisError as e:
            # Log the error
            logger.error(f"Redis error in rate limiter: {str(e)}")
            
            # Mark Redis as unavailable
            self._redis_available = False
            
            # Use fallback behavior
            if self.fallback_allowed:
                return False, 0
            else:
                # Default to rate limited if Redis is down and fallback is not allowed
                return True, 60
        
        except Exception as e:
            # Log any other errors
            logger.error(f"Unexpected error in rate limiter: {str(e)}")
            
            # Use fallback behavior
            if self.fallback_allowed:
                return False, 0
            else:
                return True, 60

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting FastAPI requests
    
    This middleware applies different rate limits based on:
    - IP address
    - User ID (if authenticated)
    - Endpoint path
    - HTTP method
    
    It supports different rate limit tiers based on user roles.
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        redis_client: Optional[redis.Redis] = None,
        global_rate_limit: Optional[Tuple[int, int]] = None,
        ip_rate_limit: Optional[Tuple[int, int]] = None,
        user_rate_limit: Optional[Dict[str, Tuple[int, int]]] = None,
        endpoint_rate_limits: Optional[Dict[str, Tuple[int, int]]] = None,
        excluded_paths: Optional[List[str]] = None,
        fallback_allowed: bool = True
    ):
        """
        Initialize the middleware
        
        Args:
            app: The ASGI application
            redis_client: Redis client
            global_rate_limit: Global rate limit as (max_requests, window_seconds)
            ip_rate_limit: Per-IP rate limit as (max_requests, window_seconds)
            user_rate_limit: Per-user rate limits by role as {"role": (max_requests, window_seconds)}
            endpoint_rate_limits: Per-endpoint rate limits as {"path:method": (max_requests, window_seconds)}
            excluded_paths: List of paths to exclude from rate limiting
            fallback_allowed: Whether to allow requests when Redis is unavailable
        """
        super().__init__(app)
        
        # Initialize Redis client if not provided
        if redis_client is None:
            try:
                self.redis = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True
                )
            except Exception as e:
                logger.error(f"Error initializing Redis client: {str(e)}")
                self.redis = None
        else:
            self.redis = redis_client
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            self.redis,
            key_prefix="rate_limit:",
            fallback_allowed=fallback_allowed
        ) if self.redis else None
        
        # Set rate limits with defaults
        self.global_rate_limit = global_rate_limit or (
            settings.GLOBAL_RATE_LIMIT_MAX_REQUESTS, 
            settings.GLOBAL_RATE_LIMIT_WINDOW_SECONDS
        )
        
        self.ip_rate_limit = ip_rate_limit or (
            settings.IP_RATE_LIMIT_MAX_REQUESTS,
            settings.IP_RATE_LIMIT_WINDOW_SECONDS
        )
        
        self.user_rate_limit = user_rate_limit or {
            "default": (
                settings.USER_RATE_LIMIT_MAX_REQUESTS,
                settings.USER_RATE_LIMIT_WINDOW_SECONDS
            ),
            "premium": (
                settings.PREMIUM_USER_RATE_LIMIT_MAX_REQUESTS,
                settings.PREMIUM_USER_RATE_LIMIT_WINDOW_SECONDS
            )
        }
        
        self.endpoint_rate_limits = endpoint_rate_limits or {}
        
        # Set excluded paths with defaults
        self.excluded_paths = set(excluded_paths or [
            "/api/v1/health",
            "/api/v1/docs",
            "/api/v1/openapi.json",
            "/ws"  # Exclude WebSocket connections
        ])
    
    def _should_exclude_path(self, path: str) -> bool:
        """
        Check if a path should be excluded from rate limiting
        
        Args:
            path: The request path
            
        Returns:
            True if the path should be excluded
        """
        # Check for exact matches
        if path in self.excluded_paths:
            return True
        
        # Check for prefix matches
        for excluded_path in self.excluded_paths:
            if excluded_path.endswith("*") and path.startswith(excluded_path[:-1]):
                return True
        
        return False
    
    def _get_endpoint_key(self, path: str, method: str) -> str:
        """
        Get a key for an endpoint
        
        Args:
            path: The request path
            method: The HTTP method
            
        Returns:
            Endpoint key for rate limit lookup
        """
        return f"{path}:{method}"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch the request with rate limiting
        
        Args:
            request: The request
            call_next: The next middleware
            
        Returns:
            The response
        """
        # Skip rate limiting if Redis is not available and fallback is allowed
        if self.rate_limiter is None:
            return await call_next(request)
        
        # Skip rate limiting for excluded paths
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Apply global rate limit
        global_key = "global"
        is_limited, retry_after = self.rate_limiter.is_rate_limited(
            key=global_key,
            max_requests=self.global_rate_limit[0],
            window_seconds=self.global_rate_limit[1]
        )
        
        if is_limited:
            return self._rate_limited_response(retry_after)
        
        # Apply IP-based rate limit
        ip_key = f"ip:{client_ip}"
        is_limited, retry_after = self.rate_limiter.is_rate_limited(
            key=ip_key,
            max_requests=self.ip_rate_limit[0],
            window_seconds=self.ip_rate_limit[1]
        )
        
        if is_limited:
            return self._rate_limited_response(retry_after)
        
        # Apply user-based rate limit if authenticated
        try:
            user_id = await get_current_user_id(request)
            if user_id:
                # Determine user role (simplified - in a real implementation,
                # this would check the user's role from the database)
                user_role = "default"  # Placeholder
                
                # Get rate limit for this role
                role_rate_limit = self.user_rate_limit.get(
                    user_role, 
                    self.user_rate_limit["default"]
                )
                
                # Apply user rate limit
                user_key = f"user:{user_id}"
                is_limited, retry_after = self.rate_limiter.is_rate_limited(
                    key=user_key,
                    max_requests=role_rate_limit[0],
                    window_seconds=role_rate_limit[1]
                )
                
                if is_limited:
                    return self._rate_limited_response(retry_after)
        except Exception as e:
            # Log the error but don't block the request
            logger.error(f"Error checking user rate limit: {str(e)}")
        
        # Apply endpoint-specific rate limit
        path = request.url.path
        method = request.method
        endpoint_key = self._get_endpoint_key(path, method)
        
        # Check if this endpoint has specific rate limits
        if endpoint_key in self.endpoint_rate_limits:
            endpoint_rate_limit = self.endpoint_rate_limits[endpoint_key]
            
            # Apply endpoint rate limit
            combined_key = f"endpoint:{endpoint_key}"
            is_limited, retry_after = self.rate_limiter.is_rate_limited(
                key=combined_key,
                max_requests=endpoint_rate_limit[0],
                window_seconds=endpoint_rate_limit[1]
            )
            
            if is_limited:
                return self._rate_limited_response(retry_after)
        
        # If all rate limits pass, proceed with the request
        response = await call_next(request)
        return response
    
    def _rate_limited_response(self, retry_after: int) -> Response:
        """
        Create a rate limited response
        
        Args:
            retry_after: Seconds until the client can retry
            
        Returns:
            JSON response with rate limit error
        """
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Rate limit exceeded, please try again later",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )
