from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.core.cache_service import CacheService
from app.core.config import settings
from app.core.rate_limiting import limiter as custom_limiter
from typing import Optional, Callable, Dict, Any, Tuple
import time
import uuid
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce API rate limits based on IP or user ID
    
    This middleware provides global rate limiting for all API endpoints
    as a protective measure against abuse. It works alongside the
    decorator-based rate limiting on individual endpoints.
    
    Features:
    - IP-based and token-based rate limiting
    - Configurable limits via settings
    - Standardized rate limit response headers
    - Failsafe handling of rate limit storage failures
    - Proper RFC 6585 response for rate limit exceeded
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Use Redis or in-memory cache for rate limiting with safe fallbacks
        self.cache_service = CacheService()
        
        # Get rate limit settings with safe defaults for production
        self.max_requests = int(getattr(settings, "RATE_LIMIT_REQUESTS", 1000))
        
        # Check for either RATE_LIMIT_WINDOW or RATE_LIMIT_PERIOD with safe default
        # Ensure we always have an integer value for time window
        time_window_value = getattr(settings, "RATE_LIMIT_PERIOD", 
                           getattr(settings, "RATE_LIMIT_WINDOW", 60))
        
        # Handle string time units like "hour", "minute", "day", "second"
        if isinstance(time_window_value, str):
            time_window_value = time_window_value.lower().strip()
            if time_window_value == "second":
                self.time_window = 1
            elif time_window_value == "minute":
                self.time_window = 60
            elif time_window_value == "hour":
                self.time_window = 3600
            elif time_window_value == "day":
                self.time_window = 86400
            elif time_window_value.isdigit():
                self.time_window = int(time_window_value)
            else:
                logger.warning(f"Invalid time window value: {time_window_value}, using default of 60 seconds")
                self.time_window = 60
        else:
            # If it's already a number, convert it directly
            self.time_window = int(time_window_value)
        
        # Initialize the custom rate limiter during middleware initialization
        env = getattr(settings, "ENVIRONMENT", "development")
        logger.info(f"Rate limiter initialized for environment: {env}")
        logger.info(f"Default rate limit set to: {self.max_requests}/{self.time_window}seconds")
        
        # List of paths to exclude from rate limiting
        self.excluded_paths = [
            "/health", 
            "/metrics", 
            "/docs", 
            "/redoc", 
            "/openapi.json",
            "/api/v1/cashflow/health"  # Exclude cashflow health endpoint
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # Skip rate limiting for certain paths (e.g., health checks, docs)
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Get user ID from request (adjust based on your auth implementation)
        user_id = self._get_user_identifier(request)
        
        # Add a unique request ID for tracking across the system
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Check rate limit
        now = int(time.time())
        rate_key = f"ratelimit:{user_id}"
        
        # Get current count
        try:
            # Try to increment the counter
            pipeline = self.cache_service._redis_async.pipeline()
            pipeline.zremrangebyscore(rate_key, 0, now - self.time_window)
            pipeline.zadd(rate_key, {str(now): now})
            pipeline.zcard(rate_key)
            pipeline.expire(rate_key, self.time_window)
            results = await pipeline.execute()
            current_count = results[2]
            
            # Calculate remaining requests and reset time
            remaining = max(0, self.max_requests - current_count)
            reset_time = now + self.time_window
            
            if current_count > self.max_requests:
                logger.warning(
                    f"Rate limit exceeded",
                    extra={
                        "user_id": user_id,
                        "request_id": request_id,
                        "path": request.url.path, 
                        "method": request.method,
                        "client_ip": request.client.host
                    }
                )
                return self._create_rate_limit_response(reset_time, request_id)
        except Exception as e:
            # Log error but continue (fail open in case of Redis issues)
            logger.error(
                f"Rate limit checking error: {str(e)}", 
                extra={
                    "user_id": user_id,
                    "request_id": request_id,
                    "error": str(e)
                },
                exc_info=True
            )
            # In production, we gracefully continue without rate limiting
            remaining = 1
            reset_time = now + self.time_window
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time for monitoring
            process_time = time.time() - request.state.start_time
            
            # Add rate limit headers
            response.headers["X-Rate-Limit-Limit"] = str(self.max_requests)
            response.headers["X-Rate-Limit-Remaining"] = str(remaining)
            response.headers["X-Rate-Limit-Reset"] = str(reset_time)
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            response.headers["X-Request-ID"] = request_id
            
            # Log successful requests for monitoring
            if settings.ENVIRONMENT == "production" and process_time > 1.0:
                # Only log slow requests in production to reduce log volume
                logger.warning(
                    f"Slow request completed in {process_time:.4f}s",
                    extra={
                        "request_id": request_id,
                        "path": request.url.path,
                        "method": request.method,
                        "process_time": process_time,
                        "status_code": response.status_code
                    }
                )
            
            return response
            
        except Exception as e:
            # Log any exception during processing
            logger.error(
                f"Request processing error: {str(e)}",
                extra={
                    "user_id": user_id,
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "error": str(e)
                },
                exc_info=True
            )
            # Re-raise to let the exception handlers deal with it
            raise
    
    def _get_user_identifier(self, request: Request) -> str:
        """
        Extract user identifier from request (JWT token, API key, or IP address)
        
        This provides a consistent way to identify users for rate limiting purposes,
        with multiple fallback mechanisms.
        
        Returns:
            A unique string identifier for the user/client
        """
        # Try to get from authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Use a hash of the token as identifier
            import hashlib
            token = auth_header.replace("Bearer ", "")
            return f"token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        
        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # Try custom API client identifier
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return f"client:{client_id}"
        
        # Fall back to IP address with proxy support
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the original client IP from proxy
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        # Last resort - direct client IP
        return f"ip:{request.client.host}"
    
    def _create_rate_limit_response(self, reset_time: int, request_id: str) -> Any:
        """
        Create a standardized rate limit exceeded response following RFC 6585
        
        Args:
            reset_time: Time when the rate limit window resets
            request_id: Unique identifier for the request for tracking
            
        Returns:
            JSONResponse with appropriate status code and headers
        """
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "detail": f"Rate limit exceeded. Please try again after {reset_time - int(time.time())} seconds.",
                "type": "rate_limit_exceeded"
            },
            headers={
                "Retry-After": str(reset_time - int(time.time())),
                "X-Rate-Limit-Reset": str(reset_time),
                "X-Request-ID": request_id
            }
        )


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track and log request details, including performance metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        request_id = str(uuid.uuid4())
        request.state.id = request_id
        
        # Track request start time
        start_time = time.time()
        
        # Set correlation ID header if one doesn't exist
        correlation_id = request.headers.get("X-Correlation-ID", request_id)
        
        # Store request information
        request_info = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
            "start_time": datetime.now().isoformat(),
        }
        
        logger.info(f"Request started", extra=request_info)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Add tracking headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{duration:.4f}"
            
            # Log request completion
            request_info.update({
                "status_code": response.status_code,
                "duration": duration,
                "completed": True
            })
            
            logger.info(f"Request completed", extra=request_info)
            
            return response
            
        except Exception as e:
            # Calculate duration even for errors
            duration = time.time() - start_time
            
            # Log the error
            request_info.update({
                "completed": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": duration
            })
            
            logger.error(f"Request failed: {str(e)}", extra=request_info)
            raise
