from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.core.cache import RateLimiter
from app.core.config import settings
from typing import Optional, Callable, Dict, Any, Tuple
import time
import uuid
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce API rate limits based on IP or user ID"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # Skip rate limiting for certain paths (e.g., health checks)
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get user ID from request (adjust based on your auth implementation)
        user_id = self._get_user_identifier(request)
        
        # Check rate limit
        remaining, reset_time = self.rate_limiter.check_rate_limit(user_id)
        if remaining <= 0:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return self._create_rate_limit_response(reset_time)
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add rate limit headers
        response.headers["X-Rate-Limit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-Rate-Limit-Remaining"] = str(remaining)
        response.headers["X-Rate-Limit-Reset"] = str(reset_time)
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        return response
    
    def _get_user_identifier(self, request: Request) -> str:
        """Extract user identifier from request (JWT token, API key, or IP address)"""
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
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the original client IP from proxy
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"
    
    def _create_rate_limit_response(self, reset_time: int) -> Any:
        """Create a standardized rate limit exceeded response"""
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests",
                "message": "Rate limit exceeded. Please try again later.",
                "resetAt": reset_time
            },
            headers={
                "Retry-After": str(reset_time - int(time.time())),
                "X-Rate-Limit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                "X-Rate-Limit-Remaining": "0",
                "X-Rate-Limit-Reset": str(reset_time)
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
