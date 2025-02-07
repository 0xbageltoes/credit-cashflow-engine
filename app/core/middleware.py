from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.cache import RateLimiter
from typing import Optional
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for certain paths (e.g., health checks)
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Get user ID from request (adjust based on your auth implementation)
        user_id = request.headers.get("user-id") or request.client.host
        
        # Check rate limit
        if self.rate_limiter.is_rate_limited(user_id):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add rate limit headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
