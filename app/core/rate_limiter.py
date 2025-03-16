"""
Rate Limiter Module for Production API Protection

This module provides a Redis-based rate limiting implementation to protect 
the API from excessive requests and ensure fair usage across clients.
"""
import time
import logging
import asyncio
from typing import Optional, Dict, Any
from redis.exceptions import RedisError

from app.core.cache_service import CacheService

# Set up logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Production-ready rate limiter implementation with Redis backend.
    Uses a sliding window algorithm for accurate rate limiting.
    """
    
    def __init__(
        self, 
        max_requests: int, 
        time_window: int, 
        redis_prefix: str = "rate_limit",
        burst_multiplier: float = 1.5
    ):
        """
        Initialize rate limiter with configurable settings
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            redis_prefix: Prefix for Redis keys
            burst_multiplier: Multiplier for burst capacity (temporary exceeding of limit)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.redis_prefix = redis_prefix
        self.burst_capacity = int(max_requests * burst_multiplier)
        self.cache_service = CacheService()
    
    async def get_redis_client(self):
        """Get Redis client with connection error handling"""
        if not hasattr(self, '_redis_client'):
            try:
                self._redis_client = self.cache_service._redis_async
            except Exception as e:
                logger.error(f"Failed to get Redis client: {str(e)}", exc_info=True)
                raise RedisError(f"Redis connection failed: {str(e)}")
        return self._redis_client
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if the request should be rate limited
        
        Args:
            identifier: Unique identifier for the client (e.g., user_id, IP)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Graceful degradation if Redis is unavailable - always allow requests
        redis_client = await self.get_redis_client()
        if redis_client is None:
            logger.warning("Rate limiter Redis unavailable - allowing request by default")
            return True
        
        # Construct key with prefix for organization
        key = f"{self.redis_prefix}:{identifier}"
        current_time = int(time.time())
        min_time = current_time - self.time_window
        
        try:
            pipe = redis_client.pipeline()
            
            # Add current request timestamp
            pipe.zadd(key, {str(current_time): current_time})
            
            # Remove outdated timestamps
            pipe.zremrangebyscore(key, 0, min_time)
            
            # Count requests in window
            pipe.zcard(key)
            
            # Set key expiration
            pipe.expire(key, self.time_window * 2)
            
            # Execute pipeline
            _, _, request_count, _ = await pipe.execute()
            
            # Check if rate limit is exceeded
            if request_count > self.max_requests:
                # Check for burst capacity - allow temporary exceeding of limit
                if request_count <= self.burst_capacity:
                    logger.info(
                        f"Rate limit soft exceeded for {identifier}: {request_count}/{self.max_requests} "
                        f"(within burst capacity {self.burst_capacity})"
                    )
                    return True
                
                logger.warning(
                    f"Rate limit exceeded for {identifier}: {request_count}/{self.max_requests}"
                )
                return False
            
            return True
        
        except RedisError as e:
            # Graceful degradation on Redis errors - allow request
            logger.error(f"Redis error in rate limiter: {str(e)} - allowing request by default")
            return True
        except Exception as e:
            logger.exception(f"Unexpected error in rate limiter: {str(e)} - allowing request by default")
            return True
    
    async def get_remaining_quota(self, identifier: str) -> Dict[str, Any]:
        """
        Get remaining request quota information
        
        Args:
            identifier: Unique identifier for the client
            
        Returns:
            Dict with quota information: remaining, limit, reset_at
        """
        # Graceful degradation if Redis is unavailable
        redis_client = await self.get_redis_client()
        if redis_client is None:
            return {
                "remaining": self.max_requests,
                "limit": self.max_requests,
                "reset_at": int(time.time()) + self.time_window
            }
        
        key = f"{self.redis_prefix}:{identifier}"
        current_time = int(time.time())
        min_time = current_time - self.time_window
        
        try:
            # Remove outdated entries and count current
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, min_time)
            pipe.zcard(key)
            _, request_count = await pipe.execute()
            
            # Calculate remaining and reset time
            remaining = max(0, self.max_requests - request_count)
            reset_at = current_time + self.time_window
            
            return {
                "remaining": remaining,
                "limit": self.max_requests,
                "reset_at": reset_at,
                "current_usage": request_count
            }
        except Exception as e:
            logger.error(f"Error getting rate limit quota: {str(e)}")
            # Return default values on error
            return {
                "remaining": self.max_requests,
                "limit": self.max_requests,
                "reset_at": int(time.time()) + self.time_window,
                "error": str(e)
            }
    
    async def reset_limit(self, identifier: str) -> bool:
        """
        Reset rate limit for an identifier
        
        Args:
            identifier: Unique identifier to reset
            
        Returns:
            True if reset successful, False otherwise
        """
        redis_client = await self.get_redis_client()
        if redis_client is None:
            return False
            
        key = f"{self.redis_prefix}:{identifier}"
        try:
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit: {str(e)}")
            return False
