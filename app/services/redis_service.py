"""
Redis Service

This module provides a production-ready Redis service implementation
that builds upon the UnifiedRedisService for robust Redis operations.

This service is designed to be used throughout the application when
Redis functionality is needed, ensuring consistent behavior and
proper error handling.
"""

from typing import Any, Dict, List, Optional, Union, Set, Tuple, TypeVar
import logging

from app.services.unified_redis_service import (
    UnifiedRedisService, 
    RedisConfig,
    REDIS_AVAILABLE
)

# Configure logging
logger = logging.getLogger("redis_service")


class RedisService(UnifiedRedisService):
    """
    Redis Service for caching and data storage
    
    This service extends the UnifiedRedisService to provide application-specific
    Redis operations and configuration. It ensures proper error handling,
    connection management, and graceful fallbacks.
    
    Use this service for all Redis operations in the application.
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis Service with production-ready defaults
        
        Args:
            config: Optional Redis configuration, uses environment variables by default
        """
        super().__init__(config)
        logger.info("Redis Service initialized with %s", 
                   "in-memory fallback only" if not REDIS_AVAILABLE else "Redis connection")
        
    # Add any application-specific methods here
    # All core Redis functionality is inherited from UnifiedRedisService