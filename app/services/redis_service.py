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
import os
from pathlib import Path

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
        # Load .env file manually if needed
        self._ensure_upstash_credentials()
        
        # If no config provided, create one that prioritizes Upstash
        if config is None:
            config = self._create_upstash_config()
            
        super().__init__(config)
        logger.info("Redis Service initialized with %s", 
                   "in-memory fallback only" if not REDIS_AVAILABLE else "Redis connection")
    
    def _ensure_upstash_credentials(self) -> None:
        """Ensure Upstash Redis credentials are loaded from .env file"""
        # Check if environment variables are already set
        upstash_host = os.environ.get("UPSTASH_REDIS_HOST")
        upstash_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
        
        # If either is missing, try to load from .env file
        if not upstash_host or not upstash_password:
            logger.info("Upstash Redis credentials missing, attempting to load from .env file")
            # Find the .env file (project root)
            current_dir = Path(__file__).resolve()
            project_root = current_dir.parent.parent.parent
            env_file = project_root / '.env'
            
            if env_file.exists():
                logger.info(f"Found .env file at {env_file}")
                try:
                    # Read .env file directly
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                                
                            if '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # Set Upstash environment variables
                                if key == "UPSTASH_REDIS_HOST" and not upstash_host:
                                    os.environ["UPSTASH_REDIS_HOST"] = value
                                    logger.info(f"Set UPSTASH_REDIS_HOST={value}")
                                    upstash_host = value
                                
                                elif key == "UPSTASH_REDIS_PASSWORD" and not upstash_password:
                                    os.environ["UPSTASH_REDIS_PASSWORD"] = value
                                    logger.info(f"Set UPSTASH_REDIS_PASSWORD=***")
                                    upstash_password = value
                                
                                elif key == "UPSTASH_REDIS_PORT" and not os.environ.get("UPSTASH_REDIS_PORT"):
                                    os.environ["UPSTASH_REDIS_PORT"] = value
                                    logger.info(f"Set UPSTASH_REDIS_PORT={value}")
                                
                                elif key == "REDIS_URL" and not os.environ.get("REDIS_URL"):
                                    os.environ["REDIS_URL"] = value
                                    masked_url = value.replace(upstash_password, "***") if upstash_password else "***"
                                    logger.info(f"Set REDIS_URL={masked_url}")
                except Exception as e:
                    logger.error(f"Error loading environment variables from .env: {e}")
            else:
                logger.warning(f".env file not found at {env_file}")
        
        # Log the result
        logger.info(f"Checking Upstash Redis credentials: host={'present' if upstash_host else 'missing'}, "
                   f"password={'present' if upstash_password else 'missing'}")
        
        if not upstash_host:
            logger.warning("UPSTASH_REDIS_HOST environment variable is missing")
        if not upstash_password:
            logger.warning("UPSTASH_REDIS_PASSWORD environment variable is missing")
    
    def _create_upstash_config(self) -> RedisConfig:
        """Create a RedisConfig that prioritizes Upstash Redis if available"""
        upstash_host = os.environ.get("UPSTASH_REDIS_HOST")
        upstash_port = os.environ.get("UPSTASH_REDIS_PORT", "6379")
        upstash_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
        redis_url = os.environ.get("REDIS_URL")
        
        # If we have Upstash credentials, use them
        if upstash_host and upstash_password:
            logger.info(f"Creating Redis config with Upstash credentials: {upstash_host}:{upstash_port}")
            
            # Construct Redis URL for Upstash if not provided
            if not redis_url:
                redis_url = f"rediss://default:{upstash_password}@{upstash_host}:{upstash_port}"
                logger.info(f"Created Redis URL for Upstash: rediss://default:***@{upstash_host}:{upstash_port}")
            
            return RedisConfig(
                host=upstash_host,
                port=int(upstash_port),
                password=upstash_password,
                ssl=True,
                ssl_cert_reqs="required",
                url=redis_url,
                socket_connect_timeout=5.0,
                socket_timeout=5.0,
                retry_on_timeout=True,
                max_connections=10
            )
        
        # If we have a Redis URL but no Upstash credentials
        elif redis_url:
            logger.info("Creating Redis config with REDIS_URL")
            # Mask password for logging
            masked_url = redis_url
            if "@" in redis_url:
                pre, post = redis_url.split("@", 1)
                masked_url = f"{pre.split(':')[0]}:***@{post}"
            logger.info(f"Using Redis URL: {masked_url}")
            
            return RedisConfig(
                url=redis_url,
                ssl="rediss://" in redis_url,
                ssl_cert_reqs="required" if "rediss://" in redis_url else None,
                socket_connect_timeout=5.0,
                socket_timeout=5.0,
                retry_on_timeout=True,
                max_connections=10
            )
        
        # Fallback to localhost config
        else:
            env = os.environ.get("ENV", "development").lower()
            if env in ["prod", "production"]:
                logger.warning("Production environment detected but no Redis credentials available!")
            
            logger.info("No Redis config provided, using defaults")
            return RedisConfig(
                host="localhost",
                port=6379,
                password=None,
                ssl=False,
                max_connections=10
            )
        
    # Add any application-specific methods here
    # All core Redis functionality is inherited from UnifiedRedisService