"""
Redis Service for Caching and Data Storage

This module provides a Redis service for caching and data storage with both
asynchronous and synchronous methods. It includes comprehensive error handling,
connection pooling, and retry mechanisms for production use.

The service is designed to be resilient with graceful fallbacks when Redis is unavailable.
"""
import logging
import json
import time
from typing import Any, Dict, List, Optional, Union
import redis
import redis.asyncio
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisConfig:
    """Configuration options for Redis connections"""
    def __init__(
        self,
        host: str = None,
        port: int = None,
        password: str = None,
        db: int = 0,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        max_connections: int = 10,
        decode_responses: bool = True
    ):
        """
        Initialize Redis configuration
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
            max_connections: Maximum number of connections in the pool
            decode_responses: Whether to decode responses as strings
        """
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.password = password or settings.REDIS_PASSWORD
        self.db = db
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.max_connections = max_connections
        self.decode_responses = decode_responses

class RedisService:
    """
    Redis service for caching and data storage
    
    This service provides methods for interacting with Redis, with both
    asynchronous and synchronous methods for flexibility in different contexts.
    """
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize the Redis service
        
        Args:
            config: Redis configuration options
        """
        self.config = config or RedisConfig()
        self._async_client = None
        self._sync_client = None
        self._connection_error = False
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Redis clients"""
        try:
            # Initialize async client
            self._async_client = redis.asyncio.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses
            )
            
            # Initialize sync client
            self._sync_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses
            )
            
            self._connection_error = False
            logger.info("Redis clients initialized successfully")
        
        except Exception as e:
            self._connection_error = True
            logger.error(f"Error initializing Redis clients: {str(e)}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis
        
        Args:
            key: The key to get
            default: Default value if key not found
            
        Returns:
            The value, or default if not found
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, returning default value")
            return default
        
        try:
            value = await self._async_client.get(key)
            return value if value is not None else default
        
        except RedisError as e:
            logger.error(f"Redis error getting key {key}: {str(e)}")
            return default
        
        except Exception as e:
            logger.error(f"Unexpected error getting key {key}: {str(e)}")
            return default
    
    def get_sync(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis synchronously
        
        Args:
            key: The key to get
            default: Default value if key not found
            
        Returns:
            The value, or default if not found
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, returning default value")
            return default
        
        try:
            value = self._sync_client.get(key)
            return value if value is not None else default
        
        except RedisError as e:
            logger.error(f"Redis error getting key {key}: {str(e)}")
            return default
        
        except Exception as e:
            logger.error(f"Unexpected error getting key {key}: {str(e)}")
            return default
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in Redis
        
        Args:
            key: The key to set
            value: The value to set
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping set operation")
            return False
        
        try:
            if ttl is not None:
                result = await self._async_client.setex(key, ttl, value)
            else:
                result = await self._async_client.set(key, value)
            
            return result
        
        except RedisError as e:
            logger.error(f"Redis error setting key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting key {key}: {str(e)}")
            return False
    
    def set_sync(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in Redis synchronously
        
        Args:
            key: The key to set
            value: The value to set
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, skipping set operation")
            return False
        
        try:
            if ttl is not None:
                result = self._sync_client.setex(key, ttl, value)
            else:
                result = self._sync_client.set(key, value)
            
            return result
        
        except RedisError as e:
            logger.error(f"Redis error setting key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping delete operation")
            return False
        
        try:
            result = await self._async_client.delete(key)
            return result > 0
        
        except RedisError as e:
            logger.error(f"Redis error deleting key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting key {key}: {str(e)}")
            return False
    
    def delete_sync(self, key: str) -> bool:
        """
        Delete a key from Redis synchronously
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, skipping delete operation")
            return False
        
        try:
            result = self._sync_client.delete(key)
            return result > 0
        
        except RedisError as e:
            logger.error(f"Redis error deleting key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis
        
        Args:
            key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, assuming key does not exist")
            return False
        
        try:
            result = await self._async_client.exists(key)
            return result > 0
        
        except RedisError as e:
            logger.error(f"Redis error checking key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error checking key {key}: {str(e)}")
            return False
    
    def exists_sync(self, key: str) -> bool:
        """
        Check if a key exists in Redis synchronously
        
        Args:
            key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, assuming key does not exist")
            return False
        
        try:
            result = self._sync_client.exists(key)
            return result > 0
        
        except RedisError as e:
            logger.error(f"Redis error checking key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error checking key {key}: {str(e)}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key
        
        Args:
            key: The key to set TTL for
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping expire operation")
            return False
        
        try:
            result = await self._async_client.expire(key, ttl)
            return result
        
        except RedisError as e:
            logger.error(f"Redis error setting TTL for key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting TTL for key {key}: {str(e)}")
            return False
    
    def expire_sync(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key synchronously
        
        Args:
            key: The key to set TTL for
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, skipping expire operation")
            return False
        
        try:
            result = self._sync_client.expire(key, ttl)
            return result
        
        except RedisError as e:
            logger.error(f"Redis error setting TTL for key {key}: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting TTL for key {key}: {str(e)}")
            return False
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the Redis connection
        
        Returns:
            True if healthy, False otherwise
        """
        if self._async_client is None:
            self._initialize_clients()
            if self._async_client is None:
                return False
        
        try:
            return await self._async_client.ping()
        
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            self._connection_error = True
            return False
    
    def health_check_sync(self) -> bool:
        """
        Perform a health check on the Redis connection synchronously
        
        Returns:
            True if healthy, False otherwise
        """
        if self._sync_client is None:
            self._initialize_clients()
            if self._sync_client is None:
                return False
        
        try:
            return self._sync_client.ping()
        
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            self._connection_error = True
            return False
    
    async def close(self):
        """Close the Redis connection"""
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis async connection: {str(e)}")
        
        if self._sync_client is not None:
            try:
                self._sync_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis sync connection: {str(e)}")

    def __del__(self):
        """Ensure connections are closed when the object is garbage collected"""
        import asyncio
        
        if self._async_client is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._async_client.close())
                else:
                    loop.run_until_complete(self._async_client.close())
            except Exception:
                pass
        
        if self._sync_client is not None:
            try:
                self._sync_client.close()
            except Exception:
                pass
