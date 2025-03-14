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
from typing import Any, Dict, List, Optional, Union, Set
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

    async def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List of matching keys
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, returning empty list for keys")
            return []
        
        try:
            # Use scan_iter for better performance with large datasets
            keys: Set[str] = set()
            async for key in self._async_client.scan_iter(match=pattern, count=1000):
                keys.add(key)
            
            return list(keys)
        
        except RedisError as e:
            logger.error(f"Redis error getting keys with pattern {pattern}: {str(e)}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error getting keys with pattern {pattern}: {str(e)}")
            return []
    
    def keys_sync(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern synchronously
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List of matching keys
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, returning empty list for keys")
            return []
        
        try:
            # Use scan_iter for better performance with large datasets
            keys: Set[str] = set()
            for key in self._sync_client.scan_iter(match=pattern, count=1000):
                keys.add(key)
            
            return list(keys)
        
        except RedisError as e:
            logger.error(f"Redis error getting keys with pattern {pattern}: {str(e)}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error getting keys with pattern {pattern}: {str(e)}")
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching a pattern
        
        Args:
            pattern: The pattern to match
            
        Returns:
            Number of keys deleted
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping delete pattern operation")
            return 0
        
        try:
            # Get all keys matching the pattern
            keys = await self.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete the keys
            count = await self._async_client.delete(*keys)
            return count
        
        except RedisError as e:
            logger.error(f"Redis error deleting keys with pattern {pattern}: {str(e)}")
            return 0
        
        except Exception as e:
            logger.error(f"Unexpected error deleting keys with pattern {pattern}: {str(e)}")
            return 0
    
    def delete_pattern_sync(self, pattern: str) -> int:
        """
        Delete keys matching a pattern synchronously
        
        Args:
            pattern: The pattern to match
            
        Returns:
            Number of keys deleted
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, skipping delete pattern operation")
            return 0
        
        try:
            # Get all keys matching the pattern
            keys = self.keys_sync(pattern)
            
            if not keys:
                return 0
            
            # Delete the keys
            count = self._sync_client.delete(*keys)
            return count
        
        except RedisError as e:
            logger.error(f"Redis error deleting keys with pattern {pattern}: {str(e)}")
            return 0
        
        except Exception as e:
            logger.error(f"Unexpected error deleting keys with pattern {pattern}: {str(e)}")
            return 0
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get the TTL of a key
        
        Args:
            key: The key to get TTL for
            
        Returns:
            TTL in seconds, None if key does not exist or has no TTL
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, returning None for TTL")
            return None
        
        try:
            ttl = await self._async_client.ttl(key)
            return ttl if ttl > 0 else None
        
        except RedisError as e:
            logger.error(f"Redis error getting TTL for key {key}: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting TTL for key {key}: {str(e)}")
            return None
    
    def get_ttl_sync(self, key: str) -> Optional[int]:
        """
        Get the TTL of a key synchronously
        
        Args:
            key: The key to get TTL for
            
        Returns:
            TTL in seconds, None if key does not exist or has no TTL
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, returning None for TTL")
            return None
        
        try:
            ttl = self._sync_client.ttl(key)
            return ttl if ttl > 0 else None
        
        except RedisError as e:
            logger.error(f"Redis error getting TTL for key {key}: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting TTL for key {key}: {str(e)}")
            return None
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """
        Increment a key by a given amount
        
        Args:
            key: The key to increment
            amount: The amount to increment by
            ttl: Optional TTL to set after incrementing
            
        Returns:
            The new value after incrementing, None if failed
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping increment operation")
            return None
        
        try:
            # Increment the key
            result = await self._async_client.incrby(key, amount)
            
            # Set TTL if provided
            if ttl is not None:
                await self.expire(key, ttl)
            
            return result
        
        except RedisError as e:
            logger.error(f"Redis error incrementing key {key}: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error incrementing key {key}: {str(e)}")
            return None
    
    def increment_sync(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """
        Increment a key by a given amount synchronously
        
        Args:
            key: The key to increment
            amount: The amount to increment by
            ttl: Optional TTL to set after incrementing
            
        Returns:
            The new value after incrementing, None if failed
        """
        if self._connection_error or self._sync_client is None:
            logger.warning("Redis connection unavailable, skipping increment operation")
            return None
        
        try:
            # Increment the key
            result = self._sync_client.incrby(key, amount)
            
            # Set TTL if provided
            if ttl is not None:
                self.expire_sync(key, ttl)
            
            return result
        
        except RedisError as e:
            logger.error(f"Redis error incrementing key {key}: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error incrementing key {key}: {str(e)}")
            return None
    
    async def invalidate_user_tokens(self, user_id: str) -> int:
        """
        Invalidate all tokens for a specific user
        
        This is critical for security operations like logout, 
        password change, or account compromise detection.
        
        Args:
            user_id: The user ID to invalidate tokens for
            
        Returns:
            Number of tokens invalidated
        """
        if self._connection_error or self._async_client is None:
            logger.warning("Redis connection unavailable, skipping token invalidation")
            return 0
        
        try:
            # Get all tokens for this user
            token_pattern = f"jwt_payload:*"
            
            # Get all token keys
            token_keys = await self.keys(token_pattern)
            
            # Check each token payload for the user ID
            invalidated_count = 0
            for key in token_keys:
                try:
                    token_data = await self.get(key)
                    if token_data:
                        try:
                            payload = json.loads(token_data)
                            if payload.get("sub") == user_id:
                                # Delete this token
                                if await self.delete(key):
                                    invalidated_count += 1
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            logger.warning(f"Invalid JSON in token payload: {key}")
                except Exception as e:
                    logger.error(f"Error processing token key {key}: {str(e)}")
            
            return invalidated_count
        
        except Exception as e:
            logger.error(f"Error invalidating tokens for user {user_id}: {str(e)}")
            return 0
    
    async def initialize_from_config(self):
        """Initialize Redis client with configuration from settings"""
        if not settings.REDIS_ENABLED:
            logger.info("Redis is disabled, skipping initialization")
            self._connection_error = True
            return
        
        try:
            # Get connection parameters from settings
            params = settings.REDIS_CONNECTION_PARAMS
            
            # Initialize async client
            self._async_client = redis.asyncio.Redis(**params)
            
            # Initialize sync client
            self._sync_client = redis.Redis(**params)
            
            # Run health check
            if await self.health_check():
                self._connection_error = False
                logger.info("Redis clients initialized successfully from config")
            else:
                self._connection_error = True
                logger.error("Redis health check failed during initialization from config")
        
        except Exception as e:
            self._connection_error = True
            logger.error(f"Error initializing Redis clients from config: {str(e)}")
