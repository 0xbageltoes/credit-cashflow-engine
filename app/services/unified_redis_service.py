"""
Unified Redis Service

A comprehensive Redis service implementation that consolidates the best practices
from all existing Redis implementations. This service provides:

1. Robust connection handling with automatic reconnection
2. Comprehensive error handling with graceful fallbacks
3. Both synchronous and asynchronous interfaces
4. Efficient serialization/deserialization with compression
5. Performance monitoring via Prometheus metrics
6. Configurable caching strategies and TTL management

This service is designed for production use and follows best practices for
resilience, performance, and maintainability.
"""

import os
import json
import time
import zlib
import pickle
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Set, Tuple, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

# Redis imports
try:
    import redis
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.exceptions import (
        RedisError, ConnectionError, TimeoutError, 
        ResponseError, LockError, WatchError
    )
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Optional Prometheus metrics
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Histogram, Summary, Gauge
    
    # Define metrics
    REDIS_OPERATIONS = Counter(
        'redis_operations_total',
        'Total number of Redis operations',
        ['operation', 'status']
    )
    
    REDIS_LATENCY = Histogram(
        'redis_operation_duration_seconds',
        'Redis operation duration in seconds',
        ['operation']
    )
    
    REDIS_CACHE_HITS = Counter(
        'redis_cache_hits_total',
        'Total number of Redis cache hits',
        ['namespace']
    )
    
    REDIS_CACHE_MISSES = Counter(
        'redis_cache_misses_total',
        'Total number of Redis cache misses',
        ['namespace']
    )
    
    REDIS_CONNECTIONS = Gauge(
        'redis_connections',
        'Number of active Redis connections'
    )
    
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Configure logging
logger = logging.getLogger("redis_service")

# Type definitions
T = TypeVar('T')
CacheKey = str
CacheValue = Any
TtlSeconds = int


class RedisConfig:
    """
    Comprehensive configuration for Redis connections
    
    This class provides a complete configuration for Redis connections with
    sensible defaults for production environments.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        db: int = 0,
        ssl: bool = False,
        ssl_cert_reqs: Optional[str] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        socket_keepalive: bool = True,
        socket_keepalive_options: Optional[Dict[int, Union[int, bytes]]] = None,
        retry_on_timeout: bool = True,
        retry_on_error: List[Exception] = None,
        health_check_interval: int = 30,
        max_connections: int = 10,
        max_idle_time: int = 300,
        wait_for_connection: bool = True,
        decode_responses: bool = False,
        encoding: str = 'utf-8',
        encoding_errors: str = 'strict',
        compression_enabled: bool = True,
        compression_threshold: int = 1024,  # in bytes
        connection_attempts: int = 3,
        connection_attempt_delay: float = 1.0,
        namespace: str = ''
    ):
        """
        Initialize Redis configuration
        
        Args:
            url: Redis URL (overrides host, port, password if provided)
            host: Redis host
            port: Redis port
            password: Redis password
            username: Redis username (for Redis 6.0+)
            db: Redis database number
            ssl: Whether to use SSL
            ssl_cert_reqs: SSL certificate requirements mode
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            socket_keepalive: Whether to use socket keepalive
            socket_keepalive_options: Socket keepalive options
            retry_on_timeout: Whether to retry on timeout
            retry_on_error: List of exceptions to retry on
            health_check_interval: Health check interval in seconds
            max_connections: Maximum number of connections in the pool
            max_idle_time: Maximum time in seconds a connection can be idle before closing
            wait_for_connection: Whether to wait for a connection from the pool
            decode_responses: Whether to decode responses (strings vs bytes)
            encoding: Encoding to use for string operations
            encoding_errors: How to handle encoding errors
            compression_enabled: Whether to enable compression for large values
            compression_threshold: Minimum size in bytes for compression
            connection_attempts: Number of connection attempts before failing
            connection_attempt_delay: Delay between connection attempts in seconds
            namespace: Namespace prefix for all keys
        """
        # Priority order: 
        # 1. Explicit URL parameter
        # 2. REDIS_URL environment variable 
        # 3. Upstash Redis URL
        # 4. Individual connection parameters (host, port, etc.)
        self.url = url or os.environ.get("REDIS_URL") or None
        
        # If using Upstash and no URL provided, construct Upstash URL
        upstash_host = os.environ.get("UPSTASH_REDIS_HOST")
        upstash_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
        if not self.url and upstash_host and upstash_password:
            upstash_port = os.environ.get("UPSTASH_REDIS_PORT", "6379")
            self.url = f"rediss://default:{upstash_password}@{upstash_host}:{upstash_port}"
            logger.info("Using Upstash Redis configuration")
            
        # Only use these if URL is not provided
        self.host = host or os.environ.get("UPSTASH_REDIS_HOST") or os.environ.get("REDIS_HOST") or "localhost"
        self.port = port or int(os.environ.get("UPSTASH_REDIS_PORT") or os.environ.get("REDIS_PORT", "6379"))
        self.password = password or os.environ.get("UPSTASH_REDIS_PASSWORD") or os.environ.get("REDIS_PASSWORD") or None
        self.username = username or os.environ.get("REDIS_USERNAME") or "default"  # Upstash typically uses "default" username
        self.db = int(os.environ.get("REDIS_DB", str(db)))
        
        # Connection settings - enable SSL by default for Upstash
        is_upstash = upstash_host is not None or self.url and "upstash.io" in self.url
        self.ssl = ssl or is_upstash or (os.environ.get("REDIS_SSL", "").lower() == "true")
        self.ssl_cert_reqs = ssl_cert_reqs
        self.socket_timeout = float(os.environ.get("REDIS_SOCKET_TIMEOUT", str(socket_timeout)))
        self.socket_connect_timeout = float(os.environ.get("REDIS_CONNECT_TIMEOUT", str(socket_connect_timeout)))
        self.socket_keepalive = socket_keepalive
        self.socket_keepalive_options = socket_keepalive_options or {}
        
        # Retry settings
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_error = retry_on_error or [ConnectionError, TimeoutError]
        
        # Pool settings
        self.health_check_interval = health_check_interval
        self.max_connections = int(os.environ.get("REDIS_MAX_CONNECTIONS", str(max_connections)))
        self.max_idle_time = max_idle_time
        self.wait_for_connection = wait_for_connection
        
        # Response settings
        self.decode_responses = decode_responses
        self.encoding = encoding
        self.encoding_errors = encoding_errors
        
        # Compression settings
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        
        # Connection retry settings
        self.connection_attempts = connection_attempts
        self.connection_attempt_delay = connection_attempt_delay
        
        # Namespace settings
        self.namespace = namespace
    
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters for Redis client
        
        Returns:
            Dictionary of connection parameters
        """
        if self.url:
            # When using URL (including Upstash), these are the critical parameters
            params = {
                "url": self.url,
                "socket_timeout": self.socket_timeout,
                "socket_connect_timeout": self.socket_connect_timeout,
                "socket_keepalive": self.socket_keepalive,
                "socket_keepalive_options": self.socket_keepalive_options,
                "retry_on_timeout": self.retry_on_timeout,
                "health_check_interval": self.health_check_interval,
                "max_connections": self.max_connections,
                "decode_responses": self.decode_responses,
                "encoding": self.encoding,
                "encoding_errors": self.encoding_errors
            }
            
            # Set SSL parameters correctly for Upstash
            if "upstash.io" in self.url or self.url.startswith("rediss://"):
                params["ssl"] = True
                # Only set ssl_cert_reqs if explicitly provided, otherwise let redis-py use defaults
                if self.ssl_cert_reqs is not None:
                    params["ssl_cert_reqs"] = self.ssl_cert_reqs
                logger.debug("SSL enabled for Redis connection")
                
            return params
        else:
            params = {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "socket_timeout": self.socket_timeout,
                "socket_connect_timeout": self.socket_connect_timeout,
                "socket_keepalive": self.socket_keepalive,
                "socket_keepalive_options": self.socket_keepalive_options,
                "retry_on_timeout": self.retry_on_timeout,
                "health_check_interval": self.health_check_interval,
                "max_connections": self.max_connections,
                "decode_responses": self.decode_responses,
                "encoding": self.encoding,
                "encoding_errors": self.encoding_errors
            }
            
            if self.password:
                params["password"] = self.password
                
            if self.username:
                params["username"] = self.username
                
            if self.ssl:
                params["ssl"] = True
                if self.ssl_cert_reqs is not None:
                    params["ssl_cert_reqs"] = self.ssl_cert_reqs
                    
            return params
    
    def __str__(self) -> str:
        """String representation of Redis configuration (sanitized)"""
        host_info = f"{self.url.split('@')[-1]}" if self.url else f"{self.host}:{self.port}"
        ssl_info = "with SSL" if self.ssl else "without SSL"
        return f"RedisConfig({host_info}, {ssl_info})"


class RedisSerialization:
    """
    Serialization utilities for Redis data
    
    This class provides methods for serializing and deserializing data for Redis,
    with optional compression for large values.
    """
    
    @staticmethod
    def serialize(value: Any, config: RedisConfig) -> bytes:
        """
        Serialize a value for storage in Redis
        
        Args:
            value: Value to serialize
            config: Redis configuration
            
        Returns:
            Serialized value as bytes
        """
        try:
            # Convert to JSON if possible
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized = json.dumps(value).encode(config.encoding)
                
                # Add JSON type marker to first byte
                serialized = b'j' + serialized
            else:
                # Use pickle for complex objects (preserves types)
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Add pickle type marker to first byte
                serialized = b'p' + serialized
            
            # Compress if enabled and value is large enough
            if config.compression_enabled and len(serialized) >= config.compression_threshold:
                compressed = zlib.compress(serialized)
                
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    # Add compression marker to first byte
                    return b'c' + compressed
            
            return serialized
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            # Return the original value as a fallback
            if isinstance(value, bytes):
                return value
            # Convert to string as a last resort
            return str(value).encode(config.encoding)
    
    @staticmethod
    def deserialize(data: bytes, config: RedisConfig) -> Any:
        """
        Deserialize a value from Redis
        
        Args:
            data: Serialized value
            config: Redis configuration
            
        Returns:
            Deserialized value
        """
        try:
            if not data:
                return None
                
            # Check the type marker
            type_marker = data[0:1]
            data_bytes = data[1:]
            
            # Decompress if marked as compressed
            if type_marker == b'c':
                data_bytes = zlib.decompress(data_bytes)
                # Get the type marker from the decompressed data
                type_marker = data_bytes[0:1]
                data_bytes = data_bytes[1:]
            
            # Deserialize based on the type marker
            if type_marker == b'j':
                return json.loads(data_bytes.decode(config.encoding))
            elif type_marker == b'p':
                return pickle.loads(data_bytes)
            else:
                # Unknown type marker - try to decode as string
                return data_bytes.decode(config.encoding, errors=config.encoding_errors)
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            # Return as bytes as a fallback
            return data


class UnifiedRedisService:
    """
    Unified Redis Service for caching and data storage
    
    This service provides a comprehensive interface for Redis operations with:
    1. Both synchronous and asynchronous methods
    2. Robust error handling and connection management
    3. Automatic reconnection and fallback mechanisms
    4. Performance monitoring and metrics collection
    5. Comprehensive logging for troubleshooting
    
    Use this service as a dependency in other services that require Redis functionality.
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize the Redis service
        
        Args:
            config: Redis configuration options
        """
        self.config = config or RedisConfig()
        self._sync_client = None
        self._async_client = None
        self._clients_initialized = False
        self._in_memory_fallback = {}  # Fallback when Redis is unavailable
        
        if METRICS_ENABLED:
            # Register connect/disconnect callbacks for metrics
            self._register_metrics_hooks()
        
        # Initialize clients
        self._initialize_clients()
    
    def _register_metrics_hooks(self) -> None:
        """Register metrics hooks for connection events"""
        if METRICS_ENABLED:
            REDIS_CONNECTIONS.set(0)
    
    def _initialize_clients(self) -> None:
        """Initialize Redis clients with retry logic"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not available. Using in-memory fallback only.")
            return
            
        try:
            # Get connection parameters
            params = self.config.get_connection_params()
            attempts = 0
            
            # Log Redis connection details (without sensitive info)
            host_info = self.config.url.split('@')[-1] if self.config.url else f"{self.config.host}:{self.config.port}"
            ssl_info = "with SSL" if self.config.ssl else "without SSL"
            logger.info(f"Initializing Redis connection to {host_info} {ssl_info}")
            
            # Determine if we're using Upstash
            is_upstash = False
            if self.config.url and ("upstash.io" in self.config.url or self.config.url.startswith("rediss://")):
                is_upstash = True
                logger.info("Detected Upstash Redis configuration")
            
            connection_error = None
            while attempts < self.config.connection_attempts:
                try:
                    attempts += 1
                    
                    # Initialize sync client
                    if self._sync_client is None:
                        self._sync_client = Redis(**params)
                    
                    # Initialize async client with the same parameters
                    # Force decode_responses=False for the async client to handle binary data properly
                    if self._async_client is None:
                        async_params = params.copy()
                        async_params["decode_responses"] = False
                        self._async_client = AsyncRedis(**async_params)
                    
                    # Test connection
                    if self._sync_client.ping():
                        logger.info(f"Redis clients initialized successfully on attempt {attempts}")
                        self._clients_initialized = True
                        
                        # Get Redis server info for logging
                        try:
                            redis_info = self._sync_client.info()
                            redis_version = redis_info.get('redis_version', 'unknown')
                            logger.info(f"Connected to Redis server version: {redis_version}")
                            
                            # Check for Upstash-specific info
                            if is_upstash:
                                logger.info("Successfully connected to Upstash Redis")
                        except Exception as e:
                            logger.warning(f"Could not retrieve Redis server info: {e}")
                        
                        if METRICS_ENABLED:
                            REDIS_CONNECTIONS.set(1)
                            
                        return
                        
                except (ConnectionError, TimeoutError, ResponseError) as e:
                    connection_error = e
                    logger.warning(f"Redis connection attempt {attempts} failed: {e}")
                    
                    if attempts < self.config.connection_attempts:
                        retry_delay = self.config.connection_attempt_delay * (2 ** (attempts - 1))  # Exponential backoff
                        logger.info(f"Retrying in {retry_delay:.2f} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to connect to Redis after {attempts} attempts")
                        break
                except Exception as e:
                    connection_error = e
                    logger.error(f"Unexpected error during Redis connection: {e}")
                    break
            
            # If we reach here, all connection attempts failed
            if connection_error:
                logger.error(f"Could not connect to Redis: {connection_error}")
                
            # Clean up any partial connections
            self._cleanup_clients()
            
            # Set up in-memory fallback
            logger.warning("Redis connection failed. Using in-memory fallback for caching.")
            
        except Exception as e:
            logger.error(f"Error initializing Redis clients: {e}")
            self._cleanup_clients()
    
    def _cleanup_clients(self) -> None:
        """Clean up Redis clients safely"""
        try:
            if self._sync_client:
                try:
                    self._sync_client.close()
                except Exception as e:
                    logger.warning(f"Error closing sync Redis client: {e}")
                self._sync_client = None
                
            if self._async_client:
                # We can't await here, so we'll just set it to None and let GC handle it
                self._async_client = None
                
            self._clients_initialized = False
            
            if METRICS_ENABLED:
                REDIS_CONNECTIONS.set(0)
                
        except Exception as e:
            logger.error(f"Error during client cleanup: {e}")
            # Reset to known state
            self._sync_client = None
            self._async_client = None
            self._clients_initialized = False
    
    def _safe_key(self, key: str) -> str:
        """
        Ensure key has the proper namespace prefix
        
        Args:
            key: The key to normalize
            
        Returns:
            Properly namespaced key
        """
        # Add namespace prefix if configured and not already present
        if self.config.namespace and not key.startswith(f"{self.config.namespace}:"):
            return f"{self.config.namespace}:{key}"
        return key
    
    def _record_metrics(self, operation: str, success: bool) -> None:
        """
        Record metrics for a Redis operation
        
        Args:
            operation: The operation name
            success: Whether the operation succeeded
        """
        if METRICS_ENABLED:
            status = "success" if success else "failure"
            REDIS_OPERATIONS.labels(operation=operation, status=status).inc()
    
    @contextmanager
    def _operation_context(self, operation: str) -> None:
        """
        Context manager for timing operations and recording metrics
        
        Args:
            operation: Name of the operation
        """
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        finally:
            if METRICS_ENABLED:
                duration = time.time() - start_time
                REDIS_LATENCY.labels(operation=operation).observe(duration)
                self._record_metrics(operation, success)
    
    def _handle_fallback(self, key: str, operation: str, default: Any = None) -> Any:
        """
        Handle fallback when Redis is unavailable
        
        Args:
            key: The key being operated on
            operation: The operation being performed
            default: Default value to return
            
        Returns:
            Fallback value based on operation
        """
        logger.warning(f"Redis unavailable for {operation} operation on key {key}, using fallback")
        
        # For get operations, check the in-memory fallback
        if operation == "get" and key in self._in_memory_fallback:
            value, expiry = self._in_memory_fallback[key]
            
            # Check if the cached value has expired
            if expiry is None or expiry > time.time():
                return value
            
            # Remove expired value
            del self._in_memory_fallback[key]
        
        return default
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Redis connection
        
        Returns:
            Dictionary with health status information
        """
        health_data = {
            "status": "unavailable",
            "connected": False,
            "error": None,
            "details": {},
            "in_memory_keys": len(self._in_memory_fallback)
        }
        
        if not REDIS_AVAILABLE:
            health_data["error"] = "Redis package not installed"
            return health_data
            
        if not self._sync_client:
            health_data["error"] = "Redis client not initialized"
            return health_data
            
        try:
            with self._operation_context("health_check"):
                # Ping Redis
                if self._sync_client.ping():
                    health_data["status"] = "healthy"
                    health_data["connected"] = True
                    
                    # Get additional info if possible
                    try:
                        info = self._sync_client.info()
                        health_data["details"] = {
                            "version": info.get("redis_version", "unknown"),
                            "uptime_seconds": info.get("uptime_in_seconds", 0),
                            "connected_clients": info.get("connected_clients", 0),
                            "used_memory_human": info.get("used_memory_human", "unknown"),
                            "total_keys": sum(info.get(f"db{i}", {}).get("keys", 0) for i in range(16) if f"db{i}" in info)
                        }
                    except Exception as e:
                        logger.warning(f"Could not retrieve Redis info: {e}")
        except Exception as e:
            health_data["error"] = str(e)
            
        return health_data
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Redis
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Attempting to reconnect to Redis...")
            self._initialize_clients()
            if self._sync_client or self._async_client:
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reconnect to Redis: {e}")
            return False
    
    async def ping(self) -> bool:
        """
        Ping Redis to check connectivity
        
        Returns:
            bool: True if Redis is available and responsive, False otherwise
        """
        # Check if Redis clients are initialized
        if not self._async_client:
            logger.warning("Redis client not initialized, cannot ping")
            return False
            
        try:
            with self._operation_context("ping"):
                result = await self._async_client.ping()
                return result == b"PONG" or result is True
        except Exception as e:
            logger.error(f"Error pinging Redis: {e}")
            return False
    
    def ping_sync(self) -> bool:
        """
        Ping Redis synchronously to check connectivity
        
        Returns:
            bool: True if Redis is available and responsive, False otherwise
        """
        # Check if Redis clients are initialized
        if not self._sync_client:
            logger.warning("Redis client not initialized, cannot ping")
            return False
            
        try:
            with self._operation_context("ping_sync"):
                result = self._sync_client.ping()
                return result == b"PONG" or result is True
        except Exception as e:
            logger.error(f"Error pinging Redis synchronously: {e}")
            return False
    
    def get_sync(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis synchronously
        
        Args:
            key: The key to get
            default: Default value if key not found or Redis unavailable
            
        Returns:
            The value, or default if not found
        """
        key = self._safe_key(key)
        
        # Check if Redis is available
        if not self._sync_client:
            return self._handle_fallback(key, "get", default)
            
        try:
            with self._operation_context("get"):
                # Get value from Redis
                value = self._sync_client.get(key)
                
                # Record cache hit/miss
                if METRICS_ENABLED:
                    namespace = self.config.namespace or "default"
                    if value is not None:
                        REDIS_CACHE_HITS.labels(namespace=namespace).inc()
                    else:
                        REDIS_CACHE_MISSES.labels(namespace=namespace).inc()
                
                # Return None if not found
                if value is None:
                    return default
                
                # Deserialize the value
                return RedisSerialization.deserialize(value, self.config)
        except Exception as e:
            logger.error(f"Error getting value for key {key}: {e}")
            return self._handle_fallback(key, "get", default)
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in Redis synchronously
        
        Args:
            key: The key to set
            value: The value to set
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = self._safe_key(key)
        
        # Store in in-memory fallback
        if ttl is not None:
            expiry = time.time() + ttl
        else:
            expiry = None
            
        self._in_memory_fallback[key] = (value, expiry)
        
        # Check if Redis is available
        if not self._sync_client:
            logger.warning(f"Redis unavailable for set operation on key {key}")
            return False
            
        try:
            with self._operation_context("set"):
                # Serialize the value
                serialized = RedisSerialization.serialize(value, self.config)
                
                # Set in Redis with TTL if provided
                if ttl is not None:
                    result = self._sync_client.setex(key, ttl, serialized)
                else:
                    result = self._sync_client.set(key, serialized)
                
                return result is True
        except Exception as e:
            logger.error(f"Error setting value for key {key}: {e}")
            return False
    
    def delete_sync(self, key: str) -> bool:
        """
        Delete a key from Redis synchronously
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        key = self._safe_key(key)
        
        # Remove from in-memory fallback
        if key in self._in_memory_fallback:
            del self._in_memory_fallback[key]
        
        # Check if Redis is available
        if not self._sync_client:
            logger.warning(f"Redis unavailable for delete operation on key {key}")
            return False
            
        try:
            with self._operation_context("delete"):
                # Delete from Redis
                result = self._sync_client.delete(key)
                return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def exists_sync(self, key: str) -> bool:
        """
        Check if a key exists in Redis synchronously
        
        Args:
            key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        key = self._safe_key(key)
        
        # Check in-memory fallback
        if key in self._in_memory_fallback:
            value, expiry = self._in_memory_fallback[key]
            if expiry is None or expiry > time.time():
                return True
            
            # Remove expired value
            del self._in_memory_fallback[key]
            return False
        
        # Check if Redis is available
        if not self._sync_client:
            logger.warning(f"Redis unavailable for exists operation on key {key}")
            return False
            
        try:
            with self._operation_context("exists"):
                # Check if key exists in Redis
                return self._sync_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
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
        key = self._safe_key(key)
        
        # Update in-memory fallback
        if key in self._in_memory_fallback:
            value, _ = self._in_memory_fallback[key]
            self._in_memory_fallback[key] = (value, time.time() + ttl)
        
        # Check if Redis is available
        if not self._sync_client:
            logger.warning(f"Redis unavailable for expire operation on key {key}")
            return False
            
        try:
            with self._operation_context("expire"):
                # Set TTL in Redis
                return self._sync_client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Error setting TTL for key {key}: {e}")
            return False
    
    # Asynchronous Redis methods
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis asynchronously
        
        Args:
            key: The key to get
            default: Default value if key not found or Redis unavailable
            
        Returns:
            The value, or default if not found
        """
        key = self._safe_key(key)
        
        # Check if Redis is available
        if not self._async_client:
            return self._handle_fallback(key, "get", default)
            
        try:
            with self._operation_context("get_async"):
                # Get value from Redis
                value = await self._async_client.get(key)
                
                # Record cache hit/miss
                if METRICS_ENABLED:
                    namespace = self.config.namespace or "default"
                    if value is not None:
                        REDIS_CACHE_HITS.labels(namespace=namespace).inc()
                    else:
                        REDIS_CACHE_MISSES.labels(namespace=namespace).inc()
                
                # Return None if not found
                if value is None:
                    return default
                
                # Deserialize the value
                return RedisSerialization.deserialize(value, self.config)
        except Exception as e:
            logger.error(f"Error getting value for key {key} asynchronously: {e}")
            return self._handle_fallback(key, "get", default)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in Redis asynchronously
        
        Args:
            key: The key to set
            value: The value to set
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = self._safe_key(key)
        
        # Store in in-memory fallback
        if ttl is not None:
            expiry = time.time() + ttl
        else:
            expiry = None
            
        self._in_memory_fallback[key] = (value, expiry)
        
        # Check if Redis is available
        if not self._async_client:
            logger.warning(f"Redis unavailable for set operation on key {key}")
            return False
            
        try:
            with self._operation_context("set_async"):
                # Serialize the value
                serialized = RedisSerialization.serialize(value, self.config)
                
                # Set in Redis with TTL if provided
                if ttl is not None:
                    result = await self._async_client.setex(key, ttl, serialized)
                else:
                    result = await self._async_client.set(key, serialized)
                
                return result is True
        except Exception as e:
            logger.error(f"Error setting value for key {key} asynchronously: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis asynchronously
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        key = self._safe_key(key)
        
        # Remove from in-memory fallback
        if key in self._in_memory_fallback:
            del self._in_memory_fallback[key]
        
        # Check if Redis is available
        if not self._async_client:
            logger.warning(f"Redis unavailable for delete operation on key {key}")
            return False
            
        try:
            with self._operation_context("delete_async"):
                # Delete from Redis
                result = await self._async_client.delete(key)
                return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key} asynchronously: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis asynchronously
        
        Args:
            key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        key = self._safe_key(key)
        
        # Check in-memory fallback
        if key in self._in_memory_fallback:
            value, expiry = self._in_memory_fallback[key]
            if expiry is None or expiry > time.time():
                return True
            
            # Remove expired value
            del self._in_memory_fallback[key]
            return False
        
        # Check if Redis is available
        if not self._async_client:
            logger.warning(f"Redis unavailable for exists operation on key {key}")
            return False
            
        try:
            with self._operation_context("exists_async"):
                # Check if key exists in Redis
                return await self._async_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence of key {key} asynchronously: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key asynchronously
        
        Args:
            key: The key to set TTL for
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = self._safe_key(key)
        
        # Update in-memory fallback
        if key in self._in_memory_fallback:
            value, _ = self._in_memory_fallback[key]
            self._in_memory_fallback[key] = (value, time.time() + ttl)
        
        # Check if Redis is available
        if not self._async_client:
            logger.warning(f"Redis unavailable for expire operation on key {key}")
            return False
            
        try:
            with self._operation_context("expire_async"):
                # Set TTL in Redis
                return await self._async_client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Error setting TTL for key {key} asynchronously: {e}")
            return False
    
    # Additional utility methods
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching a pattern
        
        Args:
            pattern: Pattern to match keys (e.g., "user:*")
            
        Returns:
            Number of keys deleted
        """
        if pattern:
            pattern = self._safe_key(pattern)
        else:
            pattern = self._safe_key("*")
            
        # Clear in-memory fallback
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys_to_delete = [k for k in self._in_memory_fallback.keys() if k.startswith(prefix)]
        else:
            keys_to_delete = [k for k in self._in_memory_fallback.keys() if k == pattern]
            
        for k in keys_to_delete:
            del self._in_memory_fallback[k]
            
        in_memory_deleted = len(keys_to_delete)
        
        # Check if Redis is available
        if not self._sync_client:
            logger.warning(f"Redis unavailable for clear_cache operation with pattern {pattern}")
            return in_memory_deleted
            
        try:
            with self._operation_context("clear_cache"):
                # Get keys matching pattern
                keys = self._sync_client.keys(pattern)
                
                if not keys:
                    return in_memory_deleted
                    
                # Delete keys in batches
                pipeline = self._sync_client.pipeline()
                for key in keys:
                    pipeline.delete(key)
                    
                results = pipeline.execute()
                return in_memory_deleted + sum(results)
        except Exception as e:
            logger.error(f"Error clearing cache with pattern {pattern}: {e}")
            return in_memory_deleted
    
    async def clear_cache_async(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching a pattern asynchronously
        
        Args:
            pattern: Pattern to match keys (e.g., "user:*")
            
        Returns:
            Number of keys deleted
        """
        if pattern:
            pattern = self._safe_key(pattern)
        else:
            pattern = self._safe_key("*")
            
        # Clear in-memory fallback
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys_to_delete = [k for k in self._in_memory_fallback.keys() if k.startswith(prefix)]
        else:
            keys_to_delete = [k for k in self._in_memory_fallback.keys() if k == pattern]
            
        for k in keys_to_delete:
            del self._in_memory_fallback[k]
            
        in_memory_deleted = len(keys_to_delete)
        
        # Check if Redis is available
        if not self._async_client:
            logger.warning(f"Redis unavailable for clear_cache operation with pattern {pattern}")
            return in_memory_deleted
            
        try:
            with self._operation_context("clear_cache_async"):
                # Get keys matching pattern
                keys = await self._async_client.keys(pattern)
                
                if not keys:
                    return in_memory_deleted
                    
                # Delete keys in batches
                pipeline = self._async_client.pipeline()
                for key in keys:
                    pipeline.delete(key)
                    
                results = await pipeline.execute()
                return in_memory_deleted + sum(results)
        except Exception as e:
            logger.error(f"Error clearing cache with pattern {pattern} asynchronously: {e}")
            return in_memory_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Redis cache
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "in_memory_keys": len(self._in_memory_fallback),
            "redis_available": self._sync_client is not None,
            "namespace": self.config.namespace or "default"
        }
        
        # Get Redis stats if available
        if self._sync_client:
            try:
                # Get memory usage
                stats["memory_usage"] = self._sync_client.info("memory").get("used_memory_human", "unknown")
                
                # Get key counts
                pattern = self._safe_key("*")
                stats["total_keys"] = len(self._sync_client.keys(pattern))
                
                # Get hit/miss ratio if metrics are enabled
                if METRICS_ENABLED:
                    namespace = self.config.namespace or "default"
                    hits = REDIS_CACHE_HITS.labels(namespace=namespace)._value.get()
                    misses = REDIS_CACHE_MISSES.labels(namespace=namespace)._value.get()
                    total = hits + misses
                    
                    stats["cache_hits"] = hits
                    stats["cache_misses"] = misses
                    stats["hit_ratio"] = hits / total if total > 0 else 0
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                
        return stats
    
    # Cache decorator utilities
    
    def cache(self, ttl: int = 3600, key_prefix: str = ""):
        """
        Decorator to cache function results in Redis
        
        Args:
            ttl: Cache TTL in seconds
            key_prefix: Prefix for cache keys
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key based on function name, args, and kwargs
                cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                cache_key = self._safe_key(cache_key)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def cache_sync(self, ttl: int = 3600, key_prefix: str = ""):
        """
        Decorator to cache synchronous function results in Redis
        
        Args:
            ttl: Cache TTL in seconds
            key_prefix: Prefix for cache keys
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key based on function name, args, and kwargs
                cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                cache_key = self._safe_key(cache_key)
                
                # Try to get from cache
                cached_result = self.get_sync(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call the function
                result = func(*args, **kwargs)
                
                # Cache the result
                self.set_sync(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
