"""Cache Service Module

This module provides a unified Redis caching implementation with the following features:
- Comprehensive configuration options
- Memory caching layer for frequently accessed items
- Both synchronous and asynchronous APIs
- Compression support for large objects
- Robust error handling
- Health check capabilities
- Cache statistics monitoring
"""

import os
import time
import json
import zlib
import pickle
import hashlib
import inspect
import logging
import warnings
import asyncio
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple
from datetime import datetime, timedelta
import random
from urllib.parse import urlparse

# Redis imports
import redis
from redis import Redis
from redis.exceptions import RedisError
import redis.asyncio 
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.connection import ConnectionPool

# Configure logger
logger = logging.getLogger(__name__)

# Custom exceptions
from app.core.exceptions import CacheError, ApplicationError

# Type variable for the cached decorator
T = TypeVar('T')

# Global cache metrics
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_ERRORS = 0
_CACHE_TOTAL_TIME = 0
_CACHE_OPERATIONS = 0

class RedisConfig:
    """Configuration for Redis connection and caching behavior
    
    This class centralizes all Redis-related configuration options to ensure
    consistent connection parameters and caching behavior across the application.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        unix_socket_path: Optional[str] = None,
        key_prefix: str = "",
        client_name: Optional[str] = None,
        encoding: str = "utf-8",
        decode_responses: bool = True,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        socket_keepalive: bool = False,
        socket_keepalive_options: Optional[Dict[int, Union[int, bytes]]] = None,
        max_connections: int = 10,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        use_connection_pooling: bool = True,
        local_cache_size: int = 100,
        local_ttl: int = 300,
        enable_compression: bool = True,
        compression_level: int = 1,
        compression_min_size: int = 1024,
        connection_retries: int = 3,
        connection_retry_delay: float = 0.5,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 30,
        retry_attempts: int = 3,
        base_backoff: float = 0.1,
        max_backoff: float = 1.0,
        jitter: float = 0.1,
        ssl: bool = False,
        replica_url: Optional[str] = None
    ):
        """Initialize Redis configuration
        
        Args:
            url: Redis URL (e.g. redis://localhost:6379/0)
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password
            unix_socket_path: Redis Unix socket path
            key_prefix: Prefix for all cache keys
            client_name: Redis client name
            encoding: Redis encoding
            decode_responses: Whether to decode responses
            socket_timeout: Redis socket timeout
            socket_connect_timeout: Redis socket connect timeout
            socket_keepalive: Whether to enable socket keepalive
            socket_keepalive_options: Socket keepalive options
            max_connections: Maximum number of connections
            health_check_interval: Health check interval
            retry_on_timeout: Whether to retry on timeout
            use_connection_pooling: Whether to use connection pooling
            local_cache_size: Local cache size (0 to disable)
            local_ttl: Local cache TTL
            enable_compression: Whether to enable compression
            compression_level: Compression level (1-9)
            compression_min_size: Minimum size for compression
            connection_retries: Number of connection retries
            connection_retry_delay: Delay between connection retries in seconds
            circuit_breaker_threshold: Circuit breaker error threshold
            circuit_breaker_timeout: Circuit breaker timeout in seconds
            retry_attempts: Number of attempts to initialize Redis clients
            base_backoff: Base backoff time for retrying Redis connections
            max_backoff: Maximum backoff time for retrying Redis connections
            jitter: Jitter for backoff time
            ssl: Whether to use SSL for Redis connection (required for Upstash)
            replica_url: Redis replica URL
        """
        # Connection URL (from env var if not provided)
        self.url = url or os.environ.get("REDIS_URL")
        
        # Basic Redis connection parameters
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.unix_socket_path = unix_socket_path
        
        # Key prefix
        self.key_prefix = key_prefix
        
        # Client options
        self.client_name = client_name
        self.encoding = encoding
        self.decode_responses = decode_responses
        
        # Socket options
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.socket_keepalive = socket_keepalive
        self.socket_keepalive_options = socket_keepalive_options
        
        # Connection pool options
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        self.retry_on_timeout = retry_on_timeout
        self.use_connection_pooling = use_connection_pooling
        
        # Local cache options
        self.local_cache_size = local_cache_size
        self.local_ttl = local_ttl
        
        # Compression options
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.compression_min_size = compression_min_size
        
        # Connection retry options
        self.connection_retries = connection_retries
        self.connection_retry_delay = connection_retry_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Retry settings
        self.retry_attempts = retry_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
        
        # SSL option
        self.ssl = ssl
        
        # Replica URL
        self.replica_url = replica_url
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get connection kwargs for Redis client
        
        Returns:
            Dict[str, Any]: Connection kwargs
        """
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_keepalive": self.socket_keepalive,
            "retry_on_timeout": self.retry_on_timeout,
            "health_check_interval": self.health_check_interval,
            "decode_responses": self.decode_responses,
        }
        
        # Add optional parameters
        if self.password:
            kwargs["password"] = self.password
            
        if self.unix_socket_path:
            kwargs["unix_socket_path"] = self.unix_socket_path
            
        if self.socket_keepalive_options:
            kwargs["socket_keepalive_options"] = self.socket_keepalive_options
            
        if self.client_name:
            kwargs["client_name"] = self.client_name
            
        # We don't add SSL parameters here as they're handled specially in _init_redis_clients
        
        return kwargs
        
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create RedisConfig from environment variables
        
        Returns:
            RedisConfig: Configuration instance
        """
        return cls(
            url=os.environ.get("REDIS_URL"),
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
            password=os.environ.get("REDIS_PASSWORD"),
            socket_timeout=float(os.environ.get("REDIS_SOCKET_TIMEOUT", "5.0")),
            socket_connect_timeout=float(os.environ.get("REDIS_SOCKET_CONNECT_TIMEOUT", "2.0")),
            default_ttl=int(os.environ.get("REDIS_DEFAULT_TTL", "3600")),
            max_connections=int(os.environ.get("REDIS_MAX_CONNECTIONS", "100")),
            key_prefix=os.environ.get("REDIS_KEY_PREFIX", ""),
            ssl=os.environ.get("REDIS_SSL", "").lower() in ("true", "1", "yes"),
            default_compression=os.environ.get("REDIS_COMPRESSION", "").lower() in ("true", "1", "yes"),
        )

class CacheService:
    """Unified Redis cache service
    
    This class provides a comprehensive Redis caching implementation with
    advanced features like compression, local caching, robust error handling,
    and both synchronous and asynchronous APIs.
    """
    
    def _init_redis_clients(self) -> None:
        """Initialize Redis clients with proper error handling and retry logic
        
        This method initializes both synchronous and asynchronous Redis clients
        with proper error handling and retry logic.
        
        Raises:
            ConnectionError: If Redis connection fails after multiple retries
        """
        # Initialize Redis clients
        if self.config.url:
            # Parse Redis URL to get host, port, etc.
            # URL format: redis://[:password@]host[:port][/database]
            redis_url = self.config.url
            
            # Check if URL is already using redis:// or rediss:// scheme
            if not (redis_url.startswith("redis://") or redis_url.startswith("rediss://")):
                # Add scheme if missing
                redis_url = f"redis://{redis_url}"
            
            # Force SSL scheme if SSL is enabled
            if self.config.ssl and redis_url.startswith("redis://"):
                redis_url = redis_url.replace("redis://", "rediss://")
            
            # Get parsed URL
            parsed_url = urlparse(redis_url)
            
            # Parse hostname and port
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 6379
            
            # Extract password if in URL
            password = parsed_url.password
            
            # Check if using SSL
            is_ssl = self.config.ssl or parsed_url.scheme == "rediss"
            
            if is_ssl:
                logger.info(f"Connecting to Redis at {host}:{port} (SSL: Enabled)")
            else:
                logger.info(f"Connecting to Redis at {host}:{port} (SSL: False)")
            
            # Initialize sync client with retry logic
            retry_attempts = 0
            max_retries = self.config.retry_attempts
            
            while retry_attempts < max_retries:
                try:
                    # Create Redis client from URL with appropriate SSL handling
                    if is_ssl:
                        # For SSL connections, we must use from_url with the rediss:// scheme
                        # The redis-py library handles SSL properly with rediss:// URLs
                        self._redis = redis.from_url(
                            redis_url,
                            socket_connect_timeout=self.config.socket_connect_timeout,
                            socket_timeout=self.config.socket_timeout,
                            health_check_interval=self.config.health_check_interval,
                            decode_responses=True
                        )
                    else:
                        self._redis = redis.from_url(
                            redis_url,
                            socket_connect_timeout=self.config.socket_connect_timeout,
                            socket_timeout=self.config.socket_timeout,
                            health_check_interval=self.config.health_check_interval,
                            decode_responses=True
                        )
                    
                    # Test the connection before continuing
                    self._redis.ping()
                    logger.debug("Redis sync client initialized successfully with URL")
                    break
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    retry_attempts += 1
                    self._connection_errors += 1
                    
                    if retry_attempts >= max_retries:
                        logger.warning(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                        self._redis = None
                        break
                    
                    # Calculate backoff with jitter
                    backoff = min(
                        self.config.max_backoff,
                        self.config.base_backoff * (2 ** (retry_attempts - 1))
                    )
                    jitter = random.uniform(0, self.config.jitter)
                    sleep_time = backoff + jitter
                    
                    logger.info(f"Redis connection attempt {retry_attempts} failed. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Initialize async client with retry logic
            retry_attempts = 0
            
            while retry_attempts < max_retries:
                try:
                    # Create Redis async client from URL with appropriate SSL handling
                    if is_ssl:
                        # For SSL connections, we must use from_url with the rediss:// scheme
                        # The redis-py library handles SSL properly with rediss:// URLs
                        self._redis_async = redis.asyncio.from_url(
                            redis_url,
                            socket_connect_timeout=self.config.socket_connect_timeout,
                            socket_timeout=self.config.socket_timeout,
                            health_check_interval=self.config.health_check_interval,
                            decode_responses=True
                        )
                    else:
                        self._redis_async = redis.asyncio.from_url(
                            redis_url,
                            socket_connect_timeout=self.config.socket_connect_timeout,
                            socket_timeout=self.config.socket_timeout,
                            health_check_interval=self.config.health_check_interval,
                            decode_responses=True
                        )
                    
                    # We can't test the async client here because it requires an event loop
                    logger.debug("Redis async client initialized successfully with URL")
                    break
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    retry_attempts += 1
                    self._connection_errors += 1
                    
                    if retry_attempts >= max_retries:
                        logger.warning(f"Failed to initialize async Redis client after {max_retries} attempts: {str(e)}")
                        self._redis_async = None
                        break
                    
                    # Calculate backoff with jitter
                    backoff = min(
                        self.config.max_backoff,
                        self.config.base_backoff * (2 ** (retry_attempts - 1))
                    )
                    jitter = random.uniform(0, self.config.jitter)
                    sleep_time = backoff + jitter
                    
                    logger.info(f"Redis async client initialization attempt {retry_attempts} failed. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
        else:
            # Fallback to using host, port, password approach
            host = os.environ.get("UPSTASH_REDIS_HOST", self.config.host)
            port = int(os.environ.get("UPSTASH_REDIS_PORT", self.config.port))
            password = os.environ.get("UPSTASH_REDIS_PASSWORD", self.config.password)
            
            # Check if we're connecting to Upstash (based on host domain)
            is_upstash = "upstash.io" in host
            ssl_enabled = is_upstash  # Upstash requires SSL
            
            logger.info(f"Connecting to Redis at {host}:{port} (SSL: {ssl_enabled})")
            
            # Prepare SSL connection parameters for Upstash
            ssl_params = None
            if ssl_enabled:
                import ssl as ssl_lib
                ssl_context = ssl_lib.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl_lib.CERT_NONE
                ssl_params = {'ssl_context': ssl_context}
            
            # Initialize sync client with retry logic
            retry_attempts = 0
            max_retries = self.config.retry_attempts
            
            while retry_attempts < max_retries:
                try:
                    # Create pool and Redis client
                    pool = redis.ConnectionPool(
                        host=host,
                        port=port,
                        password=password,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        socket_timeout=self.config.socket_timeout,
                        health_check_interval=self.config.health_check_interval,
                        retry_on_timeout=True,
                        decode_responses=True,
                        max_connections=self.config.max_connections,
                        **(ssl_params or {})
                    )
                    
                    self._redis = redis.Redis(
                        connection_pool=pool,
                        decode_responses=True
                    )
                    
                    # Test the connection before continuing
                    self._redis.ping()
                    logger.debug("Redis sync client initialized successfully")
                    break
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    retry_attempts += 1
                    self._connection_errors += 1
                    
                    if retry_attempts >= max_retries:
                        logger.warning(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                        self._redis = None
                        break
                    
                    # Calculate backoff with jitter
                    backoff = min(
                        self.config.max_backoff,
                        self.config.base_backoff * (2 ** (retry_attempts - 1))
                    )
                    jitter = random.uniform(0, self.config.jitter)
                    sleep_time = backoff + jitter
                    
                    logger.info(f"Redis connection attempt {retry_attempts} failed. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Initialize async client
            retry_attempts = 0
            
            while retry_attempts < max_retries:
                try:
                    self._redis_async = redis.asyncio.Redis(
                        host=host,
                        port=port,
                        password=password,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        socket_timeout=self.config.socket_timeout,
                        health_check_interval=self.config.health_check_interval,
                        retry_on_timeout=True,
                        decode_responses=True,
                        max_connections=self.config.max_connections,
                        **(ssl_params or {})
                    )
                    
                    # We can't test the async client here because it requires an event loop
                    logger.debug("Redis async client initialized successfully")
                    break
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    retry_attempts += 1
                    self._connection_errors += 1
                    
                    if retry_attempts >= max_retries:
                        logger.warning(f"Failed to initialize async Redis client after {max_retries} attempts: {str(e)}")
                        self._redis_async = None
                        break
                    
                    # Calculate backoff with jitter
                    backoff = min(
                        self.config.max_backoff,
                        self.config.base_backoff * (2 ** (retry_attempts - 1))
                    )
                    jitter = random.uniform(0, self.config.jitter)
                    sleep_time = backoff + jitter
                    
                    logger.info(f"Redis async client initialization attempt {retry_attempts} failed. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
        
        # Initialize replica Redis client if provided
        if self.config.replica_url:
            try:
                # Parse replica URL to get host, port, etc.
                replica_url = self.config.replica_url
                
                # Check if URL is already using redis:// or rediss:// scheme
                if not (replica_url.startswith("redis://") or replica_url.startswith("rediss://")):
                    # Add scheme if missing
                    replica_url = f"redis://{replica_url}"
                
                # Force SSL scheme if SSL is enabled
                if self.config.ssl and replica_url.startswith("redis://"):
                    replica_url = replica_url.replace("redis://", "rediss://")
                
                # Get parsed URL
                parsed_url = urlparse(replica_url)
                
                # Parse hostname and port
                host = parsed_url.hostname or "localhost"
                port = parsed_url.port or 6379
                
                # Check if using SSL
                is_ssl = self.config.ssl or parsed_url.scheme == "rediss"
                
                if is_ssl:
                    logger.info(f"Connecting to replica Redis at {host}:{port} (SSL: Enabled)")
                else:
                    logger.info(f"Connecting to replica Redis at {host}:{port} (SSL: False)")
                
                # Create replica Redis client from URL with appropriate SSL handling
                if is_ssl:
                    # For SSL connections, use from_url with the rediss:// scheme
                    # The redis-py library handles SSL properly with rediss:// URLs
                    self._redis_replica = redis.from_url(
                        replica_url,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        socket_timeout=self.config.socket_timeout,
                        health_check_interval=self.config.health_check_interval,
                        decode_responses=True
                    )
                else:
                    self._redis_replica = redis.from_url(
                        replica_url,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        socket_timeout=self.config.socket_timeout,
                        health_check_interval=self.config.health_check_interval,
                        decode_responses=True
                    )
                
                # Test connection
                self._redis_replica.ping()
                
                # Set flag to indicate we have a replica
                self._has_replica = True
                logger.info("Successfully connected to Redis replica")
            except Exception as e:
                # Log the error but continue - we'll operate without a replica
                logger.warning(f"Failed to connect to replica Redis: {str(e)}")
                self._has_replica = False
        
        # Log success or failure
        if self._redis is not None or self._redis_async is not None:
            logger.debug("Redis clients initialized successfully")
        else:
            logger.warning("Failed to initialize Redis clients, falling back to local cache only")
            self._is_fallback_mode = True
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """Initialize the cache service
        
        Args:
            config: Redis configuration (uses default if None)
        """
        # Use provided config or create default
        self.config = config or RedisConfig()
        
        # Initialize Redis clients
        self._redis = None
        self._redis_async = None
        self._redis_replica = None
        
        # Initialize local cache (for frequently accessed items)
        self._local_cache = {}
        self._local_ttl = self.config.local_ttl
        
        # Circuit breaker state
        self._circuit_open = False
        self._error_count = 0
        self._last_error_time = None
        
        # Error tracking
        self._connection_errors = 0
        self._serialization_errors = 0
        self._last_successful_connection = None
        self._is_fallback_mode = False
        
        # Initialize Redis clients with retries and error handling
        self._init_redis_clients()
        
        # Mark successful connection if clients were initialized
        if self._redis is not None or self._redis_async is not None:
            self._last_successful_connection = time.time()
            self._is_fallback_mode = False
        else:
            self._is_fallback_mode = True
            logger.warning("CacheService initialized in fallback mode (Redis unavailable)")
        
        logger.info(f"Initialized CacheService with prefix '{self.config.key_prefix}'")
    
    def is_available(self) -> bool:
        """Check if Redis is available
        
        Returns:
            bool: True if Redis is available
        """
        return (self._redis is not None or self._redis_async is not None) and not self._circuit_open

    def is_in_fallback_mode(self) -> bool:
        """Check if the service is operating in fallback mode
        
        Returns:
            bool: True if in fallback mode
        """
        return self._is_fallback_mode

    def _format_key(self, key: str) -> str:
        """Format a key with the prefix
        
        Args:
            key: Cache key
            
        Returns:
            str: Formatted key
        """
        if self.config.key_prefix:
            return f"{self.config.key_prefix}:{key}"
        return key
    
    def _should_compress(self, value: Any, compress: Optional[bool] = None) -> bool:
        """Determine if a value should be compressed
        
        Args:
            value: Value to check
            compress: Whether to compress (overrides default)
            
        Returns:
            bool: True if compression should be applied
        """
        # If compression is explicitly specified, use that
        if compress is not None:
            return compress
            
        # Otherwise use default setting, but only for objects above min size
        if not self.config.enable_compression:
            return False
            
        # Try to estimate size for common types
        try:
            size = 0
            if isinstance(value, (str, bytes)):
                size = len(value)
            else:
                size = len(pickle.dumps(value))
                
            return size >= self.config.compression_min_size
        except Exception:
            # If we can't determine the size, err on the side of no compression
            return False

    def _compress_value(self, value: Any) -> bytes:
        """Compress a value using zlib
        
        Args:
            value: Value to compress
            
        Returns:
            bytes: Compressed value
        """
        try:
            # First pickle the value to handle any Python object
            pickled_value = pickle.dumps(value)
            
            # Then compress with zlib
            compressed_value = zlib.compress(
                pickled_value, 
                level=self.config.compression_level
            )
            
            return compressed_value
        except Exception as e:
            logger.warning(f"Failed to compress value: {str(e)}")
            # If compression fails, return pickled value
            return pickle.dumps(value)
    
    def _decompress_value(self, compressed_value: bytes) -> Any:
        """Decompress a value using zlib
        
        Args:
            compressed_value: Compressed value
            
        Returns:
            Any: Decompressed value
        """
        try:
            # First decompress with zlib
            decompressed_value = zlib.decompress(compressed_value)
            
            # Then unpickle to get the original object
            return pickle.loads(decompressed_value)
        except zlib.error:
            # If decompression fails, try to unpickle directly
            try:
                return pickle.loads(compressed_value)
            except Exception as e:
                logger.error(f"Failed to decompress and unpickle value: {str(e)}")
                raise CacheError(f"Failed to decompress cached value: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to decompress value: {str(e)}")
            raise CacheError(f"Failed to decompress cached value: {str(e)}")
    
    def _serialize_value(self, value: Any, compress: bool = False) -> bytes:
        """Serialize a value for storage
        
        Args:
            value: Value to serialize
            compress: Whether to compress
            
        Returns:
            bytes: Serialized value
        """
        try:
            if compress:
                return self._compress_value(value)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {str(e)}")
            raise CacheError(f"Failed to serialize value: {str(e)}")
    
    def _deserialize_value(self, serialized_value: bytes, compressed: bool = False) -> Any:
        """Deserialize a value from storage
        
        Args:
            serialized_value: Serialized value
            compressed: Whether the value is compressed
            
        Returns:
            Any: Deserialized value
        """
        try:
            if compressed:
                return self._decompress_value(serialized_value)
            else:
                return pickle.loads(serialized_value)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {str(e)}")
            raise CacheError(f"Failed to deserialize value: {str(e)}")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if the circuit breaker is open
        
        Returns:
            bool: True if the circuit is closed and Redis can be used
        """
        # If the circuit is open, check if timeout has elapsed
        if self._circuit_open:
            elapsed = time.time() - self._last_error_time
            if elapsed > self.config.circuit_breaker_timeout:
                # Reset circuit and error count
                self._circuit_open = False
                self._error_count = 0
                logger.info("Circuit breaker reset after timeout")
                return True
            else:
                # Circuit still open
                return False
        
        # Circuit is closed
        return True
    
    def _increment_error_count(self) -> None:
        """Increment the error count and check circuit breaker
        
        This method increments the error count and opens the circuit
        if the threshold is reached.
        """
        self._error_count += 1
        self._last_error_time = time.time()
        
        # Check if threshold is reached
        if self._error_count >= self.config.circuit_breaker_threshold:
            self._circuit_open = True
            logger.warning(
                f"Circuit breaker opened after {self._error_count} errors. "
                f"Will reset in {self.config.circuit_breaker_timeout} seconds."
            )
    
    def _check_local_cache(self, key: str) -> Tuple[bool, Any]:
        """Check if a key exists in the local cache and is not expired
        
        Args:
            key: Cache key
            
        Returns:
            Tuple[bool, Any]: (hit, value) tuple
        """
        if key in self._local_cache:
            value, expiry = self._local_cache[key]
            
            # Check if expired
            if expiry > time.time():
                return True, value
            else:
                # Remove expired item
                del self._local_cache[key]
                
        return False, None
    
    def _update_local_cache(self, key: str, value: Any) -> None:
        """Update the local cache with a key-value pair
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Maintain max size
        if len(self._local_cache) >= self.config.local_cache_size:
            # Remove oldest item (simple LRU strategy)
            try:
                oldest_key = next(iter(self._local_cache))
                del self._local_cache[oldest_key]
            except (StopIteration, KeyError):
                # Handle edge case where cache is modified concurrently
                pass
        
        # Calculate expiry
        expiry = time.time() + self._local_ttl
        
        # Store value with expiry
        self._local_cache[key] = (value, expiry)
    
    async def _reset_connection(self) -> bool:
        """Reset Redis connection
        
        Returns:
            bool: True if successful
        """
        try:
            # Re-initialize Redis clients
            self._init_redis_clients()
            
            # Test connection
            if self._redis_async:
                result = await self._redis_async.ping()
                
            if self._redis:
                self._redis.ping()
                
            return True
        except Exception as e:
            logger.error(f"Failed to reset Redis connection: {str(e)}")
            return False
    
    async def get(self, key: str, default: Any = None, ignore_local_cache: bool = False) -> Any:
        """Get a value from the cache asynchronously
        
        Args:
            key: Cache key
            default: Default value if key not found
            ignore_local_cache: Whether to ignore the local cache
            
        Returns:
            Any: Cached value or default if not found
        """
        global _CACHE_HITS, _CACHE_MISSES, _CACHE_ERRORS, _CACHE_TOTAL_TIME, _CACHE_OPERATIONS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Check local cache first
            if not ignore_local_cache:
                hit, value = self._check_local_cache(formatted_key)
                if hit:
                    _CACHE_HITS += 1
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return value
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis get for key: {formatted_key}")
                return default
            
            # Check Redis
            if self._redis_async:
                result = await self._redis_async.get(formatted_key)
                
                if result is not None:
                    # Try to deserialize the result
                    try:
                        # Check if the value is compressed (simple approach)
                        is_compressed = False
                        if isinstance(result, bytes) and len(result) > 2:
                            # Check for zlib header (bytes 0 and 1 are 0x78, 0x9C for most compression levels)
                            if result[0] == 0x78 and result[1] in (0x01, 0x9C, 0xDA):
                                is_compressed = True
                        
                        # Deserialize based on compression
                        value = self._deserialize_value(result, is_compressed)
                        
                        # Update local cache
                        self._update_local_cache(formatted_key, value)
                        
                        _CACHE_HITS += 1
                        _CACHE_TOTAL_TIME += (time.time() - start_time)
                        return value
                    except Exception as e:
                        logger.error(f"Failed to deserialize cached value for key {formatted_key}: {str(e)}")
                        _CACHE_ERRORS += 1
                        _CACHE_TOTAL_TIME += (time.time() - start_time)
                        return default
                
                # Key not found
                _CACHE_MISSES += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return default
            else:
                # Redis not available
                logger.warning("Redis client not available for get operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return default
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error getting value from Redis for key {formatted_key}: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return default
    
    def get_sync(self, key: str, default: Any = None, ignore_local_cache: bool = False) -> Any:
        """Get a value from the cache synchronously
        
        Args:
            key: Cache key
            default: Default value if key not found
            ignore_local_cache: Whether to ignore the local cache
            
        Returns:
            Any: Cached value or default if not found
        """
        global _CACHE_HITS, _CACHE_MISSES, _CACHE_ERRORS, _CACHE_TOTAL_TIME, _CACHE_OPERATIONS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Check local cache first
            if not ignore_local_cache:
                hit, value = self._check_local_cache(formatted_key)
                if hit:
                    _CACHE_HITS += 1
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return value
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis get for key: {formatted_key}")
                return default
            
            # Check Redis
            if self._redis:
                result = self._redis.get(formatted_key)
                
                if result is not None:
                    # Try to deserialize the result
                    try:
                        # Check if the value is compressed (simple approach)
                        is_compressed = False
                        if isinstance(result, bytes) and len(result) > 2:
                            # Check for zlib header (bytes 0 and 1 are 0x78, 0x9C for most compression levels)
                            if result[0] == 0x78 and result[1] in (0x01, 0x9C, 0xDA):
                                is_compressed = True
                        
                        # Deserialize based on compression
                        value = self._deserialize_value(result, is_compressed)
                        
                        # Update local cache
                        self._update_local_cache(formatted_key, value)
                        
                        _CACHE_HITS += 1
                        _CACHE_TOTAL_TIME += (time.time() - start_time)
                        return value
                    except Exception as e:
                        logger.error(f"Failed to deserialize cached value for key {formatted_key}: {str(e)}")
                        _CACHE_ERRORS += 1
                        _CACHE_TOTAL_TIME += (time.time() - start_time)
                        return default
                
                # Key not found
                _CACHE_MISSES += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return default
            else:
                # Redis not available
                logger.warning("Redis client not available for get operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return default
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error getting value from Redis for key {formatted_key}: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        compress: Optional[bool] = None,
        update_local_cache: bool = True
    ) -> bool:
        """Set a value in the cache asynchronously
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            compress: Whether to compress (uses auto-detection if None)
            update_local_cache: Whether to update the local cache
            
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME, _CACHE_ERRORS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        # Use default TTL if not specified
        if ttl is None:
            # Use a reasonable default TTL (1 hour) since default_ttl is no longer in config
            ttl = 3600  # 1 hour in seconds
        
        try:
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis set for key: {formatted_key}")
                
                # Update local cache anyway
                if update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                return False
            
            # Determine if compression should be used
            should_compress = self._should_compress(value, compress)
            
            # Serialize the value
            serialized_value = self._serialize_value(value, should_compress)
            
            # Set in Redis
            if self._redis_async:
                if ttl > 0:
                    result = await self._redis_async.setex(formatted_key, ttl, serialized_value)
                else:
                    result = await self._redis_async.set(formatted_key, serialized_value)
                
                # Update local cache if successful
                if result and update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result
            else:
                # Redis not available
                logger.warning("Redis client not available for set operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                
                # Update local cache anyway
                if update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error setting value in Redis for key {formatted_key}: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            
            # Update local cache anyway in case of error
            if update_local_cache:
                self._update_local_cache(formatted_key, value)
            
            return False
            
    def set_sync(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        compress: Optional[bool] = None,
        update_local_cache: bool = True
    ) -> bool:
        """Set a value in the cache synchronously
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            compress: Whether to compress (uses auto-detection if None)
            update_local_cache: Whether to update the local cache
            
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME, _CACHE_ERRORS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        # Use default TTL if not specified
        if ttl is None:
            # Use a reasonable default TTL (1 hour) since default_ttl is no longer in config
            ttl = 3600  # 1 hour in seconds
        
        try:
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis set for key: {formatted_key}")
                
                # Update local cache anyway
                if update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                return False
            
            # Determine if compression should be used
            should_compress = self._should_compress(value, compress)
            
            # Serialize the value
            serialized_value = self._serialize_value(value, should_compress)
            
            # Set in Redis
            if self._redis:
                if ttl > 0:
                    result = self._redis.setex(formatted_key, ttl, serialized_value)
                else:
                    result = self._redis.set(formatted_key, serialized_value)
                
                # Update local cache if successful
                if result and update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result
            else:
                # Redis not available
                logger.warning("Redis client not available for set operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                
                # Update local cache anyway
                if update_local_cache:
                    self._update_local_cache(formatted_key, value)
                
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error setting value in Redis for key {formatted_key}: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            
            # Update local cache anyway in case of error
            if update_local_cache:
                self._update_local_cache(formatted_key, value)
            
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache asynchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME, _CACHE_ERRORS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Remove from local cache
            if formatted_key in self._local_cache:
                del self._local_cache[formatted_key]
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis delete for key: {formatted_key}")
                return False
            
            # Delete from Redis
            if self._redis_async:
                result = await self._redis_async.delete(formatted_key)
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result > 0
            else:
                # Redis not available
                logger.warning("Redis client not available for delete operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error deleting key {formatted_key} from Redis: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False
    
    def delete_sync(self, key: str) -> bool:
        """Delete a key from the cache synchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME, _CACHE_ERRORS
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Remove from local cache
            if formatted_key in self._local_cache:
                del self._local_cache[formatted_key]
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis delete for key: {formatted_key}")
                return False
            
            # Delete from Redis
            if self._redis:
                result = self._redis.delete(formatted_key)
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result > 0
            else:
                # Redis not available
                logger.warning("Redis client not available for delete operation")
                _CACHE_ERRORS += 1
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            
            logger.error(f"Error deleting key {formatted_key} from Redis: {str(e)}")
            _CACHE_ERRORS += 1
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache asynchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Check local cache first
            if formatted_key in self._local_cache:
                value, expiry = self._local_cache[formatted_key]
                # Check if expired
                if expiry > time.time():
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return True
                else:
                    # Remove expired item
                    del self._local_cache[formatted_key]
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis exists for key: {formatted_key}")
                return False
            
            # Check Redis
            if self._redis_async:
                result = await self._redis_async.exists(formatted_key)
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result > 0
            else:
                # Redis not available
                if self._is_fallback_mode:
                    logger.debug(f"Redis client not available for exists operation (fallback mode)")
                else:
                    logger.warning(f"Redis client not available for exists operation")
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            self._connection_errors += 1
            
            logger.error(f"Error checking if key {formatted_key} exists in Redis: {str(e)}")
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False
    
    def exists_sync(self, key: str) -> bool:
        """Check if a key exists in the cache synchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        formatted_key = self._format_key(key)
        
        try:
            # Check local cache first
            if formatted_key in self._local_cache:
                value, expiry = self._local_cache[formatted_key]
                # Check if expired
                if expiry > time.time():
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return True
                else:
                    # Remove expired item
                    del self._local_cache[formatted_key]
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug(f"Circuit breaker open, skipping Redis exists for key: {formatted_key}")
                return False
            
            # Check Redis
            if self._redis:
                result = self._redis.exists(formatted_key)
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return result > 0
            else:
                # Redis not available
                if self._is_fallback_mode:
                    logger.debug(f"Redis client not available for exists operation (fallback mode)")
                else:
                    logger.warning(f"Redis client not available for exists operation")
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count
            self._increment_error_count()
            self._connection_errors += 1
            
            logger.error(f"Error checking if key {formatted_key} exists in Redis: {str(e)}")
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False
    
    async def flush(self) -> bool:
        """Flush the entire cache asynchronously
        
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        
        try:
            # Clear local cache
            self._local_cache.clear()
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug("Circuit breaker open, skipping Redis flush")
                return False
            
            # Flush Redis
            if self._redis_async:
                try:
                    # Only flush keys with our prefix to avoid affecting other applications
                    if self.config.key_prefix:
                        # Use scan to iterate through keys with our prefix
                        cursor = 0
                        keys_to_delete = []
                        
                        while True:
                            cursor, keys = await self._redis_async.scan(
                                cursor, 
                                match=f"{self.config.key_prefix}:*", 
                                count=1000
                            )
                            
                            if keys:
                                keys_to_delete.extend(keys)
                            
                            if cursor == 0:
                                break
                        
                        if keys_to_delete:
                            await self._redis_async.delete(*keys_to_delete)
                    else:
                        # No prefix, flush entire database (use with caution)
                        await self._redis_async.flushdb()
                        
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return True
                except Exception as e:
                    # Increment error count for specific Redis operation failure
                    self._increment_error_count()
                    logger.error(f"Error executing Redis flush command: {str(e)}")
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return False
            else:
                # Redis not available
                if self._is_fallback_mode:
                    logger.debug("Redis client not available for flush operation (fallback mode)")
                else:
                    logger.warning("Redis client not available for flush operation")
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count for unexpected exceptions
            self._increment_error_count()
            self._connection_errors += 1
            
            logger.error(f"Error flushing Redis cache: {str(e)}")
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False
    
    def flush_sync(self) -> bool:
        """Flush the entire cache synchronously
        
        Returns:
            bool: True if successful
        """
        global _CACHE_OPERATIONS, _CACHE_TOTAL_TIME
        _CACHE_OPERATIONS += 1
        
        start_time = time.time()
        
        try:
            # Clear local cache
            self._local_cache.clear()
            
            # If circuit breaker is open, skip Redis
            if not self._check_circuit_breaker():
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                logger.debug("Circuit breaker open, skipping Redis flush")
                return False
            
            # Flush Redis
            if self._redis:
                try:
                    # Only flush keys with our prefix to avoid affecting other applications
                    if self.config.key_prefix:
                        # Use scan to iterate through keys with our prefix
                        cursor = 0
                        keys_to_delete = []
                        
                        while True:
                            cursor, keys = self._redis.scan(
                                cursor, 
                                match=f"{self.config.key_prefix}:*", 
                                count=1000
                            )
                            
                            if keys:
                                keys_to_delete.extend(keys)
                            
                            if cursor == 0:
                                break
                        
                        if keys_to_delete:
                            self._redis.delete(*keys_to_delete)
                    else:
                        # No prefix, flush entire database (use with caution)
                        self._redis.flushdb()
                        
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return True
                except Exception as e:
                    # Increment error count for specific Redis operation failure
                    self._increment_error_count()
                    logger.error(f"Error executing Redis flush command: {str(e)}")
                    _CACHE_TOTAL_TIME += (time.time() - start_time)
                    return False
            else:
                # Redis not available
                if self._is_fallback_mode:
                    logger.debug("Redis client not available for flush operation (fallback mode)")
                else:
                    logger.warning("Redis client not available for flush operation")
                _CACHE_TOTAL_TIME += (time.time() - start_time)
                return False
        except Exception as e:
            # Increment error count for unexpected exceptions
            self._increment_error_count()
            self._connection_errors += 1
            
            logger.error(f"Error flushing Redis cache: {str(e)}")
            _CACHE_TOTAL_TIME += (time.time() - start_time)
            return False
    
    async def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Perform a health check on the cache service
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Tuple containing (is_healthy, details)
        """
        start_time = time.time()
        is_healthy = False
        error_message = None
        
        try:
            # Try to ping Redis
            if self._redis_async:
                await self._redis_async.ping()
                is_healthy = True
                
                # Reset circuit breaker if it was open
                if self._circuit_open:
                    self._circuit_open = False
                    self._error_count = 0
                    self._last_error_time = None
            else:
                error_message = "Redis client not available"
        except Exception as e:
            error_message = str(e)
            is_healthy = False
            self._increment_error_count()
            
        # Calculate latency
        latency = time.time() - start_time
        
        # Compile results
        details = {
            "is_healthy": is_healthy,
            "latency_ms": round(latency * 1000, 2),
            "error": error_message,
            "circuit_breaker": "open" if self._circuit_open else "closed",
            "error_count": self._error_count,
            "mode": "fallback" if self._is_fallback_mode else "normal",
            "host": f"{self.config.host}:{self.config.port}",
            "ssl": "enabled" if self.config.url and self.config.url.startswith("rediss://") else "disabled",
            "timestamp": datetime.now().isoformat(),
        }
        
        return is_healthy, details
    
    def health_check_sync(self) -> Tuple[bool, Dict[str, Any]]:
        """Perform a health check on the cache service synchronously
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Tuple containing (is_healthy, details)
        """
        # Run the async version in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.health_check())
        finally:
            loop.close()
    
    def close(self) -> None:
        """Close Redis connections
        
        This method should be called when the application is shutting down
        to properly close Redis connections.
        """
        try:
            if self._redis:
                self._redis.close()
                logger.debug("Closed synchronous Redis connection")
                
            if self._redis_async and hasattr(self._redis_async, 'close'):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._redis_async.close())
                    else:
                        loop.run_until_complete(self._redis_async.close())
                    logger.debug("Closed asynchronous Redis connection")
                except Exception as e:
                    logger.warning(f"Error closing asynchronous Redis connection: {str(e)}")
        except Exception as e:
            logger.warning(f"Error closing Redis connections: {str(e)}")

# Global cache service instance
_cache_service = None

def get_cache() -> CacheService:
    """Get the global cache service instance
    
    Returns:
        CacheService: Global cache service instance
    """
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        
    return _cache_service

def calculate_cache_stats() -> Dict[str, Any]:
    """Calculate cache statistics
    
    Returns:
        Dict[str, Any]: Cache statistics
    """
    global _CACHE_HITS, _CACHE_MISSES, _CACHE_ERRORS, _CACHE_TOTAL_TIME, _CACHE_OPERATIONS
    
    # Calculate hit rate
    total_ops = _CACHE_HITS + _CACHE_MISSES + _CACHE_ERRORS
    hit_rate = (_CACHE_HITS / total_ops * 100) if total_ops > 0 else 0
    
    # Calculate average operation time
    avg_time = (_CACHE_TOTAL_TIME / _CACHE_OPERATIONS * 1000) if _CACHE_OPERATIONS > 0 else 0
    
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "errors": _CACHE_ERRORS,
        "operations": _CACHE_OPERATIONS,
        "hit_rate": hit_rate,
        "avg_time_ms": avg_time,
    }

def _generate_cache_key(func: Callable, args: Tuple, kwargs: Dict[str, Any], prefix: str = "") -> str:
    """Generate a cache key from a function and its arguments
    
    Args:
        func: Function to generate key for
        args: Positional arguments
        kwargs: Keyword arguments
        prefix: Key prefix
        
    Returns:
        str: Cache key
    """
    # Get function name and module
    func_name = func.__name__
    module_name = func.__module__
    
    # Generate consistent representation of arguments
    arg_values = []
    
    # Add positional arguments
    for arg in args:
        try:
            # Use repr for consistent string representation
            arg_values.append(repr(arg))
        except Exception:
            # For arguments that can't be repr'd, use their id
            arg_values.append(f"id:{id(arg)}")
    
    # Add keyword arguments (sorted for consistency)
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        try:
            arg_values.append(f"{key}={repr(value)}")
        except Exception:
            arg_values.append(f"{key}=id:{id(value)}")
    
    # Join everything into a string
    args_str = ",".join(arg_values)
    
    # Generate a deterministic hash
    hash_input = f"{module_name}.{func_name}({args_str})"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()
    
    # Create the final key
    if prefix:
        return f"{prefix}:{hash_value}"
    else:
        return hash_value

def cached(
    ttl: Optional[int] = None, 
    prefix: str = "", 
    compress: Optional[bool] = None,
    skip_args: List[str] = None,
    cache_service: Optional[CacheService] = None,
    cache_none: bool = True,
    fallback_to_compute: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching function results
    
    This decorator can be used with both synchronous and asynchronous functions.
    
    Args:
        ttl: TTL in seconds (uses default if None)
        prefix: Key prefix
        compress: Whether to compress (uses auto-detection if None)
        skip_args: List of argument names to exclude from the cache key
        cache_service: Custom cache service (uses global if None)
        cache_none: Whether to cache None results
        fallback_to_compute: Whether to compute the result if cache fails
        
    Returns:
        Callable: Decorated function
    """
    skip_args = skip_args or []
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_prefix = f"{prefix}:{func.__module__}.{func.__name__}" if prefix else f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            global _CACHE_OPERATIONS
            _CACHE_OPERATIONS += 1
            
            # Get or create cache service
            service = cache_service or get_cache()
            
            # Filter out arguments to skip
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in skip_args}
            
            # Filter self/cls for methods
            filter_args = list(args)
            is_method = inspect.ismethod(func) or (args and hasattr(args[0], func.__name__) and getattr(args[0], func.__name__) == func)
            if is_method and len(args) > 0:
                filter_args = args[1:]  # Skip self/cls
            
            # Generate cache key
            key = _generate_cache_key(func, filter_args, filtered_kwargs, func_prefix)
            
            # Try to get from cache
            try:
                cached_value = await service.get(key)
                if cached_value is not None:
                    return cached_value
                    
                # Special case for None values
                if cached_value is None and cache_none:
                    return None
            except Exception as e:
                if not fallback_to_compute:
                    # Re-raise if we're not falling back
                    raise CacheError(f"Error getting cached value for {func.__name__}: {str(e)}") from e
                
                logger.error(f"Error getting cached value for {func.__name__}: {str(e)}")
            
            # Cache miss, compute result
            result = await func(*args, **kwargs)
            
            # Cache result if not None or cache_none is True
            if result is not None or cache_none:
                try:
                    await service.set(key, result, ttl, compress)
                except Exception as e:
                    logger.error(f"Error caching result for {func.__name__}: {str(e)}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            global _CACHE_OPERATIONS
            _CACHE_OPERATIONS += 1
            
            # Get or create cache service
            service = cache_service or get_cache()
            
            # Filter out arguments to skip
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in skip_args}
            
            # Filter self/cls for methods
            filter_args = list(args)
            is_method = inspect.ismethod(func) or (args and hasattr(args[0], func.__name__) and getattr(args[0], func.__name__) == func)
            if is_method and len(args) > 0:
                filter_args = args[1:]  # Skip self/cls
            
            # Generate cache key
            key = _generate_cache_key(func, filter_args, filtered_kwargs, func_prefix)
            
            # Try to get from cache
            try:
                cached_value = service.get_sync(key)
                if cached_value is not None:
                    return cached_value
                    
                # Special case for None values
                if cached_value is None and cache_none:
                    return None
            except Exception as e:
                if not fallback_to_compute:
                    # Re-raise if we're not falling back
                    raise CacheError(f"Error getting cached value for {func.__name__}: {str(e)}") from e
                
                logger.error(f"Error getting cached value for {func.__name__}: {str(e)}")
            
            # Cache miss, compute result
            result = func(*args, **kwargs)
            
            # Cache result if not None or cache_none is True
            if result is not None or cache_none:
                try:
                    service.set_sync(key, result, ttl, compress)
                except Exception as e:
                    logger.error(f"Error caching result for {func.__name__}: {str(e)}")
            
            return result
        
        # Use appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

class RedisCache:
    """Legacy Redis cache implementation
    
    This class provides a compatibility layer for systems 
    still using the older RedisCache implementation. It internally
    uses the new CacheService for all operations to ensure consistent
    behavior while providing backward compatibility.
    
    Deprecated: This class is maintained for backward compatibility.
    New code should use CacheService directly.
    """
    
    def __init__(
        self, 
        redis_url: Optional[str] = None, 
        ttl: int = 3600,
        prefix: str = "",
        compress: bool = False,
        **redis_kwargs
    ):
        """Initialize the Redis cache compatibility layer
        
        Args:
            redis_url: Redis connection URL (uses env var if None)
            ttl: Default TTL in seconds
            prefix: Prefix for all cache keys
            compress: Whether to compress values
            **redis_kwargs: Additional Redis connection parameters
        """
        # Create RedisConfig from parameters
        config = RedisConfig(
            url=redis_url,
            key_prefix=prefix,
            enable_compression=compress,
            **redis_kwargs
        )
        
        # Create the underlying CacheService
        self._cache_service = CacheService(config)
        self._prefix = prefix
        self._ttl = ttl
        self._compress = compress
        
        # For compatibility
        self._loop = asyncio.get_event_loop() if not asyncio.get_event_loop().is_running() else None
        
        logger.info(f"Initialized RedisCache compatibility layer (using CacheService)")
    
    def _format_key(self, key: str) -> str:
        """Format a key with the prefix
        
        Args:
            key: Cache key
            
        Returns:
            str: Formatted key
        """
        if self._prefix:
            return f"{self._prefix}:{key}"
        return key
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        formatted_key = self._format_key(key)
        return await self._cache_service.get(formatted_key)
    
    def get_sync(self, key: str) -> Any:
        """Get a value from the cache synchronously
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        formatted_key = self._format_key(key)
        return self._cache_service.get_sync(formatted_key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            bool: True if successful
        """
        formatted_key = self._format_key(key)
        ttl = ttl if ttl is not None else self._ttl
        return await self._cache_service.set(
            formatted_key, 
            value, 
            ttl=ttl, 
            compress=self._compress
        )
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache synchronously
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            bool: True if successful
        """
        formatted_key = self._format_key(key)
        ttl = ttl if ttl is not None else self._ttl
        return self._cache_service.set_sync(
            formatted_key, 
            value, 
            ttl=ttl, 
            compress=self._compress
        )
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        formatted_key = self._format_key(key)
        return await self._cache_service.delete(formatted_key)
    
    def delete_sync(self, key: str) -> bool:
        """Delete a key from the cache synchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        formatted_key = self._format_key(key)
        return self._cache_service.delete_sync(formatted_key)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists
        """
        formatted_key = self._format_key(key)
        return await self._cache_service.exists(formatted_key)
    
    def exists_sync(self, key: str) -> bool:
        """Check if a key exists in the cache synchronously
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists
        """
        formatted_key = self._format_key(key)
        return self._cache_service.exists_sync(formatted_key)
    
    async def flush(self) -> bool:
        """Flush the entire cache
        
        Returns:
            bool: True if successful
        """
        return await self._cache_service.flush()
    
    def flush_sync(self) -> bool:
        """Flush the entire cache synchronously
        
        Returns:
            bool: True if successful
        """
        return self._cache_service.flush_sync()
    
    # Compatibility methods for systems using non-async code
    # These methods are deprecated and will be removed in future versions
    
    def get_value(self, key: str) -> Any:
        """Legacy method: Get a value from the cache
        
        Deprecated: Use get_sync instead
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        warnings.warn(
            "get_value is deprecated, use get_sync instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_sync(key)
    
    def set_value(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Legacy method: Set a value in the cache
        
        Deprecated: Use set_sync instead
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            bool: True if successful
        """
        warnings.warn(
            "set_value is deprecated, use set_sync instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_sync(key, value, ttl)
    
    def delete_value(self, key: str) -> bool:
        """Legacy method: Delete a key from the cache
        
        Deprecated: Use delete_sync instead
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        warnings.warn(
            "delete_value is deprecated, use delete_sync instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.delete_sync(key)
    
    # Methods to expose the raw client for advanced usage
    
    @property
    def redis_client(self) -> Optional[Union[Redis, AsyncRedis]]:
        """Get the underlying Redis client
        
        Returns:
            Optional[Union[Redis, AsyncRedis]]: Redis client instance
        """
        if hasattr(self._cache_service, '_redis') and self._cache_service._redis:
            return self._cache_service._redis.client
        return None
    
    @property
    def cache_service(self) -> 'CacheService':
        """Get the underlying CacheService
        
        Returns:
            CacheService: CacheService instance
        """
        return self._cache_service
