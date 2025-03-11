"""
Redis Client Service

A wrapper for Redis operations with error handling and monitoring capabilities.
This client is used for caching and distributed task management.
"""

import os
import json
import time
import logging
import pickle
from typing import Any, Dict, List, Optional, Union, Set
from datetime import timedelta

# Configure logging
logger = logging.getLogger("redis_client")

# Try importing Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis package not installed")
    REDIS_AVAILABLE = False

# Optional Prometheus metrics
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Histogram
    METRICS_ENABLED = True
    
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
    
except ImportError:
    METRICS_ENABLED = False


class RedisConfig:
    """Configuration for Redis client"""
    
    def __init__(
        self,
        url: Optional[str] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        max_connections: int = 10,
        decode_responses: bool = False
    ):
        """Initialize Redis configuration
        
        Args:
            url: Redis URL (defaults to REDIS_URL environment variable)
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
            max_connections: Maximum number of connections
            decode_responses: Whether to decode responses to strings
        """
        self.url = url or os.environ.get("REDIS_URL")
        if not self.url:
            raise ValueError("Redis URL not provided and REDIS_URL environment variable not set")
            
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.max_connections = max_connections
        self.decode_responses = decode_responses
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for Redis client
        
        Returns:
            Dict with connection parameters
        """
        return {
            "url": self.url,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
            "health_check_interval": self.health_check_interval,
            "max_connections": self.max_connections,
            "decode_responses": self.decode_responses
        }


class RedisClient:
    """Redis client with error handling and monitoring"""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """Initialize Redis client
        
        Args:
            config: Redis configuration (defaults to environment variables)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with 'pip install redis'")
        
        self.config = config or RedisConfig()
        
        try:
            # Get connection parameters
            params = self.config.get_connection_params()
            
            # Initialize Redis client
            self.client = redis.from_url(
                params["url"],
                socket_timeout=params["socket_timeout"],
                socket_connect_timeout=params["socket_connect_timeout"],
                retry_on_timeout=params["retry_on_timeout"],
                health_check_interval=params["health_check_interval"],
                decode_responses=params["decode_responses"]
            )
            
            # Test connection
            self.ping()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    def _time_operation(self, operation_name):
        """Context manager for timing operations"""
        class Timer:
            def __init__(self, operation):
                self.operation = operation
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if METRICS_ENABLED:
                    REDIS_LATENCY.labels(operation=self.operation).observe(duration)
                
                if exc_type:
                    logger.warning(
                        f"Redis {self.operation} operation failed after {duration:.3f}s",
                        extra={
                            "operation": self.operation,
                            "duration": duration,
                            "error": str(exc_val)
                        }
                    )
                    if METRICS_ENABLED:
                        REDIS_OPERATIONS.labels(
                            operation=self.operation,
                            status="error"
                        ).inc()
                else:
                    logger.debug(
                        f"Redis {self.operation} completed in {duration:.3f}s",
                        extra={
                            "operation": self.operation,
                            "duration": duration
                        }
                    )
                    if METRICS_ENABLED:
                        REDIS_OPERATIONS.labels(
                            operation=self.operation,
                            status="success"
                        ).inc()
        
        return Timer(operation_name)
    
    def ping(self) -> bool:
        """Check Redis connectivity
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._time_operation("ping"):
            try:
                return self.client.ping()
            except Exception as e:
                logger.error(f"Redis ping failed: {e}")
                return False
    
    def get(self, key: str) -> Any:
        """Get a value from Redis
        
        Args:
            key: Redis key
            
        Returns:
            Deserialized value or None if not found
        """
        with self._time_operation("get"):
            try:
                value = self.client.get(key)
                if value is None:
                    return None
                
                try:
                    # Try JSON first
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        # Then try pickle
                        return pickle.loads(value)
                    except Exception:
                        # Return raw value if all else fails
                        return value
            except Exception as e:
                logger.warning(f"Redis get failed for key {key}: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis
        
        Args:
            key: Redis key
            value: Value to store (will be serialized)
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        with self._time_operation("set"):
            try:
                # Try to serialize as JSON first
                try:
                    serialized = json.dumps(value)
                except (TypeError, OverflowError):
                    # Fall back to pickle for complex objects
                    serialized = pickle.dumps(value)
                
                if ttl:
                    return bool(self.client.setex(key, ttl, serialized))
                else:
                    return bool(self.client.set(key, serialized))
            except Exception as e:
                logger.warning(f"Redis set failed for key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from Redis
        
        Args:
            key: Redis key
            
        Returns:
            True if successful, False otherwise
        """
        with self._time_operation("delete"):
            try:
                return bool(self.client.delete(key))
            except Exception as e:
                logger.warning(f"Redis delete failed for key {key}: {e}")
                return False
    
    def exists(self, key: str) -> int:
        """Check if a key exists in Redis
        
        Args:
            key: Redis key
            
        Returns:
            1 if key exists, 0 if not, -1 on error
        """
        with self._time_operation("exists"):
            try:
                return self.client.exists(key)
            except Exception as e:
                logger.warning(f"Redis exists failed for key {key}: {e}")
                return -1
    
    def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        with self._time_operation("keys"):
            try:
                keys = self.client.keys(pattern)
                if isinstance(keys[0], bytes) if keys else False:
                    # Convert bytes to strings
                    return [k.decode('utf-8') for k in keys]
                return keys
            except Exception as e:
                logger.warning(f"Redis keys failed for pattern {pattern}: {e}")
                return []
    
    def ttl(self, key: str) -> int:
        """Get the TTL for a key
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key does not exist
        """
        with self._time_operation("ttl"):
            try:
                return self.client.ttl(key)
            except Exception as e:
                logger.warning(f"Redis ttl failed for key {key}: {e}")
                return -2
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set a TTL for a key
        
        Args:
            key: Redis key
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        with self._time_operation("expire"):
            try:
                return bool(self.client.expire(key, ttl))
            except Exception as e:
                logger.warning(f"Redis expire failed for key {key}: {e}")
                return False
    
    def info(self) -> Dict[str, Any]:
        """Get Redis server info
        
        Returns:
            Dictionary of server info
        """
        with self._time_operation("info"):
            try:
                return self.client.info()
            except Exception as e:
                logger.warning(f"Redis info failed: {e}")
                return {}
    
    def flush_all(self) -> bool:
        """Clear all keys in Redis (use with caution!)
        
        Returns:
            True if successful, False otherwise
        """
        with self._time_operation("flush_all"):
            try:
                self.client.flushall()
                return True
            except Exception as e:
                logger.warning(f"Redis flush_all failed: {e}")
                return False
    
    def flush_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace
        
        Args:
            namespace: Namespace prefix
            
        Returns:
            Number of keys deleted
        """
        with self._time_operation("flush_namespace"):
            try:
                pattern = f"{namespace}:*"
                keys = self.keys(pattern)
                if keys:
                    return self.client.delete(*keys)
                return 0
            except Exception as e:
                logger.warning(f"Redis flush_namespace failed for {namespace}: {e}")
                return -1
