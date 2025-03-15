#!/usr/bin/env python
"""
Cache Service Usage Example

This script demonstrates how to use the unified CacheService class 
for Redis caching operations in the credit-cashflow-engine.

Features demonstrated:
- Basic configuration
- Setting and getting cached values
- Using the cached decorator
- Compression capabilities
- Error handling and circuit breaker pattern
- Performance metrics and monitoring
- Graceful fallback when Redis is unavailable

Author: Credit-Cashflow-Engine team
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from app.core.cache_service import (
    CacheService, 
    RedisCache, 
    RedisConfig, 
    cached,
    get_cache,
    calculate_cache_stats
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_service_example")


def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class TestObject:
    """Test class with large data for caching demonstration"""
    def __init__(self, id: int, name: str, data: Dict[str, Any]):
        self.id = id
        self.name = name
        self.data = data
        self.created_at = datetime.now()
    
    def __repr__(self):
        return f"TestObject(id={self.id}, name={self.name}, data_size={len(json.dumps(self.data))})"


async def test_basic_operations(cache: CacheService):
    """Test basic cache operations"""
    logger.info("\n\n=== Testing Basic Operations ===")
    
    # Test set/get operations
    key = "test:basic:string"
    value = "This is a test string"
    
    logger.info(f"Setting key: {key}")
    success = await cache.set(key, value)
    logger.info(f"Set operation successful: {success}")
    
    logger.info(f"Getting key: {key}")
    result = await cache.get(key)
    logger.info(f"Got value: {result}")
    
    # Test with complex data
    key = "test:basic:dict"
    value = {
        "name": "Test Dictionary", 
        "values": [1, 2, 3, 4, 5],
        "nested": {"a": 1, "b": 2, "c": {"d": 3}}
    }
    
    logger.info(f"Setting complex value for key: {key}")
    success = await cache.set(key, value)
    logger.info(f"Set operation successful: {success}")
    
    logger.info(f"Getting complex value for key: {key}")
    result = await cache.get(key)
    logger.info(f"Got complex value with type: {type(result)}")
    
    # Test with different TTL
    key = "test:basic:short_ttl"
    value = "This will expire quickly"
    
    logger.info(f"Setting key with 5s TTL: {key}")
    success = await cache.set(key, value, ttl=5)
    logger.info(f"Set operation successful: {success}")
    
    logger.info(f"Getting key immediately: {key}")
    result = await cache.get(key)
    logger.info(f"Got value: {result}")
    
    logger.info("Waiting 6 seconds for expiration...")
    await asyncio.sleep(6)
    
    logger.info(f"Getting key after TTL: {key}")
    result = await cache.get(key)
    logger.info(f"Got value (should be None): {result}")
    
    # Test exists and delete
    key = "test:basic:exists_delete"
    value = "This will be deleted"
    
    logger.info(f"Setting key: {key}")
    await cache.set(key, value)
    
    logger.info(f"Checking if key exists: {key}")
    exists = await cache.exists(key)
    logger.info(f"Key exists: {exists}")
    
    logger.info(f"Deleting key: {key}")
    deleted = await cache.delete(key)
    logger.info(f"Delete operation successful: {deleted}")
    
    logger.info(f"Checking if key exists after delete: {key}")
    exists = await cache.exists(key)
    logger.info(f"Key exists: {exists}")


async def test_compression(cache: CacheService):
    """Test compression capabilities"""
    logger.info("\n\n=== Testing Compression ===")
    
    # Create a large dataset that benefits from compression
    large_data = {
        "id": "test_compression",
        "timestamp": datetime.now().isoformat(),
        "data": ["test data" * 100] * 100,  # Highly compressible repeated data
        "metadata": {str(i): "metadata value " * 20 for i in range(100)}
    }
    
    # Get size estimate
    data_size = len(json.dumps(large_data))
    logger.info(f"Original data size (estimate): {data_size} bytes")
    
    # Test without compression
    key = "test:compression:disabled"
    logger.info(f"Setting large data without compression: {key}")
    start_time = time.time()
    success = await cache.set(key, large_data, compress=False)
    no_compression_time = time.time() - start_time
    logger.info(f"Set without compression took {no_compression_time:.4f} seconds")
    
    # Test with compression
    key = "test:compression:enabled"
    logger.info(f"Setting large data with compression: {key}")
    start_time = time.time()
    success = await cache.set(key, large_data, compress=True)
    with_compression_time = time.time() - start_time
    logger.info(f"Set with compression took {with_compression_time:.4f} seconds")
    
    # Retrieve and check both values
    start_time = time.time()
    no_compression_result = await cache.get("test:compression:disabled")
    no_compression_get_time = time.time() - start_time
    logger.info(f"Get without compression took {no_compression_get_time:.4f} seconds")
    
    start_time = time.time()
    with_compression_result = await cache.get("test:compression:enabled")
    with_compression_get_time = time.time() - start_time
    logger.info(f"Get with compression took {with_compression_get_time:.4f} seconds")
    
    # Verify data integrity
    logger.info("Verifying data integrity...")
    original_str = json.dumps(large_data, sort_keys=True)
    no_compression_str = json.dumps(no_compression_result, sort_keys=True)
    with_compression_str = json.dumps(with_compression_result, sort_keys=True)
    
    logger.info(f"Data without compression matches: {original_str == no_compression_str}")
    logger.info(f"Data with compression matches: {original_str == with_compression_str}")


async def test_error_handling(cache: CacheService):
    """Test error handling and circuit breaker"""
    logger.info("\n\n=== Testing Error Handling ===")
    
    # First, ensure we have some data in the cache
    key = "test:error:data"
    value = "Test data for error handling"
    await cache.set(key, value)
    
    # Simulate temporary errors by making the Redis connection unavailable
    # (This is just a simulation - in a real environment, we might stop Redis)
    logger.info("Simulating Redis connection failure...")
    
    # Method 1: Force circuit breaker open by simulating errors
    original_redis = cache._redis_async
    try:
        # Temporarily replace Redis client with a mock that raises errors
        class MockRedisError:
            async def get(self, *args, **kwargs):
                raise Exception("Simulated Redis error")
                
            async def set(self, *args, **kwargs):
                raise Exception("Simulated Redis error")
        
        # Replace the Redis client
        cache._redis_async = MockRedisError()
        
        # Attempt operations and see circuit breaker in action
        for i in range(10):
            try:
                result = await cache.get(f"test:error:{i}")
                logger.info(f"Get operation {i+1} result: {result}")
            except Exception as e:
                logger.error(f"Error in get operation {i+1}: {str(e)}")
            
            try:
                success = await cache.set(f"test:error:{i}", f"Value {i}")
                logger.info(f"Set operation {i+1} success: {success}")
            except Exception as e:
                logger.error(f"Error in set operation {i+1}: {str(e)}")
            
            # Check circuit state
            logger.info(f"Circuit breaker state after operations {i+1}: {'Open' if cache._circuit_open else 'Closed'}")
            
            if cache._circuit_open:
                logger.info("Circuit breaker opened, Redis operations will be skipped")
                break
    finally:
        # Restore original Redis client
        cache._redis_async = original_redis
        
    # Allow circuit breaker to close after some time
    logger.info("Waiting for circuit breaker to close...")
    await asyncio.sleep(11)  # Circuit breaker reset time is 10 seconds
    
    # Check if circuit closed and operations work again
    logger.info("Circuit breaker reset time passed")
    logger.info(f"Circuit breaker state: {'Open' if cache._circuit_open else 'Closed'}")
    
    # Try a successful operation
    result = await cache.get(key)
    logger.info(f"Get operation after circuit reset: {result}")


async def test_local_cache(cache: CacheService):
    """Test local cache functionality"""
    logger.info("\n\n=== Testing Local Cache ===")
    
    # Set a value that should be stored in both Redis and local cache
    key = "test:local:value"
    value = {
        "id": random.randint(1000, 9999),
        "name": "Local Cache Test",
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Setting key that will be stored in both Redis and local cache: {key}")
    await cache.set(key, value)
    
    # First retrieval - should come from Redis
    logger.info("First retrieval (from Redis)...")
    start_time = time.time()
    result1 = await cache.get(key)
    first_time = time.time() - start_time
    logger.info(f"First retrieval took {first_time:.6f} seconds")
    
    # Second retrieval - should come from local cache
    logger.info("Second retrieval (should come from local cache)...")
    start_time = time.time()
    result2 = await cache.get(key)
    second_time = time.time() - start_time
    logger.info(f"Second retrieval took {second_time:.6f} seconds")
    
    # Compare times and results
    if second_time > 0:
        logger.info(f"Speed improvement: {first_time/second_time:.2f}x faster")
    else:
        logger.info(f"Speed improvement: N/A (second retrieval was too fast to measure)")
    logger.info(f"Results match: {result1 == result2}")
    
    # Test local cache eviction after TTL
    local_ttl_key = "test:local:short_ttl"
    local_ttl_value = f"Local cache test with short TTL {random.randint(1000, 9999)}"
    
    logger.info(f"Setting key with 3s local TTL: {local_ttl_key}")
    cache.config.local_ttl = 3  # Set short local TTL for testing
    await cache.set(local_ttl_key, local_ttl_value)
    
    # First retrieval - should be fast from local
    start_time = time.time()
    local_result1 = await cache.get(local_ttl_key)
    local_first_time = time.time() - start_time
    logger.info(f"First local retrieval took {local_first_time:.6f} seconds")
    
    # Wait for local TTL to expire
    logger.info("Waiting for local cache TTL to expire...")
    await asyncio.sleep(4)
    
    # Second retrieval - should come from Redis since local is expired
    start_time = time.time()
    local_result2 = await cache.get(local_ttl_key)
    local_second_time = time.time() - start_time
    logger.info(f"Second local retrieval took {local_second_time:.6f} seconds")
    
    # Restore default local TTL
    cache.config.local_ttl = 60
    
    logger.info(f"Local cache evicted: {local_second_time > local_first_time}")
    logger.info(f"Results still match: {local_result1 == local_result2}")


@cached(ttl=60, prefix="example")
async def cached_function(a: int, b: int) -> int:
    """Example function with caching using the cached decorator"""
    # Simulate a time-consuming computation
    logger.info(f"Computing result for cached_function({a}, {b})...")
    await asyncio.sleep(1)  # Simulate work
    return a + b


class CachedService:
    """Example service class with cached methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    @cached(ttl=60, prefix="service")
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with caching"""
        logger.info(f"Service {self.name} processing data {data}...")
        await asyncio.sleep(1)  # Simulate work
        
        # Example data processing
        result = {
            "service": self.name,
            "input": data,
            "processed_at": datetime.now().isoformat(),
            "result": {k: v * 2 if isinstance(v, (int, float)) else v for k, v in data.items()}
        }
        
        return result


async def test_cached_decorator():
    """Test the cached decorator functionality"""
    logger.info("\n\n=== Testing Cached Decorator ===")
    
    # Test with simple function
    logger.info("Testing cached decorator with simple function")
    
    # First call - should execute the function
    logger.info("First call to cached_function(10, 20)...")
    start_time = time.time()
    result1 = await cached_function(10, 20)
    first_time = time.time() - start_time
    logger.info(f"First call result: {result1}, took {first_time:.4f} seconds")
    
    # Second call with same parameters - should use cache
    logger.info("Second call to cached_function(10, 20)...")
    start_time = time.time()
    result2 = await cached_function(10, 20)
    second_time = time.time() - start_time
    logger.info(f"Second call result: {result2}, took {second_time:.4f} seconds")
    
    # Call with different parameters - should execute function
    logger.info("Call with different parameters cached_function(20, 30)...")
    start_time = time.time()
    result3 = await cached_function(20, 30)
    third_time = time.time() - start_time
    logger.info(f"Different params call result: {result3}, took {third_time:.4f} seconds")
    
    # Report performance improvement
    if second_time > 0:
        logger.info(f"Cache performance improvement: {first_time/second_time:.2f}x faster")
    
    # Test with class method
    logger.info("\nTesting cached decorator with class methods")
    
    # Create service instance
    service = CachedService("TestService")
    
    # First call to method
    test_data = {"value1": 10, "value2": 20, "text": "test"}
    logger.info("First call to service.process_data()...")
    start_time = time.time()
    service_result1 = await service.process_data(test_data)
    service_first_time = time.time() - start_time
    logger.info(f"First call took {service_first_time:.4f} seconds")
    
    # Second call with same data - should use cache
    logger.info("Second call to service.process_data()...")
    start_time = time.time()
    service_result2 = await service.process_data(test_data)
    service_second_time = time.time() - start_time
    logger.info(f"Second call took {service_second_time:.4f} seconds")
    
    # Different service instance but same method and parameters
    # Should still use the cache since self is excluded from key generation
    logger.info("Call with different service instance...")
    service2 = CachedService("AnotherService")
    start_time = time.time()
    service_result3 = await service2.process_data(test_data)
    service_third_time = time.time() - start_time
    logger.info(f"Different instance call took {service_third_time:.4f} seconds")
    
    # Results should match for first and second calls, but not for the
    # different service instance (since the service name is included in the result)
    logger.info(f"Results match for same instance: {service_result1 == service_result2}")
    logger.info(f"Results match for different instances: {service_result1 == service_result3}")


async def test_health_check(cache: CacheService):
    """Test health check and statistics"""
    logger.info("\n\n=== Testing Health Check and Statistics ===")
    
    # Run a health check
    logger.info("Running health check...")
    healthy, details = await cache.health_check()
    
    logger.info(f"Cache health: {'Healthy' if healthy else 'Unhealthy'}")
    logger.info(f"Health details: {json.dumps(details, indent=2)}")
    
    # Get cache statistics
    logger.info("\nCache statistics:")
    stats = calculate_cache_stats()
    logger.info(f"  Hits: {stats['hits']}")
    logger.info(f"  Misses: {stats['misses']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info(f"  Total operations: {stats['operations']}")
    logger.info(f"  Hit rate: {stats['hit_rate']:.2f}%")
    logger.info(f"  Average operation time: {stats['avg_time_ms']:.4f} ms")
    
    # Local cache information
    logger.info(f"\nLocal cache size: {len(cache._local_cache)} items")
    logger.info(f"Circuit breaker state: {'Open' if cache._circuit_open else 'Closed'}")
    logger.info(f"Error count: {cache._error_count}")


async def test_sync_async_compatibility():
    """Test compatibility between sync and async methods"""
    logger.info("\n\n=== Testing Sync/Async Compatibility ===")
    
    # Get the global cache instance
    cache = get_cache()
    
    # Test async method
    key = "test:compatibility:async"
    value = {"method": "async", "time": datetime.now().isoformat()}
    
    logger.info(f"Setting key with async method: {key}")
    await cache.set(key, value)
    
    # Get with sync method
    logger.info(f"Getting key with sync method: {key}")
    sync_result = cache.get_sync(key)
    logger.info(f"Sync retrieval result: {sync_result}")
    
    # Test sync method
    key = "test:compatibility:sync"
    value = {"method": "sync", "time": datetime.now().isoformat()}
    
    logger.info(f"Setting key with sync method: {key}")
    cache.set_sync(key, value)
    
    # Get with async method
    logger.info(f"Getting key with async method: {key}")
    async_result = await cache.get(key)
    logger.info(f"Async retrieval result: {async_result}")


async def test_legacy_redis_cache():
    """Test the legacy RedisCache compatibility layer"""
    logger.info("\n\n=== Testing Legacy RedisCache Compatibility ===")
    
    # Create a RedisCache instance (which uses CacheService internally)
    redis_cache = RedisCache(prefix="legacy")
    
    # Test basic operations
    key = "test:legacy:basic"
    value = {"legacy": True, "timestamp": datetime.now().isoformat()}
    
    # Set using legacy methods
    logger.info(f"Setting key with legacy RedisCache: {key}")
    # Test both async and sync methods
    await redis_cache.set(key, value)
    
    # Use deprecated method (should show warning)
    logger.info("Using deprecated get_value method (should show warning)...")
    legacy_result = redis_cache.get_value(key)
    logger.info(f"Legacy retrieval result: {legacy_result}")
    
    # Check accessing the underlying cache service
    logger.info("Accessing the underlying CacheService...")
    cache_service = redis_cache.cache_service
    modern_result = await cache_service.get(f"legacy:{key}")
    logger.info(f"Modern retrieval result: {modern_result}")
    
    # Verify results match
    logger.info(f"Results match: {legacy_result == modern_result}")


async def main():
    """Main async function to run all tests"""
    logger.info("Starting Cache Service Usage Example")
    
    # Configure Redis for the test - Explicitly using Upstash settings
    redis_config = RedisConfig(
        # Use explicit Upstash Redis URL - this would normally come from env vars
        url="redis://default:dd1c03a51f644a7b9aef7b6f9b59a8e6@us1-gentle-rodent-40634.upstash.io:6379",
        key_prefix="example",
        enable_compression=True,
        socket_timeout=10.0,  # Increased timeout for cloud Redis
        socket_connect_timeout=5.0,  # Increased connect timeout
        retry_on_timeout=True,
        max_connections=10,
        local_cache_size=100,
        local_ttl=60,
        health_check_interval=30,
        ssl=True,  # Enable SSL for Upstash Redis
        # Increase retry settings for better stability with remote Redis
        connection_retries=5,
        circuit_breaker_threshold=10
    )
    
    # Create cache service instance
    try:
        cache = CacheService(redis_config)
        logger.info(f"Initialized CacheService with configuration: {redis_config}")
        
        # Run all tests
        try:
            # Basic tests
            await test_basic_operations(cache)
            
            # Test compression
            await test_compression(cache)
            
            # Test local cache
            await test_local_cache(cache)
            
            # Test cached decorator
            await test_cached_decorator()
            
            # Test error handling and circuit breaker
            await test_error_handling(cache)
            
            # Test sync/async compatibility 
            await test_sync_async_compatibility()
            
            # Test legacy RedisCache compatibility
            await test_legacy_redis_cache()
            
            # Health check and statistics at the end
            await test_health_check(cache)
            
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}", exc_info=True)
        finally:
            # Clean up
            logger.info("Cleaning up...")
            await cache.flush()
    except Exception as e:
        logger.error(f"Error initializing CacheService: {str(e)}", exc_info=True)
    
    logger.info("Cache Service Usage Example Completed")


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
