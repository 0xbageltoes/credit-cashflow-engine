"""Cache Service Usage Example

This example demonstrates how to use the unified CacheService in various contexts
within the credit cashflow engine application, including:

1. Basic caching operations
2. Using the cached decorator
3. Error handling and fallback strategies
4. Performance comparison with and without caching
"""

import os
import sys
import asyncio
import time
import random
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.core.cache_service import (
    CacheService, 
    RedisConfig, 
    cached,
    calculate_cache_stats,
    get_cache
)
from app.core.cache_error_handling import (
    CircuitBreaker,
    FallbackStrategy,
    with_circuit_breaker,
    safe_cache_operation
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: Basic caching with the new CacheService
async def basic_caching_example():
    """Demonstrate basic caching operations"""
    print("\n=== Example 1: Basic Caching Operations ===")
    
    # Create cache service with custom configuration
    cache = CacheService(
        RedisConfig(
            # Default configuration from environment variables
            key_prefix="example",
            default_compression=True,
            socket_timeout=3.0,
            retry_on_timeout=True
        )
    )
    
    # Test cache operations
    key = "test_key"
    value = {
        "timestamp": datetime.now().isoformat(),
        "data": [random.random() for _ in range(5)],
        "metadata": {
            "source": "example",
            "version": "1.0"
        }
    }
    
    print(f"Setting value for key '{key}'...")
    success = await cache.set(key, value, ttl=60)
    print(f"Cache set: {success}")
    
    print(f"Getting value for key '{key}'...")
    result = await cache.get(key)
    print(f"Retrieved value: {result}")
    
    print(f"Checking if key '{key}' exists...")
    exists = await cache.exists(key)
    print(f"Key exists: {exists}")
    
    # Test memory caching
    print("\nTesting memory caching (faster access)...")
    start = time.time()
    for _ in range(100):
        await cache.get(key)
    redis_time = time.time() - start
    print(f"Time to access Redis 100 times: {redis_time:.4f}s")
    
    # Now set with memory TTL
    await cache.set(key, value, ttl=60, memory_ttl=60)
    
    start = time.time()
    for _ in range(100):
        await cache.get(key)
    memory_time = time.time() - start
    print(f"Time to access memory cache 100 times: {memory_time:.4f}s")
    print(f"Memory cache is {redis_time/memory_time:.1f}x faster")
    
    print("\nTesting compression...")
    # Create a large value
    large_value = {
        "data": [random.random() for _ in range(10000)],
        "text": "A" * 10000  # 10KB of text
    }
    
    # Set with and without compression
    await cache.set("large_uncompressed", large_value, compress=False)
    await cache.set("large_compressed", large_value, compress=True)
    
    # Get cache stats to see size difference
    stats = calculate_cache_stats(cache, key_pattern="example:large_*")
    print(f"Cache stats: {stats}")
    
    # Clean up
    await cache.delete(key)
    await cache.delete("large_uncompressed")
    await cache.delete("large_compressed")


# Example 2: Using the cached decorator
class DataService:
    """Example service that performs expensive operations"""
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service or get_cache()
    
    @cached(ttl=60)
    async def fetch_data(self, user_id: str, dataset: str, cache: CacheService = None) -> Dict[str, Any]:
        """Simulate fetching data from a database"""
        print(f"Cache miss - Fetching data for user {user_id}, dataset {dataset}")
        
        # Simulate database query delay
        await asyncio.sleep(0.5)
        
        # Generate random data
        return {
            "user_id": user_id,
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "values": [random.random() for _ in range(10)]
        }
    
    @cached(ttl=120, memory_ttl=60, compress=True)
    def compute_statistics(self, data_points: List[float], cache_service: CacheService = None) -> Dict[str, float]:
        """Compute statistics on data points"""
        print(f"Cache miss - Computing statistics on {len(data_points)} data points")
        
        # Simulate computationally expensive operation
        time.sleep(0.2)
        
        return {
            "mean": np.mean(data_points),
            "median": np.median(data_points),
            "std_dev": np.std(data_points),
            "min": min(data_points),
            "max": max(data_points),
            "timestamp": datetime.now().isoformat()
        }


async def cached_decorator_example():
    """Demonstrate using the cached decorator"""
    print("\n=== Example 2: Using the @cached Decorator ===")
    
    # Create a service instance with cache
    cache = CacheService(RedisConfig(key_prefix="example"))
    service = DataService(cache)
    
    # First call - should miss cache
    print("\nFirst call - should miss cache:")
    result1 = await service.fetch_data("user1", "transactions")
    print(f"Result timestamp: {result1['timestamp']}")
    
    # Second call with same parameters - should hit cache
    print("\nSecond call with same parameters - should hit cache:")
    result2 = await service.fetch_data("user1", "transactions")
    print(f"Result timestamp: {result2['timestamp']}")
    
    if result1['timestamp'] == result2['timestamp']:
        print("Cache hit confirmed! Both timestamps match.")
    
    # Call with different parameters - should miss cache
    print("\nCall with different parameters - should miss cache:")
    result3 = await service.fetch_data("user1", "profile")
    print(f"Result timestamp: {result3['timestamp']}")
    
    # Test with synchronous cached method
    print("\nTesting synchronous cached method:")
    data = [random.random() for _ in range(100)]
    
    # First call - should miss cache
    print("First call - should miss cache:")
    stats1 = service.compute_statistics(data)
    print(f"Statistics timestamp: {stats1['timestamp']}")
    
    # Second call with same data - should hit cache
    print("Second call with same data - should hit cache:")
    stats2 = service.compute_statistics(data)
    print(f"Statistics timestamp: {stats2['timestamp']}")
    
    if stats1['timestamp'] == stats2['timestamp']:
        print("Cache hit confirmed! Both timestamps match.")


# Example 3: Error handling and fallbacks
async def error_handling_example():
    """Demonstrate error handling and circuit breaker pattern"""
    print("\n=== Example 3: Error Handling and Fallbacks ===")
    
    # Create a circuit breaker
    circuit_breaker = CircuitBreaker(
        error_threshold=3,
        recovery_timeout=5
    )
    
    # Create a simulated "faulty" Redis client
    class FaultyRedisService:
        def __init__(self, failure_rate: float = 0.7):
            self.failure_rate = failure_rate
            self.call_count = 0
        
        @with_circuit_breaker(circuit_breaker, fallback_result=None)
        async def get_data(self, key: str) -> Optional[Dict]:
            """Simulate a Redis operation that sometimes fails"""
            self.call_count += 1
            
            # Simulate failures
            if random.random() < self.failure_rate:
                print(f"Call {self.call_count}: Simulating Redis failure")
                raise Exception("Redis connection error")
            
            print(f"Call {self.call_count}: Redis operation successful")
            return {"key": key, "value": random.random()}
        
        @safe_cache_operation(fallback_result={"status": "fallback"})
        async def safe_operation(self, key: str) -> Dict:
            """Operation with safe fallback"""
            if random.random() < self.failure_rate:
                print(f"Safe operation: Simulating Redis failure")
                raise Exception("Redis connection error")
            
            print(f"Safe operation: Redis operation successful")
            return {"key": key, "value": random.random(), "status": "success"}
    
    # Use the faulty service
    faulty_service = FaultyRedisService()
    
    # Make several calls to demonstrate circuit breaker
    print("\nTesting circuit breaker pattern:")
    for i in range(10):
        result = await faulty_service.get_data(f"key{i}")
        print(f"Result {i}: {result}")
        await asyncio.sleep(0.5)
    
    # Circuit should be open now, wait for recovery
    print("\nCircuit should be open now. Waiting for recovery...")
    await asyncio.sleep(5)
    
    # Try again after recovery timeout
    print("\nTrying again after recovery timeout:")
    for i in range(5):
        result = await faulty_service.get_data(f"key{i+10}")
        print(f"Result {i+10}: {result}")
        await asyncio.sleep(0.5)
    
    # Test safe operation with fallback
    print("\nTesting safe operation with fallback:")
    for i in range(5):
        result = await faulty_service.safe_operation(f"safe_key{i}")
        print(f"Safe operation result {i}: {result}")


# Example 4: Performance comparison
async def performance_comparison():
    """Compare performance with and without caching"""
    print("\n=== Example 4: Performance Comparison ===")
    
    # Create cache service
    cache = CacheService(RedisConfig(key_prefix="perf_test"))
    
    # Function that simulates an expensive calculation
    async def calculate_without_cache(size: int) -> float:
        """Expensive calculation without caching"""
        # Simulate complex calculation
        await asyncio.sleep(0.1)  # Fixed overhead
        
        # Create a random matrix and perform operations
        matrix = np.random.rand(size, size)
        for _ in range(3):
            matrix = matrix @ matrix  # Matrix multiplication
        
        return float(np.mean(matrix))
    
    # Same function with caching
    @cached(ttl=60)
    async def calculate_with_cache(size: int, cache: CacheService) -> float:
        """Same expensive calculation with caching"""
        # Same calculation as above
        await asyncio.sleep(0.1)  # Fixed overhead
        
        # Create a random matrix and perform operations
        matrix = np.random.rand(size, size)
        for _ in range(3):
            matrix = matrix @ matrix  # Matrix multiplication
        
        return float(np.mean(matrix))
    
    # Test parameters
    sizes = [10, 20, 30, 40, 50]  # Matrix sizes
    repeats = 3  # Number of repeats for each size
    
    # Results storage
    times_without_cache = []
    times_with_cache = []
    times_with_cache_first_call = []
    
    # Run tests
    for size in sizes:
        print(f"\nTesting with matrix size {size}x{size}:")
        
        # Test without cache
        start = time.time()
        for _ in range(repeats):
            await calculate_without_cache(size)
        time_without_cache = (time.time() - start) / repeats
        times_without_cache.append(time_without_cache)
        print(f"Average time without cache: {time_without_cache:.4f}s")
        
        # Test with cache - first call
        start = time.time()
        await calculate_with_cache(size, cache)
        time_first_call = time.time() - start
        times_with_cache_first_call.append(time_first_call)
        print(f"Time with cache (first call): {time_first_call:.4f}s")
        
        # Test with cache - subsequent calls
        start = time.time()
        for _ in range(repeats):
            await calculate_with_cache(size, cache)
        time_with_cache = (time.time() - start) / repeats
        times_with_cache.append(time_with_cache)
        print(f"Average time with cache (subsequent calls): {time_with_cache:.4f}s")
        print(f"Speedup factor: {time_without_cache / time_with_cache:.1f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_without_cache, 'o-', label='Without Cache')
    plt.plot(sizes, times_with_cache_first_call, 's-', label='With Cache (First Call)')
    plt.plot(sizes, times_with_cache, '^-', label='With Cache (Subsequent Calls)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: Cached vs. Uncached')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('cache_performance_comparison.png')
    print("\nPerformance comparison plot saved as 'cache_performance_comparison.png'")
    
    # Print summary statistics
    print("\nPerformance Summary:")
    avg_speedup = np.mean([t1/t2 for t1, t2 in zip(times_without_cache, times_with_cache)])
    print(f"Average speedup with caching: {avg_speedup:.1f}x")
    print(f"Maximum speedup with caching: {max([t1/t2 for t1, t2 in zip(times_without_cache, times_with_cache)]):.1f}x")


async def main():
    """Run all examples"""
    print("=== Cache Service Usage Examples ===")
    print("This demo shows how to use the unified CacheService in various contexts")
    
    # Run all examples
    await basic_caching_example()
    await cached_decorator_example()
    await error_handling_example()
    await performance_comparison()
    
    print("\n=== All examples completed ===")


if __name__ == "__main__":
    # Run the event loop
    asyncio.run(main())
