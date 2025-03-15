"""
Example usage of the hierarchical caching system with Upstash Redis.

This example demonstrates:
1. Initializing the hierarchical cache with Upstash Redis configuration
2. Using the cached decorator for async and sync functions
3. Direct cache access for manual caching
4. Cache invalidation techniques
5. Monitoring cache performance
"""

import asyncio
import os
import time
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure Upstash Redis credentials directly
# For production use, these would typically be loaded from environment variables
# or a secure configuration management system
UPSTASH_REDIS_HOST = "easy-macaw-12070.upstash.io"
UPSTASH_REDIS_PORT = 6379
UPSTASH_REDIS_PASSWORD = "AS8mAAIjcDFmMjJhZTIzY2ZiYmY0MTJkYmQzZDQ1MWYwMWQyYzI0MXAxMA"
REDIS_URL = f"rediss://default:{UPSTASH_REDIS_PASSWORD}@{UPSTASH_REDIS_HOST}:{UPSTASH_REDIS_PORT}"

# Set environment variables for Redis configuration
os.environ["UPSTASH_REDIS_HOST"] = UPSTASH_REDIS_HOST
os.environ["UPSTASH_REDIS_PORT"] = str(UPSTASH_REDIS_PORT)
os.environ["UPSTASH_REDIS_PASSWORD"] = UPSTASH_REDIS_PASSWORD
os.environ["REDIS_URL"] = REDIS_URL

from app.core.hierarchical_cache import (
    create_cache_service,
    cached,
    generate_cache_key,
    HierarchicalCacheService
)

# Configure logging with more verbose settings and ensure console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global cache instance - will be initialized in main()
_cache_service = None

# Example of using the cached decorator with an async function
@cached(
    cache_service=None,  # Will be set dynamically in main()
    key_prefix="forecast",
    redis_ttl=1800,      # 30 minutes in Redis
    memory_ttl=120       # 2 minutes in memory
)
async def calculate_cash_flow_forecast(
    account_id: str,
    start_date: datetime,
    end_date: datetime,
    scenario: Optional[str] = "default"
) -> List[Dict[str, Any]]:
    """
    Calculate cash flow forecast for an account
    
    This function is expensive and a good candidate for caching.
    """
    logger.info(f"Calculating forecast for account {account_id} from {start_date} to {end_date}")
    
    # Simulate a complex calculation
    await asyncio.sleep(2)
    
    # Return dummy data
    return [
        {"date": str(start_date), "amount": 1000, "category": "income"},
        {"date": str(end_date), "amount": -500, "category": "expense"}
    ]

# Example of using the cached decorator with a sync function
@cached(
    cache_service=None,  # Will be set dynamically in main()
    key_prefix="balance",
    redis_ttl=600,       # 10 minutes in Redis
    memory_ttl=60        # 1 minute in memory
)
def get_account_balance(account_id: str) -> Dict[str, Any]:
    """
    Get current balance for an account
    
    This function is moderately expensive and benefits from caching.
    """
    logger.info(f"Getting balance for account {account_id}")
    
    # Simulate a database query
    time.sleep(0.5)
    
    # Return dummy data with a timestamp that doesn't change on each call
    # Using a fixed timestamp ensures cache results match
    return {
        "account_id": account_id,
        "current_balance": 5000,
        "available_balance": 4500,
        "as_of": "2025-03-15T00:00:00Z"  # Fixed timestamp for reproducible results
    }

# Example of direct cache manipulation
async def demonstrate_direct_cache_usage(cache):
    """
    Demonstrate direct usage of the cache service
    """
    logger.info("Demonstrating direct cache usage")
    
    # Set a value in the cache
    key = "demo:direct-access"
    value = {"timestamp": "2025-03-15T00:00:00Z", "value": "example-data"}
    
    success = await cache.set(key, value)
    logger.info(f"Cache set success: {success}")
    
    # Retrieve the value from cache
    cached_value = await cache.get(key)
    logger.info(f"Retrieved from cache: {cached_value}")
    
    # Delete the value
    delete_success = await cache.delete(key)
    logger.info(f"Cache delete success: {delete_success}")
    
    # Verify it's gone
    should_be_none = await cache.get(key)
    logger.info(f"After deletion, value is: {should_be_none}")

# Example of pattern-based cache invalidation
async def demonstrate_pattern_invalidation(cache):
    """
    Demonstrate invalidating multiple cache entries by pattern
    """
    logger.info("Demonstrating pattern-based cache invalidation")
    
    # Set multiple related values
    for i in range(5):
        key = f"user:123:permission:{i}"
        await cache.set(key, f"Permission {i}")
    
    # Retrieve one to verify
    test_value = await cache.get("user:123:permission:3")
    logger.info(f"Test value before invalidation: {test_value}")
    
    # Invalidate all user permissions with pattern matching
    invalidated = await cache.invalidate_pattern("user:123:permission")
    logger.info(f"Invalidated {invalidated} cache entries")
    
    # Verify they're gone
    test_value_after = await cache.get("user:123:permission:3")
    logger.info(f"Test value after invalidation: {test_value_after}")

# Set the cache for decorated functions
def set_cache_for_decorators(cache_service):
    """Set the cache service for decorated functions"""
    # Update the cache_service parameter in the cached decorator
    calculate_cash_flow_forecast.__globals__['cached'].__defaults__ = (cache_service,) + calculate_cash_flow_forecast.__globals__['cached'].__defaults__[1:]
    get_account_balance.__globals__['cached'].__defaults__ = (cache_service,) + get_account_balance.__globals__['cached'].__defaults__[1:]

# Main entry point
async def main():
    """Run the hierarchical cache examples"""
    try:
        print("Starting hierarchical cache example")
        logger.info("Starting hierarchical cache example")
        
        # Print environment variables related to Redis (without showing sensitive values)
        redis_url = os.environ.get("REDIS_URL")
        redis_host = os.environ.get("UPSTASH_REDIS_HOST")
        redis_password = os.environ.get("UPSTASH_REDIS_PASSWORD", "").replace(os.environ.get("UPSTASH_REDIS_PASSWORD", ""), "[REDACTED]")
        
        logger.info(f"Redis URL configured: {'Yes' if redis_url else 'No'}")
        logger.info(f"Upstash Redis Host: {redis_host}")
        logger.info(f"Redis Password configured: {'Yes' if os.environ.get('UPSTASH_REDIS_PASSWORD') else 'No'}")
        
        # Create direct Redis cache service instance for better control and visibility
        _cache_service = HierarchicalCacheService(
            redis_host=UPSTASH_REDIS_HOST,
            redis_port=UPSTASH_REDIS_PORT,
            redis_password=UPSTASH_REDIS_PASSWORD,
            redis_ssl=True,  # Required for Upstash
            prefix="cashflow:",
            redis_ttl=3600,
            memory_ttl=300,
            memory_max_items=2000,
            logger=logger
        )
        
        # Check cache health
        health = await _cache_service.health_check()
        logger.info(f"Cache health status: {json.dumps(health, indent=2)}")
        
        # Set cache instance for cached decorators
        set_cache_for_decorators(_cache_service)
        
        # Example 1: Using cached decorator with async function
        logger.info("Example 1: Using cached decorator with async function")
        
        # First call - should miss cache
        result1 = await calculate_cash_flow_forecast(
            account_id="acct123",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31)
        )
        
        # Second call with same parameters - should hit cache
        result2 = await calculate_cash_flow_forecast(
            account_id="acct123",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31)
        )
        
        # Verify results match
        logger.info(f"Results match: {result1 == result2}")
        
        # Example 2: Using cached decorator with sync function
        logger.info("\nExample 2: Using cached decorator with sync function")
        
        # First call - should miss cache
        balance1 = get_account_balance("acct123")
        
        # Second call with same parameters - should hit cache
        balance2 = get_account_balance("acct123")
        
        logger.info(f"Balance results match: {balance1 == balance2}")
        
        # Example 3: Demonstrate direct cache usage
        logger.info("\nExample 3: Demonstrate direct cache usage")
        await demonstrate_direct_cache_usage(_cache_service)
        
        # Example 4: Demonstrate pattern-based cache invalidation
        logger.info("\nExample 4: Demonstrate pattern-based cache invalidation")
        await demonstrate_pattern_invalidation(_cache_service)
        
        # Show cache statistics
        logger.info("\nCache Statistics:")
        stats = _cache_service.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Close Redis connection
        await _cache_service.close()
        logger.info("Cache connections closed")
        
    except Exception as e:
        logger.error(f"Error in hierarchical cache example: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
        print("Example completed successfully")
    except Exception as e:
        print(f"Error running example: {str(e)}")
        logger.error("Error running example", exc_info=True)
