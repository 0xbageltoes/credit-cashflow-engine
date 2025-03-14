"""
Cache utilities for the credit-cashflow-engine application.

These utilities help with cache key generation, validation, and other cache-related
operations to ensure consistent and reliable caching across the application.
"""
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union
import time

logger = logging.getLogger(__name__)

def generate_cache_key(
    prefix: str,
    user_id: Optional[str] = None,
    data: Optional[Union[Dict, Any]] = None,
    additional_keys: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a cache key with consistent hashing
    
    Args:
        prefix: Prefix for the cache key
        user_id: Optional user ID to include in the key
        data: Optional data to include in the key hash
        additional_keys: Optional additional key-value pairs to include
        
    Returns:
        A cache key string
    """
    # Create a dictionary with all components that make the key unique
    key_dict = {}
    
    # Add user_id if provided
    if user_id:
        key_dict["user_id"] = user_id
    
    # Add data if provided (ensure it's serializable)
    if data:
        try:
            if hasattr(data, 'model_dump'):
                # For Pydantic v2 models
                key_dict["data"] = data.model_dump()
            elif hasattr(data, 'dict'):
                # For Pydantic v1 models
                key_dict["data"] = data.dict()
            elif isinstance(data, dict):
                # If already a dict
                key_dict["data"] = data
            else:
                # Attempt to convert to dict
                key_dict["data"] = vars(data)
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning(f"Could not convert data to dict for cache key: {e}")
            # Use string representation as fallback
            key_dict["data"] = str(data)
    
    # Add any additional keys
    if additional_keys:
        for k, v in additional_keys.items():
            key_dict[k] = v
    
    # Convert to JSON and create a hash
    try:
        # Sort keys for consistent ordering
        key_json = json.dumps(key_dict, sort_keys=True)
        hash_object = hashlib.md5(key_json.encode())
        cache_key = f"{prefix}:{hash_object.hexdigest()}"
        return cache_key
    except Exception as e:
        logger.warning(f"Error generating cache key: {e}")
        # Return a fallback key that won't match any existing cache
        return f"{prefix}:no_cache_{time.time()}"

def invalidate_cache_keys(
    redis_service,
    pattern: str
) -> int:
    """
    Invalidate all cache keys matching a pattern
    
    Args:
        redis_service: Redis service instance
        pattern: Pattern to match keys (e.g., "analytics:*")
        
    Returns:
        Number of keys invalidated
    """
    try:
        # Get all keys matching the pattern
        keys = redis_service.keys(pattern)
        
        # Delete all matched keys
        if keys:
            return redis_service.delete(*keys)
        return 0
    except Exception as e:
        logger.warning(f"Error invalidating cache keys: {e}")
        return 0

def format_cache_key_for_metrics(
    key: str
) -> str:
    """
    Format a cache key for metrics reporting
    
    This helps avoid creating too many unique metric names by
    removing the hash part of the key.
    
    Args:
        key: Original cache key
        
    Returns:
        Formatted key for metrics
    """
    if ":" not in key:
        return key
    
    # Return just the prefix part
    return key.split(":")[0]
