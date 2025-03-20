"""
Test script to verify Redis connection to Upstash
This will help diagnose any issues with the Redis connection
"""
import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_tester")

# Add app directory to path to allow importing app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Redis service
from app.services.unified_redis_service import UnifiedRedisService, RedisConfig

def main():
    """Test Redis connection with detailed logging"""
    try:
        logger.info("Starting Redis connection test")
        
        # Check environment variables
        env = os.environ.get("ENV", "").lower()
        if not env:
            env = os.environ.get("ENVIRONMENT", "development").lower()
        
        logger.info(f"Current environment: {env}")
        
        # Check Upstash Redis environment variables
        upstash_host = os.environ.get("UPSTASH_REDIS_HOST")
        upstash_port = os.environ.get("UPSTASH_REDIS_PORT")
        upstash_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
        
        if upstash_host:
            logger.info(f"Found Upstash Redis host: {upstash_host}")
        else:
            logger.warning("No Upstash Redis host found in environment variables")
            
        if upstash_port:
            logger.info(f"Found Upstash Redis port: {upstash_port}")
        else:
            logger.info("No Upstash Redis port found, will use default 6379")
            
        if upstash_password:
            logger.info("Found Upstash Redis password (masked)")
        else:
            logger.warning("No Upstash Redis password found in environment variables")
        
        # Create Redis configuration
        logger.info("Creating Redis configuration")
        config = RedisConfig()
        
        # Check if URL was created
        if config.url:
            safe_url = config.url
            if "@" in safe_url:
                parts = safe_url.split("@")
                auth_parts = parts[0].split(":")
                if len(auth_parts) > 2:  # protocol:username:password
                    safe_url = f"{auth_parts[0]}:{auth_parts[1]}:[MASKED]@{parts[1]}"
                else:  # protocol:password
                    safe_url = f"{auth_parts[0]}:[MASKED]@{parts[1]}"
            logger.info(f"Redis URL configured: {safe_url}")
        else:
            logger.info(f"Redis URL not configured, using direct connection to {config.host}:{config.port}")
        
        # Force production mode
        os.environ["ENV"] = "production"
        logger.info("Temporarily set ENV=production for testing")
        
        # Create Redis service
        logger.info("Initializing Redis service")
        redis_service = UnifiedRedisService(config)
        
        # Test connection
        logger.info("Testing Redis ping...")
        ping_result = redis_service.ping_sync()
        logger.info(f"Ping result: {ping_result}")
        
        if ping_result:
            logger.info("✅ Successfully connected to Redis")
            
            # Try setting and getting a value
            test_key = "test:connection:key"
            test_value = {"timestamp": time.time(), "test": True}
            
            logger.info(f"Setting test key: {test_key}")
            set_result = redis_service.set_sync(test_key, test_value, ttl=60)
            logger.info(f"Set result: {set_result}")
            
            logger.info(f"Getting test key: {test_key}")
            get_result = redis_service.get_sync(test_key)
            logger.info(f"Get result: {get_result}")
            
            if get_result and get_result.get("test") is True:
                logger.info("✅ Successfully set and retrieved test value")
            else:
                logger.error("❌ Failed to set and retrieve test value")
        else:
            logger.error("❌ Failed to connect to Redis")
        
        # Check health
        logger.info("Getting Redis health status")
        health = redis_service.health_check()
        logger.info(f"Health status: {health}")
        
    except Exception as e:
        logger.error(f"Error during Redis connection test: {e}", exc_info=True)

if __name__ == "__main__":
    main()
