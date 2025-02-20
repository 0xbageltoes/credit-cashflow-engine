import sys
from app.core.config import settings
from redis import Redis

def test_redis_connection():
    try:
        print("Python version:", sys.version)
        print("Connecting to Redis...")
        print(f"Redis URL: {settings.REDIS_URL}")
        
        # Connect to Redis using Upstash configuration
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
        
        print("Connected to Redis successfully")
        
        # Test set
        print("Testing SET operation...")
        redis_client.set('test_key', 'test_value')
        print("SET operation successful")
        
        # Test get
        print("Testing GET operation...")
        value = redis_client.get('test_key')
        print(f"Retrieved value: {value}")
        
        # Test delete
        print("Testing DELETE operation...")
        redis_client.delete('test_key')
        print("DELETE operation successful")
        
        print("All Redis operations completed successfully!")
        
    except Exception as e:
        print(f"Redis connection error: {str(e)}", file=sys.stderr)
        import traceback
        print("Full error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    test_redis_connection()
