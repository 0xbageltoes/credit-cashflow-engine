from app.core.config import settings
from redis import Redis
import ssl

def test_redis_connection():
    try:
        redis_client = Redis.from_url(
            settings.REDIS_URL_WITH_SSL,
            decode_responses=True,
            ssl=True,
            ssl_cert_reqs=None
        )
        
        # Test set
        redis_client.set('test_key', 'test_value')
        
        # Test get
        value = redis_client.get('test_key')
        print(f"Retrieved value: {value}")
        
        # Test delete
        redis_client.delete('test_key')
        print("Redis connection test successful!")
        
    except Exception as e:
        print(f"Redis connection error: {str(e)}")

if __name__ == "__main__":
    test_redis_connection()
