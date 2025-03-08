import json
import zlib
import asyncio
from typing import Any, Optional
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from app.core.config import settings

class RedisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                print(f"Connecting to Redis using URL: {settings.REDIS_URL}")
                if settings.REDIS_URL.startswith('rediss://'):
                    # For Upstash Redis connections with SSL
                    cls._instance.client = Redis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True,
                        ssl_cert_reqs=None  # Disable SSL verification for Upstash Redis
                    )
                    # Also create an async client
                    cls._instance.async_client = AsyncRedis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True,
                        ssl_cert_reqs=None
                    )
                else:
                    cls._instance.client = Redis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True
                    )
                    # Also create an async client
                    cls._instance.async_client = AsyncRedis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True
                    )
                
                # Test the connection
                if cls._instance.client.ping():
                    print(f"Successfully connected to Redis at {settings.REDIS_URL}")
                else:
                    print("Failed to ping Redis server, setting client to None")
                    cls._instance.client = None
                    cls._instance.async_client = None
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                # Create a dummy client for testing that doesn't error out
                cls._instance.client = None
                cls._instance.async_client = None
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache and decompress if needed (synchronous)"""
        try:
            if self.client is None:
                return None
            data = self.get_raw(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Error getting from cache: {e}")
        return None
        
    async def get_async(self, key: str) -> Optional[Any]:
        """Get a value from cache and decompress if needed (asynchronous)"""
        try:
            if self.async_client is None:
                return None
            data = await self.get_raw_async(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Error getting from cache asynchronously: {e}")
        return None

    def set(self, key: str, value: Any, expire: int = None, compress: bool = False) -> bool:
        """Set a value in cache with optional compression (synchronous)"""
        try:
            if self.client is None:
                return False
            data = json.dumps(value)
            if compress:
                data = self._compress(data)
            if expire:
                return self.client.setex(key, expire, data)
            else:
                return self.client.set(key, data)
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
            
    async def set_async(self, key: str, value: Any, expire: int = None, compress: bool = False) -> bool:
        """Set a value in cache with optional compression (asynchronous)"""
        try:
            if self.async_client is None:
                return False
            data = json.dumps(value)
            if compress:
                data = self._compress(data)
            if expire:
                return await self.async_client.setex(key, expire, data)
            else:
                return await self.async_client.set(key, data)
        except Exception as e:
            print(f"Error setting cache asynchronously: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache (synchronous)"""
        try:
            if self.client is None:
                return False
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
            
    async def delete_async(self, key: str) -> bool:
        """Delete a key from cache (asynchronous)"""
        try:
            if self.async_client is None:
                return False
            return bool(await self.async_client.delete(key))
        except Exception as e:
            print(f"Error deleting from cache asynchronously: {e}")
            return False
            
    # Async versions of task-specific methods
    async def set_task_status(self, task_id: str, status: dict) -> bool:
        """Set task status in Redis (asynchronous)"""
        key = f"task_status:{task_id}"
        return await self.set_async(key, status, expire=86400)  # Expire after 24 hours
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get task status from Redis (asynchronous)"""
        key = f"task_status:{task_id}"
        return await self.get_async(key)
    
    async def set_forecast_result(self, task_id: str, result: dict) -> bool:
        """Set forecast result in Redis (asynchronous)"""
        key = f"forecast_result:{task_id}"
        return await self.set_async(key, result, expire=604800)  # Expire after 7 days
    
    async def get_forecast_result(self, task_id: str) -> Optional[dict]:
        """Get forecast result from Redis (asynchronous)"""
        key = f"forecast_result:{task_id}"
        return await self.get_async(key)
        
    # Helper methods for raw operations
    def get_raw(self, key: str) -> Optional[str]:
        """Get raw value from Redis (synchronous)"""
        try:
            if self.client is None:
                return None
            data = self.client.get(key)
            if data and data.startswith("compressed:"):
                # Decompress data
                return self._decompress(data[11:])  # Skip "compressed:" prefix
            return data
        except Exception as e:
            print(f"Error getting raw data from cache: {e}")
            return None
            
    async def get_raw_async(self, key: str) -> Optional[str]:
        """Get raw value from Redis (asynchronous)"""
        try:
            if self.async_client is None:
                return None
            data = await self.async_client.get(key)
            if data and data.startswith("compressed:"):
                # Decompress data
                return self._decompress(data[11:])  # Skip "compressed:" prefix
            return data
        except Exception as e:
            print(f"Error getting raw data from cache asynchronously: {e}")
            return None

    def _compress(self, data: str) -> str:
        """Compress string data"""
        try:
            compressed = zlib.compress(data.encode())
            return f"compressed:{compressed.hex()}"
        except Exception as e:
            print(f"Error compressing data: {e}")
            return data

    def _decompress(self, data: str) -> str:
        """Decompress data"""
        try:
            binary_data = bytes.fromhex(data)
            return zlib.decompress(binary_data).decode()
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return data

    def clear(self) -> bool:
        """Clear all keys from cache (for testing)"""
        try:
            if self.client is None:
                return False
            return bool(self.client.flushdb())
        except Exception as e:
            print(f"Error clearing cache: {e}")
        return False

    def cleanup_websocket(self, connection_id: str) -> None:
        """Clean up WebSocket related data"""
        try:
            if self.client is None:
                return
            pattern = f"ws:{connection_id}:*"
            for key in self.client.scan_iter(match=pattern):
                self.delete(key)
        except Exception as e:
            print(f"Error cleaning up WebSocket: {e}")

    def clear_stale_forecasts(self, user_id: str, max_age: int = 86400) -> None:
        """Clear stale forecast results older than max_age seconds"""
        try:
            if self.client is None:
                return
            pattern = f"forecast_result:{user_id}:*"
            for key in self.client.scan_iter(match=pattern):
                if not self.client.ttl(key) or self.client.ttl(key) > max_age:
                    self.delete(key)
        except Exception as e:
            print(f"Error clearing stale forecasts: {e}")
