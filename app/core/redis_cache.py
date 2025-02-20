import json
import zlib
from typing import Any, Optional
from redis import Redis
from app.core.config import settings

class RedisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
        return cls._instance

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache and decompress if needed"""
        try:
            data = await self.get_raw(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Error getting from cache: {e}")
        return None

    async def get_raw(self, key: str) -> Optional[str]:
        """Get raw value from cache"""
        try:
            data = self.client.get(key)
            if data and isinstance(data, bytes):
                try:
                    # Try to decompress if it's compressed
                    return zlib.decompress(data).decode()
                except zlib.error:
                    # If decompression fails, it might be uncompressed data
                    return data.decode()
            return data
        except Exception as e:
            print(f"Error getting raw data from cache: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600, compress: bool = True) -> bool:
        """Set a value in cache with optional compression"""
        try:
            data = json.dumps(value)
            if compress:
                data = zlib.compress(data.encode())
            return bool(self.client.setex(key, ttl, data))
        except Exception as e:
            print(f"Error setting cache: {e}")
        return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting from cache: {e}")
        return False

    async def set_task_status(self, task_id: str, status: dict) -> None:
        """Set task status with default TTL"""
        await self.set(f"task_status:{task_id}", status)

    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get task status"""
        return await self.get(f"task_status:{task_id}")

    async def set_forecast_result(self, task_id: str, result: dict) -> None:
        """Store forecast result with compression for large payloads"""
        await self.set(f"forecast_result:{task_id}", result)

    async def get_forecast_result(self, task_id: str) -> Optional[dict]:
        """Get forecast result"""
        return await self.get(f"forecast_result:{task_id}")

    async def cleanup_websocket(self, connection_id: str) -> None:
        """Clean up WebSocket related data"""
        pattern = f"ws:{connection_id}:*"
        for key in self.client.scan_iter(match=pattern):
            await self.delete(key)

    async def clear_stale_forecasts(self, user_id: str, max_age: int = 86400) -> None:
        """Clear stale forecast results older than max_age seconds"""
        pattern = f"forecast_result:{user_id}:*"
        for key in self.client.scan_iter(match=pattern):
            if not self.client.ttl(key) or self.client.ttl(key) > max_age:
                await self.delete(key)
