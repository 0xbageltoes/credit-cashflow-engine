import json
import zlib
from typing import Any, Optional, Dict
from redis import Redis
from app.core.config import settings
import ssl

class RedisCache:
    _instance = None
    DEFAULT_TTL = settings.CACHE_TTL  # Default 1 hour TTL
    COMPRESSION_THRESHOLD = 1024  # Compress values larger than 1KB

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.client = Redis.from_url(
                settings.REDIS_URL_WITH_SSL,
                decode_responses=True,
                ssl=True,
                ssl_cert_reqs=None
            )
        return cls._instance

    def _compress(self, value: str) -> bytes:
        """Compress string data using zlib"""
        return zlib.compress(value.encode())

    def _decompress(self, value: bytes) -> str:
        """Decompress zlib compressed data"""
        return zlib.decompress(value).decode()

    def _should_compress(self, value: str) -> bool:
        """Check if value should be compressed based on size"""
        return len(value.encode()) > self.COMPRESSION_THRESHOLD

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key with optional TTL and automatic compression"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        # Compress large values
        if self._should_compress(value):
            compressed = self._compress(value)
            self.client.set(f"compressed:{key}", compressed, ex=ttl or self.DEFAULT_TTL)
        else:
            self.client.set(key, value, ex=ttl or self.DEFAULT_TTL)

    async def get(self, key: str) -> Optional[Any]:
        """Get a value, handling compression and JSON automatically"""
        # Try compressed key first
        compressed_value = self.client.get(f"compressed:{key}")
        if compressed_value:
            value = self._decompress(compressed_value)
        else:
            value = self.client.get(key)
            if not value:
                return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def delete(self, key: str) -> None:
        """Delete a key and its compressed version if it exists"""
        self.client.delete(key)
        self.client.delete(f"compressed:{key}")

    async def set_task_status(self, task_id: str, status: Dict) -> None:
        """Set task status with default TTL"""
        await self.set(f"task_status:{task_id}", status)

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        return await self.get(f"task_status:{task_id}")

    async def set_forecast_result(self, task_id: str, result: Dict) -> None:
        """Store forecast result with compression for large payloads"""
        await self.set(f"forecast_result:{task_id}", result)

    async def get_forecast_result(self, task_id: str) -> Optional[Dict]:
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
