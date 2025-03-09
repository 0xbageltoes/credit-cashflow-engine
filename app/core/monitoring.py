"""Monitoring and metrics collection module"""
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Callable
from fastapi import Request, Response
from redis import Redis
from app.core.config import settings
import logging

# Prometheus metrics
REQUEST_COUNT = Counter(
    'cashflow_request_count', 
    'App Request Count',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'cashflow_request_latency_seconds', 
    'Request latency in seconds',
    ['method', 'endpoint']
)
ACTIVE_REQUESTS = Gauge(
    'cashflow_active_requests', 
    'Number of active requests',
    ['method', 'endpoint']
)
FORECAST_CALCULATIONS = Counter(
    'cashflow_forecast_calculations', 
    'Number of cashflow forecasts calculated',
    ['user_id', 'success']
)
CALCULATION_TIME = Summary(
    'cashflow_calculation_time', 
    'Time spent calculating cashflow forecasts',
    ['calculation_type']
)
REDIS_CONNECTIONS = Gauge(
    'cashflow_redis_connections',
    'Number of active Redis connections'
)
SYSTEM_MEMORY = Gauge(
    'cashflow_system_memory_usage',
    'System memory usage in percent'
)
CACHE_HITS = Counter(
    'cashflow_cache_hits',
    'Number of cache hits',
    ['key_type']
)
CACHE_MISSES = Counter(
    'cashflow_cache_misses',
    'Number of cache misses',
    ['key_type']
)

class PrometheusMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        
        # Get path and method from scope
        path = scope.get("path", "unknown")
        method = scope.get("method", "unknown")
        
        # Track request count and latency
        ACTIVE_REQUESTS.labels(method=method, endpoint=path).inc()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
                REQUEST_COUNT.labels(method=method, endpoint=path, http_status=status_code).inc()
                ACTIVE_REQUESTS.labels(method=method, endpoint=path).dec()
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def check_redis_connection():
    """Check Redis connection and return status"""
    try:
        redis = Redis.from_url(settings.REDIS_URL)
        ping_result = redis.ping()
        client_count = len(redis.client_list())
        REDIS_CONNECTIONS.set(client_count)
        return {"status": "healthy", "ping": ping_result, "clients": client_count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_system_health():
    """Check overall system health"""
    try:
        import psutil
        # Get system memory usage
        memory_percent = psutil.virtual_memory().percent
        SYSTEM_MEMORY.set(memory_percent)
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get disk usage
        disk_usage = psutil.disk_usage('/').percent
        
        return {
            "status": "healthy",
            "memory_usage": memory_percent,
            "cpu_usage": cpu_percent,
            "disk_usage": disk_usage
        }
    except ImportError:
        return {"status": "healthy", "warning": "psutil not installed, system metrics unavailable"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

class CalculationTracker:
    """Context manager to track calculation time using Prometheus metrics"""
    
    def __init__(self, calculation_type: str):
        self.calculation_type = calculation_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        CALCULATION_TIME.labels(calculation_type=self.calculation_type).observe(duration)

class PrometheusMetrics:
    """Prometheus metrics wrapper for tests"""
    
    def __init__(self):
        # For test assertions
        self.request_count = REQUEST_COUNT
        self.request_latency = REQUEST_LATENCY
        self.active_requests = ACTIVE_REQUESTS
        self.calculation_time = CALCULATION_TIME
        self.cache_hits = CACHE_HITS
        self.cache_misses = CACHE_MISSES
    
    def track_request(self, endpoint: str, method: str, status_code: int):
        """Track request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
        
    def track_latency(self, endpoint: str, method: str, latency: float):
        """Track request latency"""
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        
    def track_calculation(self, calculation_type: str, duration: float):
        """Track calculation time"""
        CALCULATION_TIME.labels(calculation_type=calculation_type).observe(duration)
        
    def track_cache_hit(self, key_type: str):
        """Track cache hit"""
        CACHE_HITS.labels(key_type=key_type).inc()
        
    def track_cache_miss(self, key_type: str):
        """Track cache miss"""
        CACHE_MISSES.labels(key_type=key_type).inc()
        
    def track_cache(self, hit: bool, key_type: str = "default"):
        """Track cache hit or miss"""
        if hit:
            self.track_cache_hit(key_type)
        else:
            self.track_cache_miss(key_type)
        
    def track_request_latency(self, path):
        """Decorator for tracking request latency"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                self.track_latency(path, "GET", latency)
                return result
            return wrapper
        return decorator
        
    def track_task(self, task_name):
        """Decorator for tracking task execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.track_calculation(task_name, duration)
                return result
            return wrapper
        return decorator
        
    def update_system_info(self, info_dict):
        """Update system info metrics"""
        # This would normally set Gauge metrics with system info
        # For testing, we just log the info
        for key, value in info_dict.items():
            logging.info(f"System info: {key}={value}")
