"""Monitoring and metrics collection module"""
from prometheus_client import Counter, Histogram, Info, Gauge
import time
from functools import wraps
from typing import Optional, Dict, Any

class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        # API metrics
        self.request_count = Counter(
            'api_requests_total',
            'Total count of API requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_latency = Histogram(
            'api_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total number of cache hits'
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total number of cache misses'
        )
        
        # Task metrics
        self.task_duration = Histogram(
            'task_duration_seconds',
            'Task execution duration in seconds',
            ['task_type']
        )
        
        self.active_tasks = Gauge(
            'active_tasks',
            'Number of currently running tasks',
            ['task_type']
        )
        
        # System metrics
        self.system_info = Info(
            'system_info',
            'System information'
        )
    
    def track_request(self, endpoint: str, method: str, status: int):
        """Track API request metrics"""
        self.request_count.labels(endpoint=endpoint, method=method, status=status).inc()
    
    def track_request_latency(self, endpoint: str):
        """Decorator to track request latency"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.request_latency.labels(endpoint=endpoint).observe(duration)
            return wrapper
        return decorator
    
    def track_cache(self, hit: bool):
        """Track cache hit/miss metrics"""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
    
    def track_task(self, task_type: str):
        """Decorator to track task execution metrics"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                self.active_tasks.labels(task_type=task_type).inc()
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.task_duration.labels(task_type=task_type).observe(duration)
                    self.active_tasks.labels(task_type=task_type).dec()
            return wrapper
        return decorator
    
    def update_system_info(self, info: Dict[str, Any]):
        """Update system information metrics"""
        self.system_info.info(info)

# Global metrics instance
metrics = PrometheusMetrics()
