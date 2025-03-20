"""
Monitoring Module

This module provides monitoring utilities for the application,
including Prometheus metrics, health checks, and system diagnostics.
"""

import time
import platform
import psutil
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from redis.exceptions import RedisError

from app.core.cache_service import CacheService

# Set up logging
logger = logging.getLogger(__name__)

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total count of HTTP requests", 
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", 
    "HTTP request latency in seconds", 
    ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge(
    "http_requests_active", 
    "Number of active HTTP requests"
)

CALCULATION_TIME = Summary(
    "calculation_processing_seconds", 
    "Time spent on calculations", 
    ["operation_type"]
)

CACHE_HITS = Counter(
    "cache_hits_total", 
    "Total count of cache hits", 
    ["cache_type"]
)

CACHE_MISSES = Counter(
    "cache_misses_total", 
    "Total count of cache misses", 
    ["cache_type"]
)

class PrometheusMetrics:
    """
    Centralized metrics collection and management class for application-wide
    usage. Provides consistent metrics APIs for different services.
    """
    
    def __init__(self):
        """Initialize metrics counters and histograms"""
        # Operation counters
        self.operation_counter = Counter(
            "operations_total",
            "Count of operations performed",
            ["service", "operation", "status"]
        )
        
        # Latency tracking
        self.operation_latency = Histogram(
            "operation_latency_seconds",
            "Latency of operations in seconds",
            ["service", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0)
        )
        
        # Error counters
        self.error_counter = Counter(
            "errors_total",
            "Count of errors encountered",
            ["service", "error_type"]
        )
        
        # Data volume metrics
        self.data_processed = Counter(
            "data_processed_bytes",
            "Amount of data processed in bytes",
            ["service", "data_type"]
        )
        
        # Service-specific gauges
        self.service_state = Gauge(
            "service_state",
            "Current state of services (1=up, 0=down)",
            ["service"]
        )
    
    def record_operation(self, service: str, operation: str, status: str = "success"):
        """
        Record an operation being performed by a service
        
        Args:
            service: Name of the service performing the operation
            operation: Name of the operation performed
            status: Outcome status (success, error, etc.)
        """
        self.operation_counter.labels(
            service=service,
            operation=operation,
            status=status
        ).inc()
    
    def time_operation(self, service: str, operation: str):
        """
        Get a context manager to time an operation
        
        Args:
            service: Name of the service performing the operation
            operation: Name of the operation being timed
            
        Returns:
            A timer context manager
        """
        return self.operation_latency.labels(
            service=service,
            operation=operation
        ).time()
    
    def record_error(self, service: str, error_type: str):
        """
        Record an error encountered by a service
        
        Args:
            service: Name of the service encountering the error
            error_type: Type of error encountered
        """
        self.error_counter.labels(
            service=service,
            error_type=error_type
        ).inc()
    
    def record_data_processed(self, service: str, data_type: str, bytes_count: int):
        """
        Record amount of data processed
        
        Args:
            service: Name of the service processing the data
            data_type: Type of data being processed
            bytes_count: Amount of data in bytes
        """
        self.data_processed.labels(
            service=service,
            data_type=data_type
        ).inc(bytes_count)
    
    def set_service_state(self, service: str, is_up: bool):
        """
        Set the current state of a service
        
        Args:
            service: Name of the service
            is_up: Whether the service is operational
        """
        self.service_state.labels(service=service).set(1 if is_up else 0)

# Middleware for metrics collection
class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics for HTTP requests.
    Tracks request counts, latency, and active requests.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process an incoming request and record metrics.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint to call
            
        Returns:
            The HTTP response
        """
        # Skip metrics collection for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Track request processing time
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Record request metrics
            path = request.url.path
            method = request.method
            
            # Simplify path for common patterns to avoid metric explosion
            # E.g., /users/123 -> /users/{id}
            if "/api/" in path:
                parts = path.split("/")
                simplified_parts = []
                
                for i, part in enumerate(parts):
                    # Check if this part is likely a numeric ID
                    if part.isdigit() and i > 0 and parts[i-1] in ["users", "items", "products", "orders"]:
                        simplified_parts.append("{id}")
                    else:
                        simplified_parts.append(part)
                
                path = "/".join(simplified_parts)
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status_code=response.status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path
            ).observe(time.time() - start_time)
            
            return response
            
        finally:
            # Ensure we decrement active requests even on error
            ACTIVE_REQUESTS.dec()

class CalculationTracker:
    """
    Utility class to track calculation performance and metrics.
    
    This class provides a context manager for timing calculations and
    recording their performance metrics for monitoring and optimization.
    It integrates with Prometheus metrics for real-time monitoring.
    """
    
    def __init__(self, operation_type: str, enable_logging: bool = True):
        """
        Initialize a calculation tracker for a specific operation type
        
        Args:
            operation_type: The type of calculation being performed (e.g., 'cash_flow', 'pricing')
            enable_logging: Whether to log timing information (default: True)
        """
        self.operation_type = operation_type
        self.enable_logging = enable_logging
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Start timing the calculation"""
        self.start_time = time.time()
        logger.debug(f"Starting calculation: {self.operation_type}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        End timing and record metrics
        
        If an exception occurred during the calculation, it will be logged
        but not suppressed.
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Record to Prometheus metrics
        CALCULATION_TIME.labels(operation_type=self.operation_type).observe(duration)
        
        if self.enable_logging:
            if exc_type:
                logger.error(f"Calculation '{self.operation_type}' failed after {duration:.3f}s: {exc_val}")
            else:
                logger.info(f"Calculation '{self.operation_type}' completed in {duration:.3f}s")
                
        # Additional context can be stored here if needed
        self.duration = duration
        
        # Don't suppress exceptions
        return False
    
    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time of the calculation in seconds"""
        if self.start_time is None:
            return 0.0
            
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

async def check_redis_connection() -> Dict[str, Any]:
    """
    Check if Redis is available and responding.
    
    Returns:
        Dict with Redis connection status information
    """
    try:
        cache_service = CacheService()
        start_time = time.time()
        
        # Try a simple Redis ping operation
        ping_result = await cache_service.ping()
        
        if ping_result:
            # Get some basic Redis info
            info = await cache_service.info()
            
            latency = time.time() - start_time
            return {
                "status": "ok",
                "latency": f"{latency:.4f}s",
                "version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0)
            }
        else:
            return {
                "status": "error",
                "error": "Redis ping failed"
            }
    except Exception as e:
        logger.error(f"Redis connection check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def check_system_health() -> Dict[str, Any]:
    """
    Check and report system resource usage.
    
    Returns:
        Dict with system health information
    """
    # Get basic system information
    system_info = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uptime_seconds": int(time.time() - psutil.boot_time()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Get detailed resource information if psutil is available
    try:
        # CPU usage
        system_info["cpu"] = {
            "count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "load_avg": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
        
        # Memory usage
        mem = psutil.virtual_memory()
        system_info["memory"] = {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "percent_used": mem.percent
        }
        
        # Disk usage
        disk = psutil.disk_usage(os.getcwd())
        system_info["disk"] = {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent_used": disk.percent
        }
        
        # Network IO
        net_io = psutil.net_io_counters()
        system_info["network"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    except Exception as e:
        system_info["resource_error"] = str(e)
    
    return system_info