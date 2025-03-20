"""
Metrics Module

This module provides Prometheus-style metrics for application monitoring.
It defines counters, gauges, histograms, and other metrics used throughout the application.
"""

import logging
import os
import time
from typing import Dict, Optional, Any, Callable, Union
from functools import wraps

# Setup logging
logger = logging.getLogger(__name__)

# Check if metrics are enabled (can be disabled in tests or development)
METRICS_ENABLED = os.environ.get("ENABLE_METRICS", "1").lower() in ("1", "true", "yes", "on")

# Import Prometheus client if available, otherwise use mock implementation
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client is available for metrics")
except ImportError:
    logger.warning("Prometheus client not available, using mock metrics")
    PROMETHEUS_AVAILABLE = False
    
    # Mock implementations for Prometheus metrics
    class MockMetric:
        """Mock metric class that acts as a no-op when prometheus_client is not available"""
        
        def __init__(self, name: str, documentation: str, labelnames=None, **kwargs):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames or []
            self.values = {}
            
        def labels(self, **kwargs):
            """Return self to allow chaining"""
            key = tuple(str(kwargs.get(label, '')) for label in self.labelnames)
            if key not in self.values:
                self.values[key] = MockMetricValue(self.name)
            return self.values[key]
            
        def inc(self, amount=1):
            """Increment counter by amount"""
            pass
            
        def dec(self, amount=1):
            """Decrement counter by amount"""
            pass
            
        def set(self, value):
            """Set gauge to value"""
            pass
            
        def observe(self, value):
            """Observe a value for histogram or summary"""
            pass
    
    class MockMetricValue:
        """Mock metric value that provides inc, dec, set, observe methods"""
        
        def __init__(self, name: str):
            self.name = name
            
        def inc(self, amount=1):
            """Increment counter by amount"""
            pass
            
        def dec(self, amount=1):
            """Decrement counter by amount"""
            pass
            
        def set(self, value):
            """Set gauge to value"""
            pass
            
        def observe(self, value):
            """Observe a value for histogram or summary"""
            pass
    
    # Mock Prometheus classes
    Counter = MockMetric
    Histogram = MockMetric
    Gauge = MockMetric
    Summary = MockMetric

# Define application-wide metrics
# Request metrics
REQUEST_COUNT = Counter(
    'credit_cashflow_request_count', 
    'Count of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'credit_cashflow_request_latency_seconds', 
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
)

# Cache metrics
CACHE_HIT_COUNTER = Counter(
    'credit_cashflow_cache_hit_count', 
    'Count of cache hits',
    ['cache_type']  # redis, memory
)

CACHE_MISS_COUNTER = Counter(
    'credit_cashflow_cache_miss_count', 
    'Count of cache misses',
    ['cache_type']  # redis, memory
)

CACHE_ERROR_COUNTER = Counter(
    'credit_cashflow_cache_error_count', 
    'Count of cache errors',
    ['cache_type', 'error_type']  # redis/memory, connection/timeout/etc
)

CACHE_LATENCY = Histogram(
    'credit_cashflow_cache_operation_latency_seconds', 
    'Cache operation latency in seconds',
    ['cache_type', 'operation'],  # redis/memory, get/set/delete
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
)

# Dependency metrics
DEPENDENCY_INIT_TIME = Histogram(
    'credit_cashflow_dependency_init_seconds', 
    'Time to initialize dependencies',
    ['service'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60]
)

# AbsBox service metrics
ABSBOX_REQUEST_COUNT = Counter(
    'credit_cashflow_absbox_request_count', 
    'Count of requests to AbsBox service',
    ['endpoint', 'status']
)

ABSBOX_REQUEST_LATENCY = Histogram(
    'credit_cashflow_absbox_request_latency_seconds', 
    'AbsBox request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]
)

# Database metrics
DB_QUERY_COUNT = Counter(
    'credit_cashflow_db_query_count', 
    'Count of database queries',
    ['operation', 'status']  # select/insert/update/delete, success/error
)

DB_QUERY_LATENCY = Histogram(
    'credit_cashflow_db_query_latency_seconds', 
    'Database query latency in seconds',
    ['operation'],  # select/insert/update/delete
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)

# Calculation metrics
CALCULATION_TIME = Histogram(
    'credit_cashflow_calculation_time_seconds', 
    'Time to perform calculations',
    ['calculation_type'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600]
)

# Error metrics
ERROR_COUNT = Counter(
    'credit_cashflow_error_count', 
    'Count of application errors',
    ['service', 'error_type']
)

# System metrics
MEMORY_USAGE = Gauge(
    'credit_cashflow_memory_usage_bytes', 
    'Memory usage in bytes',
    ['process']
)

CPU_USAGE = Gauge(
    'credit_cashflow_cpu_usage_percent', 
    'CPU usage percentage',
    ['process']
)

# Utils for metrics
def track_time(metric: Histogram, labels: Dict[str, str] = None):
    """Decorator to track execution time of a function using a Histogram metric"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not METRICS_ENABLED:
                return func(*args, **kwargs)
                
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if METRICS_ENABLED:
                    duration = time.time() - start_time
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
        return wrapper
    return decorator


def count_calls(metric: Counter, labels: Dict[str, str] = None):
    """Decorator to count calls to a function using a Counter metric"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if METRICS_ENABLED and labels:
                metric.labels(**labels).inc()
            elif METRICS_ENABLED:
                metric.inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def count_errors(metric: Counter, labels: Dict[str, str] = None, error_types=None):
    """Decorator to count errors in a function using a Counter metric"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if METRICS_ENABLED:
                    if labels:
                        label_dict = labels.copy()
                        if error_types and type(e) in error_types:
                            label_dict['error_type'] = error_types[type(e)]
                        else:
                            label_dict['error_type'] = type(e).__name__
                        metric.labels(**label_dict).inc()
                    else:
                        metric.inc()
                raise
        return wrapper
    return decorator


class CalculationTracker:
    """Utility to track calculation time for different parts of a calculation"""
    
    def __init__(self, calculation_type: str):
        self.calculation_type = calculation_type
        self.start_times = {}
        self.enabled = METRICS_ENABLED
        
    def start(self, section: str):
        """Start timing a section of the calculation"""
        if self.enabled:
            self.start_times[section] = time.time()
        
    def stop(self, section: str):
        """Stop timing a section and record the duration"""
        if not self.enabled or section not in self.start_times:
            return
            
        duration = time.time() - self.start_times[section]
        CALCULATION_TIME.labels(
            calculation_type=f"{self.calculation_type}_{section}"
        ).observe(duration)
        
        # Clean up
        del self.start_times[section]
