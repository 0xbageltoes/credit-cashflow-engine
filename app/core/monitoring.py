"""Monitoring and metrics collection module"""
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Callable, Dict, List, Any, Optional
import os
import platform
from datetime import datetime
import psutil
from redis import Redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

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

# Stress testing specific metrics
STRESS_TEST_REQUESTS = Counter(
    'cashflow_stress_test_requests',
    'Number of stress test requests',
    ['user_id', 'parallel']
)
STRESS_TEST_SCENARIOS = Counter(
    'cashflow_stress_test_scenarios',
    'Number of stress test scenarios executed',
    ['scenario_name', 'success']
)
STRESS_TEST_DURATION = Histogram(
    'cashflow_stress_test_duration_seconds',
    'Duration of stress test executions',
    ['parallel', 'scenario_count']
)
ACTIVE_STRESS_TESTS = Gauge(
    'cashflow_active_stress_tests',
    'Number of currently running stress tests',
    ['user_id']
)
STRESS_TEST_ASSET_COUNT = Histogram(
    'cashflow_stress_test_asset_count',
    'Number of assets in stress test pools',
    ['user_id']
)
STRESS_TEST_ERRORS = Counter(
    'cashflow_stress_test_errors',
    'Number of errors in stress tests',
    ['error_type']
)
STRESS_WORKER_UTILIZATION = Gauge(
    'cashflow_stress_worker_utilization',
    'Worker thread utilization in stress tests',
    ['worker_count']
)
STRESS_WEBSOCKET_CONNECTIONS = Gauge(
    'cashflow_stress_websocket_connections',
    'Number of active WebSocket connections for stress test monitoring',
    ['user_id']
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

class StressTestMetrics:
    """Stress testing metrics tracker for comprehensive monitoring"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.active_trackers = {}
    
    def track_test_request(self, parallel: bool = False):
        """Track stress test request"""
        STRESS_TEST_REQUESTS.labels(user_id=self.user_id, parallel=str(parallel)).inc()
    
    def track_scenario_execution(self, scenario_name: str, success: bool = True):
        """Track stress test scenario execution"""
        STRESS_TEST_SCENARIOS.labels(scenario_name=scenario_name, success=str(success)).inc()
    
    def track_execution_duration(self, duration: float, parallel: bool, scenario_count: int):
        """Track stress test execution duration"""
        STRESS_TEST_DURATION.labels(
            parallel=str(parallel), 
            scenario_count=str(scenario_count)
        ).observe(duration)
    
    def track_asset_count(self, asset_count: int):
        """Track number of assets in stress test pool"""
        STRESS_TEST_ASSET_COUNT.labels(user_id=self.user_id).observe(asset_count)
    
    def track_error(self, error_type: str):
        """Track stress test error"""
        STRESS_TEST_ERRORS.labels(error_type=error_type).inc()
    
    def increment_active_tests(self):
        """Increment active stress test counter"""
        ACTIVE_STRESS_TESTS.labels(user_id=self.user_id).inc()
    
    def decrement_active_tests(self):
        """Decrement active stress test counter"""
        ACTIVE_STRESS_TESTS.labels(user_id=self.user_id).dec()
    
    def track_worker_utilization(self, worker_count: int, utilization_percent: float):
        """Track worker thread utilization"""
        STRESS_WORKER_UTILIZATION.labels(worker_count=str(worker_count)).set(utilization_percent)
    
    def track_websocket_connection(self, connected: bool = True):
        """Track WebSocket connection status"""
        if connected:
            STRESS_WEBSOCKET_CONNECTIONS.labels(user_id=self.user_id).inc()
        else:
            STRESS_WEBSOCKET_CONNECTIONS.labels(user_id=self.user_id).dec()
    
    def track_scenario(self, scenario_name: str):
        """Context manager to track scenario execution time"""
        class ScenarioTracker:
            def __init__(self, metrics, scenario_name):
                self.metrics = metrics
                self.scenario_name = scenario_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                success = exc_type is None
                duration = time.time() - self.start_time
                self.metrics.track_scenario_execution(self.scenario_name, success)
                if not success:
                    self.metrics.track_error(exc_type.__name__ if exc_type else "unknown")
        
        return ScenarioTracker(self, scenario_name)

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

"""
Monitoring and Performance Metrics Module

Production-grade monitoring implementation for tracking calculations, resource usage,
and performance metrics throughout the application with proper error handling.
"""
import time
import logging
import traceback
import os
import platform
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

class CalculationTracker:
    """
    Context manager for tracking execution times and resource usage
    of long-running calculations with proper error handling.
    """
    
    def __init__(self, operation_name: str):
        """Initialize tracker with operation name"""
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.error = None
        self.cpu_percent_start = None
        self.cpu_percent_end = None
        self.memory_start = None
        self.memory_end = None
        
    def __enter__(self):
        """Start tracking resources"""
        self.start_time = time.time()
        try:
            self.cpu_percent_start = psutil.cpu_percent(interval=0.1)
            self.memory_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Starting operation: {self.operation_name}")
        except Exception as e:
            logger.error(f"Error tracking resources at start: {str(e)}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracking and log results"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        try:
            self.cpu_percent_end = psutil.cpu_percent(interval=0.1)
            self.memory_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            memory_delta = self.memory_end - self.memory_start if self.memory_start and self.memory_end else None
            
            if exc_type:
                self.error = f"{exc_type.__name__}: {str(exc_val)}"
                logger.error(
                    f"Operation failed: {self.operation_name} after {self.duration:.2f}s with error: {self.error}"
                )
            else:
                logger.info(
                    f"Completed operation: {self.operation_name} in {self.duration:.2f}s "
                    f"(Memory: {memory_delta:+.2f}MB, CPU: {self.cpu_percent_end:.1f}%)"
                )
                
        except Exception as e:
            logger.error(f"Error tracking resources at end: {str(e)}")

class PerformanceMetrics:
    """
    Production-grade performance metrics collection for tracking
    detailed operation times, resource usage, and critical paths.
    Includes proper error handling and comprehensive data collection.
    """
    
    def __init__(self, operation_id: str):
        """Initialize metrics with operation ID"""
        self.operation_id = operation_id
        self.start_time = time.time()
        self.end_time = None
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.current_spans: Dict[str, Dict[str, Any]] = {}
        self.sequence: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
        # Capture initial system state
        try:
            self.system_info = {
                "hostname": platform.node(),
                "os": platform.system(),
                "cpu_count": os.cpu_count(),
                "python_version": platform.python_version(),
                "start_time": datetime.now().isoformat()
            }
            
            process = psutil.Process(os.getpid())
            self.system_info["initial_memory_mb"] = process.memory_info().rss / 1024 / 1024
            self.system_info["initial_cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Error capturing system metrics: {str(e)}")
            self.system_info = {"error": str(e)}
    
    @contextmanager
    def track(self, span_name: str):
        """
        Context manager for tracking execution time of a specific operation span
        with proper error handling.
        
        Args:
            span_name: Name of the span to track
        """
        span_id = f"{span_name}_{len(self.sequence)}"
        span_data = {
            "name": span_name,
            "id": span_id,
            "start_time": time.time(),
            "order": len(self.sequence)
        }
        
        # Store current span
        self.current_spans[span_name] = span_data
        self.sequence.append({"event": "start", "span": span_name, "time": span_data["start_time"]})
        
        try:
            # Track initial resource usage
            process = psutil.Process(os.getpid())
            span_data["initial_memory_mb"] = process.memory_info().rss / 1024 / 1024
            span_data["initial_cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.debug(f"Error capturing initial resource metrics for span {span_name}: {str(e)}")
        
        try:
            yield
        except Exception as e:
            # Track errors
            error_info = {
                "span": span_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "time": time.time(),
                "traceback": traceback.format_exc()
            }
            self.errors.append(error_info)
            span_data["error"] = error_info
            self.sequence.append({"event": "error", "span": span_name, "time": time.time(), "error": str(e)})
            raise
        finally:
            # Complete span tracking
            end_time = time.time()
            span_data["end_time"] = end_time
            span_data["duration"] = end_time - span_data["start_time"]
            
            try:
                # Track final resource usage
                process = psutil.Process(os.getpid())
                span_data["final_memory_mb"] = process.memory_info().rss / 1024 / 1024
                span_data["final_cpu_percent"] = psutil.cpu_percent(interval=0.1)
                span_data["memory_delta_mb"] = span_data["final_memory_mb"] - span_data["initial_memory_mb"]
            except Exception as e:
                logger.debug(f"Error capturing final resource metrics for span {span_name}: {str(e)}")
            
            # Store metrics
            self.metrics[span_id] = span_data
            self.sequence.append({"event": "end", "span": span_name, "time": end_time, "duration": span_data["duration"]})
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report with spans and system information
        
        Returns:
            Complete metrics report
        """
        self.end_time = time.time()
        
        # Capture final system state
        try:
            process = psutil.Process(os.getpid())
            self.system_info["final_memory_mb"] = process.memory_info().rss / 1024 / 1024
            self.system_info["final_cpu_percent"] = psutil.cpu_percent(interval=0.1)
            self.system_info["memory_delta_mb"] = self.system_info["final_memory_mb"] - self.system_info["initial_memory_mb"]
            self.system_info["end_time"] = datetime.now().isoformat()
        except Exception as e:
            logger.warning(f"Error capturing final system metrics: {str(e)}")
        
        # Prepare summary metrics
        spans_summary = {}
        for span_id, span_data in self.metrics.items():
            span_name = span_data["name"]
            if span_name not in spans_summary:
                spans_summary[span_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "min_duration": float('inf'),
                    "max_duration": 0,
                }
            
            summary = spans_summary[span_name]
            duration = span_data["duration"]
            
            summary["count"] += 1
            summary["total_duration"] += duration
            summary["min_duration"] = min(summary["min_duration"], duration)
            summary["max_duration"] = max(summary["max_duration"], duration)
        
        # Calculate averages
        for summary in spans_summary.values():
            summary["avg_duration"] = summary["total_duration"] / summary["count"]
            # Clean up infinity if no spans
            if summary["min_duration"] == float('inf'):
                summary["min_duration"] = 0
        
        # Prepare critical path analysis
        sorted_spans = sorted(self.metrics.values(), key=lambda x: x["duration"], reverse=True)
        critical_path = [
            {
                "name": span["name"],
                "duration": span["duration"],
                "percent_of_total": (span["duration"] / (self.end_time - self.start_time)) * 100
                if (self.end_time - self.start_time) > 0 else 0
            }
            for span in sorted_spans[:5]  # Top 5 spans
        ]
        
        # Return comprehensive metrics report
        return {
            "operation_id": self.operation_id,
            "total_duration": self.end_time - self.start_time,
            "spans_count": len(self.metrics),
            "error_count": len(self.errors),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "spans_summary": spans_summary,
            "critical_path": critical_path,
            "system_info": self.system_info,
            "has_errors": len(self.errors) > 0
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics including sequence of operations
        and complete span information
        
        Returns:
            Complete detailed metrics report
        """
        basic_metrics = self.get_metrics()
        return {
            **basic_metrics,
            "sequence": self.sequence,
            "spans": self.metrics,
            "errors": self.errors
        }
    
    def log_metrics(self, level: str = "info"):
        """
        Log current metrics at specified level
        
        Args:
            level: Logging level (debug, info, warning, error)
        """
        metrics = self.get_metrics()
        metrics_json = json.dumps(metrics, indent=2)
        
        log_method = getattr(logger, level.lower())
        log_method(f"Performance metrics for {self.operation_id}:\n{metrics_json}")
        
        # Log critical path separately for visibility
        if metrics["critical_path"]:
            critical_path_str = ", ".join(
                f"{span['name']}: {span['duration']:.2f}s ({span['percent_of_total']:.1f}%)"
                for span in metrics["critical_path"]
            )
            log_method(f"Critical path for {self.operation_id}: {critical_path_str}")
        
        # Log errors at error level
        if metrics["error_count"] > 0:
            logger.error(f"Operation {self.operation_id} completed with {metrics['error_count']} errors")
            for error in self.errors[:5]:  # Log first 5 errors
                logger.error(f"Error in span {error['span']}: {error['error_type']}: {error['error_message']}")
        
        return metrics

# Singleton metrics registry for application-wide tracking
class MetricsRegistry:
    """
    Singleton registry for tracking operation metrics across the application.
    Allows for centralized collection and querying of metrics.
    """
    
    def __init__(self):
        """Initialize metrics registry"""
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.active_trackers: Dict[str, PerformanceMetrics] = {}
        self._max_entries = 1000  # Prevent unbounded growth
    
    def register_metrics(self, operation_id: str, metrics: Dict[str, Any]):
        """Register metrics for an operation"""
        self.metrics[operation_id] = {
            **metrics,
            "registered_at": datetime.now().isoformat()
        }
        
        # Enforce max entries limit
        if len(self.metrics) > self._max_entries:
            # Remove oldest entries
            sorted_keys = sorted(
                self.metrics.keys(),
                key=lambda k: self.metrics[k].get("registered_at", "")
            )
            for key in sorted_keys[:len(sorted_keys) - self._max_entries]:
                del self.metrics[key]
    
    def start_tracking(self, operation_id: str) -> PerformanceMetrics:
        """Start tracking metrics for an operation"""
        tracker = PerformanceMetrics(operation_id)
        self.active_trackers[operation_id] = tracker
        return tracker
    
    def end_tracking(self, operation_id: str, log_level: str = "info") -> Optional[Dict[str, Any]]:
        """End tracking and register metrics"""
        tracker = self.active_trackers.get(operation_id)
        if not tracker:
            logger.warning(f"Attempted to end tracking for unknown operation: {operation_id}")
            return None
            
        metrics = tracker.log_metrics(level=log_level)
        self.register_metrics(operation_id, metrics)
        
        del self.active_trackers[operation_id]
        return metrics
    
    def get_metrics(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for an operation"""
        return self.metrics.get(operation_id)
    
    def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent metrics"""
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda m: m.get("registered_at", ""),
            reverse=True
        )
        return sorted_metrics[:limit]

# Create singleton registry
metrics_registry = MetricsRegistry()

def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry"""
    return metrics_registry
