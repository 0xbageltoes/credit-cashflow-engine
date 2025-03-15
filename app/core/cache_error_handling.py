"""Cache Error Handling Module

This module provides comprehensive error handling, monitoring, and fallback mechanisms
for Redis cache operations. It ensures robustness in production environments by:

1. Standardizing error types and recovery strategies
2. Providing detailed error tracking and metrics
3. Supporting graceful fallbacks when Redis is unavailable
4. Ensuring consistent circuit breaker patterns for all Redis operations
"""

import time
import logging
import traceback
from typing import Any, Dict, Optional, Callable, TypeVar, Generic, Union, List, Tuple
from functools import wraps
from dataclasses import dataclass, field
import threading
import asyncio
from datetime import datetime, timedelta

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Configure logger
logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base class for all cache-related exceptions"""
    pass


class ConnectionError(CacheError):
    """Error connecting to Redis server"""
    pass


class TimeoutError(CacheError):
    """Redis operation timed out"""
    pass


class DeserializationError(CacheError):
    """Error deserializing data from cache"""
    pass


class SerializationError(CacheError):
    """Error serializing data for cache"""
    pass


class MemoryLimitExceededError(CacheError):
    """Value too large for Redis memory limit"""
    pass


class CircuitBreakerOpenError(CacheError):
    """Circuit breaker is open - Redis operations suspended"""
    pass


@dataclass
class ErrorMetrics:
    """Tracks error rates and patterns for cache operations"""
    
    total_operations: int = 0
    failed_operations: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    first_error_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    last_error_message: Optional[str] = None
    last_error_stack: Optional[str] = None
    
    @property
    def error_rate(self) -> float:
        """Calculate the error rate as percentage"""
        if self.total_operations == 0:
            return 0.0
        return (self.failed_operations / self.total_operations) * 100
    
    def record_operation(self, success: bool, error: Optional[Exception] = None) -> None:
        """Record a cache operation result
        
        Args:
            success: Whether the operation succeeded
            error: Optional exception if operation failed
        """
        self.total_operations += 1
        
        if not success:
            self.failed_operations += 1
            self.consecutive_failures += 1
            
            now = datetime.now()
            if self.first_error_time is None:
                self.first_error_time = now
            self.last_error_time = now
            
            if error:
                error_type = type(error).__name__
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
                self.last_error_message = str(error)
                self.last_error_stack = traceback.format_exc()
        else:
            self.consecutive_failures = 0
    
    def record_recovery_attempt(self, success: bool) -> None:
        """Record a recovery attempt
        
        Args:
            success: Whether the recovery was successful
        """
        self.recovery_attempts += 1
        if success:
            self.successful_recoveries += 1
            self.consecutive_failures = 0
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.total_operations = 0
        self.failed_operations = 0
        self.errors_by_type = {}
        self.first_error_time = None
        self.last_error_time = None
        self.consecutive_failures = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.last_error_message = None
        self.last_error_stack = None


@dataclass
class CircuitBreaker:
    """Implements the circuit breaker pattern for Redis operations
    
    The circuit breaker prevents cascading failures by suspending Redis operations
    after a threshold of consecutive failures is reached. It automatically 
    attempts recovery after a specified time interval.
    """
    
    error_threshold: int = 5
    recovery_timeout: int = 30  # seconds until retry
    half_open_max_operations: int = 3
    state: str = "closed"  # closed, open, half-open
    last_state_change: datetime = field(default_factory=datetime.now)
    half_open_successes: int = 0
    metrics: ErrorMetrics = field(default_factory=ErrorMetrics)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def record_success(self) -> None:
        """Record a successful operation"""
        with self._lock:
            self.metrics.record_operation(True)
            
            if self.state == "half-open":
                self.half_open_successes += 1
                
                if self.half_open_successes >= self.half_open_max_operations:
                    # Transition to closed state
                    self.state = "closed"
                    self.last_state_change = datetime.now()
                    self.half_open_successes = 0
                    logger.info("Circuit breaker closed - Redis operations resumed")
    
    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed operation
        
        Args:
            error: The exception that occurred
        """
        with self._lock:
            self.metrics.record_operation(False, error)
            
            if self.state == "closed" and self.metrics.consecutive_failures >= self.error_threshold:
                # Transition to open state
                self.state = "open"
                self.last_state_change = datetime.now()
                logger.warning(
                    f"Circuit breaker opened after {self.error_threshold} consecutive failures. "
                    f"Redis operations suspended for {self.recovery_timeout}s"
                )
            elif self.state == "half-open":
                # Any failure in half-open state reopens the circuit
                self.state = "open"
                self.last_state_change = datetime.now()
                self.half_open_successes = 0
                logger.warning("Circuit breaker reopened after failure in half-open state")
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed
        
        Returns:
            bool: True if the request should be allowed
        """
        with self._lock:
            if self.state == "closed":
                return True
                
            if self.state == "open":
                timeout_expired = datetime.now() > self.last_state_change + timedelta(seconds=self.recovery_timeout)
                
                if timeout_expired:
                    # Transition to half-open state
                    self.state = "half-open"
                    self.last_state_change = datetime.now()
                    self.half_open_successes = 0
                    logger.info("Circuit breaker half-open - Testing Redis connection")
                    return True
                    
                return False
                
            # In half-open state, allow limited requests
            return True
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state"""
        with self._lock:
            self.state = "closed"
            self.last_state_change = datetime.now()
            self.half_open_successes = 0
            self.metrics.reset()
            logger.info("Circuit breaker reset to closed state")


class FallbackStrategy:
    """Implements fallback strategies for Redis operations when they fail
    
    This class provides methods to handle Redis operation failures gracefully
    by returning default values, using alternative data sources, or retrying.
    """
    
    @staticmethod
    def default_value(default: T = None) -> T:
        """Return a default value
        
        Args:
            default: Default value to return
            
        Returns:
            Default value
        """
        return default
    
    @staticmethod
    def retry(
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        backoff_max: float = 5.0,
        retryable_exceptions: Tuple[Exception, ...] = (ConnectionError, TimeoutError)
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that retries a function on failure
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor for retries
            backoff_max: Maximum backoff time in seconds
            retryable_exceptions: Exceptions that trigger a retry
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception = None
                for retry_count in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e
                        if retry_count < max_retries:
                            # Calculate backoff time
                            backoff_time = min(backoff_factor * (2 ** retry_count), backoff_max)
                            logger.warning(
                                f"Retrying {func.__name__} after {backoff_time}s "
                                f"(attempt {retry_count + 1}/{max_retries})"
                            )
                            time.sleep(backoff_time)
                        else:
                            logger.error(
                                f"All {max_retries} retries failed for {func.__name__}"
                            )
                
                # Re-raise the last exception
                raise last_exception
            
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception = None
                for retry_count in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e
                        if retry_count < max_retries:
                            # Calculate backoff time
                            backoff_time = min(backoff_factor * (2 ** retry_count), backoff_max)
                            logger.warning(
                                f"Retrying {func.__name__} after {backoff_time}s "
                                f"(attempt {retry_count + 1}/{max_retries})"
                            )
                            await asyncio.sleep(backoff_time)
                        else:
                            logger.error(
                                f"All {max_retries} retries failed for {func.__name__}"
                            )
                
                # Re-raise the last exception
                raise last_exception
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
    fallback_result: Any = None,
    log_failures: bool = True
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:
    """Decorator to apply circuit breaker pattern to a function
    
    Args:
        circuit_breaker: CircuitBreaker instance
        fallback_result: Result to return when circuit is open
        log_failures: Whether to log failures
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            if not circuit_breaker.allow_request():
                if log_failures:
                    logger.warning(
                        f"Circuit breaker open, skipping {func.__name__}"
                    )
                return fallback_result
                
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure(e)
                if log_failures:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}"
                    )
                raise
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            if not circuit_breaker.allow_request():
                if log_failures:
                    logger.warning(
                        f"Circuit breaker open, skipping {func.__name__}"
                    )
                return fallback_result
                
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure(e)
                if log_failures:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}"
                    )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def safe_cache_operation(
    fallback_result: Any = None,
    log_failures: bool = True,
    max_retries: int = 2,
    retryable_exceptions: Tuple[Exception, ...] = (ConnectionError, TimeoutError)
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:
    """Decorator that wraps cache operations with error handling
    
    This decorator combines retry logic with graceful fallback when
    cache operations fail completely.
    
    Args:
        fallback_result: Result to return when all retries fail
        log_failures: Whether to log failures
        max_retries: Maximum number of retry attempts
        retryable_exceptions: Exceptions that trigger a retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        # Apply retry decorator first
        retry_decorator = FallbackStrategy.retry(
            max_retries=max_retries,
            retryable_exceptions=retryable_exceptions
        )
        retrying_func = retry_decorator(func)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return retrying_func(*args, **kwargs)
            except Exception as e:
                if log_failures:
                    logger.error(
                        f"Cache operation {func.__name__} failed after {max_retries} retries: {str(e)}"
                    )
                return fallback_result
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return await retrying_func(*args, **kwargs)
            except Exception as e:
                if log_failures:
                    logger.error(
                        f"Cache operation {func.__name__} failed after {max_retries} retries: {str(e)}"
                    )
                return fallback_result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Create a global circuit breaker instance
default_circuit_breaker = CircuitBreaker()


def get_global_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance
    
    Returns:
        CircuitBreaker: Global circuit breaker instance
    """
    return default_circuit_breaker


def handle_serialization_error(
    func: Optional[Callable] = None,
    default_value: Any = None
) -> Any:
    """Decorator to handle serialization errors
    
    Args:
        func: Function to decorate
        default_value: Default value to return on error
        
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return f(*args, **kwargs)
            except (SerializationError, DeserializationError) as e:
                logger.error(f"Serialization error in {f.__name__}: {str(e)}")
                return default_value
        
        @wraps(f)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await f(*args, **kwargs)
            except (SerializationError, DeserializationError) as e:
                logger.error(f"Serialization error in {f.__name__}: {str(e)}")
                return default_value
        
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper
    
    if func is None:
        return decorator
    return decorator(func)
