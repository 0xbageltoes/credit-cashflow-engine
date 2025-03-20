"""
Unified error handling framework for the credit-cashflow-engine.

This module provides a consistent pattern for error handling across the application with:
1. Base error classes with context capturing
2. Specialized error types for different domains
3. Error handling decorators for both sync and async functions
4. Helpers for API error responses
"""

import asyncio
import functools
import inspect
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

from fastapi import status
from fastapi.responses import JSONResponse

# Type variable for function return type
T = TypeVar('T')

class ApplicationError(Exception):
    """Base class for all application errors.
    
    Provides structured error information including context, original cause,
    and timestamp for better error tracking and debugging.
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize application error with context and cause.
        
        Args:
            message: Human-readable error message
            context: Additional context information about the error
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization.
        
        Returns:
            Dictionary with error details including type, message, timestamp,
            context, and cause information
        """
        result = {
            "error_type": self.__class__.__name__,
            "error_message": str(self),
            "timestamp": self.timestamp,
            **self.context
        }
        
        if self.cause:
            result["cause"] = {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def __str__(self) -> str:
        """Get string representation of the error.
        
        Returns:
            Error message with context if available
        """
        base_message = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_message} [{context_str}]"
        return base_message


# Domain-specific error classes

class CalculationError(ApplicationError):
    """Error in financial calculations."""
    pass


class DataError(ApplicationError):
    """Error in data operations such as validation, parsing, or processing."""
    pass


class ServiceError(ApplicationError):
    """Error in service operations such as external API calls."""
    
    def __init__(
        self, 
        message: str, 
        code: int = 500, 
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.code = code
        self.details = details
        self.context = context or {}
        self.original_error = original_error
        self.cause = cause
        super().__init__(message, context, cause)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization.
        
        Returns:
            Dictionary representation of error with all safe attributes
        """
        # Create a safe copy that doesn't include message key to avoid logging conflicts
        error_dict = {
            'error_message': self.message,
            'error_code': self.code,
            'error_type': self.__class__.__name__,
            'error_context': self.context
        }
        
        if self.details:
            error_dict['error_details'] = self.details
            
        return error_dict


class ConfigurationError(ApplicationError):
    """Error in configuration settings or environment setup."""
    pass


class CacheError(ApplicationError):
    """Error in caching operations such as Redis cache access."""
    pass


class DatabaseError(ApplicationError):
    """Error in database operations."""
    pass


class ValidationError(ApplicationError):
    """Error in input validation."""
    pass


# Error handling decorators

def handle_errors(
    logger: Optional[logging.Logger] = None,
    error_mapping: Optional[Dict[Type[Exception], Type[ApplicationError]]] = None,
    default_error: Type[ApplicationError] = ServiceError,
    context: Optional[Dict[str, Any]] = None
) -> Callable:
    """Decorator for handling errors with context.
    
    This decorator wraps a function to catch exceptions, convert them to
    ApplicationError types, log them properly, and re-raise for consistent
    error handling.
    
    Args:
        logger: Logger to use for error logging
        error_mapping: Mapping from exception types to ApplicationError types
        default_error: Default ApplicationError type to use
        context: Additional context to include in errors
        
    Returns:
        Decorated function that handles errors consistently
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get logger if not provided
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        # Define sync wrapper
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to ServiceError if it's not already
                if not isinstance(e, ServiceError):
                    error = ServiceError(
                        message=str(e),
                        code=500,
                        details=traceback.format_exc(),
                        original_error=e
                    )
                else:
                    error = e
                
                # Log the error safely without conflicting log record keys
                try:
                    # Get error properties in a way that avoids 'message' key conflict
                    error_dict = error.to_dict()
                    
                    # Create a clean extra dictionary that won't conflict with LogRecord attributes
                    safe_extras = {}
                    for key, value in error_dict.items():
                        # Skip any keys that could conflict with LogRecord attributes
                        if key not in ('message', 'levelname', 'pathname', 'lineno'):
                            safe_extras[key] = value
                    
                    # Log with the error message in the message parameter
                    logger.error(
                        f"{error.__class__.__name__}: {str(error)}", 
                        exc_info=True,
                        extra=safe_extras
                    )
                except Exception as logging_error:
                    # Fallback for when logging itself fails
                    try:
                        logger.error(
                            f"Error in logging: {str(logging_error)}. Original error: {str(error)}",
                            exc_info=True
                        )
                    except:
                        # Ultimate fallback if all logging fails
                        print(f"CRITICAL: Logging failure. Error was: {str(error)}")
                    
                raise error
        
        # Define async wrapper
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except ApplicationError as e:
                # Already an application error, just log and re-raise
                logger.error(f"{e.__class__.__name__}: {str(e)}", extra=e.to_dict())
                raise
            except Exception as e:
                # Convert to application error
                error_type = error_mapping.get(type(e), default_error) if error_mapping else default_error
                
                # Create error with context
                function_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "traceback": traceback.format_exc(),
                    **(context or {})
                }
                
                # Add args and kwargs summaries (safely)
                try:
                    # Try to add args and kwargs, but don't fail if they can't be converted to string
                    if args:
                        # Summarize args to avoid huge objects
                        arg_summary = [f"{type(arg).__name__}:{str(arg)[:100]}" for arg in args]
                        function_context["args"] = arg_summary
                    
                    if kwargs:
                        # Summarize kwargs to avoid huge objects
                        kwargs_summary = {k: f"{type(v).__name__}:{str(v)[:100]}" for k, v in kwargs.items()}
                        function_context["kwargs"] = kwargs_summary
                except Exception:
                    # If summarizing args/kwargs fails, just continue without them
                    pass
                
                error = error_type(
                    message=str(e),
                    context=function_context,
                    cause=e
                )
                
                # Log error
                logger.error(f"{error.__class__.__name__}: {str(error)}", extra=error.to_dict())
                
                # Re-raise application error
                raise error
        
        # Determine if the function is async or sync
        if asyncio.iscoroutinefunction(func):
            return functools.wraps(func)(async_wrapper)
        else:
            return functools.wraps(func)(sync_wrapper)
    
    return decorator


# FastAPI exception handlers

def register_exception_handlers(app):
    """Register exception handlers for the application.
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(ApplicationError)
    async def application_error_handler(request, exc):
        """Handle application errors.
        
        Args:
            request: FastAPI request
            exc: ApplicationError instance
            
        Returns:
            JSONResponse with error details
        """
        # Log the error
        logger = logging.getLogger("api")
        logger.error(f"API error: {exc}", extra=exc.to_dict())
        
        # Determine status code based on error type
        status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, ValidationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(exc, ServiceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, DataError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, ConfigurationError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(exc, CacheError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # Return JSON response
        return JSONResponse(
            status_code=status_code,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "timestamp": exc.timestamp,
                # Only include safe context fields
                "context": {k: v for k, v in exc.context.items() if k not in ["traceback", "args", "kwargs"]}
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Handle all unhandled exceptions.
        
        Args:
            request: FastAPI request
            exc: Exception instance
            
        Returns:
            JSONResponse with error details
        """
        # Log the error
        logger = logging.getLogger("api")
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        
        # Don't expose internal details in production
        from app.core.config import settings
        is_production = settings.ENVIRONMENT == "production"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error" if is_production else str(exc),
                "type": "InternalServerError" if is_production else exc.__class__.__name__,
                "timestamp": datetime.now().isoformat()
            }
        )


# Helper functions for working with errors

def extract_error_info(exc: Exception) -> Dict[str, Any]:
    """Extract useful error information from an exception.
    
    Args:
        exc: Exception to extract information from
    
    Returns:
        Dictionary with error information
    """
    if isinstance(exc, ApplicationError):
        return exc.to_dict()
    
    return {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }


def safely_run(func: Callable[..., T], *args: Any, **kwargs: Any) -> Union[T, None]:
    """Run a function safely, returning None on error.
    
    Args:
        func: Function to run
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Function result or None on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in safely_run: {str(e)}", exc_info=True)
        return None


async def safely_run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> Union[T, None]:
    """Run an async function safely, returning None on error.
    
    Args:
        func: Async function to run
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Function result or None on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in safely_run_async: {str(e)}", exc_info=True)
        return None


def from_exception(
    exc: Exception,
    error_type: Type[ApplicationError] = ApplicationError,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> ApplicationError:
    """Create an ApplicationError from a regular exception.
    
    Args:
        exc: Exception to convert
        error_type: ApplicationError type to create
        message: Optional message override
        context: Additional context
        
    Returns:
        ApplicationError instance
    """
    return error_type(
        message=message or str(exc),
        context=context or {},
        cause=exc
    )
