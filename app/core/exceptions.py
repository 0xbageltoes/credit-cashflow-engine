"""
Custom exceptions for the application

This module defines custom exceptions used throughout the application.
"""
from typing import Any, Dict, Optional, Type
import traceback
from datetime import datetime

from fastapi import HTTPException
from starlette import status


class AuthHTTPException(HTTPException):
    """
    Custom HTTP Exception for authentication errors with additional error_code

    This exception extends the standard FastAPI HTTPException to include
    an error_code that can be used for more specific error handling.
    """
    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize the exception

        Args:
            status_code: HTTP status code
            detail: Additional details about the error
            headers: Optional HTTP headers to include in the response
            error_code: A machine-readable error code for automated handling
        """
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


# Define common authentication exceptions - these return instances not functions
not_authenticated_exception = AuthHTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing authorization credentials",
    headers={"WWW-Authenticate": "Bearer"},
    error_code="not_authenticated"
)

invalid_credentials_exception = AuthHTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid authentication credentials",
    headers={"WWW-Authenticate": "Bearer"},
    error_code="invalid_credentials"
)

invalid_token_exception = AuthHTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
    error_code="invalid_token"
)

token_blacklisted_exception = AuthHTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Token has been invalidated",
    headers={"WWW-Authenticate": "Bearer"},
    error_code="token_blacklisted"
)

insufficient_permissions_exception = AuthHTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Insufficient permissions to perform this action",
    error_code="insufficient_permissions"
)

server_error_exception = AuthHTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="An unexpected error occurred",
    error_code="server_error"
)


# Standard application exception hierarchy

class ApplicationError(Exception):
    """Base class for all application errors
    
    This provides a consistent way to handle errors across the application,
    with support for error context, cause tracking, and logging.
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the application error
        
        Args:
            message: Human-readable error message
            context: Additional context about the error (e.g. parameters, state)
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization
        
        Returns:
            Dict containing error details
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "timestamp": self.timestamp,
            **self.context
        }
        
        if self.cause:
            result["cause"] = {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def to_http_exception(self, status_code: int = 500) -> HTTPException:
        """Convert to HTTPException for API responses
        
        Args:
            status_code: HTTP status code to use
            
        Returns:
            HTTPException: Exception to return from API endpoints
        """
        return HTTPException(
            status_code=status_code,
            detail={
                "error_type": self.__class__.__name__,
                "message": str(self),
                "context": {
                    k: v for k, v in self.context.items() 
                    if not k.startswith("_")  # Don't expose internal details
                }
            }
        )


class CalculationError(ApplicationError):
    """Error in financial calculations"""
    pass


class DataError(ApplicationError):
    """Error in data operations"""
    pass


class ServiceError(ApplicationError):
    """Error in service operations"""
    pass


class ConfigurationError(ApplicationError):
    """Error in service configuration"""
    pass


class ValidationError(ApplicationError):
    """Error in data validation"""
    pass


class CacheError(ApplicationError):
    """Error in cache operations"""
    pass


def handle_errors(
    logger,
    error_mapping: Optional[Dict[Type[Exception], Type[ApplicationError]]] = None,
    default_error: Type[ApplicationError] = ServiceError,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator for handling errors with context
    
    This decorator wraps async functions to provide consistent error handling.
    It will catch exceptions, convert them to ApplicationError types if needed,
    log them appropriately, and re-raise them.
    
    Args:
        logger: Logger instance to use for error logging
        error_mapping: Mapping from exception types to ApplicationError types
        default_error: Default ApplicationError type for unmapped exceptions
        context: Additional context to include in all errors
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
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
                error = error_type(
                    message=str(e),
                    context={
                        "function": func.__name__,
                        "args_summary": str(args),
                        "kwargs_summary": str(kwargs),
                        "traceback": traceback.format_exc(),
                        **(context or {})
                    },
                    cause=e
                )
                
                # Log error
                logger.error(f"{error.__class__.__name__}: {str(error)}", extra=error.to_dict())
                
                # Re-raise application error
                raise error
        
        return wrapper
    
    return decorator


def create_auth_exception(
    status_code: int, 
    detail: str, 
    error_code: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> AuthHTTPException:
    """
    Create a custom auth exception with the given parameters

    Args:
        status_code: HTTP status code
        detail: Error message
        error_code: Optional error code
        headers: Optional HTTP headers

    Returns:
        AuthHTTPException: The created exception
    """
    return AuthHTTPException(
        status_code=status_code,
        detail=detail,
        error_code=error_code,
        headers=headers
    )
