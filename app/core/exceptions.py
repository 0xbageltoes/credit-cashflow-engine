"""
Custom exceptions for the application

This module defines custom exceptions used throughout the application.
"""
from typing import Any, Dict, Optional

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


# Helper function to create custom error exceptions when needed
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
        headers=headers,
        error_code=error_code
    )
