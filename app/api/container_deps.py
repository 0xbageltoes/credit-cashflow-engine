"""
FastAPI Dependency Integration 

This module provides FastAPI dependency functions that utilize the dependency injection container.
It bridges the gap between FastAPI's dependency injection system and our application's service container.
"""

import logging
from typing import Dict, Optional, Type, TypeVar, Callable, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.dependency_injection import container, resolve_service
from app.core.auth import decode_jwt_token
from app.core.error_handling import handle_errors, ServiceError, ApplicationError
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.services.asset_handlers.consumer_credit import ConsumerCreditHandler
from app.services.asset_handlers.commercial_loan import CommercialLoanHandler
from app.services.asset_handlers.clo_cdo import CLOCDOHandler
from app.database.supabase import SupabaseClient
from app.core.cache_service import CacheService

# Type variable for generic type hints
T = TypeVar('T')

# Set up security scheme for JWT
security = HTTPBearer()
logger = logging.getLogger(__name__)


def get_service(service_type: Type[T]) -> Callable[[], T]:
    """Create a FastAPI dependency function for resolving services from the container.
    
    This is a factory function that creates FastAPI dependency functions. It allows
    for clean dependency resolution in FastAPI endpoints.
    
    Args:
        service_type: Type of service to resolve
        
    Returns:
        A FastAPI dependency function that resolves the requested service
    
    Example:
        ```python
        # In a dependency module
        get_cache_service = get_service(CacheService)
        
        # In an API endpoint
        @router.get("/items")
        async def get_items(cache_service: CacheService = Depends(get_cache_service)):
            # Use cache_service here
            ...
        ```
    """
    @handle_errors(logger=logger, default_error=ServiceError)
    def resolve() -> T:
        """Resolve the service from the container"""
        try:
            return container.resolve(service_type)
        except Exception as e:
            logger.error(f"Failed to resolve service {service_type.__name__}: {str(e)}")
            raise ServiceError(
                message=f"Service dependency resolution failed: {service_type.__name__}",
                context={"service_type": service_type.__name__},
                cause=e
            )
    
    return resolve


# Authentication dependencies

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """Validate the JWT token and return the user information.
    
    This dependency is used to protect API endpoints that require authentication.
    It verifies the JWT token in the Authorization header and returns the user 
    information if valid.
    
    Args:
        credentials: JWT credentials from the Authorization header
        
    Returns:
        Dictionary containing user information from the JWT token
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        user_data = decode_jwt_token(token)
        
        if not user_data:
            logger.warning("Invalid token provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_data
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_admin_access(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Check if the user has admin access.
    
    This dependency is used to protect API endpoints that require admin privileges.
    It verifies that the user has the admin role.
    
    Args:
        current_user: User information from the JWT token
        
    Returns:
        User information if admin access is granted
        
    Raises:
        HTTPException: If user doesn't have admin role
    """
    if "admin" not in current_user.get("roles", []):
        logger.warning(f"User {current_user.get('id')} attempted admin access without permissions")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# Service dependencies - using the get_service factory

# Core services
get_cache_service = get_service(CacheService)
get_database_client = get_service(SupabaseClient)

# Business services
get_absbox_service = get_service(AbsBoxServiceEnhanced)
get_consumer_credit_handler = get_service(ConsumerCreditHandler)
get_commercial_loan_handler = get_service(CommercialLoanHandler)
get_clo_cdo_handler = get_service(CLOCDOHandler)


# Example of a custom dependency that combines multiple services
@handle_errors(logger=logger)
async def get_analytics_context():
    """
    Dependency to inject multiple services as an analytics context.
    
    This demonstrates how to combine multiple services into a single dependency.
    
    Returns:
        Dictionary containing service instances
    """
    return {
        "absbox_service": container.resolve(AbsBoxServiceEnhanced),
        "cache_service": container.resolve(CacheService),
        "database": container.resolve(SupabaseClient),
    }
