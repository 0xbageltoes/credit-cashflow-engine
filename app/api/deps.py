"""
Dependencies for the FastAPI application that handle
authentication and common service injection
"""
from typing import Dict, Optional
import logging
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.auth import decode_jwt_token
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.database.supabase import SupabaseClient

# Set up security scheme for JWT
security = HTTPBearer()
logger = logging.getLogger(__name__)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Validate the JWT token and return the user information
    
    This dependency is used to protect API endpoints that require authentication.
    It verifies the JWT token in the Authorization header and returns the user 
    information if valid.
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

async def get_absbox_service() -> AbsBoxServiceEnhanced:
    """
    Dependency to inject the enhanced AbsBox service
    
    This centralizes the creation of the AbsBox service and ensures
    consistent configuration across the application.
    """
    return AbsBoxServiceEnhanced()

async def get_database_client() -> SupabaseClient:
    """
    Dependency to inject the Supabase database client
    
    This centralizes the creation of the database client and ensures
    consistent configuration across the application.
    """
    return SupabaseClient()

async def check_admin_access(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Check if the user has admin access
    
    This dependency is used to protect API endpoints that require admin privileges.
    It verifies that the user has the admin role.
    """
    if "admin" not in current_user.get("roles", []):
        logger.warning(f"User {current_user.get('id')} attempted admin access without permissions")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
