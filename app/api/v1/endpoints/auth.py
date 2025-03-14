"""
Authentication endpoints for the API

This module provides endpoints for authentication and authorization functions.
It includes token verification, refresh, and user information endpoints with
comprehensive error handling and security measures.
"""
import logging
import time
import json
from typing import Dict, Optional, Any, List

from fastapi import APIRouter, Depends, Header, Request, status, Cookie, Response, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr

from app.core.security import (
    verify_supabase_jwt,
    refresh_token,
    get_current_user_id,
    get_current_user,
    require_auth,
    require_admin_auth,
    logout_user,
    blacklist_token,
    invalidate_all_user_tokens,
    create_tokens
)
from app.core.config import settings
from app.services.supabase_service import SupabaseService
from app.services.redis_service import RedisService
from app.core.exceptions import (
    AuthHTTPException, 
    invalid_credentials_exception,
    invalid_token_exception,
    token_blacklisted_exception,
    insufficient_permissions_exception,
    not_authenticated_exception,
    server_error_exception
)

# Setup logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Create router
router = APIRouter()

# Initialize services
supabase_service = SupabaseService()
redis_service = RedisService() if settings.REDIS_ENABLED else None

# Create service dependency functions for better testability
def get_redis_service():
    """Get Redis service dependency"""
    return redis_service

def get_supabase_service():
    """Get Supabase service dependency"""
    return supabase_service

# Models
class TokenResponse(BaseModel):
    """Response model for token verification"""
    user_id: str
    email: Optional[str] = None
    roles: Optional[List[str]] = Field(default_factory=list)
    exp: int = Field(..., description="Token expiration timestamp")
    issued_at: int = Field(..., description="Token issue timestamp")
    jti: str = Field(..., description="JWT ID")

class RefreshRequest(BaseModel):
    """Request model for token refresh"""
    refresh_token: str

class RefreshResponse(BaseModel):
    """Response model for token refresh"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration in seconds")

class LogoutRequest(BaseModel):
    """Request model for logout"""
    all_devices: bool = Field(False, description="Whether to logout from all devices")

class AuthErrorResponse(BaseModel):
    """Response model for authentication errors"""
    detail: str
    error_code: Optional[str] = None
    status_code: int = 401

class UserInfoResponse(BaseModel):
    """Response model for user information"""
    id: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    display_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None

@router.get(
    "/verify-token",
    response_model=TokenResponse,
    responses={
        401: {"model": AuthErrorResponse, "description": "Invalid or expired token"},
        403: {"model": AuthErrorResponse, "description": "Token blacklisted or revoked"},
        500: {"model": AuthErrorResponse, "description": "Internal server error"}
    },
    description="Verify a JWT token and return user information",
    summary="Verify JWT token"
)
async def verify_jwt_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_forwarded_for: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
):
    """
    Verify a JWT token and return user information
    
    This endpoint verifies the token signature, expiration, and issuer,
    and returns the user information if valid.
    
    Args:
        request: The request
        credentials: The authorization credentials
        x_forwarded_for: Forwarded IP address
        user_agent: User agent string
        
    Returns:
        User information from the token
        
    Raises:
        AuthHTTPException: If the token is invalid or expired
    """
    client_ip = x_forwarded_for or request.client.host
    
    if not credentials:
        logger.warning(f"Missing authorization credentials from {client_ip}")
        raise not_authenticated_exception
    
    token = credentials.credentials
    
    try:
        # Verify the token
        payload = await verify_supabase_jwt(token)
        
        if not payload:
            logger.warning(f"Invalid or expired token from {client_ip}")
            raise invalid_token_exception
    
        # Extract user information
        user_id = payload.get("sub")
        if not user_id:
            logger.warning(f"Invalid token payload from {client_ip}")
            raise invalid_token_exception
        
        # Get expiration time
        exp = payload.get("exp", 0)
        now = int(time.time())
        if exp < now:
            logger.warning(f"Token expired for user {user_id} from {client_ip}")
            raise invalid_token_exception
        
        # Extract additional information
        email = payload.get("email")
        roles = payload.get("user_roles", [])
        jti = payload.get("jti", "")
        issued_at = payload.get("iat", 0)
        
        # If needed, verify additional user information from Supabase
        # This is optional and can be done asynchronously if performance is a concern
        try:
            if settings.VERIFY_USER_ON_AUTH and supabase_service:
                user_info = await supabase_service.get_user(user_id)
                
                # Verify user status
                if not user_info or user_info.get("status") != "active":
                    logger.warning(f"User {user_id} account is suspended or deleted")
                    
                    # Blacklist the token
                    if jti:
                        await blacklist_token(payload, reason="account_suspended")
                    
                    raise token_blacklisted_exception
        
        except AuthHTTPException:
            # Re-raise AuthHTTPException
            raise
            
        except Exception as e:
            # Log the error but continue with token info
            logger.error(f"Error verifying user {user_id} in Supabase: {str(e)}")
        
        # Log successful verification
        logger.info(f"Token verified for user {user_id} from {client_ip}")
        
        # Record token usage
        if redis_service and jti:
            try:
                usage_key = f"jwt_usage:{jti}"
                usage_data = {
                    "last_used": now,
                    "ip": client_ip,
                    "user_agent": user_agent or "unknown"
                }
                
                # Store with the same TTL as the token
                ttl = max(0, exp - now)
                await redis_service.set(
                    usage_key,
                    json.dumps(usage_data),
                    ttl=ttl
                )
            except Exception as e:
                logger.error(f"Error recording token usage: {str(e)}")
        
        return {
            "user_id": user_id,
            "email": email,
            "roles": roles,
            "exp": exp,
            "issued_at": issued_at,
            "jti": jti
        }
    
    except AuthHTTPException:
        # Re-raise AuthHTTPException
        raise
    
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        
        raise server_error_exception

@router.post(
    "/refresh",
    response_model=RefreshResponse,
    responses={
        401: {"model": AuthErrorResponse, "description": "Invalid or expired refresh token"},
        403: {"model": AuthErrorResponse, "description": "Token blacklisted or revoked"},
        500: {"model": AuthErrorResponse, "description": "Internal server error"}
    },
    description="Refresh an access token using a refresh token",
    summary="Refresh access token"
)
async def refresh_access_token(
    request: Request,
    refresh_req: RefreshRequest,
    x_forwarded_for: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
):
    """
    Refresh an access token using a refresh token
    
    This endpoint refreshes an access token using a refresh token.
    The old refresh token is blacklisted after successful refresh.
    
    Args:
        request: The request object
        refresh_req: The refresh request containing the refresh token
        x_forwarded_for: Forwarded IP address
        user_agent: User agent string
        
    Returns:
        New access and refresh tokens
        
    Raises:
        AuthHTTPException: If the refresh token is invalid or expired
    """
    client_ip = x_forwarded_for or request.client.host
    
    try:
        # Refresh the token
        tokens = await refresh_token(refresh_req.refresh_token)
        
        if not tokens:
            logger.warning(f"Invalid or expired refresh token from {client_ip}")
            raise invalid_token_exception(error_code="invalid_refresh_token")
        
        # Calculate expiration time
        expires_in = settings.ACCESS_TOKEN_EXPIRE_SECONDS
        
        # Log successful refresh
        logger.info(f"Token refreshed from {client_ip}")
        
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": tokens["token_type"],
            "expires_in": expires_in
        }
    
    except AuthHTTPException:
        # Re-raise AuthHTTPException
        raise
    
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        
        raise server_error_exception(error_code="refresh_error")

@router.get(
    "/user",
    response_model=UserInfoResponse,
    responses={
        401: {"model": AuthErrorResponse, "description": "Not authenticated"},
        403: {"model": AuthErrorResponse, "description": "Not authorized"},
        500: {"model": AuthErrorResponse, "description": "Internal server error"}
    },
    description="Get current user information",
    summary="Get current user info"
)
async def get_current_user_info(
    request: Request
):
    """
    Get current user information
    
    This endpoint returns information about the currently authenticated user.
    
    Args:
        request: The request
        
    Returns:
        User information
        
    Raises:
        AuthHTTPException: If not authenticated
    """
    try:
        # Get current user ID
        user_data = await get_current_user(request)
        
        if not user_data:
            raise not_authenticated_exception
        
        user_id = user_data["id"]
        
        # Get user information from Supabase
        user_info = None
        try:
            if supabase_service:
                user_info = await supabase_service.get_user(user_id)
        except Exception as e:
            logger.error(f"Error getting user information from Supabase: {str(e)}")
            # Continue with limited information if Supabase access fails
        
        # Prepare response
        response = {
            "id": user_id,
            "email": user_data.get("email") or (user_info.get("email") if user_info else None),
            "roles": user_data.get("roles") or [],
            "display_name": (user_info.get("user_metadata", {}).get("full_name") if user_info else None),
            "metadata": user_data.get("user_metadata") or {},
            "created_at": (user_info.get("created_at") if user_info else None)
        }
        
        return response
    
    except AuthHTTPException:
        # Re-raise AuthHTTPException
        raise
    
    except Exception as e:
        logger.error(f"Error getting user information: {str(e)}")
        
        raise server_error_exception

@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": AuthErrorResponse, "description": "Not authenticated"},
        500: {"model": AuthErrorResponse, "description": "Internal server error"}
    },
    description="Logout the current user and invalidate tokens",
    summary="Logout user"
)
async def logout(
    request: Request,
    response: Response,
    logout_req: LogoutRequest = None,
    refresh_token: Optional[str] = Cookie(None, alias="refresh_token"),
    x_forwarded_for: Optional[str] = Header(None)
):
    """
    Logout the current user
    
    This endpoint invalidates the user's tokens and clears auth cookies.
    
    Args:
        request: The request
        response: The response
        logout_req: Optional logout request containing logout options
        refresh_token: The refresh token cookie
        x_forwarded_for: Forwarded IP address
        
    Returns:
        Success message
        
    Raises:
        AuthHTTPException: If an error occurs
    """
    client_ip = x_forwarded_for or request.client.host
    all_devices = False
    
    if logout_req:
        all_devices = logout_req.all_devices
    
    try:
        # Logout the user
        success = await logout_user(request, all_devices=all_devices)
        
        if not success:
            logger.warning(f"Failed to logout user from {client_ip}")
            raise server_error_exception
        
        # Also try to invalidate refresh token from cookie if present
        if refresh_token:
            try:
                # Make a request to Supabase to invalidate the refresh token
                await supabase_service.invalidate_refresh_token(refresh_token)
            except Exception as e:
                logger.warning(f"Error invalidating refresh token in Supabase: {str(e)}")
                # Continue even if this fails
        
        # Clear cookies
        response.delete_cookie(
            key="refresh_token",
            path="/",
            domain=None,
            secure=settings.SECURE_COOKIES,
            httponly=True
        )
        
        response.delete_cookie(
            key="auth_token",
            path="/",
            domain=None,
            secure=settings.SECURE_COOKIES,
            httponly=True
        )
        
        logger.info(f"User logged out from {client_ip}" + (" (all devices)" if all_devices else ""))
        
        return {"message": "Successfully logged out"}
    
    except AuthHTTPException:
        # Re-raise AuthHTTPException
        raise
    
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        
        raise server_error_exception

@router.post(
    "/invalidate-user-tokens",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": AuthErrorResponse, "description": "Not authenticated"},
        403: {"model": AuthErrorResponse, "description": "Not authorized"},
        500: {"model": AuthErrorResponse, "description": "Internal server error"}
    },
    description="Invalidate all tokens for a user (admin only)",
    summary="Invalidate user tokens"
)
async def invalidate_user_tokens(
    request: Request,
    user_id: str = Query(..., description="User ID to invalidate tokens for")
):
    """
    Invalidate all tokens for a user (admin only)
    
    This endpoint invalidates all tokens for a specific user.
    Requires admin privileges.
    
    Args:
        request: The request
        user_id: The user ID to invalidate tokens for
        
    Returns:
        Success message with count of invalidated tokens
        
    Raises:
        AuthHTTPException: If not authenticated or not authorized
    """
    try:
        # Require admin authentication
        admin_id = await require_admin_auth(request)
        
        # Log the admin action
        logger.info(f"Admin {admin_id} is invalidating tokens for user {user_id}")
        
        # Invalidate all tokens for the user
        count = await invalidate_all_user_tokens(user_id)
        
        # Also try to notify Supabase (if applicable)
        try:
            if supabase_service:
                await supabase_service.invalidate_all_user_sessions(user_id)
        except Exception as e:
            logger.warning(f"Error invalidating user sessions in Supabase: {str(e)}")
            # Continue even if this fails
        
        return {
            "message": f"Successfully invalidated tokens for user {user_id}",
            "invalidated_count": count
        }
    
    except AuthHTTPException:
        # Re-raise AuthHTTPException
        raise
    
    except Exception as e:
        logger.error(f"Error invalidating user tokens: {str(e)}")
        
        raise server_error_exception

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    description="Health check for authentication system",
    summary="Auth health check"
)
async def auth_health_check(
    redis: Optional[RedisService] = Depends(get_redis_service),
    supabase: Optional[SupabaseService] = Depends(get_supabase_service)
):
    """
    Health check for authentication system
    
    This endpoint checks the health of authentication-related services.
    
    Returns:
        Health status of authentication services
    """
    # Initialize with healthy state
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": int(time.time())
    }
    
    # Track overall system health
    all_services_healthy = True
    
    # Check Redis health if enabled
    if redis:
        try:
            redis_healthy = await redis.health_check()
            logger.info(f"Redis health check returned: {redis_healthy}")
            health_status["services"]["redis"] = "available" if redis_healthy else "unavailable"
            
            # Update overall health status
            if not redis_healthy:
                all_services_healthy = False
                logger.warning("Redis service is unhealthy")
        except Exception as e:
            logger.error(f"Error checking Redis health: {str(e)}")
            health_status["services"]["redis"] = "unavailable"
            all_services_healthy = False
    else:
        # If Redis is not configured, mark as unavailable
        health_status["services"]["redis"] = "not_configured"
        logger.warning("Redis service is not configured")
    
    # Check Supabase health if available
    if supabase:
        try:
            supabase_healthy = await supabase.health_check()
            logger.info(f"Supabase health check returned: {supabase_healthy}")
            health_status["services"]["supabase"] = "available" if supabase_healthy else "unavailable"
            
            # Update overall health status
            if not supabase_healthy:
                all_services_healthy = False
                logger.warning("Supabase service is unhealthy")
        except Exception as e:
            logger.error(f"Error checking Supabase health: {str(e)}")
            health_status["services"]["supabase"] = "unavailable"
            all_services_healthy = False
    else:
        # If Supabase is not configured, mark as unavailable
        health_status["services"]["supabase"] = "not_configured" 
        logger.warning("Supabase service is not configured")
        all_services_healthy = False
    
    # Set final status based on overall health
    if not all_services_healthy:
        health_status["status"] = "degraded"
        logger.warning("Auth service health status: DEGRADED")
    else:
        logger.info("Auth service health status: HEALTHY")
    
    return health_status
