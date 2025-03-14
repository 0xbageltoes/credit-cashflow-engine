from typing import Callable, Dict, Optional, Union, Any, List, Tuple
from datetime import datetime, timedelta
import time
import logging
import secrets
import json
import uuid
from jose import jwt, JWTError
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import httpx

from app.core.config import settings
from app.services.redis_service import RedisService

# Setup logging
logger = logging.getLogger(__name__)

# Security token scheme
security = HTTPBearer(auto_error=False)

# Cache for JWT key verification
redis_service = RedisService() if settings.REDIS_ENABLED else None

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove any headers that might leak information
        if "Server" in response.headers:
            del response.headers["Server"]
        
        return response

def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure password"""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:,.<>?"
    return "".join(secrets.choice(alphabet) for _ in range(length))

def sanitize_input(value: str) -> str:
    """Basic sanitization of user input to prevent common injection attacks"""
    if not value:
        return value
    
    # Replace potentially dangerous characters
    sanitized = value.replace("<", "&lt;").replace(">", "&gt;")
    sanitized = sanitized.replace("'", "&#39;").replace('"', "&quot;")
    sanitized = sanitized.replace(";", "&#59;")
    
    return sanitized

def validate_jwt_payload(payload: dict) -> bool:
    """Additional validation of JWT payload beyond basic signature verification"""
    required_fields = ["sub", "exp", "iat", "jti"]
    
    # Check that all required fields exist
    if not all(field in payload for field in required_fields):
        logger.warning(f"JWT missing required fields: {required_fields}")
        return False
    
    # Check that token is not too old (iat not more than 7 days ago)
    now = int(time.time())
    if now - payload.get("iat", 0) > 7 * 24 * 60 * 60:  # 7 days
        logger.warning(f"JWT too old: {payload.get('iat')}")
        return False
    
    # Validate expiration time
    if now > payload.get("exp", 0):
        logger.warning(f"JWT expired: {payload.get('exp')}")
        return False
    
    return True

async def is_token_blacklisted(jti: str) -> bool:
    """
    Check if a token is blacklisted
    
    Args:
        jti: The JWT ID to check
        
    Returns:
        True if blacklisted, False otherwise
    """
    if not redis_service:
        logger.warning("Redis not available for blacklist check")
        return False
    
    try:
        blacklist_key = f"jwt_blacklist:{jti}"
        return await redis_service.exists(blacklist_key)
    except Exception as e:
        logger.error(f"Error checking token blacklist: {str(e)}")
        return False

async def blacklist_token(payload: dict, reason: str = "user_request") -> bool:
    """
    Add a token to the blacklist
    
    Args:
        payload: The JWT payload
        reason: The reason for blacklisting
        
    Returns:
        True if successfully blacklisted, False otherwise
    """
    if not redis_service:
        logger.warning("Redis not available for token blacklisting")
        return False
    
    try:
        jti = payload.get("jti")
        if not jti:
            logger.error("Cannot blacklist token without jti claim")
            return False
        
        # Calculate time until token expiration
        exp = payload.get("exp", 0)
        now = int(time.time())
        ttl = max(0, exp - now)
        
        if ttl <= 0:
            # Token already expired, no need to blacklist
            return True
        
        # Add to blacklist with TTL
        blacklist_key = f"jwt_blacklist:{jti}"
        blacklist_data = {
            "jti": jti,
            "sub": payload.get("sub"),
            "reason": reason,
            "blacklisted_at": now
        }
        
        success = await redis_service.set(
            blacklist_key,
            json.dumps(blacklist_data),
            ttl=ttl + 300  # Add 5 minutes buffer for clock skew
        )
        
        if success:
            logger.info(f"Token blacklisted: {jti}")
            return True
        else:
            logger.error(f"Failed to blacklist token: {jti}")
            return False
    
    except Exception as e:
        logger.error(f"Error blacklisting token: {str(e)}")
        return False

async def invalidate_all_user_tokens(user_id: str) -> int:
    """
    Invalidate all tokens for a user
    
    This is critical for security operations like password change, account compromise,
    or forced logout from all devices.
    
    Args:
        user_id: The user ID
        
    Returns:
        Number of tokens invalidated
    """
    if not redis_service:
        logger.warning("Redis not available for token invalidation")
        return 0
    
    try:
        # Use Redis service's user token invalidation
        count = await redis_service.invalidate_user_tokens(user_id)
        
        # Add additional security measures - create a "logout all" record
        # with a longer TTL to catch any tokens that might be missed
        security_key = f"user_security:{user_id}:global_logout"
        timestamp = int(time.time())
        
        # Store with a 30-day TTL to ensure all tokens are eventually invalidated
        await redis_service.set(
            security_key,
            json.dumps({"timestamp": timestamp, "reason": "global_logout"}),
            ttl=30 * 24 * 60 * 60  # 30 days
        )
        
        logger.info(f"Invalidated {count} tokens for user {user_id}")
        return count
    
    except Exception as e:
        logger.error(f"Error invalidating user tokens: {str(e)}")
        return 0

async def verify_supabase_jwt(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token issued by Supabase
    
    This function verifies the token signature, expiration, and issuer.
    
    Args:
        token: The JWT token to verify
        
    Returns:
        The decoded payload if valid, None otherwise
    """
    if not token:
        return None
    
    # Extract JWT ID for blacklist checking
    try:
        # Parse the token without verification to get the JTI
        unverified_payload = jwt.get_unverified_claims(token)
        jti = unverified_payload.get("jti")
        
        # Check if token is blacklisted
        if jti and await is_token_blacklisted(jti):
            logger.warning(f"Token is blacklisted: {jti}")
            return None
        
        # Check for global user logout
        user_id = unverified_payload.get("sub")
        if user_id and redis_service:
            security_key = f"user_security:{user_id}:global_logout"
            global_logout = await redis_service.get(security_key)
            
            if global_logout:
                try:
                    logout_data = json.loads(global_logout)
                    logout_time = logout_data.get("timestamp", 0)
                    
                    # If token was issued before global logout
                    if unverified_payload.get("iat", 0) < logout_time:
                        logger.warning(f"Token was issued before global logout: {user_id}")
                        return None
                except Exception as e:
                    logger.error(f"Error checking global logout: {str(e)}")
    except Exception as e:
        logger.error(f"Error checking token blacklist status: {str(e)}")
    
    # Check if we have the payload cached
    if jti:
        cache_key = f"jwt_payload:{jti}"
        
        if redis_service:
            try:
                cached_payload = await redis_service.get(cache_key)
                if cached_payload:
                    # Return cached payload
                    return json.loads(cached_payload)
            except Exception as e:
                logger.error(f"Error checking JWT cache: {str(e)}")
                # Continue with verification if cache check fails
    
    try:
        # Get JWT signing key from Supabase JWKS endpoint
        # This should be cached to avoid repeated requests
        jwks_url = f"{settings.SUPABASE_URL}/auth/v1/jwks"
        
        jwks_cache_key = "supabase_jwks"
        jwks_json = None
        
        # Try to get JWKS from cache first
        if redis_service:
            try:
                cached_jwks = await redis_service.get(jwks_cache_key)
                if cached_jwks:
                    jwks_json = json.loads(cached_jwks)
            except Exception as e:
                logger.error(f"Error getting JWKS from cache: {str(e)}")
        
        # If not in cache, fetch from Supabase
        if not jwks_json:
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get(jwks_url)
                    if response.status_code == 200:
                        jwks_json = response.json()
                        
                        # Cache the JWKS
                        if redis_service:
                            try:
                                await redis_service.set(
                                    jwks_cache_key, 
                                    json.dumps(jwks_json),
                                    ttl=86400  # 24 hours
                                )
                            except Exception as e:
                                logger.error(f"Error caching JWKS: {str(e)}")
                    else:
                        logger.error(f"Failed to get JWKS: {response.status_code}")
                        return None
                except Exception as e:
                    logger.error(f"Error fetching JWKS: {str(e)}")
                    return None
        
        # Get the signing key from JWKS
        # In a real implementation, you would match the token's "kid" with the key in JWKS
        # For simplicity, we'll use the first key
        signing_key = None
        if jwks_json and "keys" in jwks_json and jwks_json["keys"]:
            # For a more robust implementation, match the key ID (kid) from the token
            signing_key = jwks_json["keys"][0]
        
        if not signing_key:
            logger.error("No signing key found in JWKS")
            return None
        
        # Decode and verify the token
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=settings.SUPABASE_JWT_AUDIENCE,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "verify_iat": True,
            }
        )
        
        # Additional validation of payload
        if not validate_jwt_payload(payload):
            logger.warning(f"Invalid JWT payload: {payload}")
            return None
        
        # Cache the verified payload
        if redis_service and jti:
            try:
                # Calculate TTL based on token expiration
                exp = payload.get("exp", 0)
                now = int(time.time())
                ttl = max(0, exp - now)
                
                await redis_service.set(
                    cache_key,
                    json.dumps(payload),
                    ttl=ttl
                )
            except Exception as e:
                logger.error(f"Error caching JWT payload: {str(e)}")
        
        return payload
    
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error verifying JWT: {str(e)}")
        return None

def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT token and extract the user ID (synchronous version)
    
    Args:
        token: The JWT token
        
    Returns:
        The user ID if valid, None otherwise
    """
    if not token:
        return None
    
    try:
        # Make a blocking HTTP request to the local API endpoint
        # This allows us to use our async verification in a sync context
        url = f"{settings.INTERNAL_API_BASE_URL}/api/v1/auth/verify-token"
        headers = {"Authorization": f"Bearer {token}"}
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("user_id")
            else:
                logger.warning(f"Token verification failed: {response.status_code}")
                return None
    
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return None

def verify_internal_api_key(api_key: str) -> bool:
    """
    Verify internal API key
    
    Args:
        api_key: The API key to verify
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    return api_key == settings.INTERNAL_API_KEY

async def get_current_user_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[str]:
    """
    Extract and verify the current user ID from the request
    
    Args:
        request: The request
        credentials: The authorization credentials
        
    Returns:
        The user ID if authenticated, None otherwise
    """
    # Check authorization header
    if credentials:
        token = credentials.credentials
        payload = await verify_supabase_jwt(token)
        
        if payload:
            # Extract user ID from sub claim
            return payload.get("sub")
    
    # Check for token in cookies as fallback
    token = request.cookies.get("auth_token")
    if token:
        payload = await verify_supabase_jwt(token)
        
        if payload:
            return payload.get("sub")
    
    return None

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Extract and verify the current user from the request
    
    Args:
        request: The request
        credentials: The authorization credentials
        
    Returns:
        The user data if authenticated, None otherwise
    """
    # Check authorization header
    if credentials:
        token = credentials.credentials
        payload = await verify_supabase_jwt(token)
        
        if payload:
            # Extract useful user data from payload
            return {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("user_roles", []),
                "app_metadata": payload.get("app_metadata", {}),
                "user_metadata": payload.get("user_metadata", {})
            }
    
    # Check for token in cookies as fallback
    token = request.cookies.get("auth_token")
    if token:
        payload = await verify_supabase_jwt(token)
        
        if payload:
            return {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("user_roles", []),
                "app_metadata": payload.get("app_metadata", {}),
                "user_metadata": payload.get("user_metadata", {})
            }
    
    return None

async def require_auth(request: Request) -> str:
    """
    Require authentication for a request
    
    Args:
        request: The request
        
    Returns:
        The user ID
        
    Raises:
        HTTPException: If not authenticated
    """
    user_id = await get_current_user_id(request)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_id

async def require_admin_auth(request: Request) -> str:
    """
    Require admin authentication for a request
    
    Args:
        request: The request
        
    Returns:
        The user ID
        
    Raises:
        HTTPException: If not authenticated or not admin
    """
    user = await get_current_user(request)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check if user has admin role
    roles = user.get("roles", [])
    if "admin" not in roles and "administrator" not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    
    return user["id"]

async def create_tokens(user_id: str, user_email: str, user_roles: List[str] = None) -> Tuple[str, str]:
    """
    Create access and refresh tokens
    
    Args:
        user_id: The user ID
        user_email: The user email
        user_roles: List of user roles
        
    Returns:
        Tuple of (access_token, refresh_token)
    """
    if not user_roles:
        user_roles = ["user"]
    
    # Current time
    now = int(time.time())
    
    # Create unique IDs for tokens
    access_jti = str(uuid.uuid4())
    refresh_jti = str(uuid.uuid4())
    
    # Access token payload
    access_payload = {
        "sub": user_id,
        "email": user_email,
        "user_roles": user_roles,
        "iat": now,
        "exp": now + settings.ACCESS_TOKEN_EXPIRE_SECONDS,
        "jti": access_jti
    }
    
    # Refresh token payload
    refresh_payload = {
        "sub": user_id,
        "email": user_email,
        "token_type": "refresh",
        "iat": now,
        "exp": now + settings.REFRESH_TOKEN_EXPIRE_SECONDS,
        "jti": refresh_jti
    }
    
    # Sign tokens
    access_token = jwt.encode(
        access_payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    refresh_token = jwt.encode(
        refresh_payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    # Store token payloads in Redis
    if redis_service:
        try:
            # Store access token with TTL
            access_key = f"jwt_payload:{access_jti}"
            await redis_service.set(
                access_key,
                json.dumps(access_payload),
                ttl=settings.ACCESS_TOKEN_EXPIRE_SECONDS
            )
            
            # Store refresh token with TTL
            refresh_key = f"jwt_payload:{refresh_jti}"
            await redis_service.set(
                refresh_key,
                json.dumps(refresh_payload),
                ttl=settings.REFRESH_TOKEN_EXPIRE_SECONDS
            )
        except Exception as e:
            logger.error(f"Error storing token in Redis: {str(e)}")
    
    return access_token, refresh_token

async def refresh_token(refresh_token: str) -> Optional[Dict[str, str]]:
    """
    Refresh an access token using a refresh token
    
    Args:
        refresh_token: The refresh token
        
    Returns:
        New access and refresh tokens if successful, None otherwise
    """
    if not refresh_token:
        return None
    
    try:
        # Verify the refresh token
        payload = await verify_supabase_jwt(refresh_token)
        
        if not payload:
            logger.warning("Invalid refresh token")
            return None
        
        # Check token type
        if payload.get("token_type") != "refresh":
            logger.warning("Not a refresh token")
            return None
        
        # Extract user info
        user_id = payload.get("sub")
        user_email = payload.get("email", "")
        
        if not user_id:
            logger.warning("Missing user ID in refresh token")
            return None
        
        # Get user roles from Supabase
        # This would typically require a Supabase API call
        # For now, we'll use a default role
        user_roles = ["user"]
        
        # Create new tokens
        new_access_token, new_refresh_token = await create_tokens(
            user_id,
            user_email,
            user_roles
        )
        
        # Blacklist the used refresh token
        await blacklist_token(payload, reason="token_refresh")
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
    
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        return None

async def logout_user(request: Request, all_devices: bool = False) -> bool:
    """
    Log out a user by invalidating their tokens
    
    Args:
        request: The request
        all_devices: Whether to log out from all devices
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get user ID
        user_data = await get_current_user(request)
        if not user_data:
            logger.warning("Cannot log out unauthenticated user")
            return False
        
        user_id = user_data["id"]
        
        # Get current token
        credentials = await security(request)
        if credentials:
            token = credentials.credentials
            try:
                # Get token payload without verification
                unverified_payload = jwt.get_unverified_claims(token)
                
                # Blacklist the token
                await blacklist_token(unverified_payload, reason="logout")
            except Exception as e:
                logger.error(f"Error blacklisting token: {str(e)}")
        
        # If logging out from all devices, invalidate all tokens
        if all_devices:
            await invalidate_all_user_tokens(user_id)
        
        return True
    
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return False
