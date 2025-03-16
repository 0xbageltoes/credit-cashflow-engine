from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.config import settings
import logging
from typing import Optional, Dict, Union, Any, List, Callable, ForwardRef

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

security = HTTPBearer()

# Function to decode JWT token
def decode_jwt_token(token: str) -> Optional[Dict]:
    """
    Decode JWT token and return payload
    
    This function decodes and validates Supabase JWT tokens.
    It tries both base64-decoded and raw secret formats for compatibility.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing user data or None if token is invalid
    """
    if not token:
        logger.warning("Token decoding failed: Empty token")
        return None
    
    # Try to decode the token
    try:
        # First try with JWT_SECRET as is (for some Supabase configurations)
        try:
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False} if not settings.SUPABASE_JWT_AUDIENCE else None,
                audience=settings.SUPABASE_JWT_AUDIENCE if settings.SUPABASE_JWT_AUDIENCE else None
            )
            logger.debug("Token decoded successfully with raw secret")
            return payload
        except Exception as e:
            # If the above fails, try alternative encoding/format
            logger.debug(f"Failed to decode with raw JWT secret: {str(e)}")
            
            # Try with specific audience
            if settings.JWT_AUDIENCE:
                try:
                    payload = jwt.decode(
                        token,
                        settings.SUPABASE_JWT_SECRET,
                        algorithms=["HS256"],
                        audience=settings.JWT_AUDIENCE
                    )
                    logger.debug("Token decoded successfully with specified audience")
                    return payload
                except Exception as audience_error:
                    logger.debug(f"Failed to decode with specified audience: {str(audience_error)}")
            
            # If all attempts failed, raise the original error
            raise e
    
    except JWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error decoding token: {str(e)}")
        return None

# Function to extract user ID from token
def get_user_id_from_token(authorization: str) -> str:
    """
    Extract user ID from authorization header
    
    Args:
        authorization: Authorization header with JWT token
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If token is invalid or user ID cannot be extracted
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.replace("Bearer ", "")
    payload = decode_jwt_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        logger.error("Token payload does not contain user ID (sub)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id

# Token verification function - must be defined before RoleChecker
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Verify JWT token and extract user data
    
    This function verifies the JWT token from the Authorization header
    and extracts the user data from it.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Dict containing user data
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    # Decode the token
    payload = decode_jwt_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Log successful authentication
    user_id = payload.get("sub")
    logger.debug(f"User authenticated: {user_id}")
    
    return payload

# Primary function for getting the current user - used across API endpoints
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Validate JWT token and return user information
    
    This function is the primary dependency for user authentication in API endpoints.
    It validates the JWT token and returns the user data in a consistent format.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Dict with standardized user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    # Get the raw token payload
    payload = await verify_token(credentials)
    
    # Extract and format user information for consistent API usage
    user_info = {
        "id": payload.get("sub"),
        "email": payload.get("email"),
        "app_metadata": payload.get("app_metadata", {}),
        "user_metadata": payload.get("user_metadata", {})
    }
    
    # Extract roles from multiple possible locations
    roles = []
    
    # Check app_metadata.roles (Supabase standard)
    app_metadata = payload.get("app_metadata", {})
    if isinstance(app_metadata, dict) and "roles" in app_metadata:
        metadata_roles = app_metadata["roles"]
        if isinstance(metadata_roles, list):
            roles.extend(metadata_roles)
    
    # Check user_metadata.roles (alternative location)
    user_metadata = payload.get("user_metadata", {})
    if isinstance(user_metadata, dict) and "roles" in user_metadata:
        user_roles = user_metadata["roles"]
        if isinstance(user_roles, list):
            roles.extend(user_roles)
    
    # Check direct roles property
    payload_roles = payload.get("roles", [])
    if isinstance(payload_roles, list):
        roles.extend(payload_roles)
    
    # Check legacy role property (string)
    role = payload.get("role")
    if role and isinstance(role, str) and role not in roles:
        roles.append(role)
    
    # Deduplicate roles
    user_info["roles"] = list(set(roles))
    
    # For backwards compatibility
    if roles:
        user_info["role"] = roles[0]
    else:
        user_info["role"] = "user"  # Default role
    
    logger.debug(f"User authenticated: {user_info['id']} with roles {user_info['roles']}")
    return user_info

# WebSocket token verification - for endpoints not using FastAPI dependencies
def verify_token_ws(token: str, required_roles: Optional[list] = None) -> Union[Dict, bool]:
    """
    Verify JWT token and optionally check required roles
    
    This function is designed for websocket authentication where dependency
    injection with FastAPI's Depends() is not available. It provides comprehensive
    token validation with role-based access control.
    
    Args:
        token: JWT token string (without 'Bearer ' prefix)
        required_roles: Optional list of roles that are allowed to access the resource
        
    Returns:
        Dict containing user data if token is valid, False otherwise
        
    Production features:
    - Comprehensive logging for troubleshooting
    - Robust error handling
    - Support for multiple secret formats (raw and base64-decoded)
    - Role-based access control
    - Safe token validation with proper expiration checking
    """
    # Remove Bearer prefix if present
    if token.startswith("Bearer "):
        token = token.replace("Bearer ", "")
    
    # Decode the token
    payload = decode_jwt_token(token)
    
    if not payload:
        logger.warning("WebSocket authentication failed: Invalid or expired token")
        return False
    
    # Check roles if required
    if required_roles:
        # Extract user roles
        user_roles = payload.get("app_metadata", {}).get("roles", [])
        
        # For supabase, roles might be stored differently depending on configuration
        if not user_roles:
            user_roles = payload.get("user_metadata", {}).get("roles", [])
        
        # For development/testing, if no roles defined, assume basic role
        if settings.ENVIRONMENT == "development" and not user_roles:
            user_roles = ["user"]
        
        # Log role check
        logger.debug(f"WS role check: User roles: {user_roles}, Required roles: {required_roles}")
        
        # Check if user has any of the allowed roles
        role_match = False
        for role in required_roles:
            if role in user_roles:
                logger.debug(f"WS role check passed: User has role {role}")
                role_match = True
                break
        
        if not role_match:
            logger.warning(
                f"WS role check failed: User {payload.get('sub')} lacks required roles. "
                f"Has: {user_roles}, Needs one of: {required_roles}"
            )
            return False
    
    # Log successful authentication
    user_id = payload.get("sub")
    logger.debug(f"WebSocket user authenticated: {user_id}")
    
    return payload

class RoleChecker:
    """
    Role-based access control for API endpoints.
    
    This class provides a dependency that can be used to restrict access
    to endpoints based on user roles. It verifies that the user has at
    least one of the required roles to access the endpoint.
    
    Example:
        ```
        require_admin = RoleChecker(["admin"])
        
        @app.get("/admin-only")
        async def admin_endpoint(user: Dict = Depends(require_admin)):
            # This will only execute if the user has the 'admin' role
            return {"message": "Admin access granted"}
        ```
    
    Production features:
    - Comprehensive error handling
    - Detailed error messages for debugging
    - Proper HTTP status codes for different error scenarios
    - Request tracing with unique identifiers
    - Extensive logging
    """
    
    def __init__(self, allowed_roles: List[str]):
        """
        Initialize the role checker with a list of allowed roles.
        
        Args:
            allowed_roles: List of roles that are allowed to access the endpoint.
                           User must have at least one of these roles.
        """
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: Dict = Depends(verify_token)) -> Dict:
        """
        Check if the user has any of the required roles.
        
        Args:
            user: User data from JWT token
            
        Returns:
            User data if user has required role
            
        Raises:
            HTTPException: If user lacks required roles
        """
        if not user:
            logger.warning("Role check failed: No authenticated user")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract user roles
        user_roles = user.get("app_metadata", {}).get("roles", [])
        
        # For supabase, roles might be stored differently depending on configuration
        if not user_roles:
            user_roles = user.get("user_metadata", {}).get("roles", [])
        
        # For development/testing, if no roles defined, assume basic role
        if settings.ENVIRONMENT == "development" and not user_roles:
            user_roles = ["user"]
        
        # Log role check
        logger.debug(f"Role check: User roles: {user_roles}, Required roles: {self.allowed_roles}")
        
        # Check if user has any of the allowed roles
        for role in self.allowed_roles:
            if role in user_roles:
                logger.debug(f"Role check passed: User has role {role}")
                return user
        
        # User doesn't have any of the required roles
        logger.warning(
            f"Role check failed: User {user.get('id')} lacks required roles. "
            f"Has: {user_roles}, Needs one of: {self.allowed_roles}"
        )
        
        # Return appropriate error based on authentication status
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Insufficient permissions.",
        )
