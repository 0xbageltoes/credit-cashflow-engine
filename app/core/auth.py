from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.config import settings
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Validate Supabase JWT token and return user information
    """
    try:
        token = credentials.credentials
        logger.debug(f"Received token: {token[:20]}...")  # Log first 20 chars of token
        
        # Log JWT secret length for debugging
        jwt_secret = settings.SUPABASE_JWT_SECRET
        logger.debug(f"JWT secret length: {len(jwt_secret)}")
        
        try:
            # First try with base64-decoded secret
            import base64
            decoded_secret = base64.b64decode(jwt_secret)
            payload = jwt.decode(
                token,
                decoded_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}  # Skip audience verification
            )
            logger.debug("Token decoded successfully with base64-decoded secret")
        except Exception as e:
            logger.debug(f"Failed to decode with base64 secret: {str(e)}")
            # Try with raw secret
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}  # Skip audience verification
            )
            logger.debug("Token decoded successfully with raw secret")
            
        logger.debug(f"Token payload: {payload}")
        
        # Validate token claims
        if not payload.get("sub"):
            logger.error("No 'sub' claim in token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token - no sub claim"
            )
            
        return {
            "id": payload["sub"],
            "email": payload.get("email"),
            "role": payload.get("role", "user")
        }
        
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in auth: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )

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
    try:
        # Log JWT secret length for debugging
        jwt_secret = settings.SUPABASE_JWT_SECRET
        
        try:
            # First try with base64-decoded secret
            import base64
            decoded_secret = base64.b64decode(jwt_secret)
            payload = jwt.decode(
                token,
                decoded_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}  # Skip audience verification
            )
        except Exception:
            # Try with raw secret
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}  # Skip audience verification
            )
        
        # Validate token claims
        if not payload.get("sub"):
            logger.error("No 'sub' claim in token")
            return None
            
        return {
            "id": payload["sub"],
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
            "roles": payload.get("roles", []),
            "app_metadata": payload.get("app_metadata", {})
        }
        
    except Exception as e:
        logger.error(f"Error decoding JWT token: {str(e)}")
        return None

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
            detail="Invalid authorization header format"
        )
    
    token = authorization.replace("Bearer ", "")
    user_data = decode_jwt_token(token)
    
    if not user_data or "id" not in user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return user_data["id"]
