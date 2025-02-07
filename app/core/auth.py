from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.config import settings
import logging
from typing import Optional

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
