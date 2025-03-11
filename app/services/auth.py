"""Authentication service for the credit cashflow forecasting API."""
import os
import logging
import jwt
from fastapi import HTTPException, Depends, Header
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    logger.warning("SUPABASE_JWT_SECRET not set. JWT validation will be disabled.")

def get_user_id_from_token(authorization: str) -> str:
    """
    Extract user_id from a JWT token.
    
    Args:
        authorization: Bearer token from request headers
        
    Returns:
        str: User ID from the JWT token
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    # Extract token from Bearer format
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.replace("Bearer ", "")
    
    # Skip validation in development if needed
    if os.getenv("ENVIRONMENT") == "development" and os.getenv("SKIP_AUTH_VALIDATION") == "true":
        logger.warning("Skipping JWT validation in development mode")
        try:
            # Just decode without verification to extract user_id
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload.get("sub", "")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token format: {str(e)}")
    
    # Verify token with secret
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured on server")
    
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Token validation failed")
