"""
Test utilities for authentication, caching and other core functionality tests

This module provides proper testing utilities for authentication and security features,
including mock JWT tokens, test Supabase client, and Redis test helpers.
"""

import os
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from jose import jwt
from datetime import datetime, timedelta
from fastapi import Request
from unittest.mock import AsyncMock, MagicMock

from app.core.config import settings

# ===================== JWT Authentication Testing Utilities =====================

def create_test_jwt_token(
    user_id: str = "test-user-id",
    email: str = "test@example.com", 
    roles: List[str] = None,
    expires_in_seconds: int = 3600,
    issued_at: Optional[int] = None,
    jwt_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a JWT token for testing purposes
    
    Args:
        user_id: The user ID
        email: The user email
        roles: List of user roles (defaults to ["user"])
        expires_in_seconds: Token expiration in seconds
        issued_at: Token issue time (defaults to current time)
        jwt_id: JWT ID (defaults to random UUID)
        
    Returns:
        Tuple of (token_string, token_payload)
    """
    if roles is None:
        roles = ["user"]
        
    iat = issued_at or int(time.time())
    exp = iat + expires_in_seconds
    jti = jwt_id or str(uuid.uuid4())
    
    # Create payload that matches our production structure
    payload = {
        "sub": user_id,
        "email": email,
        "user_roles": roles,
        "exp": exp,
        "iat": iat,
        "jti": jti,
        "aud": settings.SUPABASE_JWT_AUDIENCE or settings.SUPABASE_URL,
        "iss": settings.JWT_ISSUER or settings.SUPABASE_URL
    }
    
    # Sign the token with the same algorithm we use in production
    token = jwt.encode(
        payload, 
        settings.SUPABASE_JWT_SECRET, 
        algorithm=settings.HASH_ALGORITHM
    )
    
    return token, payload

def create_test_tokens(
    user_id: str = "test-user-id",
    email: str = "test@example.com", 
    roles: List[str] = None
) -> Tuple[str, str]:
    """
    Create access and refresh tokens for testing
    
    Args:
        user_id: The user ID
        email: The user email
        roles: List of user roles
        
    Returns:
        Tuple of (access_token, refresh_token)
    """
    # Create access token (1 hour)
    access_token, _ = create_test_jwt_token(
        user_id=user_id,
        email=email,
        roles=roles,
        expires_in_seconds=60 * 60  # 1 hour
    )
    
    # Create refresh token (7 days)
    refresh_token, _ = create_test_jwt_token(
        user_id=user_id,
        email=email,
        roles=roles,
        expires_in_seconds=60 * 60 * 24 * 7  # 7 days
    )
    
    return access_token, refresh_token

def create_test_request_with_token(token: str) -> Request:
    """
    Create a mock request object with a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Request object with token in Authorization header
    """
    # Create mock request object with headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"Authorization": f"Bearer {token}"}
    
    return mock_request

# ===================== Supabase Service Mocking =====================

def create_mock_supabase_service():
    """
    Create a mock Supabase service for testing
    
    Returns:
        Mock Supabase service
    """
    mock = AsyncMock()
    
    # Configure behavior for user operations
    async def mock_get_user(user_id):
        if user_id == "invalid-user":
            return None
        return {
            "id": user_id,
            "email": f"{user_id}@example.com", 
            "status": "active",
            "user_metadata": {"full_name": "Test User"},
            "app_metadata": {"roles": ["user"]},
            "created_at": "2023-01-01T00:00:00.000Z"
        }
    
    # Configure behavior for token operations
    async def mock_refresh_token(refresh_token):
        if refresh_token == "invalid-token":
            return None
        return {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "token_type": "bearer"
        }
    
    async def mock_invalidate_refresh_token(token):
        return True
    
    async def mock_invalidate_all_user_sessions(user_id):
        return True
    
    # Configure behavior for health checks
    async def mock_health_check():
        return True
    
    # Set up the mock functions
    mock.get_user.side_effect = mock_get_user
    mock.refresh_token.side_effect = mock_refresh_token
    mock.invalidate_refresh_token.side_effect = mock_invalidate_refresh_token
    mock.invalidate_all_user_sessions.side_effect = mock_invalidate_all_user_sessions
    mock.health_check.side_effect = mock_health_check
    
    return mock

# ===================== Redis Service Mocking =====================

def create_mock_redis_service():
    """
    Create a mock Redis service for testing
    
    Returns:
        Mock Redis service
    """
    mock = AsyncMock()
    
    # Create in-memory storage for the mock
    in_memory_data = {}
    in_memory_ttl = {}
    
    # Configure mock behavior
    async def mock_set(key, value, ttl=None):
        try:
            # Ensure the value can be serialized/deserialized
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            in_memory_data[key] = value
            if ttl:
                in_memory_ttl[key] = time.time() + ttl
            return True
        except Exception as e:
            # Log and return the error in a production-ready way
            print(f"Error in mock_set: {e}")
            return False
    
    async def mock_get(key, default=None):
        try:
            # Check if key has expired
            if key in in_memory_ttl and in_memory_ttl[key] < time.time():
                del in_memory_data[key]
                del in_memory_ttl[key]
                return default
            
            value = in_memory_data.get(key, default)
            
            # Try to deserialize JSON data
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except:
                    # Not JSON data, return as is
                    pass
                    
            return value
        except Exception as e:
            # Log and return the default value in case of error
            print(f"Error in mock_get: {e}")
            return default
    
    async def mock_delete(key):
        try:
            if key in in_memory_data:
                del in_memory_data[key]
                if key in in_memory_ttl:
                    del in_memory_ttl[key]
                return True
            return False
        except Exception as e:
            print(f"Error in mock_delete: {e}")
            return False
    
    async def mock_exists(key):
        try:
            # Check if key has expired
            if key in in_memory_ttl and in_memory_ttl[key] < time.time():
                del in_memory_data[key]
                del in_memory_ttl[key]
                return False
                
            return key in in_memory_data
        except Exception as e:
            print(f"Error in mock_exists: {e}")
            return False
    
    async def mock_delete_pattern(pattern):
        try:
            pattern = pattern.replace("*", "")
            keys_to_delete = [k for k in in_memory_data.keys() if pattern in k]
            for key in keys_to_delete:
                del in_memory_data[key]
                if key in in_memory_ttl:
                    del in_memory_ttl[key]
            return len(keys_to_delete)
        except Exception as e:
            print(f"Error in mock_delete_pattern: {e}")
            return 0
    
    async def mock_get_ttl(key):
        try:
            if key in in_memory_ttl:
                remaining = in_memory_ttl[key] - time.time()
                return max(0, remaining)
            return None
        except Exception as e:
            print(f"Error in mock_get_ttl: {e}")
            return None
    
    async def mock_health_check():
        # Always return True for testing
        return True
    
    # Set up the mock functions
    mock.set.side_effect = mock_set
    mock.get.side_effect = mock_get
    mock.delete.side_effect = mock_delete
    mock.exists.side_effect = mock_exists
    mock.delete_pattern.side_effect = mock_delete_pattern
    mock.get_ttl.side_effect = mock_get_ttl
    mock.health_check.side_effect = mock_health_check
    
    return mock

# ===================== Test Environment Setup =====================

def setup_test_environment():
    """
    Set up test environment variables
    
    This function ensures all required environment variables are present
    for testing the auth system. It uses the actual configuration values
    for proper production testing.
    """
    required_vars = {
        "ENVIRONMENT": "test",
        "SUPABASE_URL": "https://vszqsfntcqidghcwxeij.supabase.co",
        "SUPABASE_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzenFzZm50Y3FpZGdoY3d4ZWlqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc2MTMwNDAsImV4cCI6MjA1MzE4OTA0MH0.s4rnKZkS7Mr6nrNTml9WQIPj9OBT9C5W2vWtXPrro-g",
        "SUPABASE_ANON_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzenFzZm50Y3FpZGdoY3d4ZWlqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc2MTMwNDAsImV4cCI6MjA1MzE4OTA0MH0.s4rnKZkS7Mr6nrNTml9WQIPj9OBT9C5W2vWtXPrro-g",
        "SUPABASE_SERVICE_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzenFzZm50Y3FpZGdoY3d4ZWlqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzYxMzA0MCwiZXhwIjoyMDUzMTg5MDQwfQ.cDUilqYPNG9i7UfaHE1NW3ERCxCZc33Ppuy5ha3TOok",
        "SUPABASE_JWT_SECRET": "KAcXB7Z5Ost9OpUyX5P4hB14bQQrurCNCGj8e93ZakbUaCcRq1E4XWPvRRa1l+KyXBz+aMy+QIs2bi0E7lnDlw==",
        "SUPABASE_AUTH_EXTERNAL_URL": "https://vszqsfntcqidghcwxeij.supabase.co/auth/v1",
        "NEXT_PUBLIC_SUPABASE_URL": "https://vszqsfntcqidghcwxeij.supabase.co",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzenFzZm50Y3FpZGdoY3d4ZWlqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc2MTMwNDAsImV4cCI6MjA1MzE4OTA0MH0.s4rnKZkS7Mr6nrNTml9WQIPj9OBT9C5W2vWtXPrro-g",
        "UPSTASH_REDIS_HOST": "easy-macaw-12070.upstash.io",
        "UPSTASH_REDIS_PORT": "6379",
        "UPSTASH_REDIS_PASSWORD": "AS8mAAIjcDFmMjJhZTIzY2ZiYmY0MTJkYmQzZDQ1MWYwMWQyYzI0MXAxMA",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_PASSWORD": "mock-redis-password",
        "REDIS_DB": "0",
        "REDIS_ENABLED": "false",
        "API_V1_STR": "/api/v1",
        "BACKEND_CORS_ORIGINS": '["http://localhost:3000","http://localhost:8000"]',
        "SECRET_KEY": "mock-secret-key-for-testing-purposes-only-do-not-use-in-production",
        "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
        "REFRESH_TOKEN_EXPIRE_DAYS": "7",
        "HASH_ALGORITHM": "HS256",
        "STORAGE_BUCKET_NAME": "test-cashflow-engine-data",
        "RATE_LIMIT_ENABLED": "false"
    }
    
    # Set missing environment variables
    for key, value in required_vars.items():
        if not os.environ.get(key):
            os.environ[key] = value
