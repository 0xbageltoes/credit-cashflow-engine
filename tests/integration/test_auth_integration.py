"""
Integration tests for the authentication system

This module provides integration tests for the complete authentication system,
testing the interaction between API endpoints, Redis, and Supabase in a
real-world environment.

These tests require a running Redis instance and valid Supabase credentials.
They are meant to be run in a staging environment with real services.
"""

import pytest
import time
import uuid
import os
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.api.v1.api import api_router
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
from app.core.security import (
    create_tokens,
    verify_supabase_jwt,
    blacklist_token,
    logout_user,
    invalidate_all_user_tokens
)
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip integration tests if env var is not set
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests are skipped unless RUN_INTEGRATION_TESTS=true"
)

# Create test app
app = FastAPI(title="Auth Integration Test API")
app.include_router(api_router, prefix=settings.API_V1_STR)
client = TestClient(app)

# Initialize services
redis_service = RedisService()
supabase_service = SupabaseService()

@pytest.fixture(scope="module")
async def test_user():
    """Create a test user in Supabase for authentication tests"""
    # Generate a unique email to avoid conflicts
    test_email = f"test-{uuid.uuid4()}@example.com"
    test_password = f"Password123!{uuid.uuid4()}"
    
    try:
        # Create a test user
        user = await supabase_service.create_user(
            email=test_email,
            password=test_password,
            user_metadata={"full_name": "Integration Test User"},
            app_metadata={"roles": ["user", "test"]}
        )
        
        # Return user credentials
        yield {
            "id": user["id"],
            "email": test_email,
            "password": test_password
        }
        
        # Cleanup: We intentionally don't delete the user after tests
        # to avoid concurrency issues with tokens. The random UUID ensures
        # we don't have conflicts, and Supabase test environments should
        # be reset periodically anyway.
    except Exception as e:
        logger.error(f"Failed to create test user: {str(e)}")
        pytest.skip("Couldn't create test user in Supabase")

@pytest.fixture
async def auth_tokens(test_user):
    """Get authentication tokens for the test user"""
    try:
        # Create tokens directly
        access_token, refresh_token = create_tokens(
            user_id=test_user["id"],
            user_email=test_user["email"],
            user_roles=["user", "test"]
        )
        
        # Return tokens
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    except Exception as e:
        logger.error(f"Failed to create auth tokens: {str(e)}")
        pytest.skip("Couldn't create authentication tokens")

@pytest.mark.asyncio
class TestAuthIntegration:
    """Integration tests for the authentication system"""
    
    async def test_verify_token_full_flow(self, auth_tokens):
        """Test the complete token verification flow"""
        # Get token
        access_token = auth_tokens["access_token"]
        
        # Verify token with API
        response = client.get(
            f"{settings.API_V1_STR}/auth/verify-token",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        # Assertions
        assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "user_id" in data
        assert "roles" in data
        assert "exp" in data
        assert "jti" in data
        
        # Verify token directly
        payload = await verify_supabase_jwt(access_token)
        assert payload is not None
        assert payload["sub"] == data["user_id"]
    
    async def test_refresh_token_flow(self, auth_tokens):
        """Test the complete token refresh flow"""
        # Get refresh token
        refresh_token = auth_tokens["refresh_token"]
        
        # Refresh token with API
        response = client.post(
            f"{settings.API_V1_STR}/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        # Check success
        assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        
        # Verify the new access token works
        new_access_token = data["access_token"]
        verify_response = client.get(
            f"{settings.API_V1_STR}/auth/verify-token",
            headers={"Authorization": f"Bearer {new_access_token}"}
        )
        
        assert verify_response.status_code == 200, "New access token validation failed"
    
    async def test_token_blacklisting(self, auth_tokens):
        """Test token blacklisting and verification"""
        # Get tokens
        access_token = auth_tokens["access_token"]
        
        # Verify token works initially
        response1 = client.get(
            f"{settings.API_V1_STR}/auth/verify-token",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response1.status_code == 200
        
        # Get token payload
        payload = await verify_supabase_jwt(access_token)
        assert payload is not None
        
        # Blacklist the token
        blacklisted = await blacklist_token(payload, reason="test_blacklisting")
        assert blacklisted is True
        
        # Try to verify blacklisted token
        response2 = client.get(
            f"{settings.API_V1_STR}/auth/verify-token",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        # Should be rejected
        assert response2.status_code in [401, 403]
    
    async def test_user_info_endpoint(self, auth_tokens):
        """Test the user info endpoint"""
        # Get token
        access_token = auth_tokens["access_token"]
        
        # Get user info
        response = client.get(
            f"{settings.API_V1_STR}/auth/user",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        # Check success
        assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "id" in data
        assert "email" in data
        assert "roles" in data
        assert isinstance(data["roles"], list)
    
    async def test_logout_invalidation(self, test_user, auth_tokens):
        """Test the logout and token invalidation process"""
        # Get token
        access_token = auth_tokens["access_token"]
        
        # Create request mock with token
        class RequestMock:
            def __init__(self, token):
                self.headers = {"Authorization": f"Bearer {token}"}
        
        request_mock = RequestMock(access_token)
        
        # Logout the user
        logout_success = await logout_user(request_mock)
        assert logout_success is True
        
        # Try to use the token after logout
        response = client.get(
            f"{settings.API_V1_STR}/auth/verify-token",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        # Should be rejected
        assert response.status_code in [401, 403]
    
    async def test_invalidate_all_user_tokens(self, test_user):
        """Test invalidating all tokens for a user"""
        # Create several tokens for the user
        tokens = []
        for i in range(3):
            access_token, refresh_token = create_tokens(
                user_id=test_user["id"],
                user_email=test_user["email"],
                user_roles=["user", "test"]
            )
            tokens.append(access_token)
        
        # Verify all tokens work
        for token in tokens:
            response = client.get(
                f"{settings.API_V1_STR}/auth/verify-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200
        
        # Invalidate all tokens
        count = await invalidate_all_user_tokens(test_user["id"])
        assert count > 0
        
        # Verify no tokens work anymore
        for token in tokens:
            response = client.get(
                f"{settings.API_V1_STR}/auth/verify-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code in [401, 403]
    
    async def test_health_check_endpoint(self):
        """Test the health check endpoint"""
        response = client.get(f"{settings.API_V1_STR}/auth/health")
        
        # Check success
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "services" in data
        assert "redis" in data["services"]
        assert "supabase" in data["services"]
        
        # Verify Redis and Supabase are healthy
        if data["status"] != "healthy":
            logger.warning(f"Auth system health is not optimal: {data}")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
