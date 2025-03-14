"""
Tests for the authentication endpoints

This module provides comprehensive tests for the authentication endpoints, including:
- Token verification and validation
- Token refresh flow
- User information retrieval
- Logout functionality
- Token invalidation
- Edge cases and error handling
"""
import pytest
import json
import time
import uuid
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints.auth import router as auth_router, get_redis_service, get_supabase_service
from app.core.security import create_tokens, verify_supabase_jwt, blacklist_token, refresh_token
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
from app.core.config import settings
from app.core.exceptions import AuthHTTPException
from app.core.test_utils import (
    create_test_jwt_token, 
    create_test_tokens,
    create_test_request_with_token,
    create_mock_supabase_service,
    create_mock_redis_service
)

# Create test app with mocked dependencies
app = FastAPI()
app.include_router(auth_router, prefix="/api/v1/auth")
client = TestClient(app)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Use fixtures from test_utils
@pytest.fixture
async def mock_redis_service():
    """Create a mock Redis service for testing"""
    return create_mock_redis_service()

@pytest.fixture
async def mock_supabase_service():
    """Create a mock Supabase service for testing"""
    return create_mock_supabase_service()

@pytest.fixture
async def valid_user_token():
    """Create a valid JWT token for testing"""
    token, _ = create_test_jwt_token(
        user_id="test-user-id",
        email="test@example.com",
        roles=["user"]
    )
    return token

@pytest.fixture
async def admin_user_token():
    """Create a valid JWT token for an admin user"""
    token, _ = create_test_jwt_token(
        user_id="admin-user-id",
        email="admin@example.com",
        roles=["admin", "user"]
    )
    return token

@pytest.fixture
async def expired_token():
    """Create an expired JWT token for testing"""
    token, _ = create_test_jwt_token(
        user_id="test-user-id",
        email="test@example.com",
        roles=["user"],
        expires_in_seconds=-3600  # Expired 1 hour ago
    )
    return token

# Create a fixture for mock credentials dependency
@pytest.fixture
def mock_credentials():
    """Mock HTTPAuthorizationCredentials for dependency override"""
    class MockCredentials:
        credentials = "test-token"
        scheme = "Bearer"
    return MockCredentials()

# Create a fixture to set up all auth dependencies
@pytest.fixture
async def mock_dependencies():
    """Mock dependencies for testing"""
    async def async_mock_health_check_true():
        return True
        
    async def async_mock_health_check_false():
        return False
    
    # Set up mock for Redis
    mock_redis = AsyncMock(spec=RedisService)
    mock_redis.health_check = async_mock_health_check_true
    
    # Set up mock for Supabase
    mock_supabase = AsyncMock(spec=SupabaseService)
    mock_supabase.health_check = async_mock_health_check_true
    
    # Create patches to mock services
    redis_patch = patch("app.api.v1.endpoints.auth.RedisService", return_value=mock_redis)
    supabase_patch = patch("app.api.v1.endpoints.auth.SupabaseService", return_value=mock_supabase)
    redis_service_patch = patch("app.api.v1.endpoints.auth.redis_service", mock_redis)
    
    # Collect patches
    patches = [redis_patch, supabase_patch, redis_service_patch]
    
    yield patches, mock_redis, mock_supabase, async_mock_health_check_true, async_mock_health_check_false

@pytest.mark.asyncio
class TestAuthEndpoints:
    """Test the authentication endpoints"""
    
    async def test_verify_token_success(self, valid_user_token, mock_dependencies):
        """Test successful token verification"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure mocks for success scenario
        mock_verify_jwt = AsyncMock(return_value={
            "sub": "test-user-id",
            "email": "test@example.com",
            "user_roles": ["user"],
            "exp": int(time.time()) + 3600,
            "jti": str(uuid.uuid4())
        })
        
        # Add patch for the JWT verification
        patches.append(patch('app.api.v1.endpoints.auth.verify_supabase_jwt', mock_verify_jwt))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.get(
                "/api/v1/auth/verify-token",
                headers={"Authorization": f"Bearer {valid_user_token}"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "user_id" in data
            assert "exp" in data
            assert "jti" in data
            assert "roles" in data
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_verify_token_missing_auth(self, mock_dependencies):
        """Test token verification with missing authorization"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint without token
            response = client.get("/api/v1/auth/verify-token")
            
            # Verify response
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Missing authorization credentials" in data["detail"]
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_verify_token_expired(self, expired_token, mock_dependencies):
        """Test token verification with expired token"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure mocks for expired token scenario
        mock_verify_jwt = AsyncMock(return_value=None)
        
        # Add patch for the JWT verification
        patches.append(patch('app.api.v1.endpoints.auth.verify_supabase_jwt', mock_verify_jwt))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint with expired token
            response = client.get(
                "/api/v1/auth/verify-token",
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            
            # Verify response
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Invalid or expired token" in data["detail"]
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_verify_token_blacklisted(self, valid_user_token, mock_dependencies):
        """Test token verification with blacklisted token"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure the mock to return a value for blacklisted token
        async def mock_exists(key):
            if "jwt_blacklist" in key:
                return True
            return False
            
        mock_redis.exists.side_effect = mock_exists
        
        # Configure mocks for success scenario (token is valid but blacklisted)
        mock_verify_jwt = AsyncMock(return_value={
            "sub": "test-user-id",
            "email": "test@example.com",
            "user_roles": ["user"],
            "exp": int(time.time()) + 3600,
            "jti": str(uuid.uuid4())
        })
        
        # Add patch for the JWT verification and blacklist check
        patches.append(patch('app.api.v1.endpoints.auth.verify_supabase_jwt', mock_verify_jwt))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.get(
                "/api/v1/auth/verify-token",
                headers={"Authorization": f"Bearer {valid_user_token}"}
            )
            
            # Verify response shows token is invalid
            assert response.status_code in [401, 403]  # Either unauthorized or forbidden
            data = response.json()
            assert "detail" in data
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_refresh_token_success(self, mock_dependencies):
        """Test successful token refresh"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Mock the refresh_token function result
        mock_refresh_result = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "token_type": "bearer",
            "expires_in": 3600
        }
        
        # Add patch for the refresh token function
        patches.append(patch('app.api.v1.endpoints.auth.refresh_token', AsyncMock(return_value=mock_refresh_result)))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "valid-refresh-token"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == "new-access-token"
            assert data["refresh_token"] == "new-refresh-token"
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_refresh_token_invalid(self, mock_dependencies):
        """Test token refresh with invalid token"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Add patch to return None for the refresh token function (invalid token)
        patches.append(patch('app.api.v1.endpoints.auth.refresh_token', AsyncMock(return_value=None)))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "invalid-refresh-token"}
            )
            
            # Verify response
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Invalid or expired token" in data["detail"]
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_get_user_info_success(self, valid_user_token, mock_dependencies):
        """Test successful user info retrieval"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure mocks for the user payload
        user_payload = {
            "id": "test-user-id",
            "email": "test@example.com",
            "roles": ["user"],
            "display_name": "Test User",
            "metadata": {"full_name": "Test User"},
            "created_at": "2023-01-01T00:00:00.000Z"
        }
        
        # Create a mock for the token verification and user dependency
        patches.append(patch('app.api.v1.endpoints.auth.get_current_user', AsyncMock(return_value=user_payload)))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.get(
                "/api/v1/auth/user",
                headers={"Authorization": f"Bearer {valid_user_token}"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-user-id"
            assert data["email"] == "test@example.com"
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_get_user_info_unauthenticated(self, mock_dependencies):
        """Test user info retrieval without authentication"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Create a mock for the token verification that raises an exception
        patches.append(patch('app.api.v1.endpoints.auth.get_current_user', 
                      AsyncMock(side_effect=AuthHTTPException(
                          status_code=401, 
                          detail="Missing authorization credentials",
                          headers={"WWW-Authenticate": "Bearer"}
                      ))))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint without token
            response = client.get("/api/v1/auth/user")
            
            # Verify response
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Missing authorization credentials" in data["detail"]
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_logout_success(self, valid_user_token, mock_dependencies):
        """Test successful logout"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure mocks for successful logout
        mock_redis.set.return_value = AsyncMock(return_value=True)
        mock_supabase.invalidate_refresh_token.return_value = AsyncMock(return_value=True)
        
        # Mock the token verification to return a valid payload
        mock_verify_jwt = AsyncMock(return_value={
            "sub": "test-user-id",
            "email": "test@example.com",
            "user_roles": ["user"],
            "exp": int(time.time()) + 3600,
            "jti": str(uuid.uuid4())
        })
        
        # Add patch for the JWT verification
        patches.append(patch('app.api.v1.endpoints.auth.verify_supabase_jwt', mock_verify_jwt))
        patches.append(patch('app.api.v1.endpoints.auth.blacklist_token', AsyncMock(return_value=True)))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.post(
                "/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {valid_user_token}"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "logged out" in data["message"].lower()
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
    
    async def test_health_check(self, mock_dependencies):
        """Test health check endpoint"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure mocks to return healthy status
        mock_redis.health_check.return_value = True
        mock_supabase.health_check.return_value = True
        
        # Override dependencies for testing
        app.dependency_overrides[get_redis_service] = lambda: mock_redis
        app.dependency_overrides[get_supabase_service] = lambda: mock_supabase
        
        try:
            # Test the endpoint
            response = client.get("/api/v1/auth/health")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["services"]["redis"] == "available"
            assert data["services"]["supabase"] == "available"
        finally:
            # Clear dependency overrides
            app.dependency_overrides.clear()
            
    async def test_health_check_degraded(self, mock_dependencies):
        """Test health check with degraded services"""
        patches, mock_redis, mock_supabase, _, async_mock_health_check_false = mock_dependencies
        
        # Configure mock methods for degraded status
        mock_redis.health_check = async_mock_health_check_false
        mock_supabase.health_check = AsyncMock(return_value=True)
        
        # Override dependencies for testing
        app.dependency_overrides[get_redis_service] = lambda: mock_redis
        app.dependency_overrides[get_supabase_service] = lambda: mock_supabase
        
        try:
            # Test the endpoint with debug information
            response = client.get("/api/v1/auth/health")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            print(f"Response data: {data}")
            assert data["status"] == "degraded"
            assert data["services"]["redis"] == "unavailable"
            assert data["services"]["supabase"] == "available"
        finally:
            # Clear dependency overrides
            app.dependency_overrides.clear()
    
    async def test_graceful_error_handling(self, valid_user_token, mock_dependencies):
        """Test graceful error handling when Redis fails"""
        patches, mock_redis, mock_supabase, _, _ = mock_dependencies
        
        # Configure redis to raise an exception
        mock_redis.get.side_effect = Exception("Redis connection error")
        
        # Add JWT verification that returns a valid payload
        mock_verify_jwt = AsyncMock(return_value={
            "sub": "test-user-id",
            "email": "test@example.com",
            "user_roles": ["user"],
            "exp": int(time.time()) + 3600,
            "jti": str(uuid.uuid4())
        })
        
        # Add patch for the JWT verification
        patches.append(patch('app.api.v1.endpoints.auth.verify_supabase_jwt', mock_verify_jwt))
        
        # Patch the services
        for p in patches:
            p.start()
        
        try:
            # Test the endpoint
            response = client.get(
                "/api/v1/auth/verify-token",
                headers={"Authorization": f"Bearer {valid_user_token}"}
            )
            
            # Verify that despite Redis error, we still get a response (either success or handled error)
            # and we don't crash the application
            assert response.status_code in [200, 500]
        finally:
            # Stop all patches
            for p in patches:
                p.stop()

if __name__ == "__main__":
    pytest.main(["-xvs", "test_auth_endpoints.py"])
