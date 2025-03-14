"""Test package for credit-cashflow-engine

This module automatically sets up the testing environment before any tests run.
It ensures all required environment variables are properly set.
"""

import os
import sys
import pathlib
from dotenv import load_dotenv

# We need to initialize environment variables before any app imports happen
# This ensures proper configuration for all tests

def _init_test_environment():
    """
    Initialize the test environment before any tests run.
    This function is automatically called when the tests package is imported.
    """
    # Load .env.test file first
    current_dir = pathlib.Path(__file__).parent.absolute()
    env_test_path = current_dir / ".env.test"
    
    if env_test_path.exists():
        load_dotenv(dotenv_path=env_test_path, override=True)
        print(f"Loaded test environment from {env_test_path}")
    
    # Check if we need to set up required variables
    if os.environ.get("TEST_ENV_SETUP_DONE"):
        return
        
    # Make sure all required environment variables are set with actual values
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
    
    # Set any missing environment variables
    missing_vars = []
    for key, value in required_vars.items():
        if not os.environ.get(key):
            os.environ[key] = value
            missing_vars.append(key)
    
    if missing_vars:
        print(f"Set missing environment variables: {', '.join(missing_vars)}")
    
    # Mark as done so we don't set them again
    os.environ["TEST_ENV_SETUP_DONE"] = "1"
    
    # Validate that critical settings are present
    critical_vars = ["SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL", "UPSTASH_REDIS_HOST"]
    missing_critical = [var for var in critical_vars if not os.environ.get(var)]
    
    if missing_critical:
        print(f"WARNING: Missing critical environment variables: {', '.join(missing_critical)}")
        print("Tests may not run correctly without these variables.")
    else:
        print("Test environment initialized successfully with all critical variables")

# Initialize test environment
_init_test_environment()
