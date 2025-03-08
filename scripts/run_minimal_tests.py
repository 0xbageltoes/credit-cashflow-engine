#!/usr/bin/env python
"""
Minimal Test Runner Script for Credit Cashflow Engine

This script runs only the minimal test suite that doesn't depend on app imports.
It's designed to validate that the testing environment is working correctly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.absolute()

def set_mock_env():
    """Set mock environment variables for testing."""
    env_vars = {
        "ENV": "testing",
        "LOG_LEVEL": "debug",
        "WORKERS": "1",
        "CALCULATION_THREAD_POOL_SIZE": "1",
        "SECRET_KEY": "test_secret_key",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "dummy_key",
        "SUPABASE_SERVICE_ROLE_KEY": "dummy_service_role_key",
        "SUPABASE_JWT_SECRET": "dummy_jwt_secret",
        "NEXT_PUBLIC_SUPABASE_URL": "https://example.supabase.co",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY": "dummy_anon_key",
        "UPSTASH_REDIS_HOST": "localhost",
        "UPSTASH_REDIS_PORT": "6379",
        "UPSTASH_REDIS_PASSWORD": "dummy_password",
        # CORS_ORIGINS needs to be a JSON-formatted list as a string
        "CORS_ORIGINS": '["http://localhost:3000", "https://example.com"]'
    }
    
    # Update environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("Mock environment variables set for testing")

def run_minimal_test():
    """Run minimal tests that don't depend on app imports."""
    print("Running minimal test suite...")
    
    # Set environment variables for testing
    set_mock_env()
    
    # Run pytest on the minimal test file with environment variables set
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_minimal.py", "-v", "--no-header"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        env=os.environ.copy()  # Use the environment with our mock variables
    )
    
    # Display the output
    print("\nTest Results:")
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(result.stderr)
    
    # Check if the test was successful
    if result.returncode == 0:
        print("Minimal tests passed successfully!")
        return 0
    else:
        print(f"Minimal tests failed with exit code {result.returncode}")
        return result.returncode

if __name__ == "__main__":
    # Run the minimal test suite and exit with the same code
    exit_code = run_minimal_test()
    sys.exit(exit_code)
