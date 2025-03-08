#!/usr/bin/env python
"""
Test runner that explicitly sets PYTHONPATH and pre-configures environment variables.
"""

import os
import sys
import subprocess
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.absolute()

def run_tests():
    """Run tests with PYTHONPATH set to the root directory."""
    print(f"Running tests with PYTHONPATH set to {ROOT_DIR}")
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["ENV"] = "testing"
    env["SUPABASE_URL"] = "https://example.supabase.co"
    env["SUPABASE_KEY"] = "dummy_key"
    env["SUPABASE_SERVICE_ROLE_KEY"] = "dummy_service_role_key"
    env["SUPABASE_JWT_SECRET"] = "dummy_jwt_secret"
    env["NEXT_PUBLIC_SUPABASE_URL"] = "https://example.supabase.co"
    env["NEXT_PUBLIC_SUPABASE_ANON_KEY"] = "dummy_anon_key"
    env["UPSTASH_REDIS_HOST"] = "localhost"
    env["UPSTASH_REDIS_PORT"] = "6379"
    env["UPSTASH_REDIS_PASSWORD"] = "dummy_password"
    env["CORS_ORIGINS"] = '["http://localhost:3000","https://example.com"]'
    
    # First try with our minimal test to verify testing setup
    print("Running minimal tests...")
    minimal_cmd = [sys.executable, "-m", "pytest", "tests/test_minimal.py", "-v"]
    
    result = subprocess.run(minimal_cmd, env=env, cwd=ROOT_DIR)
    
    if result.returncode != 0:
        print(f"Minimal tests failed with exit code {result.returncode}")
        return result.returncode
    
    print("Minimal tests passed! Running selected tests...")
    
    # Run selected tests that should work
    selected_tests = [
        "tests/test_minimal.py",
        "tests/test_config.py"
    ]
    
    selected_cmd = [
        sys.executable, 
        "-m", 
        "pytest",
        *selected_tests,
        "-v", 
        "--cov=app", 
        "--cov-report=term", 
        "--cov-report=xml:coverage.xml"
    ]
    
    result = subprocess.run(selected_cmd, env=env, cwd=ROOT_DIR)
    
    if result.returncode == 0:
        print("Selected tests passed!")
    else:
        print(f"Some selected tests failed with exit code {result.returncode}")
    
    print("\nSome tests were skipped because they require additional dependencies or mock setups.")
    print("Focus on making these selected tests pass before proceeding to the full test suite.")
    
    return result.returncode

if __name__ == "__main__":
    # Run tests and exit with the same code
    sys.exit(run_tests())
