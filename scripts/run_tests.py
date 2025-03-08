#!/usr/bin/env python
"""
Test Runner Script for Credit Cashflow Engine

This script runs the test suite and reports the results.
It's designed to work in any environment, including CI/CD pipelines.
"""

import os
import sys
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

def run_basic_test():
    """Run a basic test directly to verify pytest works."""
    # Set mock environment variables first
    set_mock_env()
    
    # Create a temporary test file
    temp_test_file = ROOT_DIR / "tests" / "temp_test.py"
    
    test_code = """
import pytest
import os
import json

def test_basic():
    assert 1 + 1 == 2
    print("Basic test passed!")
    
def test_env_vars():
    # Test that our environment variables are set correctly
    assert os.environ.get("ENV") == "testing"
    assert os.environ.get("SUPABASE_URL") == "https://example.supabase.co"
    
    # Test CORS_ORIGINS parsing
    cors_str = os.environ.get("CORS_ORIGINS")
    cors = json.loads(cors_str)
    assert isinstance(cors, list)
    assert "http://localhost:3000" in cors
"""
    
    # Write the test code to a file
    with open(temp_test_file, "w") as f:
        f.write(test_code)
    
    print(f"Running basic test from {temp_test_file}...")
    try:
        # Run pytest on the temporary file with environment variables set
        result = subprocess.run(
            ["python", "-m", "pytest", str(temp_test_file), "-v"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            env=os.environ.copy()  # Use the environment with our mock variables
        )
        
        # Display the output
        print("\nStandard Output:")
        print(result.stdout)
        
        print("\nStandard Error:")
        print(result.stderr)
        
        # Check if the test was successful
        if result.returncode == 0:
            print("Basic test passed successfully!")
            return True
        else:
            print(f"Basic test failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"Error running basic test: {e}")
        return False
    finally:
        # Clean up
        if temp_test_file.exists():
            temp_test_file.unlink()

def run_test_suite():
    """Run the full test suite."""
    print("Running full test suite...")
    
    # Set environment variables for testing
    set_mock_env()
    
    # First run a simple test with no dependencies
    simple_test_result = run_basic_test()
    
    if simple_test_result != True:
        print("Simple test failed! There may be an issue with the pytest setup.")
        return simple_test_result
    
    # Create a list of tests to run, starting with the minimal tests
    test_modules = [
        "tests/test_minimal.py",  # First run tests with no dependencies
    ]
    
    # Try to find more test files
    test_dir = ROOT_DIR / "tests"
    for test_file in test_dir.glob("test_*.py"):
        # Skip the minimal and temporary test files that we've already run
        if test_file.name not in ["test_minimal.py", "temp_test.py"]:
            test_modules.append(f"tests/{test_file.name}")
    
    # Run each test module separately to isolate failures
    exit_code = 0
    for test_module in test_modules:
        print(f"\nRunning tests from {test_module}...")
        
        try:
            # Run pytest with our environment variables
            result = subprocess.run(
                ["python", "-m", "pytest", test_module, "-v"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                env=os.environ.copy()  # Use the environment with our mock variables
            )
            
            # Print the output
            print(f"\nResults for {test_module}:")
            print(f"Exit code: {result.returncode}")
            
            if result.stdout:
                print("Standard Output:")
                print(result.stdout)
            
            if result.stderr:
                print("Standard Error:")
                print(result.stderr)
            
            # Track the worst exit code
            if result.returncode != 0:
                print(f"Tests in {test_module} failed with exit code {result.returncode}")
                exit_code = max(exit_code, result.returncode)
            else:
                print(f"Tests in {test_module} passed successfully!")
                
        except Exception as e:
            print(f"Error running tests from {test_module}: {e}")
            exit_code = 1
    
    # Generate coverage report if code coverage was enabled
    try:
        print("\nGenerating coverage report...")
        subprocess.run(
            ["python", "-m", "pytest", "--cov=app", "--cov-report=xml", "--cov-report=term"],
            cwd=ROOT_DIR
        )
    except Exception as e:
        print(f"Error generating coverage report: {e}")
    
    # Return the worst exit code from all test runs
    return exit_code

if __name__ == "__main__":
    # Run the test suite and exit with the same code
    exit_code = run_test_suite()
    sys.exit(exit_code)
