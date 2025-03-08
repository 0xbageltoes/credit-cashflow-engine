#!/usr/bin/env python
"""
Isolated test runner that doesn't require conftest.py or any app module imports.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

def run_isolated_test():
    """Run a completely isolated test that doesn't depend on any app imports."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
        temp.write("""
import os
import pytest

def test_isolated():
    \"\"\"Basic test that doesn't import any app modules.\"\"\"
    assert 1 + 1 == 2
    print("Isolated test passed!")
    
def test_env_vars():
    \"\"\"Test setting environment variables.\"\"\"
    os.environ["TEST_VAR"] = "test_value"
    assert os.environ.get("TEST_VAR") == "test_value"
""")
        temp_file = temp.name
    
    try:
        # Set environment variables
        env = os.environ.copy()
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
        
        # Run pytest on the temp file with -k to specify only certain tests
        # and --no-header to reduce clutter
        print(f"Running isolated test from {temp_file}...")
        cmd = [sys.executable, "-m", "pytest", temp_file, "-v", "--no-header"]
        
        # Add -xvs for more detailed output
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Display the output
        print("\nStandard Output:")
        print(result.stdout)
        
        print("\nStandard Error:")
        print(result.stderr)
        
        # Return the exit code
        if result.returncode == 0:
            print("Isolated test passed!")
        else:
            print(f"Isolated test failed with exit code {result.returncode}")
        
        return result.returncode
    
    finally:
        # Clean up the temp file
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    # Run the test and exit with the same code
    sys.exit(run_isolated_test())
