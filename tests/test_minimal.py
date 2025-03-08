"""
Minimal test module with no external dependencies.
"""
import os
import json
import pytest

def test_basic_math():
    """A simple test that should always pass."""
    assert 1 + 1 == 2
    assert 2 * 2 == 4
    
def test_string_operations():
    """Test basic string operations."""
    assert "hello" + " world" == "hello world"
    assert "hello".upper() == "HELLO"
    
def test_list_operations():
    """Test basic list operations."""
    my_list = [1, 2, 3]
    assert len(my_list) == 3
    assert sum(my_list) == 6

def test_env_cors_parsing():
    """Test that we can parse CORS_ORIGINS correctly."""
    # Define test cases
    test_cases = [
        {
            "input": '["http://localhost:3000", "https://example.com"]',
            "expected": ["http://localhost:3000", "https://example.com"]
        },
        {
            "input": "[http://localhost:3000, https://example.com]",
            "expected": ["http://localhost:3000", "https://example.com"]
        }
    ]
    
    for test_case in test_cases:
        # Test JSON parsing
        try:
            result = json.loads(test_case["input"])
            assert result == test_case["expected"]
        except json.JSONDecodeError:
            # If JSON parsing fails, try the comma-separated approach
            result = [origin.strip() for origin in test_case["input"].strip("[]").split(",")]
            assert result == test_case["expected"]

def test_mock_environment():
    """Test that we can set environment variables for testing."""
    # Set mock environment variables
    os.environ["TEST_VAR"] = "test_value"
    os.environ["CORS_ORIGINS"] = '["http://localhost:3000", "https://example.com"]'
    
    # Check that they're set
    assert os.environ.get("TEST_VAR") == "test_value"
    
    # Parse CORS_ORIGINS
    cors_origins_str = os.environ.get("CORS_ORIGINS")
    try:
        cors_origins = json.loads(cors_origins_str)
    except json.JSONDecodeError:
        cors_origins = [origin.strip() for origin in cors_origins_str.strip("[]").split(",")]
    
    assert isinstance(cors_origins, list)
    assert "http://localhost:3000" in cors_origins
