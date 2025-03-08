"""
Basic test module to verify testing setup is working correctly.
"""

import pytest
from app.core.config import settings


def test_settings_loaded():
    """Test that settings are loaded correctly."""
    assert settings.VERSION == "1.0.0"
    assert settings.ENVIRONMENT in ["development", "testing", "production", "test"]


def test_api_path():
    """Test that API path is set correctly."""
    assert settings.API_V1_STR == "/api/v1"


def test_supabase_config():
    """Test that Supabase configuration is loaded."""
    assert settings.SUPABASE_URL is not None
    assert settings.SUPABASE_KEY is not None
    

def test_basic_math():
    """A simple test that should always pass."""
    assert 1 + 1 == 2
    assert 2 * 2 == 4
