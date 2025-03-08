"""
Test configuration loading
"""

import pytest
import os
import json
from app.core.config import settings, Settings


def test_settings_class():
    """Test that Settings class can parse CORS_ORIGINS correctly from a string."""
    # Test JSON array format
    json_value = '["http://localhost:3000", "https://example.com"]'
    assert Settings.Config.parse_env_var("CORS_ORIGINS", json_value) == ["http://localhost:3000", "https://example.com"]
    
    # Test comma-separated format
    csv_value = "[http://localhost:3000, https://example.com]"
    assert Settings.Config.parse_env_var("CORS_ORIGINS", csv_value) == ["http://localhost:3000", "https://example.com"]
    
    # Test regular string (non-CORS_ORIGINS field)
    regular_value = "some_string"
    assert Settings.Config.parse_env_var("SOME_FIELD", regular_value) == "some_string"


def test_settings_loaded():
    """Test that settings are loaded and CORS_ORIGINS is parsed correctly."""
    # Either these settings come from .env, .env.test, or our mocked values
    assert settings is not None
    assert settings.VERSION == "1.0.0"
    
    # If CORS_ORIGINS is specified, it should be a list
    assert isinstance(settings.CORS_ORIGINS, list)
