"""
Stress Testing Configuration Module

This module provides comprehensive configuration for the stress testing framework,
with production-ready defaults and environment variable support.
"""
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class StressTestSettings(BaseSettings):
    """
    Production-ready configuration settings for stress testing with proper
    defaults and environment variable support.
    """
    # Cache configuration
    STRESS_TEST_CACHE_TTL: int = Field(
        default=3600,
        description="Time-to-live for cached stress test results in seconds (Default: 1 hour)"
    )
    
    # Processing configuration
    STRESS_TEST_MAX_WORKERS: int = Field(
        default=4,
        description="Maximum number of parallel workers for stress testing"
    )
    
    STRESS_TEST_TIMEOUT: int = Field(
        default=300,
        description="Timeout in seconds for stress test operations"
    )
    
    STRESS_TEST_DEFAULT_PROJECTION_PERIODS: int = Field(
        default=360,
        description="Default number of projection periods for analysis (e.g., 360 months)"
    )
    
    # Monitoring and logging
    STRESS_TEST_LOG_LEVEL: str = Field(
        default="INFO",
        description="Log level for stress testing module (DEBUG, INFO, WARNING, ERROR)"
    )
    
    STRESS_TEST_DETAILED_METRICS: bool = Field(
        default=True,
        description="Whether to collect detailed metrics during stress testing"
    )
    
    # Resource limits
    STRESS_TEST_MAX_ASSETS_PER_POOL: int = Field(
        default=10000,
        description="Maximum number of assets allowed in a pool for stress testing"
    )
    
    STRESS_TEST_MAX_SCENARIOS: int = Field(
        default=20,
        description="Maximum number of scenarios allowed in a single stress test request"
    )
    
    # Rate limiting
    STRESS_TEST_RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        description="Maximum number of stress test requests allowed per minute per user"
    )
    
    # Defaults for market factors
    STRESS_TEST_DEFAULT_MARKET_FACTORS: Dict[str, Dict[str, float]] = Field(
        default={
            "rate_shock_up": {
                "interest_rate_shock": 0.03,
                "prepayment_multiplier": 0.7,
                "default_multiplier": 1.2
            },
            "rate_shock_down": {
                "interest_rate_shock": -0.02,
                "prepayment_multiplier": 1.5,
                "default_multiplier": 0.9
            },
            "credit_crisis": {
                "default_multiplier": 3.0,
                "recovery_multiplier": 0.6,
                "prepayment_multiplier": 0.5,
                "interest_rate_shock": 0.01
            },
            "liquidity_crisis": {
                "spread_widening": 0.05,
                "prepayment_multiplier": 0.4,
                "default_multiplier": 1.8
            }
        },
        description="Default market factor definitions for standard scenarios"
    )
    
    # Reporting configuration
    STRESS_TEST_REPORT_FORMAT: str = Field(
        default="json",
        description="Default report format (json, html, csv)"
    )
    
    STRESS_TEST_INCLUDE_CHARTS: bool = Field(
        default=True,
        description="Whether to include chart data in HTML reports"
    )
    
    # Environment-specific configuration
    STRESS_TEST_ENVIRONMENT: str = Field(
        default="production",
        description="Deployment environment (development, testing, staging, production)"
    )
    
    # Error handling and recovery
    STRESS_TEST_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of retries for failed calculations"
    )
    
    STRESS_TEST_RETRY_DELAY: int = Field(
        default=1,
        description="Delay in seconds between retries"
    )
    
    # Feature flags
    STRESS_TEST_FEATURES: Dict[str, bool] = Field(
        default={
            "enable_parallel_processing": True,
            "enable_caching": True,
            "enable_websocket_updates": True,
            "enable_report_generation": True,
            "enable_custom_scenarios": True,
            "enable_sensitivity_analysis": True
        },
        description="Feature flags for stress testing capabilities"
    )
    
    def get_feature_flag(self, flag_name: str) -> bool:
        """Get the state of a feature flag with fallback to False if not found"""
        return self.STRESS_TEST_FEATURES.get(flag_name, False)
    
    def get_market_factors(self, scenario_name: str) -> Optional[Dict[str, float]]:
        """Get the market factors for a named scenario with proper error handling"""
        return self.STRESS_TEST_DEFAULT_MARKET_FACTORS.get(scenario_name)
    
    class Config:
        """Configuration for the settings class with env file support"""
        env_prefix = ""  # Use default environment variable prefix
        env_file = ".env"  # Support .env file
        env_file_encoding = "utf-8"

# Create a singleton instance for global use
stress_test_settings = StressTestSettings()

def get_stress_test_settings() -> StressTestSettings:
    """
    Get the stress test settings instance with production defaults.
    This function allows for mocking in tests.
    
    Returns:
        StressTestSettings: The settings instance
    """
    return stress_test_settings
