from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List, Any, Dict, Union
import ssl
import os
import json
from pathlib import Path

class RedisConfig(BaseSettings):
    """Redis configuration with all connection parameters and fallback options"""
    HOST: str = os.getenv("REDIS_HOST", "localhost")
    PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    DB: int = int(os.getenv("REDIS_DB", "0"))
    SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    SSL_CERT_REQS: str = os.getenv("REDIS_SSL_CERT_REQS", "required")
    SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
    SOCKET_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))
    RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("REDIS_RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BACKOFF_FACTOR: float = float(os.getenv("REDIS_RETRY_BACKOFF_FACTOR", "0.5"))
    
    class Config:
        env_prefix = "REDIS_"
        case_sensitive = True

class Settings(BaseSettings):
    # Application settings
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENV", "development")
    API_V1_STR: str = "/api/v1"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Rate limiting settings
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD: str = os.getenv("RATE_LIMIT_PERIOD", "hour")
    
    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_KEY: str = ""  # Alias for SUPABASE_ANON_KEY for backward compatibility
    SUPABASE_ANON_KEY: str = ""  # Set from SUPABASE_KEY if not provided
    SUPABASE_SERVICE_KEY: str  # Service role key for admin operations
    SUPABASE_JWT_SECRET: str
    SUPABASE_JWT_AUDIENCE: str = ""  # Default to empty, will use base URL if not set
    NEXT_PUBLIC_SUPABASE_URL: str
    NEXT_PUBLIC_SUPABASE_ANON_KEY: str
    
    # Internal API settings
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "development_internal_key")
    INTERNAL_API_BASE_URL: str = os.getenv("INTERNAL_API_BASE_URL", "http://localhost:8000")
    
    # Redis settings
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD") 
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # Always prioritize Upstash Redis credentials if available
    UPSTASH_REDIS_HOST: Optional[str] = os.getenv("UPSTASH_REDIS_HOST")
    UPSTASH_REDIS_PORT: str = os.getenv("UPSTASH_REDIS_PORT", "6379")
    UPSTASH_REDIS_PASSWORD: Optional[str] = os.getenv("UPSTASH_REDIS_PASSWORD")
    
    # Safe getter for settings to avoid AttributeError
    def get(self, name, default=None):
        """
        Safely get a setting with a default value if it doesn't exist.
        This prevents AttributeError exceptions in production.
        
        Args:
            name: Name of the setting to get
            default: Default value if setting doesn't exist
            
        Returns:
            Setting value or default
        """
        return getattr(self, name, default)
    
    # Make settings safely accessible with dict-like syntax
    def __getitem__(self, name):
        """
        Allow dict-like access to settings with safe defaults.
        
        Args:
            name: Setting name to access
            
        Returns:
            Setting value or None if not found
        """
        return self.get(name)
    
    @property
    def REDIS_CONFIG(self) -> RedisConfig:
        """Get Redis configuration with proper prioritization of Upstash"""
        # Create base Redis config
        config = RedisConfig(
            HOST=self.REDIS_HOST,
            PORT=self.REDIS_PORT,
            PASSWORD=self.REDIS_PASSWORD,
            DB=self.REDIS_DB,
            SSL=self.REDIS_SSL,
            SSL_CERT_REQS=self.REDIS_SSL_CERT_REQS,
            ENABLED=self.REDIS_ENABLED
        )
        
        # ALWAYS prioritize Upstash Redis if available, regardless of environment
        upstash_host = os.environ.get("UPSTASH_REDIS_HOST")
        upstash_port = os.environ.get("UPSTASH_REDIS_PORT", "6379")
        upstash_password = os.environ.get("UPSTASH_REDIS_PASSWORD")
        
        # If Upstash environment variables are available, use them
        if upstash_host and upstash_password:
            print(f"Using Upstash Redis at {upstash_host}:{upstash_port}")
            config.HOST = upstash_host
            config.PORT = int(upstash_port)
            config.PASSWORD = upstash_password
            config.SSL = True
            config.SSL_CERT_REQS = "required"
            
            # Set Redis URL for Upstash
            config.REDIS_URL = f"rediss://default:{upstash_password}@{upstash_host}:{upstash_port}"
        else:
            # Log warning if Upstash credentials are missing in production
            if self.ENVIRONMENT.lower() in ["prod", "production"]:
                print("WARNING: Production environment detected but Upstash Redis credentials are missing!")
        
        return config
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set SUPABASE_ANON_KEY from SUPABASE_KEY if not provided
        if not self.SUPABASE_ANON_KEY and self.SUPABASE_KEY:
            self.SUPABASE_ANON_KEY = self.SUPABASE_KEY
        
        # Set default JWT audience if not provided
        if not self.SUPABASE_JWT_AUDIENCE and self.SUPABASE_URL:
            self.SUPABASE_JWT_AUDIENCE = self.SUPABASE_URL
    
    @property
    def REDIS_URL(self) -> str:
        """Upstash Redis URL with SSL"""
        if self.ENVIRONMENT == "production":
            return f"rediss://:{self.UPSTASH_REDIS_PASSWORD}@{self.UPSTASH_REDIS_HOST}:{self.UPSTASH_REDIS_PORT}?ssl_cert_reqs=required"
        else:
            return f"redis://:{self.UPSTASH_REDIS_PASSWORD}@{self.UPSTASH_REDIS_HOST}:{self.UPSTASH_REDIS_PORT}"
    
    @property
    def REDIS_CONNECTION_PARAMS(self) -> Dict[str, Any]:
        """Get Redis connection parameters with proper SSL configuration"""
        params = {
            "host": self.REDIS_CONFIG.HOST,
            "port": self.REDIS_CONFIG.PORT,
            "db": self.REDIS_CONFIG.DB,
            "socket_timeout": self.REDIS_CONFIG.SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_CONFIG.SOCKET_CONNECT_TIMEOUT,
            "retry_on_timeout": self.REDIS_CONFIG.RETRY_ON_TIMEOUT,
            "max_connections": self.REDIS_CONFIG.MAX_CONNECTIONS,
            "health_check_interval": self.REDIS_CONFIG.HEALTH_CHECK_INTERVAL,
        }
        
        # Add password if set
        if self.REDIS_CONFIG.PASSWORD:
            params["password"] = self.REDIS_CONFIG.PASSWORD
        
        # Configure SSL if enabled
        if self.REDIS_CONFIG.SSL:
            params["ssl"] = True
            
            if self.REDIS_CONFIG.SSL_CERT_REQS == "required":
                params["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            elif self.REDIS_CONFIG.SSL_CERT_REQS == "optional":
                params["ssl_cert_reqs"] = ssl.CERT_OPTIONAL
            else:
                params["ssl_cert_reqs"] = ssl.CERT_NONE
        
        return params
    
    @property
    def CELERY_BROKER_URL(self) -> str:
        """Celery broker URL"""
        return self.REDIS_URL
    
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        """Celery result backend URL"""
        return self.REDIS_URL
    
    class Config:
        env_file = ".env"
        # Look for .env.test if we're in a test environment
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("ENV") == "testing":
            # First try .env.test in the project root
            env_test = Path(".env.test")
            if env_test.exists():
                env_file = str(env_test)
                
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            """Parse environment variables with special handling for list types like CORS_ORIGINS."""
            if field_name in ["CORS_ORIGINS", "JWT_TOKEN_LOCATION"] and raw_val.startswith("[") and raw_val.endswith("]"):
                try:
                    # Try to parse the value as JSON
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    # If JSON parsing fails, split by commas
                    return [item.strip() for item in raw_val.strip("[]").split(",")]
            return raw_val

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

def get_supabase_client():
    """Get Supabase client with proper settings"""
    from supabase import create_client, Client
    
    client = create_client(
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_ANON_KEY,
    )
    
    return client

def get_supabase_admin_client():
    """Get Supabase client with service role key for admin operations"""
    from supabase import create_client, Client
    
    client = create_client(
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_SERVICE_KEY,
    )
    
    return client
