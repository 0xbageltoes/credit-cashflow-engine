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
    
    # Upstash Redis settings
    UPSTASH_REDIS_HOST: str
    UPSTASH_REDIS_PORT: str
    UPSTASH_REDIS_PASSWORD: str
    
    # Redis specific configuration
    REDIS_CONFIG: RedisConfig = RedisConfig()
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Monitoring and Logging
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    PROMETHEUS_ENABLED: bool = True
    LOGGING_JSON_FORMAT: bool = True
    
    # Security settings
    HASH_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ACCESS_TOKEN_EXPIRE_SECONDS: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    JWT_TOKEN_PREFIX: str = "Bearer"
    SSL_VERIFICATION: bool = True if ENVIRONMENT == "production" else False
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development_secret_key")
    
    # JWT Audience and Issuer settings
    JWT_AUDIENCE: Optional[str] = os.getenv("JWT_AUDIENCE")
    JWT_ISSUER: Optional[str] = os.getenv("JWT_ISSUER")
    JWT_TOKEN_LOCATION: List[str] = ["headers", "cookies"]
    
    # Performance settings
    RATE_LIMIT_REQUESTS: int = 100  # Number of requests
    RATE_LIMIT_WINDOW: int = 3600   # Time window in seconds (1 hour)
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    BATCH_SIZE: int = 1000  # Size of batch operations for database
    CALCULATION_THREAD_POOL_SIZE: int = int(os.getenv("CALCULATION_THREAD_POOL_SIZE", "4"))
    
    # Task queue settings
    CELERY_TASK_ALWAYS_EAGER: bool = False if ENVIRONMENT == "production" else True
    CELERY_WORKER_CONCURRENCY: int = int(os.getenv("CELERY_CONCURRENCY", "4"))
    CELERY_TASK_TIME_LIMIT: int = 1800  # 30 minutes
    CELERY_TASK_MAX_RETRIES: int = 3
    
    # Hastructure Engine Settings
    HASTRUCTURE_URL: Optional[str] = os.getenv("HASTRUCTURE_URL", "http://hastructure:8081")
    HASTRUCTURE_TIMEOUT: int = int(os.getenv("HASTRUCTURE_TIMEOUT", "300"))
    HASTRUCTURE_MAX_POOL_SIZE: int = int(os.getenv("HASTRUCTURE_MAX_POOL_SIZE", "10"))
    
    # Market Data API settings
    FRED_API_KEY: Optional[str] = None
    BLOOMBERG_API_KEY: Optional[str] = None
    
    # AWS settings (for production)
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_S3_BUCKET: Optional[str] = os.getenv("AWS_S3_BUCKET")
    
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
