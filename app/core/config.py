from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List, Any, Dict
import ssl
import os
import json
from pathlib import Path

class Settings(BaseSettings):
    # Application settings
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENV", "development")
    API_V1_STR: str = "/api/v1"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str
    NEXT_PUBLIC_SUPABASE_URL: str
    NEXT_PUBLIC_SUPABASE_ANON_KEY: str
    
    # Upstash Redis settings
    UPSTASH_REDIS_HOST: str
    UPSTASH_REDIS_PORT: str
    UPSTASH_REDIS_PASSWORD: str
    
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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SSL_VERIFICATION: bool = True if ENVIRONMENT == "production" else False
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development_secret_key")
    
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
    
    @property
    def REDIS_URL(self) -> str:
        """Upstash Redis URL with SSL"""
        if self.ENVIRONMENT == "production":
            return f"rediss://:{self.UPSTASH_REDIS_PASSWORD}@{self.UPSTASH_REDIS_HOST}:{self.UPSTASH_REDIS_PORT}?ssl_cert_reqs=required"
        else:
            return f"redis://:{self.UPSTASH_REDIS_PASSWORD}@{self.UPSTASH_REDIS_HOST}:{self.UPSTASH_REDIS_PORT}"
    
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
            if field_name == "CORS_ORIGINS" and raw_val.startswith("[") and raw_val.endswith("]"):
                try:
                    # Try to parse the value as JSON
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    # If JSON parsing fails, split by commas
                    return [origin.strip() for origin in raw_val.strip("[]").split(",")]
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
        supabase_key=settings.SUPABASE_KEY,
    )
    
    return client
