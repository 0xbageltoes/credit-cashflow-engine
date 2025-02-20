from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import ssl

class Settings(BaseSettings):
    # Environment variables as they appear in .env
    NEXT_PUBLIC_SUPABASE_URL: str
    NEXT_PUBLIC_SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # Upstash Redis settings
    UPSTASH_REDIS_HOST: str
    UPSTASH_REDIS_PORT: str
    UPSTASH_REDIS_PASSWORD: str
    
    @property
    def REDIS_URL(self) -> str:
        """Upstash Redis URL with SSL"""
        return f"rediss://:{self.UPSTASH_REDIS_PASSWORD}@{self.UPSTASH_REDIS_HOST}:{self.UPSTASH_REDIS_PORT}?ssl_cert_reqs=required"
    
    @property
    def CELERY_BROKER_URL(self) -> str:
        """Celery broker URL"""
        return self.REDIS_URL
    
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        """Celery result backend URL"""
        return self.REDIS_URL
    
    # Computed properties
    @property
    def SUPABASE_URL(self) -> str:
        return self.NEXT_PUBLIC_SUPABASE_URL
    
    @property
    def SUPABASE_KEY(self) -> str:
        return self.NEXT_PUBLIC_SUPABASE_ANON_KEY
    
    # Rate limiting settings
    RATE_LIMIT_REQUESTS: int = 100  # Number of requests
    RATE_LIMIT_WINDOW: int = 3600   # Time window in seconds (1 hour)
    
    # Cache settings
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    BATCH_SIZE: int = 1000  # Size of batch operations for database
    
    # Market Data API settings
    FRED_API_KEY: Optional[str] = None
    FRED_CACHE_TTL: int = 3600  # 1 hour cache for market data
    BLOOMBERG_API_KEY: Optional[str] = None
    ECB_API_URL: str = "https://sdw-wsrest.ecb.europa.eu/service"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Credit Cashflow Engine"
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields in environment variables

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
