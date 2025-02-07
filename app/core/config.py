from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Environment variables as they appear in .env
    NEXT_PUBLIC_SUPABASE_URL: str
    NEXT_PUBLIC_SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # Computed properties
    @property
    def SUPABASE_URL(self) -> str:
        return self.NEXT_PUBLIC_SUPABASE_URL
    
    @property
    def SUPABASE_KEY(self) -> str:
        return self.NEXT_PUBLIC_SUPABASE_ANON_KEY
    
    # Redis settings (optional)
    REDIS_URL: Optional[str] = None
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Credit Cashflow Engine"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
