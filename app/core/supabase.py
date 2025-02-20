from supabase import create_client as supabase_create_client
from app.core.config import settings

def create_client():
    """Create a Supabase client with configured settings"""
    return supabase_create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )
