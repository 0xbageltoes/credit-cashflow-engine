"""
Health check and monitoring endpoints for the API
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import platform
import datetime
from fastapi.responses import Response
from app.core.config import settings
from app.core.monitoring import check_redis_connection, check_system_health
from app.core.auth import get_current_user
from app.services.analytics import AnalyticsService

router = APIRouter()

# Try to import psutil, but use fallbacks if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@router.get("/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring services like Kubernetes.
    Returns basic health information and status.
    """
    start_time = time.time()
    
    # Check database connection
    db_status = {"status": "healthy"}
    try:
        # Test Supabase connection
        from app.core.config import get_supabase_client
        supabase = get_supabase_client()
        response = supabase.table("health_check").select("*").limit(1).execute()
        db_status["message"] = "Connected to Supabase"
    except Exception as e:
        db_status = {"status": "unhealthy", "error": str(e)}
    
    # Check Redis connection
    redis_status = check_redis_connection()
    
    # Get system info
    system_info = {}
    if HAS_PSUTIL:
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }
    else:
        system_info = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_percent": 0,
            "message": "psutil not installed, system metrics unavailable"
        }
    
    # Calculate response time
    response_time = time.time() - start_time
    
    return {
        "status": "ok",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime": check_system_health(),
        "database": db_status,
        "cache": redis_status,
        "system": system_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "response_time_ms": round(response_time * 1000, 2),
    }


@router.get("/ready", summary="Readiness Check")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes and other orchestration systems.
    Verifies that the application is ready to accept traffic.
    """
    # Check if critical components are initialized
    is_ready = True
    checks = {
        "database": {"status": "ok"},
        "cache": {"status": "ok"},
        "apis": {"status": "ok"},
    }
    
    # Check Redis connection (simplistic check)
    try:
        redis_status = check_redis_connection()
        if redis_status["status"] != "healthy":
            checks["cache"]["status"] = "error"
            checks["cache"]["message"] = "Redis connection failed"
            is_ready = False
    except Exception as e:
        checks["cache"]["status"] = "error"
        checks["cache"]["message"] = str(e)
        is_ready = False
    
    # Check database connection (simplistic check)
    try:
        from app.core.config import get_supabase_client
        supabase = get_supabase_client()
        supabase.table("health_check").select("*").limit(1).execute()
    except Exception as e:
        checks["database"]["status"] = "error"
        checks["database"]["message"] = str(e)
        is_ready = False
    
    return {
        "status": "ok" if is_ready else "error",
        "checks": checks,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@router.get("/metrics", summary="Prometheus Metrics")
async def metrics_endpoint(current_user: dict = Depends(get_current_user)):
    """Prometheus metrics endpoint for monitoring"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

@router.get("/api-metrics", summary="API Usage Metrics")
async def api_metrics(current_user: dict = Depends(get_current_user)):
    """Get API usage metrics"""
    analytics_service = AnalyticsService()
    metrics = await analytics_service.get_api_metrics()
    return metrics
