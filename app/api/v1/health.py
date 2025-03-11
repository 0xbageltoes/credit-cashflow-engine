"""API endpoints for health checks and monitoring"""
import logging
from typing import Dict, Any, Optional
import time
import platform
import psutil
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user
from app.core.monitoring import CALCULATION_TIME, request_counter
from app.core.config import settings
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.core.cache import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", summary="Check service health")
async def health_check():
    """
    Check health of the service
    
    Returns basic system information, API version, and environment.
    """
    try:
        # Get basic system info
        system_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "memory_available": psutil.virtual_memory().available,
            "memory_total": psutil.virtual_memory().total,
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "disk_usage": psutil.disk_usage('/').percent,
        }
        
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "system_info": system_info
        }
    except Exception as e:
        logger.exception("Error in basic health check")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/absbox", summary="Check AbsBox service health")
async def absbox_health_check(current_user: Dict = Depends(get_current_user)):
    """
    Check health of the AbsBox analytics service
    
    Performs basic calculations to verify that the AbsBox library is functioning correctly.
    """
    try:
        start_time = time.time()
        service = AbsBoxServiceEnhanced()
        health_status = service.health_check()
        
        # Add execution time
        execution_time = time.time() - start_time
        health_status["execution_time"] = execution_time
        
        # Record the calculation time in Prometheus
        CALCULATION_TIME.labels(calculation="health_check").observe(execution_time)
        request_counter.labels(endpoint="/api/v1/health/absbox").inc()
        
        return health_status
    except Exception as e:
        logger.exception("Error in AbsBox health check")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/cache", summary="Check Redis cache status")
async def cache_health_check(current_user: Dict = Depends(get_current_user)):
    """
    Check health of the Redis cache
    
    Tests connection to Redis and reports cache statistics.
    """
    try:
        redis_client = get_redis_client()
        
        # Try a basic Redis operation
        if redis_client is None:
            return {
                "status": "disabled",
                "message": "Redis cache is not configured"
            }
            
        start_time = time.time()
        info = redis_client.info()
        ping_result = redis_client.ping()
        execution_time = time.time() - start_time
        
        cache_stats = {
            "status": "healthy" if ping_result else "error",
            "connected_clients": info.get("connected_clients", "unknown"),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "total_commands_processed": info.get("total_commands_processed", "unknown"),
            "uptime_in_seconds": info.get("uptime_in_seconds", "unknown"),
            "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1) + 0.001),
            "execution_time": execution_time
        }
        
        return cache_stats
    except Exception as e:
        logger.exception("Error in cache health check")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/metrics", summary="Get service metrics")
async def get_metrics(current_user: Dict = Depends(get_current_user)):
    """
    Get metrics about service usage and performance
    
    Returns count of calculations performed and execution times.
    """
    try:
        # Get metrics from the enhanced service
        service = AbsBoxServiceEnhanced()
        metrics = service.get_usage_metrics()
        
        return metrics
    except Exception as e:
        logger.exception("Error getting service metrics")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "detail": str(e)}
        )

@router.post("/clear-cache", summary="Clear service cache")
async def clear_cache(current_user: Dict = Depends(get_current_user)):
    """
    Clear the Redis cache for AbsBox calculations
    
    This endpoint requires admin privileges.
    """
    # Check if user has admin privileges
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    try:
        service = AbsBoxServiceEnhanced()
        result = service.clear_cache()
        
        return {"status": "success", "cleared_keys": result}
    except Exception as e:
        logger.exception("Error clearing cache")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
