from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from datetime import datetime
import json
import logging
from app.api.routes import router as api_router
from app.core.middleware import RateLimitMiddleware, RequestTrackingMiddleware
from app.core.monitoring import PrometheusMiddleware
from app.core.security import SecurityHeadersMiddleware
from app.core.error_tracking import init_sentry, setup_logging, log_exception
from app.core.error_handling import ApplicationError, ServiceError, register_exception_handlers, extract_error_info
from app.core.dependency_injection import container
from app.core.service_registry import register_all_services
from app.core.websocket import manager
from app.core.config import settings
from app.tasks.forecasting import generate_forecast, run_stress_test
from app.services.market_data import MarketDataService
import uuid

# Initialize error tracking and logging
init_sentry()
logger = setup_logging()

# Initialize service container
logger.info("Initializing service container")
try:
    register_all_services()
    logger.info("Service container initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize service container: {str(e)}", exc_info=True)
    raise

app = FastAPI(
    title="Credit Cashflow Engine",
    description="API for credit cashflow forecasting and scenario analysis with absbox integration",
    version="2.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Add request tracking middleware
app.add_middleware(RequestTrackingMiddleware)

# Add Prometheus monitoring middleware
app.add_middleware(PrometheusMiddleware)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include API router - Using the updated router with enhanced analytics endpoints
app.include_router(api_router, prefix="/api")

# Register the application error handlers
register_exception_handlers(app)

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    """Global exception handler with logging and error tracking"""
    # Log the exception with context
    context = {
        "path": request.url.path,
        "method": request.method,
        "client_ip": request.client.host if request.client else None,
    }
    
    # Convert to ApplicationError for consistent handling if it's not already
    if not isinstance(exc, ApplicationError) and not isinstance(exc, HTTPException):
        error_details = extract_error_info(exc)
        log_exception(exc, context)
    else:
        # ApplicationErrors already have context
        log_exception(exc, context)
    
    # Return appropriate response
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    # For production, don't expose internal error details
    if settings.ENVIRONMENT == "production":
        error_message = "Internal server error"
    else:
        error_message = str(exc)
    
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        content=jsonable_encoder({
            "error": error_message,
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        content=jsonable_encoder({
            "error": exc.detail,
            "type": "HTTPException",
            "status_code": exc.status_code
        }),
        status_code=exc.status_code
    )

@app.exception_handler(ApplicationError)
async def application_error_handler(request, exc: ApplicationError):
    """Handler for ApplicationError exceptions"""
    # Already logged in the error handling decorator
    
    # Determine status code based on error type
    status_code = status.HTTP_400_BAD_REQUEST
    
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder({
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": exc.timestamp,
            # Only include safe context fields
            "context": {k: v for k, v in exc.context.items() if k not in ["traceback", "args", "kwargs"]}
        })
    )

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    connection_id = str(uuid.uuid4())
    try:
        await manager.connect(websocket, user_id, connection_id)
        
        while True:
            try:
                # Wait for messages from the client
                data = await websocket.receive_json()
                
                # Handle different message types
                message_type = data.get("type")
                
                if message_type == "subscribe":
                    # Subscribe to task updates
                    task_id = data.get("task_id")
                    if task_id:
                        await manager.subscribe_to_task(task_id, user_id)
                        await websocket.send_json({
                            "type": "subscription_confirmed",
                            "task_id": task_id
                        })
                
                elif message_type == "market_data_request":
                    # Handle real-time market data requests
                    market_service = MarketDataService()
                    yields = await market_service.get_treasury_yields()
                    await websocket.send_json({
                        "type": "market_data_update",
                        "data": yields
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    finally:
        manager.disconnect(user_id, connection_id)

@app.post("/api/v1/forecast/async")
async def create_async_forecast(request: dict, user_id: str):
    """Create asynchronous forecast"""
    try:
        # Queue the forecast task
        task = generate_forecast.delay(request, user_id)
        
        return {
            "task_id": task.id,
            "status": "queued",
            "websocket_url": f"/ws/{user_id}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/v1/forecast/stress-test")
async def create_stress_test(request: dict, user_id: str):
    """Run stress test scenarios"""
    try:
        # Get stress scenarios from market data service
        market_service = MarketDataService()
        scenarios = market_service.generate_stress_scenarios()
        
        # Queue the stress test task
        task = run_stress_test.delay(request, user_id, scenarios)
        
        return {
            "task_id": task.id,
            "status": "queued",
            "num_scenarios": len(scenarios),
            "websocket_url": f"/ws/{user_id}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for production monitoring"""
    from app.core.monitoring import check_redis_connection, check_system_health
    
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
    
    # Check system health (CPU, memory, disk)
    system_status = check_system_health()
    
    # Check Celery worker status
    from app.core.celery_app import celery_app
    worker_status = {"status": "unknown"}
    try:
        celery_ping = celery_app.control.ping()
        if celery_ping:
            worker_status = {"status": "healthy", "workers": len(celery_ping)}
        else:
            worker_status = {"status": "unhealthy", "error": "No Celery workers found"}
    except Exception as e:
        worker_status = {"status": "unhealthy", "error": str(e)}
    
    # Determine overall status
    components = [db_status, redis_status, system_status, worker_status]
    if any(component["status"] == "unhealthy" for component in components):
        overall_status = "unhealthy"
    elif any(component["status"] == "degraded" for component in components):
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": db_status,
            "redis": redis_status,
            "celery": worker_status,
            "system": system_status
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for monitoring"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    content = generate_latest()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)
