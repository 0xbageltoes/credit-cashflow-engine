from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from datetime import datetime
import json
from app.core.auth import get_current_user
from app.services.cashflow import CashflowService
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse
)
from app.core.middleware import RateLimitMiddleware
from app.core.websocket import manager
from app.api.v1.api import api_router
from app.core.config import settings
from app.tasks.forecasting import generate_forecast, run_stress_test
from app.services.market_data import MarketDataService
import uuid

app = FastAPI(
    title="Credit Cashflow Engine",
    description="API for credit cashflow forecasting and scenario analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Initialize services
cashflow_service = CashflowService()

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(
        content=jsonable_encoder({"error": str(exc)}),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        content=jsonable_encoder({"error": exc.detail}),
        status_code=exc.status_code
    )

@app.post("/cashflow/forecast", response_model=CashflowForecastResponse)
async def generate_forecast(
    request: CashflowForecastRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate cash flow projections for a set of loans
    """
    try:
        result = await cashflow_service.generate_forecast(request, current_user["id"])
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/cashflow/scenario/save")
async def save_scenario(
    scenario: ScenarioSaveRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Save a scenario for future reference
    """
    try:
        result = await cashflow_service.save_scenario(scenario, current_user["id"])
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/cashflow/scenario/load", response_model=list[ScenarioResponse])
async def load_scenarios(
    current_user: dict = Depends(get_current_user)
):
    """
    Load saved scenarios
    """
    try:
        result = await cashflow_service.load_scenarios(current_user["id"])
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/cashflow/history")
async def get_forecast_history(
    current_user: dict = Depends(get_current_user)
):
    """
    Get forecast history
    """
    try:
        result = await cashflow_service.get_forecast_history(current_user["id"])
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "endpoints": {
            "/api/v1/forecast": "Generate cashflow forecasts",
            "/api/v1/scenarios": "Manage forecast scenarios",
            "/health": "Health check endpoint",
            "/metrics": "Metrics endpoint",
            "/ws/{user_id}": "WebSocket connection for real-time updates",
            "/api/v1/forecast/async": "Generate async cashflow forecasts",
            "/api/v1/forecast/stress-test": "Run stress test scenarios"
        }
    }
