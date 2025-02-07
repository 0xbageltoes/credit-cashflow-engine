from fastapi import FastAPI, Depends, HTTPException, status
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
            "/metrics": "Metrics endpoint"
        }
    }
