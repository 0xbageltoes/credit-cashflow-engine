from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_current_user
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse
)
from app.services.cashflow import CashflowService
from typing import List

cashflow_router = APIRouter()

@cashflow_router.post("/forecast", response_model=CashflowForecastResponse)
async def create_forecast(
    request: CashflowForecastRequest,
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """
    Generate a cash flow forecast based on input parameters
    """
    try:
        result = await cashflow_service.generate_forecast(request, current_user["id"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.post("/scenario/save")
async def save_scenario(
    scenario: ScenarioSaveRequest,
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """
    Save a forecasting scenario
    """
    try:
        return await cashflow_service.save_scenario(scenario, current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/scenario/load", response_model=List[ScenarioResponse])
async def load_scenarios(
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """
    Load saved scenarios for the current user
    """
    try:
        return await cashflow_service.load_scenarios(current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/history")
async def get_history(
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """
    Get forecast history for the current user
    """
    try:
        return await cashflow_service.get_forecast_history(current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
