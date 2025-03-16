"""
Forecasting API Endpoints

This module provides REST API endpoints for forecasting operations
such as time series predictions, scenario analysis, and what-if modeling.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, date
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.core.logging import logger
from app.services.forecasting import ForecastingService
from app.models.forecasting import (
    ForecastRequest,
    ScenarioRequest,
    ForecastingResult,
    TimeSeriesData,
    ForecastParameters
)

# Create router for forecasting endpoints
router = APIRouter()

class TimeSeriesRequest(BaseModel):
    """Time series data for forecasting"""
    series_id: str = Field(..., description="Unique identifier for the time series")
    series_name: str = Field(..., description="Display name for the time series")
    data_points: List[Dict[str, Any]] = Field(..., description="Time series data points")
    parameters: ForecastParameters = Field(..., description="Forecasting parameters")
    
class ForecastResponse(BaseModel):
    """Forecast response model"""
    forecast_id: str
    created_at: datetime
    forecast_data: List[Dict[str, Any]]
    confidence_intervals: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]

@router.post("/time-series", response_model=ForecastResponse)
async def forecast_time_series(
    request: TimeSeriesRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate time series forecasts based on historical data
    
    This endpoint accepts time series data and forecasting parameters
    to generate future projections with confidence intervals.
    """
    try:
        forecasting_service = ForecastingService()
        result = await forecasting_service.forecast_time_series(
            series_id=request.series_id,
            series_name=request.series_name,
            data_points=request.data_points,
            parameters=request.parameters,
            user_id=current_user["id"]
        )
        return result
    except Exception as e:
        logger.error(f"Error in time series forecasting: {str(e)}", extra={
            "user_id": current_user["id"],
            "series_id": request.series_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenarios", response_model=List[ForecastResponse])
async def run_scenarios(
    request: ScenarioRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Run multiple forecasting scenarios with different parameters
    
    This endpoint allows running what-if analyses by varying parameters
    across multiple scenarios to compare different potential outcomes.
    """
    try:
        forecasting_service = ForecastingService()
        results = await forecasting_service.run_scenario_analysis(
            base_data=request.base_data,
            scenarios=request.scenarios,
            user_id=current_user["id"]
        )
        return results
    except Exception as e:
        logger.error(f"Error in scenario forecasting: {str(e)}", extra={
            "user_id": current_user["id"],
            "scenario_count": len(request.scenarios),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{series_id}", response_model=TimeSeriesData)
async def get_historical_data(
    series_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Retrieve historical time series data for analysis
    
    This endpoint retrieves stored time series data for a given identifier,
    optionally filtered by date range.
    """
    try:
        forecasting_service = ForecastingService()
        result = await forecasting_service.get_historical_data(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            user_id=current_user["id"]
        )
        return result
    except Exception as e:
        logger.error(f"Error retrieving historical data: {str(e)}", extra={
            "user_id": current_user["id"],
            "series_id": series_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecasts/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Retrieve a previously generated forecast
    
    This endpoint retrieves a previously generated forecast by its ID.
    """
    try:
        forecasting_service = ForecastingService()
        result = await forecasting_service.get_forecast(
            forecast_id=forecast_id,
            user_id=current_user["id"]
        )
        if not result:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving forecast: {str(e)}", extra={
            "user_id": current_user["id"],
            "forecast_id": forecast_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/forecasts/{forecast_id}", response_model=Dict[str, bool])
async def delete_forecast(
    forecast_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a previously generated forecast
    
    This endpoint deletes a previously generated forecast by its ID.
    """
    try:
        forecasting_service = ForecastingService()
        result = await forecasting_service.delete_forecast(
            forecast_id=forecast_id,
            user_id=current_user["id"]
        )
        if not result:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting forecast: {str(e)}", extra={
            "user_id": current_user["id"],
            "forecast_id": forecast_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))