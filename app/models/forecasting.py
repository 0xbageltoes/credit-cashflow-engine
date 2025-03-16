"""
Forecasting Models

This module defines data models for the forecasting service.
These models define the structure of requests and responses for
forecasting operations such as time series analysis and scenario modeling.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from enum import Enum

class ForecastMethod(str, Enum):
    """Methods available for forecasting"""
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    LSTM = "lstm"
    REGRESSION = "regression"
    ENSEMBLE = "ensemble"

class ForecastParameters(BaseModel):
    """Parameters for forecasting operations"""
    horizon: int = Field(..., description="Number of periods to forecast")
    method: ForecastMethod = Field(..., description="Forecasting method to use")
    confidence_level: float = Field(0.95, description="Confidence level for intervals (0-1)")
    seasonality: Optional[Dict[str, Any]] = Field(None, description="Seasonality parameters")
    exogenous_variables: Optional[List[Dict[str, Any]]] = Field(None, description="External variables to include")
    
    @validator('confidence_level')
    def check_confidence_level(cls, v):
        if not 0 < v < 1:
            raise ValueError('Confidence level must be between 0 and 1')
        return v

class TimeSeriesPoint(BaseModel):
    """Individual point in a time series"""
    timestamp: Union[datetime, date, str]
    value: float
    metadata: Optional[Dict[str, Any]] = None

class TimeSeriesData(BaseModel):
    """Time series data container"""
    series_id: str
    series_name: str
    data_points: List[TimeSeriesPoint]
    metadata: Optional[Dict[str, Any]] = None

class ScenarioDefinition(BaseModel):
    """Definition of a single forecasting scenario"""
    scenario_id: str
    scenario_name: str
    parameters: ForecastParameters
    adjustments: Optional[Dict[str, Any]] = None

class ScenarioRequest(BaseModel):
    """Request to run multiple forecasting scenarios"""
    base_data: TimeSeriesData
    scenarios: List[ScenarioDefinition]
    metadata: Optional[Dict[str, Any]] = None

class ForecastPoint(BaseModel):
    """Individual forecast point"""
    timestamp: Union[datetime, date, str]
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class ForecastingResult(BaseModel):
    """Result of a forecasting operation"""
    forecast_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    series_id: str
    series_name: str
    method: ForecastMethod
    parameters: Dict[str, Any]
    forecast_points: List[ForecastPoint]
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    """Request to generate a forecast"""
    series_data: TimeSeriesData
    parameters: ForecastParameters
    metadata: Optional[Dict[str, Any]] = None
