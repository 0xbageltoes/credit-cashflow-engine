"""
Forecasting Tasks

This module serves as a bridge between the forecasting service implementation
and async task invocation. It provides task-friendly interfaces to the underlying 
forecasting functionality without duplicating implementation details.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

from app.core.config import settings
from app.services.forecasting import ForecastingService

# Configure logging
logger = logging.getLogger(__name__)

# Create a singleton instance of the forecasting service
_forecasting_service = ForecastingService()

async def generate_forecast(
    series_id: str,
    series_name: str,
    data_points: List[Dict[str, Any]],
    parameters: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """
    Task-friendly wrapper for the forecasting service
    
    Args:
        series_id: Unique identifier for the time series
        series_name: Display name for the time series
        data_points: Historical time series data points
        parameters: Forecasting parameters
        user_id: ID of the user requesting the forecast
        
    Returns:
        Forecast results including predictions and confidence intervals
    """
    logger.info(f"Starting forecast generation for series {series_id}")
    
    try:
        # Convert dict parameters to the proper object if needed
        from app.models.forecasting import ForecastParameters
        
        if isinstance(parameters, dict):
            # Convert dict to ForecastParameters
            from app.models.forecasting import ForecastMethod
            method = parameters.get("method", "auto")
            if isinstance(method, str):
                method = ForecastMethod(method)
                
            forecast_params = ForecastParameters(
                horizon=parameters.get("horizon", 12),
                method=method,
                confidence_level=parameters.get("confidence_level", 0.95),
                seasonality=parameters.get("seasonality"),
                transformation=parameters.get("transformation"),
            )
        else:
            forecast_params = parameters
            
        # Call the service method
        result = await _forecasting_service.forecast_time_series(
            series_id=series_id,
            series_name=series_name,
            data_points=data_points,
            parameters=forecast_params,
            user_id=user_id
        )
        
        logger.info(f"Forecast generation completed for series {series_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
        raise

async def run_stress_test(
    portfolio_id: str,
    scenario_id: str,
    stress_config: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """
    Run a stress test scenario on a portfolio
    
    Args:
        portfolio_id: ID of the portfolio to stress test
        scenario_id: ID of the scenario to apply
        stress_config: Stress test configuration parameters
        user_id: ID of the user requesting the stress test
        
    Returns:
        Results of the stress test including adjusted cash flows
    """
    logger.info(f"Starting stress test for portfolio {portfolio_id} with scenario {scenario_id}")
    
    try:
        # Set up a mock response for now
        # In a real implementation, this would call the appropriate service
        result = {
            "portfolio_id": portfolio_id,
            "scenario_id": scenario_id,
            "status": "completed",
            "message": "Stress test simulation (placeholder)",
            "timestamp": asyncio.get_event_loop().time(),
            "results": {
                "original_npv": 1000000.0,
                "stressed_npv": 950000.0,
                "impact_percentage": -5.0,
                "risk_metrics": {
                    "var_95": 50000.0,
                    "expected_shortfall": 75000.0
                }
            }
        }
        
        logger.info(f"Stress test completed for portfolio {portfolio_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error running stress test: {str(e)}", exc_info=True)
        raise
