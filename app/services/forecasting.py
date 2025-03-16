"""
Forecasting Service

This module provides a forecasting service that uses simple time series techniques
and handles graceful fallbacks for production environments without requiring additional dependencies.
"""

import logging
import uuid
import json
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio

from app.core.config import settings
from app.core.logging import logger
from app.core.cache_service import CacheService
from app.models.forecasting import (
    ForecastRequest,
    ScenarioRequest,
    ForecastingResult,
    TimeSeriesData,
    ForecastParameters,
    ForecastMethod,
    TimeSeriesPoint,
    ForecastPoint,
    ScenarioDefinition
)

class ForecastingService:
    """
    Service for time series forecasting using simple, robust algorithms.
    
    This implementation avoids external dependencies to ensure it works
    in all environments and provides graceful degradation when advanced
    statistical libraries are not available.
    """
    
    def __init__(self):
        """Initialize the forecasting service with required dependencies"""
        self.cache_service = CacheService()
        self.logger = logging.getLogger(__name__)
        
        # Configure forecasting defaults
        self.default_confidence_level = 0.95
        self.forecast_cache_ttl = settings.FORECAST_CACHE_TTL if hasattr(settings, 'FORECAST_CACHE_TTL') else 3600
        self.max_forecast_horizon = settings.MAX_FORECAST_HORIZON if hasattr(settings, 'MAX_FORECAST_HORIZON') else 60
        
    async def forecast_time_series(
        self,
        series_id: str,
        series_name: str, 
        data_points: List[Dict[str, Any]],
        parameters: ForecastParameters,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate time series forecasts based on historical data
        
        Args:
            series_id: Unique identifier for the time series
            series_name: Display name for the time series
            data_points: Historical time series data points
            parameters: Forecasting parameters
            user_id: ID of the user requesting the forecast
            
        Returns:
            Forecast results including predictions and confidence intervals
        """
        # Generate a unique forecast ID
        forecast_id = f"forecast-{uuid.uuid4()}"
        
        try:
            # Extract timestamps and values from data points
            timestamps = []
            values = []
            
            for point in data_points:
                # Handle different timestamp formats
                if isinstance(point["timestamp"], str):
                    try:
                        timestamps.append(datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00')))
                    except ValueError:
                        # Try different format
                        timestamps.append(datetime.strptime(point["timestamp"], "%Y-%m-%d"))
                else:
                    timestamps.append(point["timestamp"])
                
                values.append(float(point["value"]))
            
            # Perform the forecast using basic methods
            forecast_values, confidence_intervals, metrics = self._simple_forecast(
                values, 
                parameters.horizon, 
                parameters.method.value, 
                parameters.confidence_level
            )
            
            # Generate dates for forecast points
            last_date = timestamps[-1]
            time_diff = (timestamps[-1] - timestamps[-2]) if len(timestamps) > 1 else timedelta(days=1)
            
            forecast_data = []
            for i, value in enumerate(forecast_values):
                forecast_date = last_date + time_diff * (i + 1)
                forecast_data.append({
                    "timestamp": forecast_date.isoformat(),
                    "value": value
                })
            
            # Format confidence intervals
            confidence_data = []
            if confidence_intervals:
                for i, (lower, upper) in enumerate(confidence_intervals):
                    forecast_date = last_date + time_diff * (i + 1)
                    confidence_data.append({
                        "timestamp": forecast_date.isoformat(),
                        "lower": lower,
                        "upper": upper
                    })
            
            # Prepare the response
            result = {
                "forecast_id": forecast_id,
                "created_at": datetime.utcnow(),
                "forecast_data": forecast_data,
                "confidence_intervals": confidence_data,
                "metadata": {
                    "series_id": series_id,
                    "series_name": series_name,
                    "user_id": user_id,
                    "method": parameters.method.value,
                    "horizon": parameters.horizon,
                    "confidence_level": parameters.confidence_level,
                    "metrics": metrics
                }
            }
            
            # Cache the result
            cache_key = f"forecast:{forecast_id}"
            await self.cache_service.set(
                cache_key,
                json.dumps(self._make_json_serializable(result)),
                ttl=self.forecast_cache_ttl
            )
            
            return result
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error generating forecast: {str(e)}",
                extra={
                    "series_id": series_id,
                    "user_id": user_id,
                    "method": parameters.method.value if hasattr(parameters, 'method') else None,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception for the API to handle
            raise
    
    async def run_scenario_analysis(
        self, 
        base_data: TimeSeriesData,
        scenarios: List[ScenarioDefinition],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Run multiple forecasting scenarios with different parameters
        
        Args:
            base_data: Base time series data for all scenarios
            scenarios: List of scenario definitions with parameters
            user_id: ID of the user requesting the analysis
            
        Returns:
            List of forecast results for each scenario
        """
        try:
            # Convert base data to the format needed by forecast_time_series
            data_points = [{"timestamp": point.timestamp, "value": point.value} for point in base_data.data_points]
            
            # Run forecasts for each scenario sequentially (for simplicity without dependencies)
            results = []
            for scenario in scenarios:
                result = await self.forecast_time_series(
                    series_id=f"{base_data.series_id}-{scenario.scenario_id}",
                    series_name=f"{base_data.series_name} - {scenario.scenario_name}",
                    data_points=data_points,
                    parameters=scenario.parameters,
                    user_id=user_id
                )
                
                # Add scenario information
                result["metadata"]["scenario_id"] = scenario.scenario_id
                result["metadata"]["scenario_name"] = scenario.scenario_name
                if scenario.adjustments:
                    result["metadata"]["adjustments"] = scenario.adjustments
                
                results.append(result)
            
            return results
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error running scenario analysis: {str(e)}",
                extra={
                    "series_id": base_data.series_id,
                    "user_id": user_id,
                    "scenario_count": len(scenarios),
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception for the API to handle
            raise
    
    async def get_historical_data(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        user_id: str = None
    ) -> TimeSeriesData:
        """
        Retrieve historical time series data
        
        Args:
            series_id: Unique identifier for the time series
            start_date: Optional start date filter
            end_date: Optional end date filter
            user_id: ID of the user requesting the data
            
        Returns:
            Historical time series data
        """
        try:
            # Attempt to retrieve from cache first
            cache_key = f"timeseries:{series_id}"
            cached_data = await self.cache_service.get(cache_key)
            
            if cached_data:
                # Parse the cached data
                data = json.loads(cached_data)
                
                # Apply date filters if provided
                if start_date or end_date:
                    filtered_points = []
                    for point in data["data_points"]:
                        point_date = datetime.fromisoformat(point["timestamp"]).date()
                        if (not start_date or point_date >= start_date) and \
                           (not end_date or point_date <= end_date):
                            filtered_points.append(point)
                    data["data_points"] = filtered_points
                
                return TimeSeriesData(**data)
            
            # Return empty result if no cached data
            return TimeSeriesData(
                series_id=series_id,
                series_name=f"Series {series_id}",
                data_points=[],
                metadata={"message": "Historical data not found"}
            )
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error retrieving historical data: {str(e)}",
                extra={
                    "series_id": series_id,
                    "user_id": user_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception for the API to handle
            raise
    
    async def get_forecast(
        self,
        forecast_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previously generated forecast
        
        Args:
            forecast_id: Unique identifier for the forecast
            user_id: ID of the user requesting the forecast
            
        Returns:
            Forecast data if found, None otherwise
        """
        try:
            # Retrieve from cache
            cache_key = f"forecast:{forecast_id}"
            cached_forecast = await self.cache_service.get(cache_key)
            
            if cached_forecast:
                return json.loads(cached_forecast)
            
            # Not found
            return None
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error retrieving forecast: {str(e)}",
                extra={
                    "forecast_id": forecast_id,
                    "user_id": user_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception for the API to handle
            raise
    
    async def delete_forecast(
        self,
        forecast_id: str,
        user_id: str
    ) -> bool:
        """
        Delete a previously generated forecast
        
        Args:
            forecast_id: Unique identifier for the forecast
            user_id: ID of the user requesting the deletion
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            # Delete from cache
            cache_key = f"forecast:{forecast_id}"
            exists = await self.cache_service.exists(cache_key)
            
            if exists:
                await self.cache_service.delete(cache_key)
                return True
            
            # Not found
            return False
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error deleting forecast: {str(e)}",
                extra={
                    "forecast_id": forecast_id,
                    "user_id": user_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception for the API to handle
            raise
    
    # ---------- Private helper methods ----------
    
    def _simple_forecast(
        self,
        values: List[float],
        horizon: int,
        method: str,
        confidence_level: float
    ) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """
        Perform a simple forecast using basic algorithms
        
        Args:
            values: Historical values to forecast from
            horizon: Number of periods to forecast
            method: Forecasting method to use
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of forecast values, confidence intervals, and metrics
        """
        # Cap the horizon to prevent excessive forecasting
        horizon = min(horizon, self.max_forecast_horizon)
        
        # Choose the forecast method based on the parameter or available data
        if method == "arima" and len(values) > 10:
            return self._simple_arima(values, horizon, confidence_level)
        elif method == "exponential_smoothing" and len(values) > 3:
            return self._simple_exponential_smoothing(values, horizon, confidence_level)
        elif method == "regression" and len(values) > 2:
            return self._simple_regression(values, horizon, confidence_level)
        else:
            # Default to simple moving average for limited data
            return self._simple_moving_average(values, horizon, confidence_level)
    
    def _simple_moving_average(
        self,
        values: List[float],
        horizon: int,
        confidence_level: float
    ) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """
        Simple moving average forecast
        
        Args:
            values: Historical values
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of forecast values, confidence intervals, and metrics
        """
        # Use the average of the last few values as the forecast
        window_size = min(10, len(values))
        recent_values = values[-window_size:]
        forecast_value = sum(recent_values) / len(recent_values)
        
        # Create a flat forecast
        forecast = [forecast_value] * horizon
        
        # Calculate standard deviation for confidence intervals
        std_dev = 0
        if len(values) > 1:
            mean = sum(values) / len(values)
            squared_diffs = [(v - mean) ** 2 for v in values]
            std_dev = (sum(squared_diffs) / len(values)) ** 0.5
        
        # Calculate z-score for confidence intervals
        z = 1.96  # Approximate z-score for 95% confidence
        if confidence_level != 0.95:
            # Approximate z-scores for other common confidence levels
            if confidence_level > 0.99:
                z = 2.58
            elif confidence_level > 0.98:
                z = 2.33
            elif confidence_level > 0.95:
                z = 2.17
            elif confidence_level > 0.90:
                z = 1.65
            elif confidence_level > 0.85:
                z = 1.44
            elif confidence_level > 0.80:
                z = 1.28
        
        # Create confidence intervals
        margin = z * std_dev
        intervals = [(max(0, forecast_value - margin), forecast_value + margin) for _ in range(horizon)]
        
        # Calculate basic metrics
        metrics = {
            "method": "simple_moving_average",
            "window_size": window_size,
            "std_dev": std_dev
        }
        
        return forecast, intervals, metrics
    
    def _simple_exponential_smoothing(
        self,
        values: List[float],
        horizon: int,
        confidence_level: float
    ) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """
        Simple exponential smoothing forecast
        
        Args:
            values: Historical values
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of forecast values, confidence intervals, and metrics
        """
        # Set the smoothing factor
        alpha = 0.3
        
        # Initialize with the first value
        smoothed = values[0]
        
        # Apply exponential smoothing
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Create a flat forecast
        forecast = [smoothed] * horizon
        
        # Calculate RMSE for the model
        sse = sum([(values[i] - (alpha * values[i-1] + (1-alpha) * values[i-2])) ** 2 
                  for i in range(2, len(values))])
        rmse = (sse / (len(values) - 2)) ** 0.5 if len(values) > 2 else 0
        
        # Calculate z-score for confidence intervals
        z = 1.96  # Approximate z-score for 95% confidence
        if confidence_level != 0.95:
            # Approximate z-scores for other common confidence levels
            if confidence_level > 0.99:
                z = 2.58
            elif confidence_level > 0.98:
                z = 2.33
            elif confidence_level > 0.95:
                z = 2.17
            elif confidence_level > 0.90:
                z = 1.65
            elif confidence_level > 0.85:
                z = 1.44
            elif confidence_level > 0.80:
                z = 1.28
        
        # Create confidence intervals that increase with horizon
        intervals = []
        for i in range(horizon):
            # This is a simplification of ARIMA confidence intervals
            # The variance grows with the horizon for AR processes
            variance = rmse * math.sqrt(i + 1)
            margin = z * math.sqrt(variance)
            intervals.append((max(0, smoothed - margin), smoothed + margin))
        
        # Calculate metrics
        metrics = {
            "method": "simple_exponential_smoothing",
            "alpha": alpha,
            "rmse": rmse
        }
        
        return forecast, intervals, metrics
    
    def _simple_regression(
        self,
        values: List[float],
        horizon: int,
        confidence_level: float
    ) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """
        Simple linear regression forecast
        
        Args:
            values: Historical values
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of forecast values, confidence intervals, and metrics
        """
        # Calculate trend with simple linear regression
        n = len(values)
        
        if n < 2:
            # Not enough data, fall back to moving average
            return self._simple_moving_average(values, horizon, confidence_level)
        
        # Calculate means
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        # Calculate slope and intercept
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        # Avoid division by zero
        slope = numerator / denominator if denominator != 0 else 0
        intercept = mean_y - slope * mean_x
        
        # Create forecast
        forecast = [intercept + slope * (n + i) for i in range(horizon)]
        
        # Calculate standard error of regression
        y_pred = [intercept + slope * i for i in range(n)]
        sse = sum([(values[i] - y_pred[i]) ** 2 for i in range(n)])
        standard_error = (sse / (n - 2)) ** 0.5 if n > 2 else 0
        
        # Calculate z-score for confidence intervals
        z = 1.96  # Approximate z-score for 95% confidence
        if confidence_level != 0.95:
            # Approximate z-scores for other common confidence levels
            if confidence_level > 0.99:
                z = 2.58
            elif confidence_level > 0.98:
                z = 2.33
            elif confidence_level > 0.95:
                z = 2.17
            elif confidence_level > 0.90:
                z = 1.65
            elif confidence_level > 0.85:
                z = 1.44
            elif confidence_level > 0.80:
                z = 1.28
        
        # Create confidence intervals that increase with horizon
        intervals = []
        for i in range(horizon):
            # Margin of error increases with forecast horizon
            margin = z * standard_error * math.sqrt(1 + 1/n + 
                    ((n + i - mean_x) ** 2) / denominator)
            
            forecast_value = forecast[i]
            intervals.append((max(0, forecast_value - margin), forecast_value + margin))
        
        # Calculate metrics
        if n > 2:
            r_squared = 1 - (sse / sum([(y - mean_y) ** 2 for y in values]))
        else:
            r_squared = 0
        
        metrics = {
            "method": "simple_regression",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "standard_error": standard_error
        }
        
        return forecast, intervals, metrics
    
    def _simple_arima(
        self,
        values: List[float],
        horizon: int,
        confidence_level: float
    ) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """
        Simplified ARIMA-like model using autoregression
        
        Args:
            values: Historical values
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of forecast values, confidence intervals, and metrics
        """
        # Use a simple AR(1) model: y_t = c + φ * y_{t-1} + ε_t
        n = len(values)
        
        if n < 2:
            # Not enough data, fall back to moving average
            return self._simple_moving_average(values, horizon, confidence_level)
        
        # Calculate AR(1) coefficient
        y_t = values[1:]
        y_t_1 = values[:-1]
        
        mean_y_t = sum(y_t) / len(y_t)
        mean_y_t_1 = sum(y_t_1) / len(y_t_1)
        
        numerator = sum((y_t[i] - mean_y_t) * (y_t_1[i] - mean_y_t_1) for i in range(len(y_t)))
        denominator = sum((y_t_1[i] - mean_y_t_1) ** 2 for i in range(len(y_t_1)))
        
        phi = numerator / denominator if denominator != 0 else 0
        c = mean_y_t - phi * mean_y_t_1
        
        # Ensure stability
        phi = max(min(phi, 0.99), -0.99)
        
        # Calculate residuals
        y_pred = [c + phi * y_t_1[i] for i in range(len(y_t_1))]
        residuals = [y_t[i] - y_pred[i] for i in range(len(y_t))]
        residual_variance = sum([r ** 2 for r in residuals]) / len(residuals)
        
        # Generate forecast
        forecast = []
        last_value = values[-1]
        
        for _ in range(horizon):
            next_value = c + phi * last_value
            forecast.append(next_value)
            last_value = next_value
        
        # Calculate confidence intervals
        z = 1.96  # Approximate z-score for 95% confidence
        if confidence_level != 0.95:
            # Approximate z-scores for other common confidence levels
            if confidence_level > 0.99:
                z = 2.58
            elif confidence_level > 0.98:
                z = 2.33
            elif confidence_level > 0.95:
                z = 2.17
            elif confidence_level > 0.90:
                z = 1.65
            elif confidence_level > 0.85:
                z = 1.44
            elif confidence_level > 0.80:
                z = 1.28
        
        # ARIMA confidence intervals increase with horizon
        intervals = []
        for i in range(horizon):
            # This is a simplification of ARIMA confidence intervals
            # The variance grows with the horizon for AR processes
            variance = residual_variance * sum([phi ** (2 * j) for j in range(i + 1)])
            margin = z * math.sqrt(variance)
            intervals.append((max(0, forecast[i] - margin), forecast[i] + margin))
        
        # Calculate AIC
        aic = n * math.log(residual_variance) + 2 * 2  # 2 parameters (c, phi)
        
        metrics = {
            "method": "simple_arima",
            "ar_coefficient": phi,
            "constant": c,
            "residual_variance": residual_variance,
            "aic": aic
        }
        
        return forecast, intervals, metrics
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert an object to be JSON serializable
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (ForecastMethod)):
            return obj.value
        else:
            return obj
