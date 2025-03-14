"""
Analytics Service for credit cashflow engine.

This module provides a comprehensive analytics service that:
1. Calculates financial metrics for structured finance deals
2. Handles caching of analytical results
3. Integrates economic factors into calculations
4. Supports both synchronous and asynchronous interfaces
5. Runs sensitivity analysis and Monte Carlo simulations
"""

import json
import logging
import time
import traceback
import hashlib
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date

import numpy as np
from pydantic import BaseModel
from fastapi import Depends

from app.core.config import settings
from app.core.metrics import (
    ANALYTICS_CACHE_HITS,
    ANALYTICS_CACHE_MISSES,
    ANALYTICS_CALCULATION_TIME,
    METRICS_ENABLED
)
from app.services.redis_service import RedisService
from app.services.unified_absbox_service import UnifiedAbsBoxService
from app.models.analytics import (
    AnalyticsResult,
    AnalyticsRequest,
    CashflowProjection,
    EconomicFactors,
    RiskMetrics,
    SensitivityAnalysisResult,
    StatisticalOutputs
)

logger = logging.getLogger(__name__)

class AnalyticsService:
    """
    Service for advanced financial analytics and metrics calculations.
    
    This service leverages the UnifiedAbsBoxService for core calculations
    and adds additional analytical capabilities with efficient caching.
    """
    
    def __init__(
        self, 
        absbox_service: UnifiedAbsBoxService,
        redis: RedisService,
        cache_ttl: int = settings.CACHE_TTL
    ):
        """
        Initialize the AnalyticsService.
        
        Args:
            absbox_service: Client for AbsBox calculations
            redis: Redis service for caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.absbox = absbox_service
        self.redis = redis
        self.cache_ttl = cache_ttl
        self.use_cache = settings.USE_REDIS_CACHE
    
    @staticmethod
    def _generate_cache_key(*args, **kwargs) -> str:
        """
        Generate a deterministic cache key from inputs.
        
        Args:
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            A deterministic cache key string
        """
        # Convert all inputs to JSON-serializable format
        def serialize(obj):
            if hasattr(obj, 'dict'):
                return obj.dict()
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create serialized representation
        serialized_args = json.dumps([serialize(arg) for arg in args], sort_keys=True)
        serialized_kwargs = json.dumps({k: serialize(v) for k, v in kwargs.items()}, sort_keys=True)
        
        # Create hash
        combined = serialized_args + serialized_kwargs
        hash_obj = hashlib.md5(combined.encode())
        
        return f"analytics:{hash_obj.hexdigest()}"
    
    def calculate_metrics(
        self,
        cashflows: Union[List[Dict], CashflowProjection],
        discount_rate: float,
        economic_factors: Optional[Union[Dict, EconomicFactors]] = None,
        cache: bool = True
    ) -> AnalyticsResult:
        """
        Calculate comprehensive analytics for cashflows with economic factors.
        
        Args:
            cashflows: The projected cashflows to analyze
            discount_rate: The base discount rate for calculations
            economic_factors: Optional economic factor adjustments
            cache: Whether to use caching
            
        Returns:
            Comprehensive analytics results
        """
        start_time = time.time()
        
        try:
            # Generate cache key from inputs
            use_cache = cache and self.use_cache
            cache_key = None
            
            if use_cache:
                cache_key = self._generate_cache_key(
                    cashflows, discount_rate, economic_factors
                )
                
                # Try to get from cache
                try:
                    cached_result = self.redis.get_sync(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for analytics calculation")
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_HITS.inc()
                        return AnalyticsResult.model_validate(cached_result)
                    else:
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_MISSES.inc()
                except Exception as e:
                    logger.warning(f"Error checking cache: {e}")
            
            # Parse the cashflows if provided as a list of dicts
            cf_data = cashflows
            if isinstance(cashflows, CashflowProjection):
                cf_data = cashflows.model_dump()
            
            # Parse economic factors if needed
            econ_factors = None
            if economic_factors:
                econ_factors = economic_factors
                if isinstance(economic_factors, EconomicFactors):
                    econ_factors = economic_factors.model_dump()
            
            # Apply economic factors to cashflows if provided
            adjusted_cashflows = self._apply_economic_factors(cf_data, econ_factors)
            
            # Convert to numpy arrays for efficient calculation
            dates = np.array([datetime.fromisoformat(cf["date"]) for cf in adjusted_cashflows])
            amounts = np.array([cf["amount"] for cf in adjusted_cashflows])
            
            # Calculate basic metrics
            npv = self._calculate_npv(dates, amounts, discount_rate)
            irr = self._calculate_irr(dates, amounts)
            duration = self._calculate_duration(dates, amounts, discount_rate)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(dates, amounts, discount_rate)
            
            # Create the result
            result = AnalyticsResult(
                npv=npv,
                irr=irr,
                duration=duration,
                convexity=risk_metrics.convexity,
                yield_to_maturity=risk_metrics.yield_to_maturity,
                average_life=risk_metrics.average_life,
                volatility=risk_metrics.volatility,
                calculation_time=time.time() - start_time
            )
            
            # Cache the result
            if use_cache and cache_key:
                try:
                    self.redis.set_sync(cache_key, result.model_dump(), self.cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {e}")
            
            # Update metrics
            if METRICS_ENABLED:
                ANALYTICS_CALCULATION_TIME.observe(time.time() - start_time)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Return error result
            return AnalyticsResult(
                error=str(e),
                calculation_time=time.time() - start_time
            )
    
    async def calculate_metrics_async(
        self,
        cashflows: Union[List[Dict], CashflowProjection],
        discount_rate: float,
        economic_factors: Optional[Union[Dict, EconomicFactors]] = None,
        cache: bool = True
    ) -> AnalyticsResult:
        """
        Calculate comprehensive analytics for cashflows asynchronously.
        
        Args:
            cashflows: The projected cashflows to analyze
            discount_rate: The base discount rate for calculations
            economic_factors: Optional economic factor adjustments
            cache: Whether to use caching
            
        Returns:
            Comprehensive analytics results
        """
        start_time = time.time()
        
        try:
            # Generate cache key from inputs
            use_cache = cache and self.use_cache
            cache_key = None
            
            if use_cache:
                cache_key = self._generate_cache_key(
                    cashflows, discount_rate, economic_factors
                )
                
                # Try to get from cache
                try:
                    cached_result = await self.redis.get(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for analytics calculation")
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_HITS.inc()
                        return AnalyticsResult.model_validate(cached_result)
                    else:
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_MISSES.inc()
                except Exception as e:
                    logger.warning(f"Error checking cache: {e}")
            
            # Run calculation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.calculate_metrics(
                    cashflows, discount_rate, economic_factors, False
                )
            )
            
            # Cache the result
            if use_cache and cache_key:
                try:
                    await self.redis.set(cache_key, result.model_dump(), self.cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error calculating metrics asynchronously: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Return error result
            return AnalyticsResult(
                error=str(e),
                calculation_time=time.time() - start_time
            )
    
    def _apply_economic_factors(
        self,
        cashflows: List[Dict],
        economic_factors: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Apply economic factors to cashflow projections.
        
        Args:
            cashflows: The original cashflow projections
            economic_factors: Economic factors to apply
            
        Returns:
            Adjusted cashflow projections
        """
        if not economic_factors:
            return cashflows
        
        # Create deep copy to avoid modifying original
        adjusted_cashflows = [cf.copy() for cf in cashflows]
        
        # Extract economic factors
        inflation_rate = economic_factors.get("inflation_rate", 0.0)
        gdp_growth = economic_factors.get("gdp_growth", 0.0)
        unemployment_rate = economic_factors.get("unemployment_rate", 0.0)
        housing_price_index = economic_factors.get("housing_price_index", 0.0)
        
        # Create adjustment factor based on correlation model
        # This is a simplified model - in production you'd use a more sophisticated 
        # econometric model with proper correlations
        for i, cf in enumerate(adjusted_cashflows):
            # Calculate time index (0 for first cashflow, increasing for later ones)
            time_index = i / len(cashflows)
            
            # Calculate adjustment factor
            # Here we use a simplified model that considers multiple economic factors
            # with different weights based on common financial correlations
            inflation_effect = -0.3 * inflation_rate * time_index  # Inflation decreases future real value
            gdp_effect = 0.2 * gdp_growth * time_index  # GDP growth increases cashflows
            unemployment_effect = -0.15 * unemployment_rate * time_index  # Unemployment decreases payment ability
            housing_effect = 0.25 * housing_price_index * time_index  # Housing price increases improve recovery values
            
            # Combined adjustment factor
            adjustment = 1 + inflation_effect + gdp_effect + unemployment_effect + housing_effect
            
            # Ensure adjustment is reasonable (don't allow extreme adjustments)
            adjustment = max(0.5, min(1.5, adjustment))
            
            # Apply adjustment to the cashflow amount
            adjusted_cashflows[i]["amount"] = cf["amount"] * adjustment
            
            # Store the adjustment factor for reference
            adjusted_cashflows[i]["adjustment_factor"] = adjustment
        
        return adjusted_cashflows
    
    def _calculate_npv(
        self, 
        dates: np.ndarray,
        amounts: np.ndarray,
        discount_rate: float
    ) -> float:
        """
        Calculate Net Present Value using vectorized operations.
        
        Args:
            dates: Array of cashflow dates
            amounts: Array of cashflow amounts
            discount_rate: Annual discount rate
            
        Returns:
            NPV value
        """
        # Calculate time periods in years from the first date
        base_date = dates[0]
        years = np.array([(date - base_date).days / 365.0 for date in dates])
        
        # Calculate discount factors
        discount_factors = np.power(1 + discount_rate, -years)
        
        # Calculate NPV
        npv = np.sum(amounts * discount_factors)
        
        return float(npv)
    
    def _calculate_irr(
        self, 
        dates: np.ndarray,
        amounts: np.ndarray
    ) -> float:
        """
        Calculate Internal Rate of Return.
        
        Args:
            dates: Array of cashflow dates
            amounts: Array of cashflow amounts
            
        Returns:
            IRR value
        """
        try:
            # Calculate time periods in years from the first date
            base_date = dates[0]
            years = np.array([(date - base_date).days / 365.0 for date in dates])
            
            # Initial guess for IRR
            guess = 0.05
            
            # IRR function to solve
            def npv_function(rate):
                discount_factors = np.power(1 + rate, -years)
                return np.sum(amounts * discount_factors)
            
            # Solve for IRR using numerical methods
            from scipy import optimize
            irr = optimize.newton(npv_function, guess)
            
            return float(irr)
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return 0.0
    
    def _calculate_duration(
        self, 
        dates: np.ndarray,
        amounts: np.ndarray,
        discount_rate: float
    ) -> float:
        """
        Calculate Macaulay Duration.
        
        Args:
            dates: Array of cashflow dates
            amounts: Array of cashflow amounts
            discount_rate: Annual discount rate
            
        Returns:
            Duration value in years
        """
        # Calculate time periods in years from the first date
        base_date = dates[0]
        years = np.array([(date - base_date).days / 365.0 for date in dates])
        
        # Calculate discount factors
        discount_factors = np.power(1 + discount_rate, -years)
        
        # Calculate present values of each cashflow
        present_values = amounts * discount_factors
        
        # Calculate NPV
        npv = np.sum(present_values)
        
        # Calculate duration (weighted average time)
        if npv == 0:
            return 0.0
        
        duration = np.sum(present_values * years) / npv
        
        return float(duration)
    
    def _calculate_risk_metrics(
        self, 
        dates: np.ndarray,
        amounts: np.ndarray,
        discount_rate: float
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            dates: Array of cashflow dates
            amounts: Array of cashflow amounts
            discount_rate: Annual discount rate
            
        Returns:
            Risk metrics including convexity, YTM, average life, and volatility
        """
        # Calculate time periods in years from the first date
        base_date = dates[0]
        years = np.array([(date - base_date).days / 365.0 for date in dates])
        
        # Calculate discount factors
        discount_factors = np.power(1 + discount_rate, -years)
        
        # Calculate present values of each cashflow
        present_values = amounts * discount_factors
        
        # Calculate NPV
        npv = np.sum(present_values)
        
        # Calculate duration (weighted average time)
        duration = 0.0
        if npv > 0:
            duration = np.sum(present_values * years) / npv
        
        # Calculate convexity (second derivative of price with respect to yield)
        convexity = 0.0
        if npv > 0:
            convexity = np.sum(present_values * years * (years + 1)) / (npv * (1 + discount_rate)**2)
        
        # Calculate average life (weighted average time using nominal cashflows)
        total_amount = np.sum(amounts)
        average_life = 0.0
        if total_amount > 0:
            average_life = np.sum(amounts * years) / total_amount
        
        # Approximate yield to maturity (YTM) as discount rate that gives NPV = 0
        # For simplicity, we'll use the provided discount rate as YTM estimate
        yield_to_maturity = discount_rate
        
        # Calculate volatility as standard deviation of adjusted returns
        # This is a simplified approach - in production you'd use more sophisticated models
        cashflow_returns = amounts / (np.mean(amounts) if np.mean(amounts) > 0 else 1)
        volatility = float(np.std(cashflow_returns))
        
        return RiskMetrics(
            convexity=convexity,
            yield_to_maturity=yield_to_maturity,
            average_life=average_life,
            volatility=volatility
        )
    
    def run_sensitivity_analysis(
        self,
        cashflows: Union[List[Dict], CashflowProjection],
        discount_rate: float,
        parameter_ranges: Dict[str, List[float]],
        cache: bool = True
    ) -> SensitivityAnalysisResult:
        """
        Run sensitivity analysis on key parameters.
        
        Args:
            cashflows: The projected cashflows to analyze
            discount_rate: The base discount rate for calculations
            parameter_ranges: Ranges of parameters to test
            cache: Whether to use caching
            
        Returns:
            Sensitivity analysis results
        """
        start_time = time.time()
        
        try:
            # Generate cache key from inputs
            use_cache = cache and self.use_cache
            cache_key = None
            
            if use_cache:
                cache_key = self._generate_cache_key(
                    cashflows, discount_rate, parameter_ranges
                )
                
                # Try to get from cache
                try:
                    cached_result = self.redis.get_sync(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for sensitivity analysis")
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_HITS.inc()
                        return SensitivityAnalysisResult.model_validate(cached_result)
                    else:
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_MISSES.inc()
                except Exception as e:
                    logger.warning(f"Error checking cache: {e}")
            
            # Parse the cashflows if provided as a list of dicts
            cf_data = cashflows
            if isinstance(cashflows, CashflowProjection):
                cf_data = cashflows.model_dump()
            
            # Calculate base case metrics
            base_metrics = self.calculate_metrics(cf_data, discount_rate, cache=False)
            
            # Run sensitivity analysis for each parameter
            results = {}
            
            for param_name, param_values in parameter_ranges.items():
                param_results = {}
                
                for value in param_values:
                    # Create modified inputs based on parameter
                    if param_name == "discount_rate":
                        metrics = self.calculate_metrics(cf_data, value, cache=False)
                    elif param_name.startswith("economic."):
                        # Handle economic factor parameters
                        econ_param = param_name.split(".")[1]
                        econ_factors = {econ_param: value}
                        metrics = self.calculate_metrics(cf_data, discount_rate, econ_factors, cache=False)
                    else:
                        # Skip unknown parameters
                        continue
                    
                    # Calculate differences from base case
                    diffs = {
                        "npv": metrics.npv - base_metrics.npv,
                        "irr": metrics.irr - base_metrics.irr,
                        "duration": metrics.duration - base_metrics.duration
                    }
                    
                    # Store results
                    param_results[str(value)] = {
                        "metrics": metrics.model_dump(exclude={"calculation_time", "error"}),
                        "diff_from_base": diffs
                    }
                
                results[param_name] = param_results
            
            # Create result
            sensitivity_result = SensitivityAnalysisResult(
                base_case=base_metrics.model_dump(exclude={"calculation_time", "error"}),
                sensitivity_results=results,
                calculation_time=time.time() - start_time
            )
            
            # Cache the result
            if use_cache and cache_key:
                try:
                    self.redis.set_sync(cache_key, sensitivity_result.model_dump(), self.cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {e}")
            
            return sensitivity_result
        except Exception as e:
            logger.error(f"Error running sensitivity analysis: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            return SensitivityAnalysisResult(
                base_case={},
                sensitivity_results={},
                calculation_time=time.time() - start_time,
                error=f"Error running sensitivity analysis: {str(e)}"
            )

    def run_monte_carlo_simulation(
        self,
        cashflows: Union[List[Dict], CashflowProjection],
        discount_rate: float,
        parameter_ranges: Dict[str, List[float]],
        cache: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on key parameters.
        
        Args:
            cashflows: The projected cashflows to analyze
            discount_rate: The base discount rate for calculations
            parameter_ranges: Ranges of parameters to test
            cache: Whether to use caching
            
        Returns:
            Monte Carlo simulation results
        """
        start_time = time.time()
        
        try:
            # Generate cache key from inputs
            use_cache = cache and self.use_cache
            cache_key = None
            
            if use_cache:
                cache_key = self._generate_cache_key(
                    cashflows, discount_rate, parameter_ranges
                )
                
                # Try to get from cache
                try:
                    cached_result = self.redis.get_sync(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for Monte Carlo simulation")
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_HITS.inc()
                        return MonteCarloResult.model_validate(cached_result)
                    else:
                        if METRICS_ENABLED:
                            ANALYTICS_CACHE_MISSES.inc()
                except Exception as e:
                    logger.warning(f"Error checking cache: {e}")
            
            # Parse the cashflows if provided as a list of dicts
            cf_data = cashflows
            if isinstance(cashflows, CashflowProjection):
                cf_data = cashflows.model_dump()
            
            # Calculate base case metrics
            base_metrics = self.calculate_metrics(cf_data, discount_rate, cache=False)
            
            # Run Monte Carlo simulation for each parameter
            results = {}
            
            for param_name, param_values in parameter_ranges.items():
                param_results = {}
                
                for value in param_values:
                    # Create modified inputs based on parameter
                    if param_name == "discount_rate":
                        metrics = self.calculate_metrics(cf_data, value, cache=False)
                    elif param_name.startswith("economic."):
                        # Handle economic factor parameters
                        econ_param = param_name.split(".")[1]
                        econ_factors = {econ_param: value}
                        metrics = self.calculate_metrics(cf_data, discount_rate, econ_factors, cache=False)
                    else:
                        # Skip unknown parameters
                        continue
                    
                    # Calculate differences from base case
                    diffs = {
                        "npv": metrics.npv - base_metrics.npv,
                        "irr": metrics.irr - base_metrics.irr,
                        "duration": metrics.duration - base_metrics.duration
                    }
                    
                    # Store results
                    param_results[str(value)] = {
                        "metrics": metrics.model_dump(exclude={"calculation_time", "error"}),
                        "diff_from_base": diffs
                    }
                
                results[param_name] = param_results
            
            # Create result
            monte_carlo_result = MonteCarloResult(
                base_case=base_metrics.model_dump(exclude={"calculation_time", "error"}),
                simulation_results=results,
                calculation_time=time.time() - start_time
            )
            
            # Cache the result
            if use_cache and cache_key:
                try:
                    self.redis.set_sync(cache_key, monte_carlo_result.model_dump(), self.cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {e}")
            
            return monte_carlo_result
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            return MonteCarloResult(
                base_case={},
                simulation_results={},
                calculation_time=time.time() - start_time,
                error=f"Error running Monte Carlo simulation: {str(e)}"
            )
