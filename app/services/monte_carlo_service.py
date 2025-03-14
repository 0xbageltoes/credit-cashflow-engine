"""
Monte Carlo Simulation Service

This module provides a comprehensive service for running Monte Carlo simulations
for financial analytics. It includes functionality for generating random variables
with specified distributions and correlations, running simulations, and calculating
statistics on the results.

The service is designed for production use with:
- Proper error handling and logging
- Redis caching integration
- Integration with Celery for background processing
- Full integration with asset class handlers
"""
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import uuid
import traceback
import scipy.stats as stats
import hashlib
import asyncio
import functools

from app.models.monte_carlo import (
    MonteCarloSimulationRequest, 
    MonteCarloSimulationResult,
    SimulationStatus,
    SimulationResult,
    DistributionType,
    ScenarioDefinition,
    SavedSimulation,
    CorrelationMatrix
)
from app.models.analytics import (
    StatisticalOutputs,
    EconomicFactors,
    MonteCarloResult,
    RiskMetrics
)
from app.core.config import settings
from app.services.redis_service import RedisService
from app.core.monitoring import CalculationTracker
from app.core.exceptions import ValidationError, CalculationError, handle_exceptions

# Import asset handlers to use in simulations
from app.services.asset_handlers import (
    ConsumerCreditHandler,
    CommercialLoanHandler,
    CLOCDOHandler
)

logger = logging.getLogger(__name__)

class MonteCarloSimulationService:
    """
    Service for running Monte Carlo simulations for financial analytics
    
    This service provides methods for generating random variables, running simulations,
    and calculating statistics on the results. It integrates with Redis for caching
    and Celery for background processing.
    """
    def __init__(self, redis_service: Optional[RedisService] = None):
        """
        Initialize the Monte Carlo simulation service
        
        Args:
            redis_service: Optional Redis service for caching
        """
        self.redis_service = redis_service or RedisService()
        self._rng = np.random.default_rng()  # High-quality random number generator
        
        # Initialize asset handlers
        self.consumer_credit_handler = ConsumerCreditHandler()
        self.commercial_loan_handler = CommercialLoanHandler()
        self.clo_cdo_handler = CLOCDOHandler()

    @handle_exceptions
    def _generate_correlated_variables(self, request: MonteCarloSimulationRequest) -> Dict[str, np.ndarray]:
        """
        Generate correlated random variables for Monte Carlo simulation using Cholesky decomposition
        
        Args:
            request: The simulation request containing variables and correlation matrix
            
        Returns:
            Dictionary mapping variable names to arrays of random values
            
        Raises:
            ValueError: If correlation matrix is not positive definite or other validation issues
        """
        # Extract variables and number of simulations
        variables = request.variables
        num_simulations = request.num_simulations
        
        # Check if correlation matrix is provided
        correlation_matrix = getattr(request, 'correlation_matrix', None)
        
        # Create result dictionary
        result = {}
        
        # Generate uncorrelated random variables first
        uncorrelated_variables = {}
        for var in variables:
            # Generate random variables based on distribution type
            if var.distribution_type == DistributionType.NORMAL:
                uncorrelated_variables[var.name] = self._rng.normal(
                    loc=var.mean, 
                    scale=var.std_dev, 
                    size=num_simulations
                )
            elif var.distribution_type == DistributionType.UNIFORM:
                uncorrelated_variables[var.name] = self._rng.uniform(
                    low=var.min_value, 
                    high=var.max_value, 
                    size=num_simulations
                )
            elif var.distribution_type == DistributionType.LOGNORMAL:
                uncorrelated_variables[var.name] = self._rng.lognormal(
                    mean=var.mean, 
                    sigma=var.std_dev, 
                    size=num_simulations
                )
            elif var.distribution_type == DistributionType.BETA:
                if not hasattr(var, 'alpha') or not hasattr(var, 'beta'):
                    raise ValueError(f"Beta distribution for {var.name} requires alpha and beta parameters")
                uncorrelated_variables[var.name] = self._rng.beta(
                    a=var.alpha,
                    b=var.beta,
                    size=num_simulations
                )
            elif var.distribution_type == DistributionType.GAMMA:
                if not hasattr(var, 'shape') or not hasattr(var, 'scale'):
                    raise ValueError(f"Gamma distribution for {var.name} requires shape and scale parameters")
                uncorrelated_variables[var.name] = self._rng.gamma(
                    shape=var.shape,
                    scale=var.scale,
                    size=num_simulations
                )
            else:
                raise ValueError(f"Unsupported distribution type: {var.distribution_type}")
        
        # If no correlation matrix, return uncorrelated variables
        if not correlation_matrix:
            return uncorrelated_variables
        
        # If correlation matrix is provided, apply Cholesky decomposition
        try:
            # Extract variable names in the order they appear in the correlation matrix
            var_names = [var.name for var in variables]
            num_vars = len(var_names)
            
            # Build numpy correlation matrix
            corr_matrix = np.eye(num_vars)  # Start with identity matrix
            
            # Fill in correlation matrix
            if isinstance(correlation_matrix, dict):
                # Handle correlation matrix as dictionary mapping (var1, var2) -> correlation
                for i, var1 in enumerate(var_names):
                    for j, var2 in enumerate(var_names):
                        if i != j:  # Skip diagonal (self-correlation is always 1)
                            # Try both orderings of variable names
                            key1 = f"{var1},{var2}"
                            key2 = f"{var2},{var1}"
                            
                            if key1 in correlation_matrix:
                                corr_matrix[i, j] = correlation_matrix[key1]
                            elif key2 in correlation_matrix:
                                corr_matrix[i, j] = correlation_matrix[key2]
                            # If no correlation specified, default to 0 (already set by np.eye)
            elif isinstance(correlation_matrix, CorrelationMatrix):
                # Handle correlation matrix from Pydantic model
                for corr in correlation_matrix.correlations:
                    try:
                        i = var_names.index(corr.variable1)
                        j = var_names.index(corr.variable2)
                        corr_matrix[i, j] = corr.correlation
                        corr_matrix[j, i] = corr.correlation  # Ensure symmetry
                    except ValueError:
                        logger.warning(f"Correlation specified for unknown variable: {corr.variable1} or {corr.variable2}")
            elif isinstance(correlation_matrix, np.ndarray):
                # Direct numpy array
                if correlation_matrix.shape == (num_vars, num_vars):
                    corr_matrix = correlation_matrix
                else:
                    raise ValueError(f"Correlation matrix has wrong shape: {correlation_matrix.shape}, expected {(num_vars, num_vars)}")
            else:
                # Try generic iterable conversion
                try:
                    if len(correlation_matrix) == num_vars and all(len(row) == num_vars for row in correlation_matrix):
                        corr_matrix = np.array(correlation_matrix)
                    else:
                        raise ValueError(f"Correlation matrix dimensions mismatch with variables")
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid correlation matrix format: {e}")
            
            # Ensure the matrix is symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            
            # Check if the matrix is positive definite
            try:
                # Try to compute Cholesky decomposition
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                # Not positive definite, try to fix using nearest positive definite matrix
                logger.warning("Correlation matrix is not positive definite. Attempting to find nearest positive definite matrix.")
                
                # Compute eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(corr_matrix)
                
                # Fix negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-6)
                
                # Reconstruct the correlation matrix
                corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Ensure diagonal is exactly 1
                np.fill_diagonal(corr_matrix, 1.0)
                
                # Try Cholesky again
                try:
                    L = np.linalg.cholesky(corr_matrix)
                except np.linalg.LinAlgError as e:
                    raise ValueError(f"Failed to find a valid correlation matrix: {e}")
            
            # Create matrix of uncorrelated standard normal variates
            Z = np.column_stack([
                stats.norm.ppf(
                    stats.rankdata(uncorrelated_variables[var_name]) / (num_simulations + 1)
                ) 
                for var_name in var_names
            ])
            
            # Apply Cholesky decomposition to get correlated standard normal variables
            Y = Z @ L.T
            
            # Transform back to original distributions using inverse CDF
            for i, var_name in enumerate(var_names):
                # Get the percentiles from the correlated standard normal
                u = stats.norm.cdf(Y[:, i])
                
                # Get original uncorrelated values for this variable
                original_var = uncorrelated_variables[var_name]
                
                # Create new variable with same distribution but correlated
                var = variables[i]
                if var.distribution_type == DistributionType.NORMAL:
                    result[var_name] = stats.norm.ppf(u, loc=var.mean, scale=var.std_dev)
                elif var.distribution_type == DistributionType.UNIFORM:
                    result[var_name] = stats.uniform.ppf(u, loc=var.min_value, scale=var.max_value - var.min_value)
                elif var.distribution_type == DistributionType.LOGNORMAL:
                    result[var_name] = stats.lognorm.ppf(u, s=var.std_dev, scale=np.exp(var.mean))
                elif var.distribution_type == DistributionType.BETA:
                    result[var_name] = stats.beta.ppf(u, a=var.alpha, b=var.beta)
                elif var.distribution_type == DistributionType.GAMMA:
                    result[var_name] = stats.gamma.ppf(u, a=var.shape, scale=var.scale)
                else:
                    # Fallback for unsupported distributions: keep original
                    result[var_name] = np.quantile(original_var, u)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating correlated variables: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            raise CalculationError(f"Failed to generate correlated variables: {e}")

    @handle_exceptions
    def apply_economic_factors(
        self, 
        simulation_variables: Dict[str, float],
        economic_factors: Optional[EconomicFactors] = None
    ) -> Dict[str, float]:
        """
        Apply economic factors to simulation variables based on correlation model
        
        Args:
            simulation_variables: The base simulation variables
            economic_factors: Economic factors to apply
            
        Returns:
            Modified simulation variables
        """
        if not economic_factors:
            return simulation_variables
        
        # Create a copy to avoid modifying the original
        modified_variables = simulation_variables.copy()
        
        # Extract economic factors
        factors_dict = economic_factors.model_dump() if hasattr(economic_factors, 'model_dump') else economic_factors
        
        # Get main economic indicators
        unemployment_rate = factors_dict.get("unemployment_rate", 0.0)
        gdp_growth = factors_dict.get("gdp_growth", 0.0)
        housing_price_index = factors_dict.get("housing_price_index", 0.0)
        interest_rates = factors_dict.get("interest_rates", {})
        
        # Get interest rate environment and specific rates
        interest_env = factors_dict.get("interest_rate_environment", "neutral")
        fed_funds_rate = interest_rates.get("fed_funds", 0.0)
        term_10y = interest_rates.get("term_10y", 0.0)
        term_spread = term_10y - fed_funds_rate if term_10y and fed_funds_rate else 0.0
        
        # Credit industry factors
        credit_spread = factors_dict.get("credit_spread", 0.0)
        market_volatility = factors_dict.get("market_volatility", 0.0)
        liquidity_index = factors_dict.get("liquidity_index", 0.0)
        
        # --- Variable adjustments based on economic research ---
        
        # Default rate adjustments
        if "default_rate" in modified_variables:
            base_default = modified_variables["default_rate"]
            
            # Unemployment has strong positive correlation with defaults
            unemployment_effect = 0.3 * unemployment_rate
            
            # GDP growth has negative correlation with defaults
            gdp_effect = -0.25 * gdp_growth
            
            # Housing prices have negative correlation with defaults (especially for mortgages)
            housing_effect = -0.15 * housing_price_index
            
            # Credit spreads widening increases defaults (market stress indicator)
            credit_effect = 0.2 * credit_spread
            
            # Term spread (yield curve) - inverted yield curve (negative spread) predicts recessions
            term_effect = -0.15 * term_spread if term_spread < 0 else -0.05 * term_spread
            
            # Combined adjustment (weighted based on economic research)
            default_adjustment = 1.0 + unemployment_effect + gdp_effect + housing_effect + credit_effect + term_effect
            
            # Apply adjustment with reasonable bounds and smoothing
            modified_variables["default_rate"] = max(0.0001, min(0.99, base_default * default_adjustment))
        
        # Prepayment rate adjustments
        if "prepayment_rate" in modified_variables:
            base_prepayment = modified_variables["prepayment_rate"]
            
            # Interest rate environment effect - falling rates increase prepayments
            interest_effect = -0.4 if interest_env == "falling" else 0.3 if interest_env == "rising" else 0.0
            
            # Housing prices have positive correlation with prepayments (refinance opportunity)
            housing_effect = 0.25 * housing_price_index
            
            # Unemployment has negative correlation with prepayments (financial stress)
            unemployment_effect = -0.2 * unemployment_rate
            
            # Liquidity conditions affect refinancing ability
            liquidity_effect = 0.15 * liquidity_index
            
            # Combined adjustment
            prepayment_adjustment = 1.0 + interest_effect + housing_effect + unemployment_effect + liquidity_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["prepayment_rate"] = max(0.0001, min(0.99, base_prepayment * prepayment_adjustment))
        
        # Recovery rate adjustments
        if "recovery_rate" in modified_variables:
            base_recovery = modified_variables["recovery_rate"]
            
            # Housing price index strongly affects recovery (especially for secured loans)
            housing_effect = 0.3 * housing_price_index
            
            # GDP growth affects general ability to recover value
            gdp_effect = 0.15 * gdp_growth
            
            # Market liquidity affects ability to sell recovered assets
            liquidity_effect = 0.2 * liquidity_index
            
            # Market volatility negatively impacts recoveries
            volatility_effect = -0.1 * market_volatility
            
            # Combined adjustment
            recovery_adjustment = 1.0 + housing_effect + gdp_effect + liquidity_effect + volatility_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["recovery_rate"] = max(0.0001, min(0.99, base_recovery * recovery_adjustment))
            
        # Loss given default (LGD) adjustments (if separate from recovery rate)
        if "lgd" in modified_variables:
            # LGD is often inverse of recovery rate, but can have its own factors
            if "recovery_rate" in modified_variables:
                # Ensure consistency with recovery rate if both are present
                modified_variables["lgd"] = max(0.01, min(0.99, 1.0 - modified_variables["recovery_rate"]))
            else:
                # Apply similar factors as recovery but inverted
                base_lgd = modified_variables["lgd"]
                lgd_adjustment = 1.0 - (housing_effect + gdp_effect + liquidity_effect - volatility_effect)
                modified_variables["lgd"] = max(0.01, min(0.99, base_lgd * lgd_adjustment))
                
        # Credit spread adjustments
        if "credit_spread" in modified_variables:
            base_spread = modified_variables["credit_spread"]
            
            # Market volatility increases spreads
            volatility_effect = 0.3 * market_volatility
            
            # Unemployment increases spreads
            unemployment_effect = 0.15 * unemployment_rate
            
            # GDP growth decreases spreads
            gdp_effect = -0.2 * gdp_growth
            
            # Liquidity decreases spreads
            liquidity_effect = -0.25 * liquidity_index
            
            # Combined adjustment
            spread_adjustment = 1.0 + volatility_effect + unemployment_effect + gdp_effect + liquidity_effect
            
            # Apply adjustment
            modified_variables["credit_spread"] = max(0.0001, base_spread * spread_adjustment)
            
        # Loan growth rate adjustments
        if "loan_growth" in modified_variables:
            base_growth = modified_variables["loan_growth"]
            
            # GDP strongly correlates with loan growth
            gdp_effect = 0.4 * gdp_growth
            
            # Lower interest rates increase loan demand
            interest_effect = -0.2 * fed_funds_rate if fed_funds_rate is not None else 0.0
            
            # Higher unemployment decreases loan growth
            unemployment_effect = -0.3 * unemployment_rate
            
            # Combined adjustment
            growth_adjustment = 1.0 + gdp_effect + interest_effect + unemployment_effect
            
            # Apply adjustment
            modified_variables["loan_growth"] = base_growth * growth_adjustment
            
        return modified_variables

    async def run_simulation(
        self, 
        request: MonteCarloSimulationRequest, 
        user_id: str,
        use_cache: bool = True,
        progress_callback = None
    ) -> MonteCarloSimulationResult:
        """
        Run a Monte Carlo simulation based on the provided request
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            use_cache: Whether to use caching
            progress_callback: Optional callback function for progress updates
            
        Returns:
            The simulation result
        """
        # Generate a unique ID for this simulation
        simulation_id = str(uuid.uuid4())
        
        # Initialize the result object
        result = MonteCarloSimulationResult(
            id=simulation_id,
            name=request.name,
            description=request.description,
            status=SimulationStatus.PENDING,
            start_time=datetime.now(),
            num_simulations=request.num_simulations,
            num_completed=0,
            summary_statistics={},
            percentiles={},
            detailed_paths=[]
        )
        
        # Check cache if enabled
        cache_hit = False
        if use_cache:
            cache_key = self._generate_cache_key(request, user_id)
            try:
                cached_data = await self.redis_service.get(cache_key)
                if cached_data:
                    try:
                        logger.info(f"Found cached result for simulation {simulation_id}")
                        result_dict = json.loads(cached_data)
                        return MonteCarloSimulationResult(**result_dict)
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.warning(f"Error parsing cached result: {str(e)}. Proceeding with calculation.")
                        # Cache is corrupted, will be overwritten
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}. Proceeding with calculation.")
                # Continue with the calculation if cache check fails
        
        try:
            logger.info(f"Starting Monte Carlo simulation {simulation_id} with {request.num_simulations} iterations")
            
            # Update status to running
            result.status = SimulationStatus.RUNNING
            
            # Generate correlated random variables for all simulations
            start_time = time.time()
            with CalculationTracker(f"monte_carlo_generate_variables_{simulation_id}"):
                variables_matrix = self._generate_correlated_variables(request)
            
            # Initialize result arrays
            all_metrics = {}
            all_cashflows = []
            
            # Run the simulations
            with CalculationTracker(f"monte_carlo_run_simulations_{simulation_id}"):
                for i in range(request.num_simulations):
                    # Extract the variables for this simulation
                    simulation_variables = {
                        var.name: variables_matrix[var.name][i] 
                        for var in request.variables
                    }
                    
                    # Apply economic factors if provided
                    if request.economic_factors:
                        simulation_variables = self.apply_economic_factors(simulation_variables, request.economic_factors)
                    
                    # Run the single simulation
                    sim_result = self._run_single_simulation(
                        request, 
                        simulation_variables, 
                        i
                    )
                    
                    # Collect metrics
                    for metric, value in sim_result.metrics.items():
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)
                    
                    # Collect cashflows if detailed paths are requested
                    if request.include_detailed_paths and sim_result.cashflows:
                        all_cashflows.append(sim_result)
                    
                    # Update progress
                    result.num_completed = i + 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, request.num_simulations)
            
            # Calculate summary statistics
            with CalculationTracker(f"monte_carlo_calculate_statistics_{simulation_id}"):
                for metric, values in all_metrics.items():
                    # Calculate basic statistics
                    result.summary_statistics[metric] = {
                        "mean": float(np.mean(values)),
                        "std_dev": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }
                    
                    # Calculate percentiles
                    result.percentiles[metric] = {
                        p: float(np.percentile(values, p * 100))
                        for p in request.percentiles
                    }
            
            # Update result
            result.status = SimulationStatus.COMPLETED
            result.end_time = datetime.now()
            result.execution_time_seconds = time.time() - start_time
            
            # Add detailed paths if requested
            if request.include_detailed_paths:
                result.detailed_paths = all_cashflows
            
            # Generate enhanced Monte Carlo result
            enhanced_result = self.generate_enhanced_monte_carlo_result(
                simulation_id,
                request.num_simulations,
                request.projection_months,
                all_metrics,
                all_cashflows,
                result.execution_time_seconds
            )
            
            # Cache the result if enabled
            if use_cache and not cache_hit:
                try:
                    cache_key = self._generate_cache_key(request, user_id)
                    result_json = result.json()
                    ttl = settings.CACHE_TTL_SECONDS or 3600  # Default to 1 hour
                    await self.cache_result(cache_key, result_json, ttl=ttl)
                    logger.debug(f"Cached result for simulation {simulation_id} with TTL {ttl}s")
                except Exception as e:
                    logger.warning(f"Error saving result to cache: {str(e)}. Continuing without caching.")
            
            logger.info(f"Completed Monte Carlo simulation {simulation_id} in {result.execution_time_seconds:.2f} seconds")
            return enhanced_result
        
        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error running Monte Carlo simulation {simulation_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update result with error information
            result.status = SimulationStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.execution_time_seconds = time.time() - start_time
            
            return result
    
    @handle_exceptions
    def calculate_enhanced_statistics(
        self, 
        values: np.ndarray, 
        percentiles: List[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99], 
        confidence_level: float = 0.95
    ) -> StatisticalOutputs:
        """
        Calculate comprehensive statistical outputs from simulation values
        
        Args:
            values: Array of simulation values
            percentiles: Percentiles to calculate
            confidence_level: Confidence level for intervals
            
        Returns:
            StatisticalOutputs with comprehensive statistics
        """
        try:
            if len(values) < 2:
                raise ValueError("At least two values are required to calculate statistics")
            
            # Basic statistics
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            std_dev = float(np.std(values, ddof=1))  # Use sample standard deviation
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            
            # Calculate percentiles
            percentile_values = {}
            for p in percentiles:
                percentile_values[str(p)] = float(np.percentile(values, p))
            
            # Higher-order moments
            skewness = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))
            
            # Confidence intervals
            alpha = 1 - confidence_level
            dof = len(values) - 1  # Degrees of freedom
            t_crit = stats.t.ppf(1 - alpha/2, dof)
            
            # Standard error of the mean
            sem = std_dev / np.sqrt(len(values))
            
            # Confidence intervals
            ci_lower = mean_value - t_crit * sem
            ci_upper = mean_value + t_crit * sem
            
            confidence_intervals = {
                "mean": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper)
                }
            }
            
            # Create result
            return StatisticalOutputs(
                mean=mean_value,
                median=median_value,
                std_dev=std_dev,
                min_value=min_value,
                max_value=max_value,
                percentiles=percentile_values,
                skewness=skewness,
                kurtosis=kurtosis,
                confidence_intervals=confidence_intervals
            )
        except Exception as e:
            logger.error(f"Error calculating enhanced statistics: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Return basic statistics if calculation fails
            return StatisticalOutputs(
                mean=float(np.mean(values)) if len(values) > 0 else 0.0,
                median=float(np.median(values)) if len(values) > 0 else 0.0,
                std_dev=float(np.std(values)) if len(values) > 0 else 0.0,
                min_value=float(np.min(values)) if len(values) > 0 else 0.0,
                max_value=float(np.max(values)) if len(values) > 0 else 0.0,
                percentiles={"50": float(np.median(values)) if len(values) > 0 else 0.0}
            )

    @handle_exceptions
    def apply_economic_factors(
        self, 
        simulation_variables: Dict[str, float],
        economic_factors: Optional[EconomicFactors] = None
    ) -> Dict[str, float]:
        """
        Apply economic factors to simulation variables based on correlation model
        
        Args:
            simulation_variables: The base simulation variables
            economic_factors: Economic factors to apply
            
        Returns:
            Modified simulation variables
        """
        if not economic_factors:
            return simulation_variables
        
        # Create a copy to avoid modifying the original
        modified_variables = simulation_variables.copy()
        
        # Extract economic factors
        factors_dict = economic_factors.model_dump() if hasattr(economic_factors, 'model_dump') else economic_factors
        
        # Get main economic indicators
        unemployment_rate = factors_dict.get("unemployment_rate", 0.0)
        gdp_growth = factors_dict.get("gdp_growth", 0.0)
        housing_price_index = factors_dict.get("housing_price_index", 0.0)
        interest_rates = factors_dict.get("interest_rates", {})
        
        # Get interest rate environment and specific rates
        interest_env = factors_dict.get("interest_rate_environment", "neutral")
        fed_funds_rate = interest_rates.get("fed_funds", 0.0)
        term_10y = interest_rates.get("term_10y", 0.0)
        term_spread = term_10y - fed_funds_rate if term_10y and fed_funds_rate else 0.0
        
        # Credit industry factors
        credit_spread = factors_dict.get("credit_spread", 0.0)
        market_volatility = factors_dict.get("market_volatility", 0.0)
        liquidity_index = factors_dict.get("liquidity_index", 0.0)
        
        # --- Variable adjustments based on economic research ---
        
        # Default rate adjustments
        if "default_rate" in modified_variables:
            base_default = modified_variables["default_rate"]
            
            # Unemployment has strong positive correlation with defaults
            unemployment_effect = 0.3 * unemployment_rate
            
            # GDP growth has negative correlation with defaults
            gdp_effect = -0.25 * gdp_growth
            
            # Housing prices have negative correlation with defaults (especially for mortgages)
            housing_effect = -0.15 * housing_price_index
            
            # Credit spreads widening increases defaults (market stress indicator)
            credit_effect = 0.2 * credit_spread
            
            # Term spread (yield curve) - inverted yield curve (negative spread) predicts recessions
            term_effect = -0.15 * term_spread if term_spread < 0 else -0.05 * term_spread
            
            # Combined adjustment (weighted based on economic research)
            default_adjustment = 1.0 + unemployment_effect + gdp_effect + housing_effect + credit_effect + term_effect
            
            # Apply adjustment with reasonable bounds and smoothing
            modified_variables["default_rate"] = max(0.0001, min(0.99, base_default * default_adjustment))
        
        # Prepayment rate adjustments
        if "prepayment_rate" in modified_variables:
            base_prepayment = modified_variables["prepayment_rate"]
            
            # Interest rate environment effect - falling rates increase prepayments
            interest_effect = -0.4 if interest_env == "falling" else 0.3 if interest_env == "rising" else 0.0
            
            # Housing prices have positive correlation with prepayments (refinance opportunity)
            housing_effect = 0.25 * housing_price_index
            
            # Unemployment has negative correlation with prepayments (financial stress)
            unemployment_effect = -0.2 * unemployment_rate
            
            # Liquidity conditions affect refinancing ability
            liquidity_effect = 0.15 * liquidity_index
            
            # Combined adjustment
            prepayment_adjustment = 1.0 + interest_effect + housing_effect + unemployment_effect + liquidity_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["prepayment_rate"] = max(0.0001, min(0.99, base_prepayment * prepayment_adjustment))
        
        # Recovery rate adjustments
        if "recovery_rate" in modified_variables:
            base_recovery = modified_variables["recovery_rate"]
            
            # Housing price index strongly affects recovery (especially for secured loans)
            housing_effect = 0.3 * housing_price_index
            
            # GDP growth affects general ability to recover value
            gdp_effect = 0.15 * gdp_growth
            
            # Market liquidity affects ability to sell recovered assets
            liquidity_effect = 0.2 * liquidity_index
            
            # Market volatility negatively impacts recoveries
            volatility_effect = -0.1 * market_volatility
            
            # Combined adjustment
            recovery_adjustment = 1.0 + housing_effect + gdp_effect + liquidity_effect + volatility_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["recovery_rate"] = max(0.0001, min(0.99, base_recovery * recovery_adjustment))
            
        # Loss given default (LGD) adjustments (if separate from recovery rate)
        if "lgd" in modified_variables:
            # LGD is often inverse of recovery rate, but can have its own factors
            if "recovery_rate" in modified_variables:
                # Ensure consistency with recovery rate if both are present
                modified_variables["lgd"] = max(0.01, min(0.99, 1.0 - modified_variables["recovery_rate"]))
            else:
                # Apply similar factors as recovery but inverted
                base_lgd = modified_variables["lgd"]
                lgd_adjustment = 1.0 - (housing_effect + gdp_effect + liquidity_effect - volatility_effect)
                modified_variables["lgd"] = max(0.01, min(0.99, base_lgd * lgd_adjustment))
                
        # Credit spread adjustments
        if "credit_spread" in modified_variables:
            base_spread = modified_variables["credit_spread"]
            
            # Market volatility increases spreads
            volatility_effect = 0.3 * market_volatility
            
            # Unemployment increases spreads
            unemployment_effect = 0.15 * unemployment_rate
            
            # GDP growth decreases spreads
            gdp_effect = -0.2 * gdp_growth
            
            # Liquidity decreases spreads
            liquidity_effect = -0.25 * liquidity_index
            
            # Combined adjustment
            spread_adjustment = 1.0 + volatility_effect + unemployment_effect + gdp_effect + liquidity_effect
            
            # Apply adjustment
            modified_variables["credit_spread"] = max(0.0001, base_spread * spread_adjustment)
            
        # Loan growth rate adjustments
        if "loan_growth" in modified_variables:
            base_growth = modified_variables["loan_growth"]
            
            # GDP strongly correlates with loan growth
            gdp_effect = 0.4 * gdp_growth
            
            # Lower interest rates increase loan demand
            interest_effect = -0.2 * fed_funds_rate if fed_funds_rate is not None else 0.0
            
            # Higher unemployment decreases loan growth
            unemployment_effect = -0.3 * unemployment_rate
            
            # Combined adjustment
            growth_adjustment = 1.0 + gdp_effect + interest_effect + unemployment_effect
            
            # Apply adjustment
            modified_variables["loan_growth"] = base_growth * growth_adjustment
            
        return modified_variables

    async def run_enhanced_simulation(
        self, 
        request: Any, 
        user_id: str,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        progress_callback = None
    ) -> MonteCarloResult:
        """
        Run an enhanced Monte Carlo simulation with proper statistical outputs
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            progress_callback: Optional callback function for progress updates
            
        Returns:
            MonteCarloResult with comprehensive statistics
        """
        # Generate a unique ID for this simulation
        simulation_id = str(uuid.uuid4())
        
        # Start time tracking
        start_time = time.time()
        
        # Check cache if enabled
        cache_hit = False
        if use_cache:
            try:
                # Generate cache key
                from app.utils.cache_utils import generate_cache_key
                cache_key = generate_cache_key(
                    prefix="monte_carlo_enhanced",
                    user_id=user_id,
                    data=request.model_dump() if hasattr(request, 'model_dump') else request
                )
                
                # Try to get from cache
                cached_result = await self.get_cached_result(cache_key, MonteCarloResult)
                if cached_result:
                    logger.info(f"Found cached result for enhanced simulation {simulation_id}")
                    return cached_result
                    
            except Exception as e:
                # Log but continue - cache failures should not stop execution
                logger.warning(f"Error checking cache: {str(e)}. Proceeding with calculation.")
                logger.debug(f"Cache check error - traceback: {traceback.format_exc()}")
        
        try:
            logger.info(f"Starting enhanced Monte Carlo simulation {simulation_id} with {request.num_iterations} iterations")
            
            # Set up variable tracking
            all_metrics = {}
            all_cashflows = []
            
            # Generate correlated random variables for all simulations
            with CalculationTracker(f"monte_carlo_generate_variables_{simulation_id}"):
                try:
                    variables_matrix = self._generate_correlated_variables(request)
                except Exception as e:
                    logger.error(f"Error generating correlated variables: {str(e)}")
                    logger.debug(f"Variables generation error - traceback: {traceback.format_exc()}")
                    raise ValueError(f"Failed to generate simulation variables: {str(e)}")
            
            # Run the simulations
            completed_simulations = 0
            with CalculationTracker(f"monte_carlo_run_simulations_{simulation_id}"):
                for i in range(request.num_iterations):
                    try:
                        # Extract the variables for this simulation
                        simulation_variables = {
                            var.name: variables_matrix[var.name][i] 
                            for var in request.variables
                        }
                        
                        # Apply economic factors if provided
                        if hasattr(request, 'economic_factors') and request.economic_factors:
                            simulation_variables = self.apply_economic_factors(
                                simulation_variables, 
                                request.economic_factors
                            )
                        
                        # Run the single simulation
                        sim_result = self._run_single_simulation(
                            request, 
                            simulation_variables, 
                            i
                        )
                        
                        # Collect metrics
                        for metric, value in sim_result.metrics.items():
                            if metric not in all_metrics:
                                all_metrics[metric] = []
                            all_metrics[metric].append(value)
                        
                        # Collect cashflows if detailed paths are requested
                        include_cashflows = getattr(request, 'include_detailed_paths', False)
                        if include_cashflows and hasattr(sim_result, 'cashflows') and sim_result.cashflows:
                            all_cashflows.append(sim_result)
                        
                        # Update progress
                        completed_simulations += 1
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(i + 1, request.num_iterations)
                            
                    except Exception as e:
                        # Log error but continue with other simulations
                        logger.error(f"Error in simulation iteration {i}: {str(e)}")
                        logger.debug(f"Simulation iteration error - traceback: {traceback.format_exc()}")
            
            # Get total calculation time
            calculation_time = time.time() - start_time
            
            # Generate final result with statistics
            with CalculationTracker(f"monte_carlo_calculate_statistics_{simulation_id}"):
                result = self.generate_enhanced_monte_carlo_result(
                    simulation_id=simulation_id,
                    num_iterations=request.num_iterations,
                    time_horizon=getattr(request, 'projection_months', 120),
                    all_metrics=all_metrics,
                    all_cashflows=all_cashflows,
                    calculation_time=calculation_time
                )
            
            # Cache the result if enabled
            if use_cache and not cache_hit:
                try:
                    # Generate cache key
                    from app.utils.cache_utils import generate_cache_key
                    cache_key = generate_cache_key(
                        prefix="monte_carlo_enhanced",
                        user_id=user_id,
                        data=request.model_dump() if hasattr(request, 'model_dump') else request
                    )
                    
                    # Cache the result
                    await self.cache_result(cache_key, result, ttl=cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {str(e)}. Continuing without caching.")
            
            logger.info(f"Completed enhanced Monte Carlo simulation {simulation_id} in {calculation_time:.2f} seconds")
            return result
            
        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error in enhanced Monte Carlo simulation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Calculate time even for failed simulations
            calculation_time = time.time() - start_time
            
            # Create error result
            error_result = MonteCarloResult(
                simulation_id=simulation_id,
                num_iterations=getattr(request, 'num_iterations', 0),
                time_horizon=getattr(request, 'projection_months', 0),
                calculation_time=calculation_time,
                npv_stats=StatisticalOutputs(
                    mean=0.0,
                    median=0.0,
                    std_dev=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    percentiles={"50": 0.0}
                ),
                error=f"Simulation failed: {str(e)}"
            )
            
            return error_result

    def generate_enhanced_monte_carlo_result(
        self,
        simulation_id: str,
        num_iterations: int,
        time_horizon: int,
        all_metrics: Dict[str, List[float]],
        all_cashflows: Optional[List[Any]] = None,
        calculation_time: float = 0.0,
        error: Optional[str] = None
    ) -> MonteCarloResult:
        """
        Generate a comprehensive MonteCarloResult with enhanced statistical outputs and risk metrics
        
        Args:
            simulation_id: Unique identifier for this simulation
            num_iterations: Number of iterations performed
            time_horizon: Time horizon in months
            all_metrics: Dictionary of metric names to lists of values
            all_cashflows: Optional list of cashflow projections
            calculation_time: Calculation time in seconds
            error: Optional error message
            
        Returns:
            Comprehensive MonteCarloResult
        """
        try:
            # Calculate enhanced statistics for NPV
            npv_values = np.array(all_metrics.get("npv", [0.0]))
            npv_stats = self.calculate_enhanced_statistics(npv_values)
            
            # Calculate risk metrics for NPV
            risk_metrics = self.calculate_risk_metrics(npv_values)
            
            # Create base result object
            result = MonteCarloResult(
                simulation_id=simulation_id,
                num_iterations=num_iterations,
                time_horizon=time_horizon,
                calculation_time=calculation_time,
                npv_stats=npv_stats,
                risk_metrics=risk_metrics
            )
            
            # Add additional statistics if available
            if "irr" in all_metrics:
                irr_values = np.array(all_metrics["irr"])
                result.irr_stats = self.calculate_enhanced_statistics(irr_values)
                # IRR is a rate, so risk metrics are appropriate
                result.irr_risk_metrics = self.calculate_risk_metrics(irr_values)
            
            if "duration" in all_metrics:
                duration_values = np.array(all_metrics["duration"])
                result.duration_stats = self.calculate_enhanced_statistics(duration_values)
            
            if "default" in all_metrics:
                default_values = np.array(all_metrics["default"])
                result.default_stats = self.calculate_enhanced_statistics(default_values)
            
            if "prepayment" in all_metrics:
                prepayment_values = np.array(all_metrics["prepayment"])
                result.prepayment_stats = self.calculate_enhanced_statistics(prepayment_values)
            
            # Calculate loss distribution and risk metrics
            if "loss" in all_metrics:
                loss_values = np.array(all_metrics["loss"])
                result.loss_stats = self.calculate_enhanced_statistics(loss_values)
                result.loss_risk_metrics = self.calculate_risk_metrics(loss_values)
                
                # Create detailed loss distribution
                percentiles = list(range(1, 100))
                result.loss_distribution = {
                    str(p): float(np.percentile(loss_values, p))
                    for p in percentiles
                }
                
                # Calculate expected loss
                result.expected_loss = float(np.mean(loss_values))
                
                # Calculate unexpected loss (UL = standard deviation of loss)
                result.unexpected_loss = float(np.std(loss_values, ddof=1))
                
                # Calculate economic capital (EC = UL * confidence factor)
                # Typically using 99.9% confidence for bank capital calculations
                confidence_factor = stats.norm.ppf(0.999)  # ~3.09
                result.economic_capital = result.unexpected_loss * confidence_factor
            
            # Add portfolio-specific metrics if available
            if "diversification_benefit" in all_metrics:
                div_values = np.array(all_metrics["diversification_benefit"])
                result.diversification_benefit = float(np.mean(div_values))
                
            if "correlation_effect" in all_metrics:
                corr_values = np.array(all_metrics["correlation_effect"])
                result.correlation_effect = float(np.mean(corr_values))
            
            # Add cashflow projections if available
            if all_cashflows and len(all_cashflows) > 0:
                # Find best and worst case
                if "npv" in all_metrics and len(npv_values) == len(all_cashflows):
                    npv_values = np.array(all_metrics["npv"])
                    best_idx = np.argmax(npv_values)
                    worst_idx = np.argmin(npv_values)
                    median_idx = np.argsort(npv_values)[len(npv_values)//2]
                    
                    if 0 <= best_idx < len(all_cashflows):
                        result.best_case_cashflows = all_cashflows[best_idx]
                    
                    if 0 <= worst_idx < len(all_cashflows):
                        result.worst_case_cashflows = all_cashflows[worst_idx]
                        
                    if 0 <= median_idx < len(all_cashflows):
                        result.median_case_cashflows = all_cashflows[median_idx]
                    
                # Calculate representative paths for visualization
                # Taking samples at different percentiles to show the distribution
                if len(all_cashflows) >= 10 and "npv" in all_metrics and len(npv_values) == len(all_cashflows):
                    percentile_idxs = []
                    for p in [5, 25, 50, 75, 95]:
                        sorted_indices = np.argsort(npv_values)
                        idx = sorted_indices[int(len(sorted_indices) * p / 100)]
                        if 0 <= idx < len(all_cashflows):
                            percentile_idxs.append(idx)
                    
                    result.representative_paths = [all_cashflows[idx] for idx in percentile_idxs]
            
            # Add error information if provided
            if error:
                result.error = error
                
            # Record completed iterations
            result.completed_iterations = num_iterations
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating enhanced Monte Carlo result: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Return basic result with error information
            return MonteCarloResult(
                simulation_id=simulation_id,
                num_iterations=num_iterations,
                time_horizon=time_horizon,
                calculation_time=calculation_time,
                npv_stats=self.calculate_enhanced_statistics(np.array([0.0])),
                error=f"Error generating result: {str(e)}"
            )

    async def cache_result(
        self, 
        key: str, 
        result: Any, 
        ttl: int = 3600,
        max_retries: int = 3
    ) -> bool:
        """
        Cache a result with robust error handling and retry logic
        
        Args:
            key: Cache key
            result: Result to cache
            ttl: Time to live in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_service:
            logger.warning("Redis service not available. Skipping caching.")
            return False
            
        try:
            # Serialize the result
            try:
                if hasattr(result, 'json'):
                    result_json = result.json()
                else:
                    import json
                    result_json = json.dumps(result)
            except Exception as e:
                logger.warning(f"Error serializing result for caching: {str(e)}. Skipping cache.")
                logger.debug(f"Serialization error - traceback: {traceback.format_exc()}")
                return False
                
            # Set with retry logic
            success = False
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    await self.redis_service.set(key, result_json, ttl=ttl)
                    logger.debug(f"Successfully cached result with key {key} and TTL {ttl}s")
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:  # Don't log if last attempt
                        logger.warning(f"Attempt {attempt+1} failed to cache result: {str(e)}. Retrying...")
                    # Exponential backoff with jitter
                    backoff_time = 0.1 * (2 ** attempt) * (0.75 + 0.5 * np.random.random())
                    await asyncio.sleep(backoff_time)
            
            if not success:
                logger.warning(f"All {max_retries} attempts to cache result failed: {str(last_error)}. Continuing without caching.")
                return False
                
            return True
                
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}. Continuing without caching.")
            logger.debug(f"Error caching result - traceback: {traceback.format_exc()}")
            return False
    
    async def get_cached_result(
        self, 
        key: str, 
        result_type: Any = None,
        fallback_value: Any = None
    ) -> Optional[Any]:
        """
        Get a cached result with robust error handling and graceful fallbacks
        
        Args:
            key: Cache key
            result_type: Optional type to parse the result into
            fallback_value: Value to return if cache retrieval fails
            
        Returns:
            Cached result if found and valid, fallback_value if specified, or None otherwise
        """
        if not self.redis_service:
            logger.debug("Redis service not available. Skipping cache check.")
            return fallback_value
            
        try:
            # Try to get the cached result
            cached_data = await self.redis_service.get(key)
            if not cached_data:
                logger.debug(f"No cached data found for key {key}")
                return fallback_value
                
            # Parse the result
            try:
                if result_type and hasattr(result_type, 'parse_raw'):
                    # Parse into Pydantic model
                    try:
                        return result_type.parse_raw(cached_data)
                    except Exception as e:
                        logger.warning(f"Error parsing cached data into {result_type.__name__}: {str(e)}")
                        logger.debug(f"Parse error traceback: {traceback.format_exc()}")
                        logger.debug(f"Invalid cached data: {cached_data[:100]}...")
                        # Try to delete corrupted cache
                        try:
                            await self.redis_service.delete(key)
                            logger.debug(f"Deleted corrupted cache entry with key {key}")
                        except Exception as del_err:
                            logger.debug(f"Failed to delete corrupted cache entry: {str(del_err)}")
                        return fallback_value
                else:
                    # Parse as JSON
                    try:
                        import json
                        return json.loads(cached_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding cached JSON: {str(e)}")
                        logger.debug(f"Invalid JSON data: {cached_data[:100]}...")
                        # Try to delete corrupted cache
                        try:
                            await self.redis_service.delete(key)
                            logger.debug(f"Deleted corrupted cache entry with key {key}")
                        except Exception as del_err:
                            logger.debug(f"Failed to delete corrupted cache entry: {str(del_err)}")
                        return fallback_value
            except Exception as e:
                logger.warning(f"Unexpected error parsing cached data: {str(e)}")
                logger.debug(f"Parse error traceback: {traceback.format_exc()}")
                return fallback_value
                
        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}. Proceeding without using cache.")
            logger.debug(f"Error retrieving from cache - traceback: {traceback.format_exc()}")
            return fallback_value

    async def run_enhanced_simulation(
        self, 
        request: Any, 
        user_id: str,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        progress_callback = None
    ) -> MonteCarloResult:
        """
        Run an enhanced Monte Carlo simulation with proper statistical outputs
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            progress_callback: Optional callback function for progress updates
            
        Returns:
            MonteCarloResult with comprehensive statistics
        """
        # Generate a unique ID for this simulation
        simulation_id = str(uuid.uuid4())
        
        # Start time tracking
        start_time = time.time()
        
        # Check cache if enabled
        cache_hit = False
        if use_cache:
            try:
                # Generate cache key
                from app.utils.cache_utils import generate_cache_key
                cache_key = generate_cache_key(
                    prefix="monte_carlo_enhanced",
                    user_id=user_id,
                    data=request.model_dump() if hasattr(request, 'model_dump') else request
                )
                
                # Try to get from cache
                cached_result = await self.get_cached_result(cache_key, MonteCarloResult)
                if cached_result:
                    logger.info(f"Found cached result for enhanced simulation {simulation_id}")
                    return cached_result
                    
            except Exception as e:
                # Log but continue - cache failures should not stop execution
                logger.warning(f"Error checking cache: {str(e)}. Proceeding with calculation.")
                logger.debug(f"Cache check error - traceback: {traceback.format_exc()}")
        
        try:
            logger.info(f"Starting enhanced Monte Carlo simulation {simulation_id} with {request.num_iterations} iterations")
            
            # Set up variable tracking
            all_metrics = {}
            all_cashflows = []
            
            # Generate correlated random variables for all simulations
            with CalculationTracker(f"monte_carlo_generate_variables_{simulation_id}"):
                try:
                    variables_matrix = self._generate_correlated_variables(request)
                except Exception as e:
                    logger.error(f"Error generating correlated variables: {str(e)}")
                    logger.debug(f"Variables generation error - traceback: {traceback.format_exc()}")
                    raise ValueError(f"Failed to generate simulation variables: {str(e)}")
            
            # Run the simulations
            completed_simulations = 0
            with CalculationTracker(f"monte_carlo_run_simulations_{simulation_id}"):
                for i in range(request.num_iterations):
                    try:
                        # Extract the variables for this simulation
                        simulation_variables = {
                            var.name: variables_matrix[var.name][i] 
                            for var in request.variables
                        }
                        
                        # Apply economic factors if provided
                        if hasattr(request, 'economic_factors') and request.economic_factors:
                            simulation_variables = self.apply_economic_factors(
                                simulation_variables, 
                                request.economic_factors
                            )
                        
                        # Run the single simulation
                        sim_result = self._run_single_simulation(
                            request, 
                            simulation_variables, 
                            i
                        )
                        
                        # Collect metrics
                        for metric, value in sim_result.metrics.items():
                            if metric not in all_metrics:
                                all_metrics[metric] = []
                            all_metrics[metric].append(value)
                        
                        # Collect cashflows if detailed paths are requested
                        include_cashflows = getattr(request, 'include_detailed_paths', False)
                        if include_cashflows and hasattr(sim_result, 'cashflows') and sim_result.cashflows:
                            all_cashflows.append(sim_result)
                        
                        # Update progress
                        completed_simulations += 1
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(i + 1, request.num_iterations)
                            
                    except Exception as e:
                        # Log error but continue with other simulations
                        logger.error(f"Error in simulation iteration {i}: {str(e)}")
                        logger.debug(f"Simulation iteration error - traceback: {traceback.format_exc()}")
            
            # Get total calculation time
            calculation_time = time.time() - start_time
            
            # Generate final result with statistics
            with CalculationTracker(f"monte_carlo_calculate_statistics_{simulation_id}"):
                result = self.generate_enhanced_monte_carlo_result(
                    simulation_id=simulation_id,
                    num_iterations=request.num_iterations,
                    time_horizon=getattr(request, 'projection_months', 120),
                    all_metrics=all_metrics,
                    all_cashflows=all_cashflows,
                    calculation_time=calculation_time
                )
            
            # Cache the result if enabled
            if use_cache and not cache_hit:
                try:
                    # Generate cache key
                    from app.utils.cache_utils import generate_cache_key
                    cache_key = generate_cache_key(
                        prefix="monte_carlo_enhanced",
                        user_id=user_id,
                        data=request.model_dump() if hasattr(request, 'model_dump') else request
                    )
                    
                    # Cache the result
                    await self.cache_result(cache_key, result, ttl=cache_ttl)
                except Exception as e:
                    logger.warning(f"Error caching result: {str(e)}. Continuing without caching.")
            
            logger.info(f"Completed enhanced Monte Carlo simulation {simulation_id} in {calculation_time:.2f} seconds")
            return result
            
        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error in enhanced Monte Carlo simulation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Calculate time even for failed simulations
            calculation_time = time.time() - start_time
            
            # Create error result
            error_result = MonteCarloResult(
                simulation_id=simulation_id,
                num_iterations=getattr(request, 'num_iterations', 0),
                time_horizon=getattr(request, 'projection_months', 0),
                calculation_time=calculation_time,
                npv_stats=StatisticalOutputs(
                    mean=0.0,
                    median=0.0,
                    std_dev=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    percentiles={"50": 0.0}
                ),
                error=f"Simulation failed: {str(e)}"
            )
            
            return error_result

    def _generate_cache_key(self, request: Any, user_id: str, scenario_id: str = None) -> str:
        """
        Generate a cache key for a simulation request
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            scenario_id: Optional scenario ID if running with a scenario
            
        Returns:
            Cache key string
        """
        # Create a dict with all the key elements that make this simulation unique
        try:
            key_dict = {
                "user_id": user_id,
                "num_simulations": getattr(request, "num_iterations", getattr(request, "num_simulations", 1000)),
                "asset_class": getattr(request, "asset_class", "generic"),
                "projection_months": getattr(request, "projection_months", getattr(request, "time_horizon", 120)),
            }
            
            # Add scenario_id if present
            if scenario_id:
                key_dict["scenario_id"] = scenario_id
                
            # Add any variables if present
            if hasattr(request, "variables"):
                if hasattr(request.variables, "model_dump"):
                    key_dict["variables"] = request.variables.model_dump()
                elif hasattr(request.variables, "dict"):
                    key_dict["variables"] = request.variables.dict()
                else:
                    key_dict["variables"] = str(request.variables)
                    
            # Add economic factors if present
            if hasattr(request, "economic_factors") and request.economic_factors:
                if hasattr(request.economic_factors, "model_dump"):
                    key_dict["economic_factors"] = request.economic_factors.model_dump()
                elif hasattr(request.economic_factors, "dict"):
                    key_dict["economic_factors"] = request.economic_factors.dict()
                else:
                    key_dict["economic_factors"] = str(request.economic_factors)
            
            # Convert to JSON and create a hash
            key_json = json.dumps(key_dict, sort_keys=True)
            hash_object = hashlib.md5(key_json.encode())
            cache_key = f"monte_carlo:{hash_object.hexdigest()}"
            return cache_key
        except Exception as e:
            logger.warning(f"Error generating cache key: {str(e)}. Proceeding without caching.")
            # Return a fallback key that won't match any existing cache
            return f"monte_carlo:no_cache_{time.time()}"

    @handle_exceptions
    def calculate_risk_metrics(
        self, 
        values: np.ndarray, 
        confidence_levels: List[float] = [0.95, 0.99],
        risk_free_rate: float = 0.02
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics including VaR, CVaR, Sharpe ratio
        
        Args:
            values: Array of simulation values (typically returns or NPVs)
            confidence_levels: Confidence levels for VaR and CVaR
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            
        Returns:
            RiskMetrics object with comprehensive risk metrics
        """
        try:
            if len(values) < 10:
                raise ValueError("At least 10 values required for risk metric calculation")
                
            # Sort values for percentile-based calculations
            sorted_values = np.sort(values)
            
            # Calculate Value at Risk (VaR) for each confidence level
            var_metrics = {}
            cvar_metrics = {}
            
            for cl in confidence_levels:
                # Calculate the index for VaR
                index = int(len(sorted_values) * (1 - cl))
                
                # Handle edge case
                if index >= len(sorted_values):
                    index = len(sorted_values) - 1
                
                # VaR is the loss at the specified confidence level
                # We negate because VaR is typically expressed as a positive number
                # representing the potential loss
                var_value = -sorted_values[index]
                var_metrics[str(cl)] = float(var_value)
                
                # Conditional VaR (Expected Shortfall) - average of losses beyond VaR
                tail_values = sorted_values[:index+1]
                cvar_value = -np.mean(tail_values) if len(tail_values) > 0 else -sorted_values[0]
                cvar_metrics[str(cl)] = float(cvar_value)
            
            # Calculate other risk metrics
            mean_return = float(np.mean(values))
            volatility = float(np.std(values, ddof=1))
            
            # Calculate downside deviation (semi-deviation)
            # Only consider returns below the mean
            downside_returns = values[values < mean_return]
            downside_deviation = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else volatility
            
            # Calculate Sharpe ratio if we're dealing with returns
            # Only makes sense for returns, not absolute values like NPV
            sharpe_ratio = float((mean_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
            
            # Calculate Sortino ratio (using downside deviation)
            sortino_ratio = float((mean_return - risk_free_rate) / downside_deviation) if downside_deviation > 0 else 0.0
            
            # Maximum drawdown
            # For a series of returns, this would be calculated differently
            # Here we just use the maximum loss relative to the mean
            max_drawdown = float(np.min(values) - mean_return) if len(values) > 0 else 0.0
            
            # Create and return RiskMetrics object
            return RiskMetrics(
                var=var_metrics,
                cvar=cvar_metrics,
                volatility=volatility,
                downside_deviation=downside_deviation,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Return basic risk metrics with defaults
            return RiskMetrics(
                var={"0.95": 0.0},
                cvar={"0.95": 0.0},
                volatility=0.0,
                downside_deviation=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0
            )

    @handle_exceptions
    def run_stress_test(
        self, 
        base_request: MonteCarloSimulationRequest,
        stress_scenarios: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, MonteCarloResult]:
        """
        Run stress tests by adjusting simulation parameters across multiple scenarios
        
        Args:
            base_request: The base simulation request
            stress_scenarios: List of stress scenario definitions, each containing parameter adjustments
            user_id: ID of the user running the simulation
            
        Returns:
            Dictionary mapping scenario names to MonteCarloResult objects
        """
        results = {}
        
        # Run the base scenario first
        logger.info(f"Running base scenario stress test")
        try:
            base_result = asyncio.run(self.run_enhanced_simulation(
                request=base_request, 
                user_id=user_id,
                use_cache=True
            ))
            results["base_scenario"] = base_result
        except Exception as e:
            logger.error(f"Error running base stress test scenario: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            results["base_scenario"] = self._create_error_result(str(e))
        
        # Run each stress scenario
        for i, scenario in enumerate(stress_scenarios):
            scenario_name = scenario.get("name", f"stress_scenario_{i+1}")
            logger.info(f"Running stress test scenario: {scenario_name}")
            
            try:
                # Create a deep copy of the base request to modify
                import copy
                request_copy = copy.deepcopy(base_request)
                
                # Apply scenario adjustments
                self._apply_stress_scenario(request_copy, scenario)
                
                # Run the simulation with the adjusted parameters
                result = asyncio.run(self.run_enhanced_simulation(
                    request=request_copy, 
                    user_id=user_id,
                    use_cache=True
                ))
                
                results[scenario_name] = result
                
            except Exception as e:
                logger.error(f"Error running stress test scenario {scenario_name}: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                results[scenario_name] = self._create_error_result(f"Stress test {scenario_name} failed: {str(e)}")
        
        return results
    
    def _apply_stress_scenario(self, request: MonteCarloSimulationRequest, scenario: Dict[str, Any]) -> None:
        """
        Apply stress scenario adjustments to a simulation request
        
        Args:
            request: The simulation request to modify
            scenario: Stress scenario definition
        """
        # Apply economic factors adjustments
        if "economic_factors" in scenario and hasattr(request, "economic_factors"):
            for factor_name, value in scenario["economic_factors"].items():
                if hasattr(request.economic_factors, factor_name):
                    setattr(request.economic_factors, factor_name, value)
                elif isinstance(request.economic_factors, dict):
                    request.economic_factors[factor_name] = value
        
        # Apply variable adjustments
        if "variables" in scenario and hasattr(request, "variables"):
            var_adjustments = scenario["variables"]
            for var in request.variables:
                var_name = var.name
                if var_name in var_adjustments:
                    adjustments = var_adjustments[var_name]
                    for param, value in adjustments.items():
                        if hasattr(var, param):
                            # Apply multiplicative or additive adjustment
                            if isinstance(value, dict) and "factor" in value:
                                # Multiplicative adjustment
                                current = getattr(var, param)
                                if isinstance(current, (int, float)):
                                    setattr(var, param, current * value["factor"])
                            elif isinstance(value, dict) and "add" in value:
                                # Additive adjustment
                                current = getattr(var, param)
                                if isinstance(current, (int, float)):
                                    setattr(var, param, current + value["add"])
                            else:
                                # Direct replacement
                                setattr(var, param, value)
        
        # Apply simulation parameter adjustments
        for param, value in scenario.items():
            if param not in ["name", "economic_factors", "variables"] and hasattr(request, param):
                setattr(request, param, value)
    
    def _create_error_result(self, error_message: str) -> MonteCarloResult:
        """Create a MonteCarloResult with error information"""
        return MonteCarloResult(
            simulation_id=str(uuid.uuid4()),
            num_iterations=0,
            time_horizon=0,
            calculation_time=0.0,
            npv_stats=StatisticalOutputs(
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                percentiles={"50": 0.0}
            ),
            error=error_message
        )

    @handle_exceptions
    def perform_sensitivity_analysis(
        self, 
        base_request: MonteCarloSimulationRequest,
        parameters_to_test: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Dict[str, MonteCarloResult]]:
        """
        Perform sensitivity analysis by varying parameters one at a time
        
        Args:
            base_request: The base simulation request
            parameters_to_test: List of parameter definitions, each containing name, path, and values to test
            user_id: ID of the user running the simulation
            
        Returns:
            Nested dictionary mapping parameter names and values to MonteCarloResult objects
        """
        results = {}
        
        # Run the base scenario first
        logger.info(f"Running base scenario for sensitivity analysis")
        try:
            base_result = asyncio.run(self.run_enhanced_simulation(
                request=base_request, 
                user_id=user_id,
                use_cache=True
            ))
            results["base_scenario"] = {"base": base_result}
        except Exception as e:
            logger.error(f"Error running base sensitivity scenario: {e}")
            results["base_scenario"] = {"base": self._create_error_result(str(e))}
        
        # For each parameter to test
        for param_def in parameters_to_test:
            param_name = param_def.get("name", "unknown_parameter")
            param_path = param_def.get("path", "")
            param_values = param_def.get("values", [])
            
            logger.info(f"Performing sensitivity analysis for parameter: {param_name}")
            
            # Store results for this parameter
            param_results = {}
            
            # Test each value
            for value in param_values:
                value_str = str(value)
                logger.info(f"Testing {param_name} = {value_str}")
                
                try:
                    # Create a deep copy of the base request to modify
                    import copy
                    request_copy = copy.deepcopy(base_request)
                    
                    # Apply the parameter change
                    self._apply_parameter_change(request_copy, param_path, value)
                    
                    # Run the simulation with the adjusted parameters
                    result = asyncio.run(self.run_enhanced_simulation(
                        request=request_copy, 
                        user_id=user_id,
                        use_cache=True
                    ))
                    
                    param_results[value_str] = result
                    
                except Exception as e:
                    logger.error(f"Error in sensitivity test {param_name}={value_str}: {e}")
                    param_results[value_str] = self._create_error_result(
                        f"Sensitivity test {param_name}={value_str} failed: {str(e)}"
                    )
            
            # Store all results for this parameter
            results[param_name] = param_results
        
        return results
    
    def _apply_parameter_change(self, request: Any, param_path: str, value: Any) -> None:
        """
        Apply a parameter change to a nested object structure
        
        Args:
            request: The object to modify
            param_path: Dot-separated path to the parameter (e.g., "economic_factors.unemployment_rate")
            value: New value to set
        """
        parts = param_path.split('.')
        target = request
        
        # Navigate to the parent object
        for i, part in enumerate(parts[:-1]):
            if hasattr(target, part):
                target = getattr(target, part)
            elif isinstance(target, dict) and part in target:
                target = target[part]
            else:
                raise ValueError(f"Invalid parameter path: {param_path} (failed at '{part}')")
        
        # Set the attribute or dictionary key
        last_part = parts[-1]
        if hasattr(target, last_part):
            setattr(target, last_part, value)
        elif isinstance(target, dict):
            target[last_part] = value
        else:
            raise ValueError(f"Cannot set {last_part} on target object of type {type(target)}")

class CalculationTracker:
    """Context manager for tracking calculation time and handling errors"""
    
    def __init__(self, operation_name: str):
        """
        Initialize a calculation tracker
        
        Args:
            operation_name: Name of the operation being tracked
        """
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        """Start timing the operation"""
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        End timing the operation and log results
        
        Also handles any exceptions by logging them but allowing them to propagate
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        if exc_type is None:
            # Operation completed successfully
            logger.debug(f"Completed operation: {self.operation_name} in {duration:.4f}s")
        else:
            # Operation failed
            logger.error(f"Operation {self.operation_name} failed after {duration:.4f}s: {exc_val}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Note: We're not suppressing the exception here, just logging it
            # The exception will propagate up to the caller
        
        # Return False to propagate exceptions
        return False

class CalculationError(Exception):
    """Custom exception for calculation errors"""
    pass

def handle_exceptions(func):
    """
    Decorator to handle exceptions in Monte Carlo service methods
    
    Provides comprehensive error logging and graceful degradation
    for production environments.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Include function name and arguments in error details (excluding self)
            # This helps with debugging but filters out sensitive information
            safe_args = [f"arg{i}" for i in range(len(args) - 1)] if len(args) > 1 else []
            safe_kwargs = {k: "..." for k in kwargs.keys()}
            error_context = f"Function: {func.__name__}, Args: {safe_args}, Kwargs: {safe_kwargs}"
            logger.debug(f"Error context: {error_context}")
            
            # Re-raise as a CalculationError with the original exception info
            raise CalculationError(f"Calculation failed in {func.__name__}: {str(e)}") from e
    
    return wrapper
