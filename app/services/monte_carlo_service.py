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
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
import traceback
import scipy.stats as stats
import hashlib

from app.models.monte_carlo import (
    MonteCarloSimulationRequest, 
    MonteCarloSimulationResult,
    SimulationStatus,
    SimulationResult,
    DistributionType,
    ScenarioDefinition,
    SavedSimulation
)
from app.models.analytics import (
    StatisticalOutputs,
    EconomicFactors,
    MonteCarloResult
)
from app.core.config import settings
from app.services.redis_service import RedisService
from app.core.monitoring import CalculationTracker

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
                    await self.redis_service.set(cache_key, result_json, ttl=ttl)
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
        
        # Apply economic factor adjustments to simulation variables
        # Implement correlation model between economic factors and credit performance variables
        
        # Example: Modify default rate based on economic factors
        if "default_rate" in modified_variables:
            base_default = modified_variables["default_rate"]
            
            # Create adjustment based on economic factors
            unemployment_effect = 0.2 * factors_dict.get("unemployment_rate", 0.0)
            gdp_effect = -0.15 * factors_dict.get("gdp_growth", 0.0)
            housing_effect = -0.1 * factors_dict.get("housing_price_index", 0.0)
            
            # Combined adjustment
            default_adjustment = 1 + unemployment_effect + gdp_effect + housing_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["default_rate"] = max(0.0, min(1.0, base_default * default_adjustment))
        
        # Example: Modify prepayment rate based on economic factors
        if "prepayment_rate" in modified_variables:
            base_prepayment = modified_variables["prepayment_rate"]
            
            # Create adjustment based on economic factors
            interest_env = factors_dict.get("interest_rate_environment", "neutral")
            interest_effect = -0.2 if interest_env == "rising" else 0.1 if interest_env == "falling" else 0.0
            housing_effect = 0.15 * factors_dict.get("housing_price_index", 0.0)
            
            # Combined adjustment
            prepayment_adjustment = 1 + interest_effect + housing_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["prepayment_rate"] = max(0.0, min(1.0, base_prepayment * prepayment_adjustment))
        
        # Example: Modify recovery rate based on economic factors
        if "recovery_rate" in modified_variables:
            base_recovery = modified_variables["recovery_rate"]
            
            # Create adjustment based on economic factors
            housing_effect = 0.25 * factors_dict.get("housing_price_index", 0.0)
            gdp_effect = 0.1 * factors_dict.get("gdp_growth", 0.0)
            
            # Combined adjustment
            recovery_adjustment = 1 + housing_effect + gdp_effect
            
            # Apply adjustment with reasonable bounds
            modified_variables["recovery_rate"] = max(0.0, min(1.0, base_recovery * recovery_adjustment))
        
        return modified_variables
    
    def generate_enhanced_monte_carlo_result(
        self,
        simulation_id: str,
        num_iterations: int,
        time_horizon: int,
        all_metrics: Dict[str, List[float]],
        all_cashflows: Optional[List[Dict]] = None,
        calculation_time: float = 0.0,
        error: Optional[str] = None
    ) -> MonteCarloResult:
        """
        Generate a comprehensive MonteCarloResult with enhanced statistical outputs
        
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
            result = MonteCarloResult(
                simulation_id=simulation_id,
                num_iterations=num_iterations,
                time_horizon=time_horizon,
                calculation_time=calculation_time,
                npv_stats=self.calculate_enhanced_statistics(np.array(all_metrics.get("npv", [0.0])))
            )
            
            # Add additional statistics if available
            if "irr" in all_metrics:
                result.irr_stats = self.calculate_enhanced_statistics(np.array(all_metrics["irr"]))
            
            if "duration" in all_metrics:
                result.duration_stats = self.calculate_enhanced_statistics(np.array(all_metrics["duration"]))
            
            if "default" in all_metrics:
                result.default_stats = self.calculate_enhanced_statistics(np.array(all_metrics["default"]))
            
            if "prepayment" in all_metrics:
                result.prepayment_stats = self.calculate_enhanced_statistics(np.array(all_metrics["prepayment"]))
            
            # Create loss distribution
            if "loss" in all_metrics:
                loss_values = np.array(all_metrics["loss"])
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                result.loss_distribution = {
                    str(p): float(np.percentile(loss_values, p))
                    for p in percentiles
                }
            
            # Add cashflow projections if available
            if all_cashflows:
                # Find best and worst case
                if "npv" in all_metrics:
                    npv_values = np.array(all_metrics["npv"])
                    best_idx = np.argmax(npv_values)
                    worst_idx = np.argmin(npv_values)
                    
                    if 0 <= best_idx < len(all_cashflows):
                        result.best_case_cashflows = all_cashflows[best_idx]
                    
                    if 0 <= worst_idx < len(all_cashflows):
                        result.worst_case_cashflows = all_cashflows[worst_idx]
            
            if error:
                result.error = error
            
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
                error=str(e)
            )
    
    async def cache_result(
        self, 
        key: str, 
        result: Any, 
        ttl: int = 3600
    ) -> bool:
        """
        Cache a result with robust error handling and retry logic
        
        Args:
            key: Cache key
            result: Result to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_service:
            logger.warning("Redis service not available. Skipping caching.")
            return False
            
        try:
            # Serialize the result
            if hasattr(result, 'json'):
                result_json = result.json()
            else:
                import json
                result_json = json.dumps(result)
                
            # Set with retry logic
            for attempt in range(3):  # Try up to 3 times
                try:
                    await self.redis_service.set(key, result_json, ttl=ttl)
                    logger.debug(f"Successfully cached result with key {key} and TTL {ttl}s")
                    return True
                except Exception as e:
                    if attempt < 2:  # Don't log if last attempt
                        logger.warning(f"Attempt {attempt+1} failed to cache result: {str(e)}. Retrying...")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
            # If we get here, all attempts failed
            logger.warning("All attempts to cache result failed. Continuing without caching.")
            return False
                
        except Exception as e:
            logger.warning(f"Error serializing or caching result: {str(e)}. Continuing without caching.")
            logger.debug(f"Error caching result - traceback: {traceback.format_exc()}")
            return False
    
    async def get_cached_result(
        self, 
        key: str, 
        result_type: Any = None
    ) -> Optional[Any]:
        """
        Get a cached result with robust error handling
        
        Args:
            key: Cache key
            result_type: Optional type to parse the result into
            
        Returns:
            Cached result if found and valid, None otherwise
        """
        if not self.redis_service:
            logger.debug("Redis service not available. Skipping cache check.")
            return None
            
        try:
            # Try to get the cached result
            cached_data = await self.redis_service.get(key)
            if not cached_data:
                logger.debug(f"No cached data found for key {key}")
                return None
                
            # Parse the result
            if result_type and hasattr(result_type, 'parse_raw'):
                # Parse into Pydantic model
                try:
                    return result_type.parse_raw(cached_data)
                except Exception as e:
                    logger.warning(f"Error parsing cached data into {result_type.__name__}: {str(e)}")
                    return None
            else:
                # Parse as JSON
                try:
                    import json
                    return json.loads(cached_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding cached JSON: {str(e)}")
                    return None
                
        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}. Proceeding without using cache.")
            logger.debug(f"Error retrieving from cache - traceback: {traceback.format_exc()}")
            return None
    
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
                            all_cashflows.append(sim_result.cashflows)
                        
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
