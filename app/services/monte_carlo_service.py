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

from app.models.monte_carlo import (
    MonteCarloSimulationRequest, 
    MonteCarloSimulationResult,
    SimulationStatus,
    SimulationResult,
    DistributionType
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
        use_cache: bool = True
    ) -> MonteCarloSimulationResult:
        """
        Run a Monte Carlo simulation based on the provided request
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            use_cache: Whether to use caching
            
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
            percentiles={}
        )
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._generate_cache_key(request, user_id)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Found cached result for simulation {simulation_id}")
                return cached_result
        
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
                        float(p): float(np.percentile(values, p * 100))
                        for p in request.percentiles
                    }
            
            # Update result
            result.status = SimulationStatus.COMPLETED
            result.end_time = datetime.now()
            result.execution_time_seconds = time.time() - start_time
            
            # Add detailed paths if requested
            if request.include_detailed_paths:
                result.detailed_paths = all_cashflows
            
            # Cache the result if enabled
            if use_cache:
                await self._save_to_cache(cache_key, result)
            
            logger.info(f"Completed Monte Carlo simulation {simulation_id} in {result.execution_time_seconds:.2f} seconds")
            return result
        
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
    
    def _generate_correlated_variables(
        self, 
        request: MonteCarloSimulationRequest
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated random variables for all simulations
        
        Args:
            request: The simulation request
            
        Returns:
            Dictionary mapping variable names to arrays of random values
        """
        num_simulations = request.num_simulations
        num_variables = len(request.variables)
        
        # Generate uncorrelated random variables
        uncorrelated = np.zeros((num_variables, num_simulations))
        for i, var in enumerate(request.variables):
            uncorrelated[i] = self._generate_random_variable(
                var.distribution,
                var.parameters,
                num_simulations
            )
        
        # If no correlation matrix provided, return uncorrelated variables
        if not request.correlation_matrix:
            return {
                var.name: uncorrelated[i] 
                for i, var in enumerate(request.variables)
            }
        
        # Create correlation matrix
        correlation_matrix = np.eye(num_variables)
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                var1 = request.variables[i].name
                var2 = request.variables[j].name
                key = f"{var1}:{var2}"
                reverse_key = f"{var2}:{var1}"
                
                # Look up correlation coefficient
                if key in request.correlation_matrix.correlations:
                    corr = request.correlation_matrix.correlations[key]
                elif reverse_key in request.correlation_matrix.correlations:
                    corr = request.correlation_matrix.correlations[reverse_key]
                else:
                    corr = 0.0
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Calculate Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, adjust it
            logger.warning("Correlation matrix is not positive definite. Applying adjustment.")
            
            # Calculate nearest positive definite matrix
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            adjusted_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Ensure the diagonal is 1
            d = np.sqrt(np.diag(adjusted_matrix))
            adjusted_matrix = adjusted_matrix / np.outer(d, d)
            
            # Calculate Cholesky decomposition
            L = np.linalg.cholesky(adjusted_matrix)
        
        # Generate correlated random variables
        correlated = L @ uncorrelated
        
        # Transform back to original distributions if needed
        # (This is a simplified approach; more complex methods might be needed)
        
        # Return dictionary mapping variable names to arrays of values
        return {
            var.name: correlated[i] 
            for i, var in enumerate(request.variables)
        }
    
    def _generate_random_variable(
        self, 
        distribution: DistributionType,
        parameters: Dict[str, float],
        size: int
    ) -> np.ndarray:
        """
        Generate random variables with the specified distribution
        
        Args:
            distribution: The type of distribution
            parameters: The parameters for the distribution
            size: The number of random variables to generate
            
        Returns:
            Array of random variables
        """
        if distribution == DistributionType.NORMAL:
            mean = parameters.get('mean', 0.0)
            std_dev = parameters.get('std_dev', 1.0)
            return self._rng.normal(mean, std_dev, size=size)
        
        elif distribution == DistributionType.LOGNORMAL:
            mean = parameters.get('mean', 0.0)
            std_dev = parameters.get('std_dev', 1.0)
            return self._rng.lognormal(mean, std_dev, size=size)
        
        elif distribution == DistributionType.UNIFORM:
            min_val = parameters.get('min', 0.0)
            max_val = parameters.get('max', 1.0)
            return self._rng.uniform(min_val, max_val, size=size)
        
        elif distribution == DistributionType.TRIANGULAR:
            min_val = parameters.get('min', 0.0)
            mode = parameters.get('mode', 0.5)
            max_val = parameters.get('max', 1.0)
            return self._rng.triangular(min_val, mode, max_val, size=size)
        
        elif distribution == DistributionType.BETA:
            alpha = parameters.get('alpha', 1.0)
            beta = parameters.get('beta', 1.0)
            min_val = parameters.get('min', 0.0)
            max_val = parameters.get('max', 1.0)
            raw = self._rng.beta(alpha, beta, size=size)
            return min_val + (max_val - min_val) * raw
        
        elif distribution == DistributionType.CUSTOM:
            values = parameters.get('values', [0.0, 1.0])
            probabilities = parameters.get('probabilities', [0.5, 0.5])
            return self._rng.choice(values, size=size, p=probabilities)
        
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")
            
    def _run_single_simulation(
        self,
        request: MonteCarloSimulationRequest,
        simulation_variables: Dict[str, float],
        simulation_id: int
    ) -> SimulationResult:
        """
        Run a single simulation with the given variables
        
        Args:
            request: The simulation request
            simulation_variables: The values of the random variables for this simulation
            simulation_id: The ID of this simulation (for tracking)
            
        Returns:
            The result of the simulation
        """
        # Apply the simulation variables to the asset parameters
        modified_parameters = self._apply_variables_to_parameters(
            request.asset_parameters,
            simulation_variables
        )
        
        # Run the appropriate asset class handler
        metrics = {}
        cashflows = None
        
        # Get the appropriate handler for the asset class
        if request.asset_class.lower() == "consumer_credit":
            result = self._run_consumer_credit_simulation(request, modified_parameters)
            metrics = result["metrics"]
            cashflows = result.get("cashflows")
        
        elif request.asset_class.lower() == "commercial_loan":
            result = self._run_commercial_loan_simulation(request, modified_parameters)
            metrics = result["metrics"]
            cashflows = result.get("cashflows")
        
        elif request.asset_class.lower() == "clo_cdo":
            result = self._run_clo_cdo_simulation(request, modified_parameters)
            metrics = result["metrics"]
            cashflows = result.get("cashflows")
        
        else:
            raise ValueError(f"Unsupported asset class: {request.asset_class}")
        
        # Create and return the simulation result
        return SimulationResult(
            simulation_id=simulation_id,
            variables=simulation_variables,
            metrics=metrics,
            cashflows=cashflows
        )
    
    def _apply_variables_to_parameters(
        self,
        parameters: Dict[str, Any],
        variables: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply the simulation variables to the asset parameters
        
        This method modifies the asset parameters based on the simulation variables.
        It supports both direct parameter replacement and parameterized formulas.
        
        Args:
            parameters: The original asset parameters
            variables: The simulation variables
            
        Returns:
            The modified asset parameters
        """
        # Make a deep copy of the parameters to avoid modifying the original
        import copy
        result = copy.deepcopy(parameters)
        
        # Apply variable replacements
        # This is a simple implementation; more complex formulas could be supported
        for param_key, param_value in result.items():
            # Check if this parameter should be replaced by a variable
            if isinstance(param_value, str) and param_value.startswith("$"):
                var_name = param_value[1:]  # Remove the $ prefix
                if var_name in variables:
                    result[param_key] = variables[var_name]
            
            # Check if this parameter should be modified by a formula
            # Format: "$formula:param_name*factor+offset"
            elif isinstance(param_value, str) and param_value.startswith("$formula:"):
                formula = param_value[9:]  # Remove the $formula: prefix
                
                # Parse and evaluate the formula
                try:
                    # Replace variable references with their values
                    for var_name, var_value in variables.items():
                        formula = formula.replace(f"${var_name}", str(var_value))
                    
                    # Evaluate the formula
                    result[param_key] = eval(formula)
                except Exception as e:
                    logger.error(f"Error evaluating formula {formula}: {str(e)}")
                    # Keep the original value if there's an error
            
            # Recursively process nested dictionaries
            elif isinstance(param_value, dict):
                result[param_key] = self._apply_variables_to_parameters(param_value, variables)
            
            # Process lists of dictionaries
            elif isinstance(param_value, list) and all(isinstance(item, dict) for item in param_value):
                result[param_key] = [
                    self._apply_variables_to_parameters(item, variables)
                    for item in param_value
                ]
        
        return result
    
    def _run_consumer_credit_simulation(
        self,
        request: MonteCarloSimulationRequest,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a simulation for consumer credit assets
        
        Args:
            request: The simulation request
            parameters: The modified asset parameters
            
        Returns:
            Dictionary with metrics and optional cashflows
        """
        # Use the consumer credit handler to perform the analysis
        try:
            # Create a request object for the handler
            from app.models.asset_classes import AssetPoolAnalysisRequest, AssetPool
            
            # Extract assets from parameters
            assets = parameters.get("assets", [])
            
            # Create asset pool
            pool = AssetPool(
                pool_name=f"Simulation_{request.name}",
                assets=assets,
                cut_off_date=request.analysis_date
            )
            
            # Create analysis request
            analysis_request = AssetPoolAnalysisRequest(
                pool=pool,
                analysis_date=request.analysis_date,
                discount_rate=request.discount_rate,
                include_cashflows=request.include_detailed_paths
            )
            
            # Run analysis
            analysis_result = self.consumer_credit_handler.analyze_pool(analysis_request)
            
            # Extract metrics and cashflows
            result = {
                "metrics": analysis_result.metrics.dict() if analysis_result.metrics else {},
            }
            
            if request.include_detailed_paths and analysis_result.cashflows:
                result["cashflows"] = [cf.dict() for cf in analysis_result.cashflows]
            
            return result
        
        except Exception as e:
            logger.error(f"Error in consumer credit simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return {"metrics": {"error": str(e)}}
    
    def _run_commercial_loan_simulation(
        self,
        request: MonteCarloSimulationRequest,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a simulation for commercial loan assets
        
        Args:
            request: The simulation request
            parameters: The modified asset parameters
            
        Returns:
            Dictionary with metrics and optional cashflows
        """
        # Similar implementation to consumer credit but using commercial loan handler
        try:
            # Create a request object for the handler
            from app.models.asset_classes import AssetPoolAnalysisRequest, AssetPool
            
            # Extract assets from parameters
            assets = parameters.get("assets", [])
            
            # Create asset pool
            pool = AssetPool(
                pool_name=f"Simulation_{request.name}",
                assets=assets,
                cut_off_date=request.analysis_date
            )
            
            # Create analysis request
            analysis_request = AssetPoolAnalysisRequest(
                pool=pool,
                analysis_date=request.analysis_date,
                discount_rate=request.discount_rate,
                include_cashflows=request.include_detailed_paths
            )
            
            # Run analysis
            analysis_result = self.commercial_loan_handler.analyze_pool(analysis_request)
            
            # Extract metrics and cashflows
            result = {
                "metrics": analysis_result.metrics.dict() if analysis_result.metrics else {},
            }
            
            if request.include_detailed_paths and analysis_result.cashflows:
                result["cashflows"] = [cf.dict() for cf in analysis_result.cashflows]
            
            return result
        
        except Exception as e:
            logger.error(f"Error in commercial loan simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return {"metrics": {"error": str(e)}}
    
    def _run_clo_cdo_simulation(
        self,
        request: MonteCarloSimulationRequest,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a simulation for CLO/CDO assets
        
        Args:
            request: The simulation request
            parameters: The modified asset parameters
            
        Returns:
            Dictionary with metrics and optional cashflows
        """
        # Similar implementation to consumer credit but using CLO/CDO handler
        try:
            # Create a request object for the handler
            from app.models.asset_classes import AssetPoolAnalysisRequest, AssetPool
            
            # Extract assets from parameters
            assets = parameters.get("assets", [])
            
            # Create asset pool
            pool = AssetPool(
                pool_name=f"Simulation_{request.name}",
                assets=assets,
                cut_off_date=request.analysis_date
            )
            
            # Create analysis request
            analysis_request = AssetPoolAnalysisRequest(
                pool=pool,
                analysis_date=request.analysis_date,
                discount_rate=request.discount_rate,
                include_cashflows=request.include_detailed_paths
            )
            
            # Run analysis
            analysis_result = self.clo_cdo_handler.analyze_pool(analysis_request)
            
            # Extract metrics and cashflows
            result = {
                "metrics": analysis_result.metrics.dict() if analysis_result.metrics else {},
            }
            
            if request.include_detailed_paths and analysis_result.cashflows:
                result["cashflows"] = [cf.dict() for cf in analysis_result.cashflows]
            
            return result
        
        except Exception as e:
            logger.error(f"Error in CLO/CDO simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return {"metrics": {"error": str(e)}}
    
    def _generate_cache_key(
        self, 
        request: MonteCarloSimulationRequest,
        user_id: str
    ) -> str:
        """
        Generate a cache key for the simulation request
        
        Args:
            request: The simulation request
            user_id: The user ID
            
        Returns:
            A unique cache key
        """
        # Create a simplified version of the request for caching
        cache_data = {
            "user_id": user_id,
            "name": request.name,
            "num_simulations": request.num_simulations,
            "asset_class": request.asset_class,
            "analysis_date": request.analysis_date.isoformat(),
            "projection_months": request.projection_months,
            "variables": [
                {
                    "name": var.name,
                    "distribution": str(var.distribution),
                    "parameters": var.parameters
                }
                for var in request.variables
            ]
        }
        
        # Generate a hash of the request
        import hashlib
        cache_json = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_json.encode()).hexdigest()
        
        return f"monte_carlo_simulation:{cache_hash}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[MonteCarloSimulationResult]:
        """
        Get a simulation result from the cache
        
        Args:
            cache_key: The cache key
            
        Returns:
            The cached simulation result, or None if not found
        """
        try:
            # Try to get from cache
            cached_data = await self.redis_service.get(cache_key)
            
            if not cached_data:
                return None
            
            # Deserialize the cached data
            result_dict = json.loads(cached_data)
            
            # Convert to MonteCarloSimulationResult
            return MonteCarloSimulationResult(**result_dict)
        
        except Exception as e:
            logger.error(f"Error getting simulation from cache: {str(e)}")
            return None
    
    async def _save_to_cache(
        self, 
        cache_key: str, 
        result: MonteCarloSimulationResult
    ) -> bool:
        """
        Save a simulation result to the cache
        
        Args:
            cache_key: The cache key
            result: The simulation result to cache
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Serialize the result to JSON
            result_json = result.json()
            
            # Save to cache with TTL
            ttl = settings.CACHE_TTL_SECONDS or 3600  # Default to 1 hour
            await self.redis_service.set(cache_key, result_json, ttl=ttl)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving simulation to cache: {str(e)}")
            return False
            
    def run_simulation_sync(
        self, 
        request: MonteCarloSimulationRequest, 
        user_id: str,
        use_cache: bool = True
    ) -> MonteCarloSimulationResult:
        """
        Synchronous version of run_simulation for use with Celery tasks
        
        Args:
            request: The simulation request
            user_id: ID of the user running the simulation
            use_cache: Whether to use caching
            
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
            percentiles={}
        )
        
        # Check cache if enabled (using synchronous Redis client)
        if use_cache:
            cache_key = self._generate_cache_key(request, user_id)
            try:
                cached_data = self.redis_service.get_sync(cache_key)
                if cached_data:
                    logger.info(f"Found cached result for simulation {simulation_id}")
                    result_dict = json.loads(cached_data)
                    return MonteCarloSimulationResult(**result_dict)
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
                # Continue with the calculation if cache check fails
        
        try:
            logger.info(f"Starting Monte Carlo simulation {simulation_id} with {request.num_simulations} iterations")
            
            # Update status to running
            result.status = SimulationStatus.RUNNING
            
            # Generate correlated random variables for all simulations
            start_time = time.time()
            variables_matrix = self._generate_correlated_variables(request)
            
            # Initialize result arrays
            all_metrics = {}
            all_cashflows = []
            
            # Run the simulations
            for i in range(request.num_simulations):
                # Extract the variables for this simulation
                simulation_variables = {
                    var.name: variables_matrix[var.name][i] 
                    for var in request.variables
                }
                
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
            
            # Calculate summary statistics
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
                    str(p): float(np.percentile(values, p * 100))
                    for p in request.percentiles
                }
            
            # Update result
            result.status = SimulationStatus.COMPLETED
            result.end_time = datetime.now()
            result.execution_time_seconds = time.time() - start_time
            
            # Add detailed paths if requested
            if request.include_detailed_paths:
                result.detailed_paths = all_cashflows
            
            # Cache the result if enabled (using synchronous Redis client)
            if use_cache:
                try:
                    cache_key = self._generate_cache_key(request, user_id)
                    result_json = result.json()
                    ttl = settings.CACHE_TTL_SECONDS or 3600  # Default to 1 hour
                    self.redis_service.set_sync(cache_key, result_json, ttl=ttl)
                except Exception as e:
                    logger.error(f"Error saving result to cache: {str(e)}")
                    # Continue even if caching fails
            
            logger.info(f"Completed Monte Carlo simulation {simulation_id} in {result.execution_time_seconds:.2f} seconds")
            return result
        
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
