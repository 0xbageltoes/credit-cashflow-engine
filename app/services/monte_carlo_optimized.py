"""
Optimized Monte Carlo Simulation Services

This module provides optimized implementations for Monte Carlo simulations
with advanced correlation modeling, memory efficiency, and parallel processing.
It includes:
- Correlation-aware scenario generation
- Memory-efficient simulation processing
- Parallel scenario evaluation 

The optimizations focus on:
1. Proper correlation modeling between economic factors
2. Efficient random number generation
3. Memory efficiency for large simulations
4. Parallel processing for performance
"""
import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from app.models.monte_carlo import (
    MonteCarloSimulationRequest, 
    MonteCarloSimulationResult,
    SimulationStatus,
    SimulationResult,
    DistributionType
)
from app.models.analytics import (
    StatisticalOutputs,
    EconomicFactors,
    MonteCarloResult,
    RiskMetrics
)
from app.core.config import settings
from app.core.monitoring import CalculationTracker
from app.core.exceptions import ValidationError, CalculationError, handle_exceptions

logger = logging.getLogger(__name__)

def generate_correlated_scenarios(
    num_scenarios: int,
    base_factors: Dict[str, float],
    correlation_matrix: Optional[np.ndarray] = None,
    volatilities: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, float]]:
    """Generate correlated economic scenarios
    
    Creates economic scenarios with proper correlation structure between factors.
    Uses Cholesky decomposition to maintain correlations while generating
    random variations.
    
    Args:
        num_scenarios: Number of scenarios to generate
        base_factors: Base economic factors dict
        correlation_matrix: Optional correlation matrix
        volatilities: Optional volatilities dict
        seed: Optional random seed
        
    Returns:
        List of scenario dictionaries
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Extract factor names and values
    factor_names = list(base_factors.keys())
    base_values = np.array([base_factors[name] for name in factor_names])
    
    # Default volatilities if not provided
    if volatilities is None:
        volatilities = {name: 0.2 for name in factor_names}  # 20% default volatility
    
    vol_values = np.array([volatilities[name] for name in factor_names])
    
    # Default correlation matrix if not provided
    if correlation_matrix is None:
        n_factors = len(factor_names)
        # Moderate positive correlation (0.3) between all factors
        correlation_matrix = np.ones((n_factors, n_factors)) * 0.3
        np.fill_diagonal(correlation_matrix, 1.0)
    
    # Ensure correlation matrix is valid
    try:
        # Attempt Cholesky decomposition
        cholesky = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, adjust
        logger.warning("Correlation matrix not positive definite, adjusting...")
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)  # Ensure positive eigenvalues
        correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Try again
        cholesky = np.linalg.cholesky(correlation_matrix)
    
    # Generate uncorrelated random normal variables
    uncorrelated = np.random.normal(0, 1, size=(num_scenarios, len(factor_names)))
    
    # Apply correlation
    correlated = uncorrelated @ cholesky.T
    
    # Apply volatility and base values
    scenarios_array = correlated * vol_values + base_values
    
    # Convert to list of dictionaries
    scenarios = []
    for i in range(num_scenarios):
        scenario = {
            factor_names[j]: max(0, scenarios_array[i, j]) 
            if factor_names[j] != "gdp_growth" else scenarios_array[i, j]
            for j in range(len(factor_names))
        }
        scenarios.append(scenario)
    
    return scenarios

async def process_scenario(
    loan_data: Dict,
    scenario: Dict[str, float],
    calculation_function: callable
) -> Tuple[float, float, float]:
    """Process a single Monte Carlo scenario
    
    Handles a single scenario processing using asyncio to run CPU-bound tasks
    in a separate thread for improved concurrency.
    
    Args:
        loan_data: Loan data dictionary
        scenario: Economic scenario
        calculation_function: Function to calculate scenario metrics
        
    Returns:
        Tuple of (npv, irr, duration)
    """
    # Use asyncio to run CPU-bound task in a separate thread
    result = await asyncio.to_thread(
        calculation_function,
        loan_data,
        scenario
    )
    return result

def _calculate_scenario_metrics(
    loan_data: Dict,
    scenario: Dict[str, float]
) -> Tuple[float, float, float]:
    """Calculate metrics for a scenario (CPU-bound)
    
    Performs the actual calculation of financial metrics based on the loan data
    and economic scenario. This is typically a CPU-intensive operation that 
    should be executed in a separate thread or process.
    
    Args:
        loan_data: Loan data dictionary
        scenario: Economic scenario
        
    Returns:
        Tuple of (npv, irr, duration)
    """
    # Here we'd implement the actual metric calculations
    # based on the loan data and economic scenario
    
    # Adjust interest rate based on scenario
    base_rate = loan_data.get("interest_rate", 0.05)
    market_rate = scenario.get("market_rate", 0.04)
    inflation = scenario.get("inflation_rate", 0.02)
    
    # Blend rates with some factors from scenario
    adjusted_rate = (base_rate * 0.4 + market_rate * 0.6) + max(0, inflation - 0.02)
    
    # Calculate metrics (simplified example)
    npv = loan_data.get("principal", 100000) * (1 - adjusted_rate)
    irr = adjusted_rate * 0.9  # Simplified IRR calculation
    duration = 5.0 * (1 + adjusted_rate * 2)  # Simplified duration calculation
    
    return npv, irr, duration

async def run_monte_carlo_simulation(
    loan_data: Dict,
    base_economic_factors: Dict[str, float],
    num_scenarios: int = 1000,
    correlation_matrix: Optional[np.ndarray] = None,
    volatilities: Optional[Dict[str, float]] = None,
    batch_size: int = 100,
    seed: Optional[int] = None,
    calculation_function: Optional[callable] = None
) -> Dict:
    """Run Monte Carlo simulation with batched processing
    
    Executes a Monte Carlo simulation with optimized memory usage through batched
    processing. Properly handles correlation between economic factors and
    provides comprehensive risk metrics.
    
    Args:
        loan_data: Loan data dictionary
        base_economic_factors: Base economic factors
        num_scenarios: Number of scenarios to run
        correlation_matrix: Optional correlation matrix
        volatilities: Optional volatilities dict
        batch_size: Batch size for processing
        seed: Optional random seed
        calculation_function: Optional custom function for scenario calculations
        
    Returns:
        Dictionary of simulation results
    """
    start_time = time.time()
    
    # Use default calculation function if none provided
    calc_func = calculation_function or _calculate_scenario_metrics
    
    # Generate all scenarios first
    scenarios = generate_correlated_scenarios(
        num_scenarios=num_scenarios,
        base_factors=base_economic_factors,
        correlation_matrix=correlation_matrix,
        volatilities=volatilities,
        seed=seed
    )
    
    # Initialize arrays for results
    npvs = np.zeros(num_scenarios)
    irrs = np.zeros(num_scenarios)
    durations = np.zeros(num_scenarios)
    
    # Process in batches to limit memory usage
    for batch_start in range(0, num_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, num_scenarios)
        batch_size_actual = batch_end - batch_start
        batch_scenarios = scenarios[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1} of {(num_scenarios + batch_size - 1)//batch_size}")
        
        # Process each scenario in batch
        batch_results = await asyncio.gather(*[
            process_scenario(loan_data, scenario, calc_func)
            for scenario in batch_scenarios
        ])
        
        # Store batch results
        for i, (npv, irr, duration) in enumerate(batch_results):
            npvs[batch_start + i] = npv
            irrs[batch_start + i] = irr
            durations[batch_start + i] = duration
    
    # Calculate risk metrics
    var_95 = np.percentile(npvs, 5)  # 95% VaR is the 5th percentile
    var_99 = np.percentile(npvs, 1)  # 99% VaR is the 1st percentile
    es_95 = npvs[npvs <= var_95].mean() if any(npvs <= var_95) else var_95  # Expected shortfall
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Format results
    result = {
        "npvs": npvs.tolist(),
        "irrs": irrs.tolist(),
        "durations": durations.tolist(),
        "summary": {
            "npv_mean": float(np.mean(npvs)),
            "npv_std": float(np.std(npvs)),
            "npv_min": float(np.min(npvs)),
            "npv_max": float(np.max(npvs)),
            "irr_mean": float(np.mean(irrs)),
            "duration_mean": float(np.mean(durations)),
            "var_95": float(var_95),
            "var_99": float(var_99),
            "expected_shortfall_95": float(es_95)
        },
        "scenarios": scenarios,
        "execution_info": {
            "num_scenarios": num_scenarios,
            "batch_size": batch_size,
            "execution_time_seconds": execution_time
        }
    }
    
    logger.info(f"Completed Monte Carlo simulation with {num_scenarios} scenarios in {execution_time:.2f} seconds")
    
    return result

class OptimizedMonteCarloService:
    """
    Service for optimized Monte Carlo simulations
    
    This service provides memory-efficient and performance-optimized methods
    for running Monte Carlo simulations with proper correlation modeling.
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the optimized Monte Carlo service
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or min(32, (asyncio.get_event_loop().get_default_executor()._max_workers))
        self._rng = np.random.default_rng()  # High-quality random number generator
    
    @handle_exceptions
    async def run_simulation(
        self,
        loan_data: Dict,
        base_economic_factors: Dict[str, float],
        num_scenarios: int = 1000,
        correlation_matrix: Optional[np.ndarray] = None,
        volatilities: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        calculation_function: Optional[callable] = None
    ) -> Dict:
        """
        Run an optimized Monte Carlo simulation
        
        Args:
            loan_data: Loan data dictionary
            base_economic_factors: Base economic factors
            num_scenarios: Number of scenarios to run
            correlation_matrix: Optional correlation matrix
            volatilities: Optional volatilities dict
            batch_size: Batch size for processing (defaults to max_workers * 2)
            seed: Optional random seed
            calculation_function: Optional custom function for scenario calculations
            
        Returns:
            Dictionary of simulation results
        """
        with CalculationTracker("Optimized Monte Carlo Simulation"):
            # Set default batch size based on number of workers
            if batch_size is None:
                batch_size = max(100, self.max_workers * 2)
            
            # Run the simulation
            result = await run_monte_carlo_simulation(
                loan_data=loan_data,
                base_economic_factors=base_economic_factors,
                num_scenarios=num_scenarios,
                correlation_matrix=correlation_matrix,
                volatilities=volatilities,
                batch_size=batch_size,
                seed=seed,
                calculation_function=calculation_function
            )
            
            return result
    
    @handle_exceptions
    def generate_scenarios(
        self,
        base_factors: Dict[str, float],
        num_scenarios: int = 1000,
        correlation_matrix: Optional[np.ndarray] = None,
        volatilities: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Generate correlated economic scenarios
        
        Convenience method to access the scenario generation functionality directly.
        
        Args:
            base_factors: Base economic factors
            num_scenarios: Number of scenarios to generate
            correlation_matrix: Optional correlation matrix
            volatilities: Optional volatilities dict
            seed: Optional random seed
            
        Returns:
            List of scenario dictionaries
        """
        return generate_correlated_scenarios(
            num_scenarios=num_scenarios,
            base_factors=base_factors,
            correlation_matrix=correlation_matrix,
            volatilities=volatilities,
            seed=seed
        )
"""
