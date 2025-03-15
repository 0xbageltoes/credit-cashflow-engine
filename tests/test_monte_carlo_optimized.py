"""
Tests for optimized Monte Carlo simulation implementation

These tests verify that the optimized Monte Carlo simulation functionality works
correctly, including correlation modeling, memory efficiency, and parallel processing.
"""
import asyncio
import pytest
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from app.services.monte_carlo_optimized import (
    generate_correlated_scenarios,
    run_monte_carlo_simulation,
    process_scenario,
    _calculate_scenario_metrics,
    OptimizedMonteCarloService
)

# Sample test data
@pytest.fixture
def sample_loan_data():
    return {
        "principal": 500000,
        "interest_rate": 0.045,
        "term_months": 360,
        "origination_date": "2024-01-01",
        "loan_type": "mortgage",
        "credit_score": 720,
        "ltv_ratio": 0.8,
        "dti_ratio": 0.36,
    }

@pytest.fixture
def sample_economic_factors():
    return {
        "market_rate": 0.04,
        "inflation_rate": 0.02,
        "unemployment_rate": 0.042,
        "gdp_growth": 0.025,
        "house_price_index_growth": 0.03,
    }

@pytest.fixture
def sample_correlation_matrix():
    # 5x5 correlation matrix for the 5 economic factors
    return np.array([
        [1.0,  0.3,  0.2, -0.4,  0.5],
        [0.3,  1.0,  0.6,  0.1,  0.2],
        [0.2,  0.6,  1.0,  0.3, -0.1],
        [-0.4, 0.1,  0.3,  1.0,  0.0],
        [0.5,  0.2, -0.1,  0.0,  1.0],
    ])

@pytest.fixture
def sample_volatilities():
    return {
        "market_rate": 0.1,
        "inflation_rate": 0.05,
        "unemployment_rate": 0.15,
        "gdp_growth": 0.2,
        "house_price_index_growth": 0.12,
    }

def test_scenario_generation(sample_economic_factors, sample_correlation_matrix, sample_volatilities):
    """Test that scenario generation produces the correct number of scenarios with expected properties"""
    num_scenarios = 1000
    seed = 42
    
    # Generate scenarios
    scenarios = generate_correlated_scenarios(
        num_scenarios=num_scenarios,
        base_factors=sample_economic_factors,
        correlation_matrix=sample_correlation_matrix,
        volatilities=sample_volatilities,
        seed=seed
    )
    
    # Check we have the correct number of scenarios
    assert len(scenarios) == num_scenarios
    
    # Check that each scenario has all the factors
    assert all(set(scenario.keys()) == set(sample_economic_factors.keys()) for scenario in scenarios)
    
    # Convert to numpy array for statistical analysis
    scenario_array = np.array([[scenario[k] for k in sample_economic_factors.keys()] for scenario in scenarios])
    
    # Calculate empirical correlations
    empirical_corr = np.corrcoef(scenario_array.T)
    
    # Check that empirical correlations are reasonably close to the specified correlations
    # We can't expect exact matches due to randomness, but they should be close
    assert np.allclose(empirical_corr, sample_correlation_matrix, atol=0.15)
    
    # Make sure unemployment rate is never negative
    unemployment_rates = [scenario["unemployment_rate"] for scenario in scenarios]
    assert all(rate >= 0 for rate in unemployment_rates)
    
    # GDP growth can be negative, so no need to check non-negativity for that

def test_scenario_generation_defaults(sample_economic_factors):
    """Test scenario generation with default values for correlation matrix and volatilities"""
    num_scenarios = 500
    seed = 42
    
    # Generate scenarios with defaults
    scenarios = generate_correlated_scenarios(
        num_scenarios=num_scenarios,
        base_factors=sample_economic_factors,
        seed=seed
    )
    
    # Check we have the correct number of scenarios
    assert len(scenarios) == num_scenarios
    
    # Check that each scenario has all the factors
    assert all(set(scenario.keys()) == set(sample_economic_factors.keys()) for scenario in scenarios)
    
    # Check that factors vary from the base values
    for factor in sample_economic_factors.keys():
        factor_values = [scenario[factor] for scenario in scenarios]
        # At least some values should be different from the base
        assert len(set(factor_values)) > 1

@pytest.mark.asyncio
async def test_process_scenario(sample_loan_data):
    """Test that processing a single scenario works correctly"""
    scenario = {
        "market_rate": 0.04,
        "inflation_rate": 0.02,
    }
    
    # Process the scenario
    npv, irr, duration = await process_scenario(
        loan_data=sample_loan_data,
        scenario=scenario,
        calculation_function=_calculate_scenario_metrics
    )
    
    # Basic validation of results
    assert isinstance(npv, float)
    assert isinstance(irr, float)
    assert isinstance(duration, float)
    
    # Reasonable range checks
    assert -1000000 < npv < 1000000  # Reasonable NPV range
    assert 0 < irr < 0.2  # Reasonable IRR range
    assert 0 < duration < 30  # Reasonable duration range

@pytest.mark.asyncio
async def test_monte_carlo_simulation(sample_loan_data, sample_economic_factors, sample_correlation_matrix, sample_volatilities):
    """Test the full Monte Carlo simulation process"""
    num_scenarios = 200  # Smaller number for testing
    batch_size = 50
    seed = 42
    
    # Run the simulation
    result = await run_monte_carlo_simulation(
        loan_data=sample_loan_data,
        base_economic_factors=sample_economic_factors,
        num_scenarios=num_scenarios,
        correlation_matrix=sample_correlation_matrix,
        volatilities=sample_volatilities,
        batch_size=batch_size,
        seed=seed
    )
    
    # Check result structure
    assert "npvs" in result
    assert "irrs" in result
    assert "durations" in result
    assert "summary" in result
    assert "scenarios" in result
    assert "execution_info" in result
    
    # Check array sizes
    assert len(result["npvs"]) == num_scenarios
    assert len(result["irrs"]) == num_scenarios
    assert len(result["durations"]) == num_scenarios
    
    # Check summary statistics
    summary = result["summary"]
    assert "npv_mean" in summary
    assert "npv_std" in summary
    assert "var_95" in summary
    assert "expected_shortfall_95" in summary
    
    # Check that VaR is less than or equal to the 5th percentile of NPVs
    sorted_npvs = sorted(result["npvs"])
    percentile_5_index = int(0.05 * num_scenarios)
    assert summary["var_95"] <= sorted_npvs[percentile_5_index]
    
    # Check execution info
    assert result["execution_info"]["num_scenarios"] == num_scenarios
    assert result["execution_info"]["batch_size"] == batch_size
    assert result["execution_info"]["execution_time_seconds"] > 0

@pytest.mark.asyncio
async def test_optimized_monte_carlo_service(sample_loan_data, sample_economic_factors):
    """Test the OptimizedMonteCarloService class"""
    # Create service
    service = OptimizedMonteCarloService(max_workers=4)
    
    # Run simulation through service
    result = await service.run_simulation(
        loan_data=sample_loan_data,
        base_economic_factors=sample_economic_factors,
        num_scenarios=100,
        seed=42
    )
    
    # Basic validation of result
    assert "npvs" in result
    assert "summary" in result
    assert len(result["npvs"]) == 100
    
    # Test scenario generation directly
    scenarios = service.generate_scenarios(
        base_factors=sample_economic_factors,
        num_scenarios=50,
        seed=42
    )
    
    assert len(scenarios) == 50
    assert all(set(scenario.keys()) == set(sample_economic_factors.keys()) for scenario in scenarios)

@pytest.mark.asyncio
async def test_monte_carlo_simulation_invalid_correlation_matrix(sample_loan_data, sample_economic_factors):
    """Test handling of an invalid correlation matrix"""
    # Create an invalid correlation matrix (not positive definite)
    invalid_corr_matrix = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]
    ])
    
    # Create volatilities dictionary with same keys as economic factors
    volatilities = {k: 0.1 for k in list(sample_economic_factors.keys())[:3]}
    
    # Only use first 3 economic factors to match matrix size
    reduced_factors = {k: v for i, (k, v) in enumerate(sample_economic_factors.items()) if i < 3}
    
    # Run simulation with invalid matrix - should adjust automatically
    result = await run_monte_carlo_simulation(
        loan_data=sample_loan_data,
        base_economic_factors=reduced_factors,
        num_scenarios=100,
        correlation_matrix=invalid_corr_matrix,
        volatilities=volatilities,
        batch_size=50,
        seed=42
    )
    
    # Check that it completed successfully
    assert len(result["npvs"]) == 100
    assert "summary" in result

def test_correlation_matrix_positive_definite_adjustment():
    """Test that invalid correlation matrices are adjusted to be positive definite"""
    # Create a correlation matrix that is not positive definite
    invalid_matrix = np.array([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, 0.95],
        [0.95, 0.95, 1.0]
    ])
    
    # Create economic factors
    factors = {"factor1": 0.1, "factor2": 0.2, "factor3": 0.3}
    
    # This should not raise an exception, as the matrix should be adjusted
    scenarios = generate_correlated_scenarios(
        num_scenarios=50,
        base_factors=factors,
        correlation_matrix=invalid_matrix,
        seed=42
    )
    
    # Check we have the correct number of scenarios
    assert len(scenarios) == 50
    
    # Check that each scenario has all the factors
    assert all(set(scenario.keys()) == set(factors.keys()) for scenario in scenarios)
"""
