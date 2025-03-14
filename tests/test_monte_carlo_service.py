"""
Comprehensive tests for the Monte Carlo Simulation Service

These tests validate that the Monte Carlo simulation service:
1. Correctly calculates statistical outputs
2. Properly handles economic factors 
3. Implements robust error handling
4. Uses caching effectively
5. Scales properly with different simulation sizes
6. Handles various edge cases appropriately
"""

import pytest
import json
import time
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.redis_service import RedisService
from app.models.monte_carlo import (
    MonteCarloSimulationRequest, 
    MonteCarloSimulationResult,
    Variable,
    DistributionType, 
    CorrelationMatrix,
    DistributionParameters
)
from app.models.analytics import (
    EconomicFactors,
    StatisticalOutputs
)

# Test data
SAMPLE_CASHFLOWS = [
    {"date": "2023-01-01", "amount": 100.0},
    {"date": "2023-02-01", "amount": 200.0},
    {"date": "2023-03-01", "amount": 300.0},
    {"date": "2023-04-01", "amount": 400.0},
    {"date": "2023-05-01", "amount": 500.0}
]

@pytest.fixture
def redis_service():
    """Mock Redis service fixture"""
    mock_redis = MagicMock(spec=RedisService)
    mock_redis.is_available = True
    mock_redis.get.return_value = None  # Default to cache miss
    mock_redis.set.return_value = True  # Default to successful cache set
    return mock_redis

@pytest.fixture
def monte_carlo_service(redis_service):
    """Monte Carlo service fixture"""
    return MonteCarloSimulationService(redis_service=redis_service)

@pytest.fixture
def simulation_request():
    """Sample simulation request fixture"""
    return MonteCarloSimulationRequest(
        name="Test Simulation",
        description="Test simulation for unit tests",
        num_simulations=100,
        variables=[
            Variable(
                name="default_rate",
                distribution=DistributionType.BETA,
                parameters=DistributionParameters(
                    alpha=2.0,
                    beta=5.0
                )
            ),
            Variable(
                name="recovery_rate",
                distribution=DistributionType.BETA,
                parameters=DistributionParameters(
                    alpha=5.0,
                    beta=2.0
                )
            )
        ],
        correlation_matrix=CorrelationMatrix(
            correlations={
                "default_rate:recovery_rate": -0.7
            }
        ),
        asset_class="consumer_loans",
        asset_parameters={
            "cashflows": SAMPLE_CASHFLOWS,
            "discount_rate": 0.05
        },
        projection_months=12,
        analysis_date=datetime.now(),
        include_detailed_paths=True,
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        random_seed=42
    )

@pytest.mark.asyncio
async def test_run_simulation_basic(monte_carlo_service, simulation_request):
    """Test basic simulation execution"""
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify the result contains expected fields
    assert result.simulation_id is not None
    assert result.status == "completed"
    assert result.num_iterations == 100
    assert result.time_horizon == 12
    assert result.calculation_time > 0
    
    # Verify statistical outputs
    assert result.npv_stats is not None
    assert result.npv_stats.mean is not None
    assert result.npv_stats.std_dev is not None
    assert result.npv_stats.percentiles is not None
    assert "50" in result.npv_stats.percentiles  # Median should be present
    
    # Verify paths are included
    assert result.simulation_paths is not None
    assert len(result.simulation_paths) == 100  # Should match num_simulations

@pytest.mark.asyncio
async def test_run_enhanced_simulation(monte_carlo_service, simulation_request):
    """Test enhanced simulation with additional statistics"""
    # Add economic factors
    simulation_request.economic_factors = EconomicFactors(
        inflation_rate=0.03,
        unemployment_rate=0.05,
        housing_price_index=0.02,
        interest_rate_environment="neutral"
    )
    
    # Run the enhanced simulation
    result = await monte_carlo_service.run_enhanced_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify basic result structure
    assert result.simulation_id is not None
    assert result.status == "completed"
    
    # Verify enhanced statistical outputs
    assert result.npv_stats is not None
    assert result.npv_stats.mean is not None
    assert result.npv_stats.median is not None
    assert result.npv_stats.std_dev is not None
    assert result.npv_stats.skewness is not None  # Enhanced stat
    assert result.npv_stats.kurtosis is not None  # Enhanced stat
    assert result.npv_stats.var_95 is not None  # Enhanced stat
    assert result.npv_stats.cvar_95 is not None  # Enhanced stat
    
    # Verify economic factor effects are included
    assert result.economic_factor_effects is not None
    assert "inflation_rate" in result.economic_factor_effects

@pytest.mark.asyncio
async def test_cache_hit(monte_carlo_service, redis_service, simulation_request):
    """Test cache hit functionality"""
    # Setup mock for cache hit
    cached_result = MonteCarloSimulationResult(
        simulation_id="cached_sim_123",
        status="completed",
        num_iterations=100,
        time_horizon=12,
        calculation_time=0.5,
        npv_stats=StatisticalOutputs(
            mean=1000.0,
            median=950.0,
            std_dev=200.0,
            min_value=500.0,
            max_value=1500.0,
            percentiles={"50": 950.0}
        )
    )
    
    # Convert to JSON for Redis mock
    cached_json = json.dumps(
        cached_result.model_dump() if hasattr(cached_result, 'model_dump') else 
        cached_result.dict()
    )
    
    # Setup Redis mock to return the cached result
    redis_service.get.return_value = cached_json
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=True
    )
    
    # Verify we got the cached result
    assert result.simulation_id == "cached_sim_123"
    assert result.npv_stats.mean == 1000.0
    assert result.cache_hit == True  # Should indicate cache hit
    
    # Verify Redis was called correctly
    redis_service.get.assert_called_once()
    redis_service.set.assert_not_called()  # Should not set cache on a hit

@pytest.mark.asyncio
async def test_cache_miss_then_set(monte_carlo_service, redis_service, simulation_request):
    """Test cache miss followed by cache set"""
    # Setup Redis mock for cache miss but successful set
    redis_service.get.return_value = None
    redis_service.set.return_value = True
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=True
    )
    
    # Verify cache behavior
    assert result.cache_hit == False  # Should indicate cache miss
    
    # Verify Redis was called correctly
    redis_service.get.assert_called_once()
    redis_service.set.assert_called_once()  # Should set cache after calculation

@pytest.mark.asyncio
async def test_disabled_cache(monte_carlo_service, redis_service, simulation_request):
    """Test with caching disabled"""
    # Run the simulation with cache disabled
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Verify cache behavior
    assert result.cache_hit == False  # Should indicate no cache hit
    
    # Verify Redis was not called
    redis_service.get.assert_not_called()
    redis_service.set.assert_not_called()

@pytest.mark.asyncio
async def test_redis_unavailable(monte_carlo_service, redis_service, simulation_request):
    """Test resilience when Redis is unavailable"""
    # Setup Redis mock to be unavailable
    redis_service.is_available = False
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=True  # Try to use cache
    )
    
    # Verify the service still worked without Redis
    assert result.simulation_id is not None
    assert result.status == "completed"
    assert result.cache_hit == False
    
    # Verify Redis was not called when unavailable
    redis_service.get.assert_not_called()
    redis_service.set.assert_not_called()

@pytest.mark.asyncio
async def test_error_handling_invalid_parameters(monte_carlo_service, simulation_request):
    """Test error handling for invalid parameters"""
    # Create invalid parameters (negative alpha)
    simulation_request.variables[0].parameters = DistributionParameters(
        alpha=-1.0,  # Invalid alpha
        beta=5.0
    )
    
    # Run the simulation and expect error handling
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify error is captured properly
    assert result.status == "error"
    assert result.error is not None
    assert "invalid parameters" in result.error.lower()

@pytest.mark.asyncio
async def test_error_handling_correlation_matrix(monte_carlo_service, simulation_request):
    """Test error handling for invalid correlation matrix"""
    # Create invalid correlation (value outside -1 to 1)
    simulation_request.correlation_matrix = CorrelationMatrix(
        correlations={
            "default_rate:recovery_rate": -1.5  # Invalid correlation
        }
    )
    
    # Run the simulation and expect error handling
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify error is captured properly
    assert result.status == "error"
    assert result.error is not None
    assert "correlation" in result.error.lower()

@pytest.mark.asyncio
async def test_simulation_paths_count(monte_carlo_service, simulation_request):
    """Test that simulation paths match the requested count"""
    # Set specific number of simulations
    simulation_request.num_simulations = 50
    simulation_request.include_detailed_paths = True
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify path count
    assert len(result.simulation_paths) == 50

@pytest.mark.asyncio
async def test_exclude_paths(monte_carlo_service, simulation_request):
    """Test option to exclude detailed paths"""
    # Set to exclude paths
    simulation_request.include_detailed_paths = False
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify paths are not included
    assert result.simulation_paths is None or len(result.simulation_paths) == 0

@pytest.mark.asyncio
async def test_reproducibility_with_seed(monte_carlo_service, simulation_request):
    """Test that results are reproducible with the same seed"""
    # Set specific seed
    simulation_request.random_seed = 12345
    
    # Run first simulation
    result1 = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False  # Ensure we're not using cache
    )
    
    # Run second simulation with same seed
    result2 = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False  # Ensure we're not using cache
    )
    
    # Verify results are the same
    assert result1.npv_stats.mean == result2.npv_stats.mean
    assert result1.npv_stats.median == result2.npv_stats.median
    assert result1.npv_stats.std_dev == result2.npv_stats.std_dev

@pytest.mark.asyncio
async def test_different_seeds_different_results(monte_carlo_service, simulation_request):
    """Test that different seeds produce different results"""
    # First simulation with seed 12345
    simulation_request.random_seed = 12345
    result1 = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False  # Ensure we're not using cache
    )
    
    # Second simulation with different seed
    simulation_request.random_seed = 54321
    result2 = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False  # Ensure we're not using cache
    )
    
    # Verify results are different
    # Note: There's a small chance they could be the same, but very unlikely
    assert result1.npv_stats.mean != result2.npv_stats.mean

@pytest.mark.asyncio
async def test_performance_scaling(monte_carlo_service, simulation_request):
    """Test performance scaling with simulation size"""
    # Small simulation
    simulation_request.num_simulations = 10
    start_time = time.time()
    small_result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    small_duration = time.time() - start_time
    
    # Larger simulation
    simulation_request.num_simulations = 100
    start_time = time.time()
    large_result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    large_duration = time.time() - start_time
    
    # Verify larger simulation took longer but not excessively so
    assert large_duration > small_duration
    # Should scale somewhat linearly (allow some overhead)
    assert large_duration < (small_duration * 15)  # Allow for some non-linearity

@pytest.mark.asyncio
async def test_percentile_calculations(monte_carlo_service, simulation_request):
    """Test that percentiles are calculated correctly"""
    # Set specific percentiles to calculate
    simulation_request.percentiles = [0.05, 0.5, 0.95]
    
    # Run the simulation
    result = await monte_carlo_service.run_simulation(
        request=simulation_request,
        user_id="test_user"
    )
    
    # Verify percentiles are correctly calculated
    percentiles = result.npv_stats.percentiles
    assert "5" in percentiles
    assert "50" in percentiles
    assert "95" in percentiles
    
    # Verify order is correct
    assert percentiles["5"] <= percentiles["50"] <= percentiles["95"]

@pytest.mark.asyncio
async def test_concurrent_simulations(monte_carlo_service, simulation_request):
    """Test running multiple simulations concurrently"""
    # Run multiple simulations concurrently
    tasks = []
    for i in range(3):
        simulation_request.random_seed = 1000 + i  # Different seed each time
        task = monte_carlo_service.run_simulation(
            request=simulation_request,
            user_id=f"test_user_{i}",
            use_cache=False
        )
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # Verify all completed successfully
    for result in results:
        assert result.status == "completed"
        assert result.npv_stats is not None
    
    # Verify they produced different results
    means = [result.npv_stats.mean for result in results]
    assert len(set(means)) == 3  # All should be different

@pytest.mark.asyncio
async def test_economic_factors_sensitivity(monte_carlo_service, simulation_request):
    """Test sensitivity to economic factors"""
    # Base case - no economic factors
    base_result = await monte_carlo_service.run_enhanced_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Run with high inflation
    simulation_request.economic_factors = EconomicFactors(
        inflation_rate=0.1,  # 10% inflation
        unemployment_rate=0.05,
        housing_price_index=0.0,
        interest_rate_environment="neutral"
    )
    
    inflation_result = await monte_carlo_service.run_enhanced_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Run with high unemployment
    simulation_request.economic_factors = EconomicFactors(
        inflation_rate=0.03,
        unemployment_rate=0.15,  # 15% unemployment
        housing_price_index=0.0,
        interest_rate_environment="neutral"
    )
    
    unemployment_result = await monte_carlo_service.run_enhanced_simulation(
        request=simulation_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Verify economic factors had different effects
    assert base_result.npv_stats.mean != inflation_result.npv_stats.mean
    assert base_result.npv_stats.mean != unemployment_result.npv_stats.mean
    assert inflation_result.npv_stats.mean != unemployment_result.npv_stats.mean

# Integration tests with actual Redis (requires Redis to be running)
@pytest.mark.integration
@pytest.mark.skipif("not os.environ.get('RUN_INTEGRATION_TESTS')")
@pytest.mark.asyncio
async def test_redis_integration():
    """Integration test with real Redis"""
    # Create real services
    redis_service = RedisService()
    monte_carlo_service = MonteCarloSimulationService(redis_service=redis_service)
    
    # Create test request
    request = MonteCarloSimulationRequest(
        name="Redis Integration Test",
        description="Testing with real Redis instance",
        num_simulations=50,
        variables=[
            Variable(
                name="default_rate",
                distribution=DistributionType.NORMAL,
                parameters=DistributionParameters(
                    mean=0.05,
                    std_dev=0.01
                )
            )
        ],
        asset_class="mortgage",
        asset_parameters={
            "cashflows": SAMPLE_CASHFLOWS,
            "discount_rate": 0.04
        },
        projection_months=60,
        random_seed=42
    )
    
    # First run - should be a cache miss
    result1 = await monte_carlo_service.run_simulation(
        request=request,
        user_id="integration_test",
        use_cache=True
    )
    assert result1.cache_hit == False
    
    # Second run - should be a cache hit
    result2 = await monte_carlo_service.run_simulation(
        request=request,
        user_id="integration_test",
        use_cache=True
    )
    assert result2.cache_hit == True
    assert result2.simulation_id == result1.simulation_id
    
    # Clear cache
    cache_key = monte_carlo_service._generate_cache_key(request, "integration_test")
    await redis_service.delete(cache_key)
    
    # Verify cache miss after clearing
    result3 = await monte_carlo_service.run_simulation(
        request=request,
        user_id="integration_test",
        use_cache=True
    )
    assert result3.cache_hit == False
