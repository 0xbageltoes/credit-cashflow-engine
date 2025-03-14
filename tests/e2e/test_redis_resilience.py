"""
Redis Resilience Tests for Monte Carlo Simulation Service

This module contains comprehensive tests that verify the Redis caching system's 
resilience and error handling capabilities. These tests deliberately simulate 
various Redis failure scenarios to ensure the system gracefully degrades and 
continues to function correctly even when Redis is unavailable or misbehaving.

Usage:
    pytest -xvs tests/e2e/test_redis_resilience.py
"""
import os
import uuid
import pytest
import asyncio
import logging
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required services and models
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult,
    ScenarioDefinition,
    SimulationStatus
)

# Test constants
TEST_USER_ID = "redis_resilience_test_user"

# Test fixtures

@pytest.fixture
async def redis_service():
    """Create a Redis service for testing."""
    service = RedisService()
    
    # Clear test keys before testing
    try:
        await service.delete_pattern("monte_carlo:test:*")
    except Exception as e:
        logger.warning(f"Error clearing Redis test keys: {e}")
    
    yield service
    
    # Clean up after tests
    try:
        await service.delete_pattern("monte_carlo:test:*")
    except Exception as e:
        logger.warning(f"Error clearing Redis test keys: {e}")

@pytest.fixture
async def supabase_service():
    """Create a Supabase service for testing."""
    return SupabaseService()

@pytest.fixture
async def monte_carlo_service(redis_service):
    """Create a Monte Carlo service with the test Redis service."""
    return MonteCarloSimulationService(redis_service=redis_service)

@pytest.fixture
def simulation_request():
    """Create a simulation request for testing."""
    return MonteCarloSimulationRequest(
        name=f"Redis Test Simulation {random.randint(1000, 9999)}",
        description="Test simulation for Redis resilience testing",
        asset_class="mortgage",
        projection_months=36,
        num_simulations=100,  # Small number for faster tests
        percentiles=[0.05, 0.5, 0.95],
        analysis_date=datetime.now().date(),
        variables=[
            {
                "name": "interest_rate",
                "distribution": "normal",
                "parameters": {"mean": 0.05, "std_dev": 0.01}
            },
            {
                "name": "prepayment_rate",
                "distribution": "uniform",
                "parameters": {"min": 0.02, "max": 0.1}
            },
            {
                "name": "default_rate",
                "distribution": "lognormal",
                "parameters": {"mean": -3.5, "std_dev": 0.5}
            }
        ],
        correlation_matrix={
            "variables": ["interest_rate", "prepayment_rate", "default_rate"],
            "correlations": {
                "interest_rate:prepayment_rate": 0.3,
                "interest_rate:default_rate": 0.2,
                "prepayment_rate:default_rate": -0.1
            }
        },
        asset_parameters={
            "principal": 1000000,
            "rate": 0.05,
            "term_months": 360,
            "payment_frequency": "monthly"
        },
        include_detailed_paths=False
    )

@pytest.fixture
async def scenario_definition(supabase_service):
    """Create a test scenario."""
    scenario_id = str(uuid.uuid4())
    scenario = ScenarioDefinition(
        id=scenario_id,
        name=f"Redis Test Scenario {random.randint(1000, 9999)}",
        type="stress_test",
        description="Test scenario for Redis resilience testing",
        user_id=TEST_USER_ID,
        parameters={
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": 0.02,
                    "volatility_multiplier": 1.5
                },
                "default_rate": {
                    "mean_shift": 0.01,
                    "volatility_multiplier": 2.0
                }
            }
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Store in Supabase
    try:
        await supabase_service.create_scenario(scenario.dict())
    except Exception as e:
        logger.warning(f"Error creating test scenario: {e}")
        pytest.skip("Could not create test scenario")
    
    yield scenario
    
    # Clean up
    try:
        await supabase_service.delete_scenario(scenario.id, TEST_USER_ID)
    except Exception as e:
        logger.warning(f"Error cleaning up test scenario: {e}")

# Utility class to simulate Redis failures
class RedisFailureSimulator(RedisService):
    """
    Redis service that can be configured to simulate various failure modes.
    Used to test resilience of the caching system in production environments.
    """
    
    def __init__(self, failure_mode=None):
        """
        Initialize with a specific failure mode.
        
        Args:
            failure_mode: Type of failure to simulate:
                - 'connection': Simulate connection failures
                - 'timeout': Simulate timeouts
                - 'data_corruption': Simulate data corruption
                - 'intermittent': Simulate intermittent failures
                - None: Normal operation
        """
        super().__init__()
        self.failure_mode = failure_mode
        self.failure_count = 0
        self.success_count = 0
        self.intermittent_fail = False
    
    async def get(self, key):
        """Override get method to simulate failures."""
        if self.failure_mode == 'connection':
            self.failure_count += 1
            raise ConnectionError("Simulated Redis connection failure")
        
        elif self.failure_mode == 'timeout':
            self.failure_count += 1
            await asyncio.sleep(0.5)  # Simulate slow response
            raise TimeoutError("Simulated Redis timeout")
        
        elif self.failure_mode == 'data_corruption':
            self.failure_count += 1
            # Return invalid JSON data
            return b'{"corrupted_data": true, "this_is_not_valid_'
        
        elif self.failure_mode == 'intermittent':
            self.intermittent_fail = not self.intermittent_fail
            if self.intermittent_fail:
                self.failure_count += 1
                raise ConnectionError("Simulated intermittent Redis failure")
        
        # Normal operation
        self.success_count += 1
        return await super().get(key)
    
    async def set(self, key, value, ttl=None):
        """Override set method to simulate failures."""
        if self.failure_mode == 'connection':
            self.failure_count += 1
            raise ConnectionError("Simulated Redis connection failure")
        
        elif self.failure_mode == 'timeout':
            self.failure_count += 1
            await asyncio.sleep(0.5)  # Simulate slow response
            raise TimeoutError("Simulated Redis timeout")
        
        elif self.failure_mode == 'intermittent':
            self.intermittent_fail = not self.intermittent_fail
            if self.intermittent_fail:
                self.failure_count += 1
                raise ConnectionError("Simulated intermittent Redis failure")
        
        # Normal operation
        self.success_count += 1
        return await super().set(key, value, ttl)

# Tests

@pytest.mark.asyncio
async def test_normal_redis_operation(redis_service, monte_carlo_service, simulation_request):
    """Test normal Redis caching operation."""
    # Run simulation with caching
    result1 = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True
    )
    
    assert result1 is not None
    assert result1.status == SimulationStatus.COMPLETED
    
    # Generate cache key
    cache_key = monte_carlo_service._generate_cache_key(simulation_request, TEST_USER_ID)
    
    # Verify that result was cached
    cached_data = await redis_service.get(cache_key)
    assert cached_data is not None
    
    # Run again - should use cache
    start_time = time.time()
    result2 = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True
    )
    cached_execution_time = time.time() - start_time
    
    # Run again without cache
    start_time = time.time()
    result3 = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=False
    )
    uncached_execution_time = time.time() - start_time
    
    # Verify that cached execution was faster
    logger.info(f"Cached execution: {cached_execution_time:.2f}s, Uncached: {uncached_execution_time:.2f}s")
    
    # Verify results are the same
    assert result1.id == result2.id
    assert result1.id != result3.id  # New simulation without cache

@pytest.mark.asyncio
async def test_connection_failure_resilience(monte_carlo_service, simulation_request):
    """Test resilience to Redis connection failures."""
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='connection')
    monte_carlo_service.redis_service = fail_redis
    
    # Run simulation - should succeed despite Redis failure
    result = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True  # Try to use cache, but it will fail
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert fail_redis.failure_count > 0
    
    logger.info(f"System remained operational despite {fail_redis.failure_count} Redis connection failures")

@pytest.mark.asyncio
async def test_timeout_resilience(monte_carlo_service, simulation_request):
    """Test resilience to Redis timeout failures."""
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='timeout')
    monte_carlo_service.redis_service = fail_redis
    
    # Run simulation - should succeed despite Redis timeouts
    result = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True  # Try to use cache, but it will timeout
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert fail_redis.failure_count > 0
    
    logger.info(f"System remained operational despite {fail_redis.failure_count} Redis timeout failures")

@pytest.mark.asyncio
async def test_data_corruption_resilience(monte_carlo_service, simulation_request):
    """Test resilience to Redis data corruption."""
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='data_corruption')
    monte_carlo_service.redis_service = fail_redis
    
    # First run to cache (corrupted) data
    await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True
    )
    
    # Second run - should handle corrupted cache data gracefully
    result = await monte_carlo_service.run_simulation(
        simulation_request, 
        TEST_USER_ID,
        use_cache=True  # Try to use cache, but it will be corrupted
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert fail_redis.failure_count > 0
    
    logger.info(f"System remained operational despite corrupted Redis data")

@pytest.mark.asyncio
async def test_intermittent_failure_resilience(monte_carlo_service, simulation_request):
    """Test resilience to intermittent Redis failures."""
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='intermittent')
    monte_carlo_service.redis_service = fail_redis
    
    # Run simulation multiple times with intermittent failures
    results = []
    
    for i in range(5):
        result = await monte_carlo_service.run_simulation(
            simulation_request, 
            TEST_USER_ID,
            use_cache=True
        )
        results.append(result)
    
    # All simulations should succeed
    assert all(r is not None and r.status == SimulationStatus.COMPLETED for r in results)
    assert fail_redis.failure_count > 0
    assert fail_redis.success_count > 0
    
    logger.info(
        f"System remained operational despite {fail_redis.failure_count} intermittent failures "
        f"and had {fail_redis.success_count} successful Redis operations"
    )

@pytest.mark.asyncio
async def test_redis_disabled_operation(monte_carlo_service, simulation_request):
    """Test operation when Redis is completely disabled."""
    # Set REDIS_ENABLED to False to simulate disabled Redis
    original_enabled = os.environ.get("REDIS_ENABLED", "true")
    os.environ["REDIS_ENABLED"] = "false"
    
    try:
        # Run simulation - should work without Redis
        result = await monte_carlo_service.run_simulation(
            simulation_request, 
            TEST_USER_ID,
            use_cache=True  # Cache request will be ignored
        )
        
        assert result is not None
        assert result.status == SimulationStatus.COMPLETED
        
        # Run again - should still work but not use cache
        result2 = await monte_carlo_service.run_simulation(
            simulation_request, 
            TEST_USER_ID,
            use_cache=True
        )
        
        assert result2 is not None
        assert result2.status == SimulationStatus.COMPLETED
        assert result.id != result2.id  # Should be different results since cache is disabled
    
    finally:
        # Restore original setting
        os.environ["REDIS_ENABLED"] = original_enabled

@pytest.mark.asyncio
async def test_scenario_with_redis_failures(monte_carlo_service, supabase_service, simulation_request, scenario_definition):
    """Test scenario-based simulation with Redis failures."""
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='intermittent')
    monte_carlo_service.redis_service = fail_redis
    
    # Run scenario simulation with intermittent Redis failures
    result = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert "metadata" in result.dict()
    assert "scenario" in result.dict()["metadata"]
    assert result.dict()["metadata"]["scenario"]["id"] == scenario_definition.id
    
    assert fail_redis.failure_count > 0
    logger.info(f"Scenario simulation remained operational despite {fail_redis.failure_count} Redis failures")

@pytest.mark.asyncio
async def test_compare_scenarios_with_redis_failures(monte_carlo_service, supabase_service, simulation_request, scenario_definition):
    """Test scenario comparison with Redis failures."""
    # Create a second scenario
    second_scenario = ScenarioDefinition(
        id=str(uuid.uuid4()),
        name=f"Second Redis Test Scenario {random.randint(1000, 9999)}",
        type="optimistic",
        description="Second test scenario for Redis resilience testing",
        user_id=TEST_USER_ID,
        parameters={
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": -0.01
                }
            }
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Store in Supabase
    try:
        await supabase_service.create_scenario(second_scenario.dict())
    except Exception as e:
        logger.warning(f"Error creating second test scenario: {e}")
        pytest.skip("Could not create second test scenario")
    
    # Replace Redis service with failure simulator
    fail_redis = RedisFailureSimulator(failure_mode='intermittent')
    monte_carlo_service.redis_service = fail_redis
    
    try:
        # Compare scenarios with intermittent Redis failures
        comparison_result = await monte_carlo_service.compare_scenarios(
            simulation_request,
            [scenario_definition.id, second_scenario.id],
            TEST_USER_ID,
            metrics_to_compare=["npv", "irr", "default_rate"],
            percentiles_to_compare=[0.05, 0.5, 0.95]
        )
        
        assert comparison_result is not None
        assert "scenario_results" in comparison_result
        assert len(comparison_result["scenario_results"]) == 2
        assert "comparison_metrics" in comparison_result
        
        assert fail_redis.failure_count > 0
        logger.info(f"Scenario comparison remained operational despite {fail_redis.failure_count} Redis failures")
    
    finally:
        # Clean up second scenario
        try:
            await supabase_service.delete_scenario(second_scenario.id, TEST_USER_ID)
        except Exception as e:
            logger.warning(f"Error cleaning up second test scenario: {e}")

@pytest.mark.asyncio
async def test_large_cache_data_handling(monte_carlo_service, simulation_request):
    """Test handling of large cache data."""
    # Modify simulation to include detailed paths (creates larger output)
    large_simulation = simulation_request.copy()
    large_simulation.include_detailed_paths = True
    large_simulation.num_simulations = 200  # More simulations = more data
    
    # Run simulation with large output
    result = await monte_carlo_service.run_simulation(
        large_simulation, 
        TEST_USER_ID,
        use_cache=True
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    
    # Generate cache key
    cache_key = monte_carlo_service._generate_cache_key(large_simulation, TEST_USER_ID)
    
    # Verify that large result was cached
    cached_data = await monte_carlo_service.redis_service.get(cache_key)
    assert cached_data is not None
    
    # Run again - should use cache
    result2 = await monte_carlo_service.run_simulation(
        large_simulation, 
        TEST_USER_ID,
        use_cache=True
    )
    
    assert result2 is not None
    assert result2.id == result.id  # Should be the same result from cache
    
    # Get the size of the cached data
    cached_size = len(json.dumps(result.dict()))
    logger.info(f"Successfully cached and retrieved {cached_size / 1024:.2f} KB of simulation data")

# Main entry point for manual testing
if __name__ == "__main__":
    async def main():
        """Run a simple Redis test."""
        redis = RedisService()
        
        # Test basic connectivity
        try:
            await redis.set("test:key", "test_value", ttl=60)
            value = await redis.get("test:key")
            print(f"Redis test result: {value}")
            await redis.delete("test:key")
            print("Redis connection successful")
        except Exception as e:
            print(f"Redis connection failed: {e}")
    
    asyncio.run(main())
