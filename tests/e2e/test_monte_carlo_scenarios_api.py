"""
End-to-End Tests for Monte Carlo Scenario API Integration

This module contains realistic test cases that simulate the frontend application's interaction
with the Monte Carlo scenario backend using actual services (no mocks). These tests use
the real Supabase database, Redis cache, and service layer to verify the entire system
works correctly in a production-like environment.

Usage:
    pytest -xvs tests/e2e/test_monte_carlo_scenarios_api.py
"""
import os
import pytest
import asyncio
import json
import uuid
import time
import random
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the services directly for testing
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.supabase_service import SupabaseService
from app.services.redis_service import RedisService
from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult,
    ScenarioDefinition,
    SimulationStatus
)

# Test constants
TEST_USER_EMAIL = "test_user@example.com"
TEST_USER_PASSWORD = "SecureTestPassword123!"
TEST_USER_ID = TEST_USER_EMAIL  # Using email as user ID for testing

# Test fixtures for services
@pytest.fixture
async def redis_service():
    """Get a real Redis service instance."""
    service = RedisService()
    
    # Clear test-related keys
    try:
        await service.delete_pattern("monte_carlo:test*")
    except Exception as e:
        logger.warning(f"Error clearing Redis test keys: {e}")
    
    return service

@pytest.fixture
async def supabase_service():
    """Get a real Supabase service instance."""
    return SupabaseService()

@pytest.fixture
async def monte_carlo_service(redis_service):
    """Get a Monte Carlo service with the real Redis service."""
    return MonteCarloSimulationService(redis_service=redis_service)

# Test fixture for scenario creation
@pytest.fixture
async def test_scenario(supabase_service):
    """Create a test scenario that will be used and then cleaned up."""
    scenario_id = str(uuid.uuid4())
    scenario = ScenarioDefinition(
        id=scenario_id,
        name=f"E2E Test Scenario {random.randint(1000, 9999)}",
        type="stress_test",
        description="Test scenario for E2E API testing",
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
            },
            "correlation_modifiers": {
                "interest_rate:default_rate": 0.3
            },
            "additional_parameters": {
                "stress_level": "high",
                "description": "High interest rate environment",
                "tags": ["stress_test", "high_rate", "e2e_test"]
            }
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Create the scenario using direct service call
    try:
        await supabase_service.create_scenario(scenario.dict(), TEST_USER_ID)
    except Exception as e:
        logger.error(f"Failed to create test scenario: {e}")
        pytest.skip("Could not create test scenario")
    
    yield scenario
    
    # Clean up the scenario after tests
    try:
        await supabase_service.delete_scenario(scenario.id, TEST_USER_ID)
    except Exception as e:
        logger.warning(f"Failed to delete test scenario {scenario.id}: {e}")

# Test fixture for simulation request
@pytest.fixture
def simulation_request():
    """Create a realistic simulation request similar to what the frontend would send."""
    return MonteCarloSimulationRequest(
        name=f"E2E Test Simulation {random.randint(1000, 9999)}",
        description="Test simulation for E2E API testing",
        asset_class="mortgage",
        projection_months=36,
        num_simulations=100,  # Use smaller number for faster tests
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95],
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
        include_detailed_paths=False  # Keeping response size smaller for tests
    )

# Test creating a scenario
@pytest.mark.asyncio
async def test_create_scenario(supabase_service):
    """Test creating a new scenario through the service."""
    scenario_id = str(uuid.uuid4())
    scenario = {
        "id": scenario_id,
        "name": f"API Create Test Scenario {random.randint(1000, 9999)}",
        "type": "baseline",
        "description": "Test scenario creation via service",
        "user_id": TEST_USER_ID,
        "parameters": {
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": 0.01
                }
            }
        }
    }
    
    # Create scenario using service
    try:
        result = await supabase_service.create_scenario(scenario, TEST_USER_ID)
        assert result is not None
        assert "id" in result
    except Exception as e:
        pytest.fail(f"Failed to create scenario: {e}")
    
    # Clean up
    await supabase_service.delete_scenario(scenario_id, TEST_USER_ID)

# Test retrieving scenarios
@pytest.mark.asyncio
async def test_list_scenarios(supabase_service, test_scenario):
    """Test retrieving scenarios through the service."""
    scenarios = await supabase_service.list_scenarios(user_id=TEST_USER_ID)
    
    assert scenarios is not None
    assert isinstance(scenarios, list)
    
    # Verify our test scenario is in the list
    scenario_ids = [s.get("id") for s in scenarios]
    assert test_scenario.id in scenario_ids

# Test retrieving a specific scenario
@pytest.mark.asyncio
async def test_get_scenario(supabase_service, test_scenario):
    """Test retrieving a specific scenario through the service."""
    scenario = await supabase_service.get_scenario(test_scenario.id, TEST_USER_ID)
    
    assert scenario is not None
    assert scenario.get("id") == test_scenario.id
    assert scenario.get("name") == test_scenario.name
    assert "parameters" in scenario
    assert "risk_factor_modifiers" in scenario["parameters"]

# Test updating a scenario
@pytest.mark.asyncio
async def test_update_scenario(supabase_service, test_scenario):
    """Test updating an existing scenario through the service."""
    # Update scenario data
    updated_data = test_scenario.dict()
    updated_data["name"] = f"Updated E2E Test Scenario {random.randint(1000, 9999)}"
    updated_data["parameters"]["risk_factor_modifiers"]["interest_rate"]["mean_shift"] = 0.03
    
    # Update scenario using service
    result = await supabase_service.update_scenario(updated_data, TEST_USER_ID)
    
    assert result is not None
    assert result.get("name") == updated_data["name"]
    assert result["parameters"]["risk_factor_modifiers"]["interest_rate"]["mean_shift"] == 0.03

# Test running a simulation with a scenario
@pytest.mark.asyncio
async def test_run_simulation_with_scenario(monte_carlo_service, supabase_service, test_scenario, simulation_request):
    """Test running a simulation with a specific scenario applied."""
    # Run simulation with scenario
    result = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        test_scenario.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert result.metadata is not None
    assert "scenario" in result.metadata
    assert result.metadata["scenario"]["id"] == test_scenario.id
    
    # Verify that the scenario was actually applied
    assert "summary_statistics" in result.dict()
    
    return result.id

# Test running an async simulation with a scenario
@pytest.mark.asyncio
async def test_run_async_simulation_with_scenario(monte_carlo_service, supabase_service, test_scenario, simulation_request):
    """Test running an async simulation with a specific scenario applied."""
    # Using the internal _run_async method to simulate an async task
    # This would normally be done by a Celery worker, but for testing we can do it directly
    task_id = str(uuid.uuid4())
    
    # Start the task
    asyncio.create_task(
        monte_carlo_service._run_async_scenario_simulation(
            simulation_request,
            test_scenario.id,
            TEST_USER_ID,
            task_id,
            use_cache=False,
            supabase_service=supabase_service
        )
    )
    
    # Poll for completion
    max_wait = 120  # Maximum wait time in seconds
    poll_interval = 5  # Check every 5 seconds
    total_wait = 0
    result = None
    
    while total_wait < max_wait:
        # Check for task completion
        status_data = await supabase_service.get_simulation_task_status(task_id)
        
        if status_data and status_data.get("status") in ["COMPLETED", "FAILED"]:
            logger.info(f"Simulation task {task_id} finished with status: {status_data.get('status')}")
            if status_data.get("status") == "COMPLETED":
                result = await supabase_service.get_simulation_result(status_data.get("result_id"))
                break
            else:
                pytest.fail(f"Simulation failed: {status_data.get('error')}")
        
        logger.info(f"Simulation task {task_id} still running")
        await asyncio.sleep(poll_interval)
        total_wait += poll_interval
    
    assert result is not None, f"Simulation did not complete within the expected time"
    assert "metadata" in result
    assert "scenario" in result["metadata"]
    assert result["metadata"]["scenario"]["id"] == test_scenario.id

# Test comparing scenarios
@pytest.mark.asyncio
async def test_compare_scenarios(monte_carlo_service, supabase_service, test_scenario, simulation_request):
    """Test comparing multiple scenarios."""
    # Create a second scenario for comparison
    second_scenario = ScenarioDefinition(
        id=str(uuid.uuid4()),
        name=f"Comparison Test Scenario {random.randint(1000, 9999)}",
        type="optimistic",
        description="Second test scenario for comparison testing",
        user_id=TEST_USER_ID,
        parameters={
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": -0.01,
                    "volatility_multiplier": 0.8
                },
                "default_rate": {
                    "mean_shift": -0.02,
                    "volatility_multiplier": 0.7
                }
            }
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Create the second scenario
    try:
        await supabase_service.create_scenario(second_scenario.dict(), TEST_USER_ID)
    except Exception as e:
        pytest.fail(f"Failed to create second scenario: {e}")
    
    # Compare scenarios
    comparison_result = await monte_carlo_service.compare_scenarios(
        simulation_request,
        [test_scenario.id, second_scenario.id],
        TEST_USER_ID,
        metrics_to_compare=["npv", "irr", "default_rate", "prepayment_rate"],
        percentiles_to_compare=[0.05, 0.5, 0.95]
    )
    
    # Verify comparison results
    assert comparison_result is not None
    assert "scenario_results" in comparison_result
    assert len(comparison_result["scenario_results"]) == 2
    assert "comparison_metrics" in comparison_result
    
    # Verify each scenario has results
    scenario_ids = [res["scenario_id"] for res in comparison_result["scenario_results"]]
    assert test_scenario.id in scenario_ids
    assert second_scenario.id in scenario_ids
    
    # Clean up the second scenario
    await supabase_service.delete_scenario(second_scenario.id, TEST_USER_ID)

# Test error handling for non-existent scenario
@pytest.mark.asyncio
async def test_scenario_not_found(monte_carlo_service, supabase_service, simulation_request):
    """Test error handling when a scenario is not found."""
    fake_id = str(uuid.uuid4())
    
    # Try to run simulation with non-existent scenario
    with pytest.raises(ValueError) as excinfo:
        await monte_carlo_service.run_simulation_with_scenario(
            simulation_request,
            fake_id,
            TEST_USER_ID,
            use_cache=True,
            supabase_service=supabase_service
        )
    
    assert "not found" in str(excinfo.value).lower()

# Test validation errors with invalid request
@pytest.mark.asyncio
async def test_validation_error(monte_carlo_service, supabase_service, test_scenario):
    """Test validation error handling with invalid simulation request."""
    # Create an invalid request (missing required fields)
    invalid_request = MonteCarloSimulationRequest(
        name="Invalid Test Simulation",
        description="Invalid simulation for testing",
        asset_class="mortgage",
        projection_months=36,
        # Missing variables
        variables=[],
        num_simulations=100,
        percentiles=[0.05, 0.5, 0.95],
        analysis_date=datetime.now().date()
    )
    
    # Try to apply scenario to invalid request - should raise validation error
    with pytest.raises(Exception) as excinfo:
        await monte_carlo_service.apply_scenario(
            invalid_request,
            test_scenario
        )
    
    error_str = str(excinfo.value).lower()
    assert "invalid" in error_str or "validation" in error_str or "missing" in error_str

# Test authorization errors
@pytest.mark.asyncio
async def test_authorization_error(supabase_service, test_scenario):
    """Test authorization error handling with invalid user."""
    # Try to get scenario with wrong user
    with pytest.raises(Exception) as excinfo:
        await supabase_service.get_scenario(test_scenario.id, "invalid_user_id")
    
    error_str = str(excinfo.value).lower()
    assert "not found" in error_str or "permission" in error_str or "unauthorized" in error_str

# Test Redis caching behavior
@pytest.mark.asyncio
async def test_cache_behavior(monte_carlo_service, supabase_service, test_scenario, simulation_request):
    """Test that caching works properly by running the same simulation twice."""
    # First run - should calculate and cache
    start_time = time.time()
    result1 = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        test_scenario.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    first_run_time = time.time() - start_time
    
    # Generate cache key
    cache_key = monte_carlo_service._generate_cache_key(
        simulation_request, 
        TEST_USER_ID, 
        test_scenario.id
    )
    
    # Verify the result was cached
    cached_data = await monte_carlo_service.redis_service.get(cache_key)
    assert cached_data is not None
    
    # Second run - should use cache
    start_time = time.time()
    result2 = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        test_scenario.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    second_run_time = time.time() - start_time
    
    # Verify results are the same
    assert result1.id == result2.id
    
    # Log timing info, but don't make this a hard assertion as it could be affected
    # by test environment variables
    logger.info(f"First run time: {first_run_time:.2f}s, Second run time: {second_run_time:.2f}s")
    logger.info(f"Cache speedup: {first_run_time / second_run_time if second_run_time > 0 else 'infinite'}x")
    
    # Run again with cache disabled
    result3 = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        test_scenario.id,
        TEST_USER_ID,
        use_cache=False,
        supabase_service=supabase_service
    )
    
    assert result3.id != result1.id  # Should be a new simulation

# Test bulk scenario operations
@pytest.mark.asyncio
async def test_bulk_scenario_operations(supabase_service):
    """Test creating, listing, and deleting multiple scenarios in bulk."""
    # Create multiple scenarios
    scenario_ids = []
    for i in range(3):
        scenario_id = str(uuid.uuid4())
        scenario = {
            "id": scenario_id,
            "name": f"Bulk Test Scenario {i}",
            "type": "bulk_test",
            "description": f"Scenario {i} for bulk testing",
            "user_id": TEST_USER_ID,
            "parameters": {
                "risk_factor_modifiers": {
                    "interest_rate": {
                        "mean_shift": 0.01 * i
                    }
                }
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        result = await supabase_service.create_scenario(scenario, TEST_USER_ID)
        assert result is not None
        scenario_ids.append(scenario_id)
    
    # List all scenarios with a specific type
    scenarios = await supabase_service.list_scenarios(user_id=TEST_USER_ID, type="bulk_test")
    
    assert len([s for s in scenarios if s["type"] == "bulk_test"]) >= 3
    
    # Clean up all test scenarios
    for scenario_id in scenario_ids:
        await supabase_service.delete_scenario(scenario_id, TEST_USER_ID)

# Main cleanup function
def pytest_sessionfinish(session, exitstatus):
    """Clean up any remaining test resources after all tests complete."""
    logger.info("Cleaning up any remaining test resources...")
    
    # Create an event loop for cleanup
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Clean up test data
    async def cleanup():
        try:
            # Initialize services
            supabase = SupabaseService()
            redis = RedisService()
            
            # Clean up test scenarios in Supabase
            scenarios = await supabase.list_scenarios(user_id=TEST_USER_ID)
            for scenario in scenarios:
                if "e2e_test" in scenario.get("description", "").lower() or "test scenario" in scenario.get("name", "").lower():
                    try:
                        await supabase.delete_scenario(scenario["id"], TEST_USER_ID)
                        logger.info(f"Cleaned up test scenario: {scenario['id']}")
                    except Exception as e:
                        logger.warning(f"Failed to delete test scenario {scenario['id']}: {e}")
            
            # Clean up Redis test keys
            try:
                await redis.delete_pattern("monte_carlo:test*")
                logger.info("Cleaned up Redis test keys")
            except Exception as e:
                logger.warning(f"Failed to clean up Redis: {e}")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    loop.run_until_complete(cleanup())
    logger.info("Cleanup complete")
