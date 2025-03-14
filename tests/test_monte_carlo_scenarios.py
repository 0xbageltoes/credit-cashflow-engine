"""
Test cases for Monte Carlo simulation service with scenario support

This module provides comprehensive tests for the Monte Carlo simulation 
scenario functionality, including:
1. Creating and running simulations with scenarios
2. Verifying the correct application of scenarios to variables
3. Testing Redis caching for scenario-based simulations
4. Testing error handling and resilience
5. Testing scenario comparison functionality

The tests use real Redis and Supabase services to verify production readiness.
"""
import asyncio
import json
import os
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult,
    RandomVariable,
    DistributionType,
    CorrelationMatrix,
    ScenarioDefinition,
    SimulationStatus
)
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.supabase_service import SupabaseService
from app.services.redis_service import RedisService

# Test data
TEST_USER_ID = "test_user_1"

@pytest.fixture
def simulation_request():
    """Fixture for a basic simulation request"""
    return MonteCarloSimulationRequest(
        name="Test Simulation",
        description="Test simulation for unit tests",
        asset_class="mortgage",
        projection_months=36,
        num_simulations=100,
        percentiles=[0.05, 0.5, 0.95],
        analysis_date=datetime.now().date(),
        variables=[
            RandomVariable(
                name="interest_rate",
                distribution=DistributionType.NORMAL,
                parameters={"mean": 0.05, "std_dev": 0.01}
            ),
            RandomVariable(
                name="prepayment_rate",
                distribution=DistributionType.UNIFORM,
                parameters={"min": 0.02, "max": 0.1}
            ),
            RandomVariable(
                name="default_rate",
                distribution=DistributionType.LOGNORMAL,
                parameters={"mean": -3.5, "std_dev": 0.5}
            )
        ],
        correlation_matrix=CorrelationMatrix(
            variables=["interest_rate", "prepayment_rate", "default_rate"],
            correlations={
                "interest_rate:prepayment_rate": 0.3,
                "interest_rate:default_rate": 0.2,
                "prepayment_rate:default_rate": -0.1
            }
        ),
        asset_parameters={
            "principal": 1000000,
            "rate": 0.05,
            "term_months": 360,
            "payment_frequency": "monthly"
        },
        include_detailed_paths=False
    )

@pytest.fixture
def scenario_definition():
    """Fixture for a test scenario definition"""
    scenario_id = str(uuid.uuid4())
    return ScenarioDefinition(
        id=scenario_id,
        name="Stress Test Scenario",
        type="stress_test",
        description="High interest rate, high default scenario",
        user_id=TEST_USER_ID,
        parameters={
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": 0.02,
                    "volatility_multiplier": 1.5
                },
                "default_rate": {
                    "mean_shift": 0.5,
                    "volatility_multiplier": 2.0
                }
            },
            "correlation_modifiers": {
                "interest_rate:default_rate": 0.3
            },
            "additional_parameters": {
                "stress_level": "high"
            }
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )

@pytest.fixture
async def redis_service():
    """Fixture for Redis service with proper error handling"""
    service = RedisService()
    
    # Clear the test cache before each test
    try:
        await service.delete_pattern("monte_carlo:test*")
    except Exception as e:
        print(f"Warning: Error clearing Redis cache: {str(e)}")
    
    return service

@pytest.fixture
async def supabase_service():
    """Fixture for Supabase service"""
    return SupabaseService()

@pytest.fixture
async def monte_carlo_service(redis_service):
    """Fixture for Monte Carlo service using the test Redis service"""
    return MonteCarloSimulationService(redis_service=redis_service)

@pytest.mark.asyncio
async def test_apply_scenario(monte_carlo_service, simulation_request, scenario_definition):
    """Test applying a scenario to a simulation request"""
    # Apply the scenario
    modified_request = await monte_carlo_service.apply_scenario(
        simulation_request, 
        scenario_definition
    )
    
    # Check that scenario was applied correctly
    assert modified_request is not None
    assert modified_request.metadata is not None
    assert "applied_scenario" in modified_request.metadata
    assert modified_request.metadata["applied_scenario"]["id"] == scenario_definition.id
    
    # Check variable modifications
    for var in modified_request.variables:
        if var.name == "interest_rate":
            # Should have had mean_shift and volatility_multiplier applied
            assert var.parameters["mean"] == pytest.approx(0.07)  # 0.05 + 0.02
            assert var.parameters["std_dev"] == pytest.approx(0.015)  # 0.01 * 1.5
        elif var.name == "default_rate":
            # Should have had mean_shift and volatility_multiplier applied
            assert var.parameters["mean"] == pytest.approx(-3.0)  # -3.5 + 0.5
            assert var.parameters["std_dev"] == pytest.approx(1.0)  # 0.5 * 2.0
    
    # Check correlation modifications
    if modified_request.correlation_matrix:
        assert modified_request.correlation_matrix.correlations["interest_rate:default_rate"] == pytest.approx(0.5)  # 0.2 + 0.3
    
    # Check additional parameters
    assert modified_request.additional_parameters is not None
    assert "stress_level" in modified_request.additional_parameters
    assert modified_request.additional_parameters["stress_level"] == "high"

@pytest.mark.asyncio
async def test_generate_cache_key(monte_carlo_service, simulation_request, scenario_definition):
    """Test cache key generation for normal and scenario-based simulations"""
    # Generate a normal cache key
    normal_key = monte_carlo_service._generate_cache_key(
        simulation_request, 
        TEST_USER_ID
    )
    assert normal_key is not None
    assert normal_key.startswith("monte_carlo:")
    
    # Generate a scenario-based cache key
    scenario_key = monte_carlo_service._generate_cache_key(
        simulation_request, 
        TEST_USER_ID, 
        scenario_definition.id
    )
    assert scenario_key is not None
    assert scenario_key.startswith("monte_carlo:")
    
    # Keys should be different
    assert normal_key != scenario_key

@pytest.mark.asyncio
async def test_run_simulation_with_scenario(monte_carlo_service, supabase_service, simulation_request, scenario_definition):
    """Test running a simulation with a scenario applied"""
    # Store the scenario in Supabase
    try:
        await supabase_service.create_scenario(scenario_definition.dict())
    except Exception as e:
        pytest.skip(f"Error setting up test scenario in Supabase: {str(e)}")
    
    # Run the simulation with scenario
    result = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    
    # Verify the result
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED
    assert result.metadata is not None
    assert "scenario" in result.metadata
    assert result.metadata["scenario"]["id"] == scenario_definition.id

@pytest.mark.asyncio
async def test_scenario_caching(monte_carlo_service, supabase_service, redis_service, simulation_request, scenario_definition):
    """Test caching for scenario-based simulations"""
    # Store the scenario in Supabase
    try:
        await supabase_service.create_scenario(scenario_definition.dict())
    except Exception as e:
        pytest.skip(f"Error setting up test scenario in Supabase: {str(e)}")
    
    # Run the simulation with scenario (first run, no cache)
    start_time = datetime.now()
    result1 = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    first_run_time = (datetime.now() - start_time).total_seconds()
    
    # Get the cache key
    cache_key = monte_carlo_service._generate_cache_key(
        simulation_request, 
        TEST_USER_ID, 
        scenario_definition.id
    )
    
    # Verify that result was cached
    cached_data = await redis_service.get(cache_key)
    assert cached_data is not None
    
    # Run again with cache
    start_time = datetime.now()
    result2 = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    cached_run_time = (datetime.now() - start_time).total_seconds()
    
    # Verify results are the same
    assert result1.id == result2.id
    assert result1.summary_statistics == result2.summary_statistics
    
    # Cached run should be faster (skipping due to test environment variability)
    # assert cached_run_time < first_run_time

@pytest.mark.asyncio
async def test_compare_scenarios(monte_carlo_service, supabase_service, simulation_request):
    """Test scenario comparison functionality"""
    # Create multiple test scenarios
    scenarios = []
    for i in range(3):
        scenario = ScenarioDefinition(
            id=str(uuid.uuid4()),
            name=f"Test Scenario {i}",
            type="test",
            description=f"Test scenario {i} for comparison",
            user_id=TEST_USER_ID,
            parameters={
                "risk_factor_modifiers": {
                    "interest_rate": {
                        "mean_shift": 0.01 * i,
                    },
                    "default_rate": {
                        "mean_shift": 0.1 * i,
                    }
                }
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        scenarios.append(scenario)
        
        # Store in Supabase
        try:
            await supabase_service.create_scenario(scenario.dict())
        except Exception as e:
            pytest.skip(f"Error setting up test scenarios in Supabase: {str(e)}")
    
    # Compare scenarios
    scenario_ids = [s.id for s in scenarios]
    comparison_result = await monte_carlo_service.compare_scenarios(
        simulation_request,
        scenario_ids,
        TEST_USER_ID,
        metrics_to_compare=["npv", "irr", "default_rate"],
        percentiles_to_compare=[0.05, 0.5, 0.95]
    )
    
    # Verify comparison results
    assert comparison_result is not None
    assert "scenario_results" in comparison_result
    assert len(comparison_result["scenario_results"]) == len(scenarios)
    assert "comparison_metrics" in comparison_result
    
    # Each scenario should have results
    for scenario_id in scenario_ids:
        assert any(r["scenario_id"] == scenario_id for r in comparison_result["scenario_results"])

@pytest.mark.asyncio
async def test_error_handling(monte_carlo_service, supabase_service, simulation_request):
    """Test error handling in scenario-based simulations"""
    # Test with non-existent scenario ID
    fake_scenario_id = str(uuid.uuid4())
    
    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        await monte_carlo_service.run_simulation_with_scenario(
            simulation_request,
            fake_scenario_id,
            TEST_USER_ID,
            use_cache=True,
            supabase_service=supabase_service
        )
    assert "not found" in str(excinfo.value)
    
    # Test with invalid request
    invalid_request = simulation_request.copy()
    # Create an invalid request by setting an impossible correlation
    if invalid_request.correlation_matrix:
        invalid_request.correlation_matrix.correlations["interest_rate:prepayment_rate"] = 2.0
    
    # Create a valid scenario
    scenario = ScenarioDefinition(
        id=str(uuid.uuid4()),
        name="Test Error Scenario",
        type="test",
        description="Test scenario for error handling",
        user_id=TEST_USER_ID,
        parameters={},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Store in Supabase
    try:
        await supabase_service.create_scenario(scenario.dict())
    except Exception as e:
        pytest.skip(f"Error setting up test scenario in Supabase: {str(e)}")
    
    # Should raise validation error when applying scenario
    with pytest.raises(Exception) as excinfo:
        await monte_carlo_service.run_simulation_with_scenario(
            invalid_request,
            scenario.id,
            TEST_USER_ID,
            use_cache=True,
            supabase_service=supabase_service
        )

@pytest.mark.asyncio
async def test_redis_resilience(monte_carlo_service, supabase_service, simulation_request, scenario_definition, monkeypatch):
    """Test resilience when Redis is unavailable"""
    # Store the scenario in Supabase
    try:
        await supabase_service.create_scenario(scenario_definition.dict())
    except Exception as e:
        pytest.skip(f"Error setting up test scenario in Supabase: {str(e)}")
    
    # Mock Redis service to raise exception on get
    async def mock_redis_get(*args, **kwargs):
        raise Exception("Simulated Redis failure")
    
    # Apply the mock
    monkeypatch.setattr(monte_carlo_service.redis_service, "get", mock_redis_get)
    
    # Run the simulation with scenario - should still work despite Redis failure
    result = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,  # Even with cache enabled, should gracefully handle Redis failure
        supabase_service=supabase_service
    )
    
    # Verify the result
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED

    # Now mock Redis service to fail on set
    async def mock_redis_set(*args, **kwargs):
        raise Exception("Simulated Redis failure")
    
    # Apply the mock
    monkeypatch.setattr(monte_carlo_service.redis_service, "set", mock_redis_set)
    
    # Run again - should still complete despite Redis set failure
    result = await monte_carlo_service.run_simulation_with_scenario(
        simulation_request,
        scenario_definition.id,
        TEST_USER_ID,
        use_cache=True,
        supabase_service=supabase_service
    )
    
    # Verify the result
    assert result is not None
    assert result.status == SimulationStatus.COMPLETED

# Integration tests for the API endpoints

@pytest.mark.asyncio
async def test_api_simulation_with_scenario(client, supabase_service, simulation_request, scenario_definition):
    """Test the API endpoint for running a simulation with a scenario"""
    # Skip this test if client fixture is not available
    if not client:
        pytest.skip("API client fixture not available")
    
    # Store the scenario in Supabase
    try:
        await supabase_service.create_scenario(scenario_definition.dict())
    except Exception as e:
        pytest.skip(f"Error setting up test scenario in Supabase: {str(e)}")
    
    # Make API request
    response = await client.post(
        f"/v1/monte-carlo/simulations/with-scenario?scenario_id={scenario_definition.id}&run_async=false",
        json=simulation_request.dict(),
        headers={"Authorization": f"Bearer {TEST_USER_ID}"}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert "result" in data

@pytest.mark.asyncio
async def test_api_compare_scenarios(client, supabase_service, simulation_request):
    """Test the API endpoint for comparing scenarios"""
    # Skip this test if client fixture is not available
    if not client:
        pytest.skip("API client fixture not available")
    
    # Create multiple test scenarios
    scenario_ids = []
    for i in range(2):
        scenario = ScenarioDefinition(
            id=str(uuid.uuid4()),
            name=f"API Test Scenario {i}",
            type="test",
            description=f"API test scenario {i} for comparison",
            user_id=TEST_USER_ID,
            parameters={
                "risk_factor_modifiers": {
                    "interest_rate": {
                        "mean_shift": 0.01 * i,
                    }
                }
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        scenario_ids.append(scenario.id)
        
        # Store in Supabase
        try:
            await supabase_service.create_scenario(scenario.dict())
        except Exception as e:
            pytest.skip(f"Error setting up test scenarios in Supabase: {str(e)}")
    
    # Make API request
    response = await client.post(
        "/v1/monte-carlo/scenarios/compare",
        json={
            "base_request": simulation_request.dict(),
            "scenario_ids": scenario_ids,
            "metrics_to_compare": ["npv", "irr"]
        },
        headers={"Authorization": f"Bearer {TEST_USER_ID}"}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "scenario_results" in data
    assert len(data["scenario_results"]) == len(scenario_ids)

# Cleanup test data after all tests
@pytest.fixture(scope="module", autouse=True)
async def cleanup(request):
    """Cleanup test data after all tests"""
    def finalizer():
        # Create an event loop for cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Clean up Redis
        redis_service = RedisService()
        try:
            loop.run_until_complete(redis_service.delete_pattern("monte_carlo:test*"))
        except Exception as e:
            print(f"Warning: Error cleaning up Redis test data: {str(e)}")
        
        # Clean up Supabase
        supabase_service = SupabaseService()
        try:
            # Delete test scenarios
            scenarios = loop.run_until_complete(
                supabase_service.list_scenarios(user_id=TEST_USER_ID, limit=100)
            )
            for scenario in scenarios:
                if scenario.get("id"):
                    try:
                        loop.run_until_complete(
                            supabase_service.delete_scenario(scenario["id"], TEST_USER_ID)
                        )
                    except Exception as e:
                        print(f"Warning: Error deleting test scenario: {str(e)}")
            
            # Delete test simulations
            simulations = loop.run_until_complete(
                supabase_service.list_simulations(user_id=TEST_USER_ID, limit=100)
            )
            for simulation in simulations:
                if simulation.get("id"):
                    try:
                        loop.run_until_complete(
                            supabase_service.delete_item("monte_carlo_simulations", simulation["id"])
                        )
                    except Exception as e:
                        print(f"Warning: Error deleting test simulation: {str(e)}")
        except Exception as e:
            print(f"Warning: Error cleaning up Supabase test data: {str(e)}")
        
        # Close the event loop
        loop.close()
    
    # Register the finalizer
    request.addfinalizer(finalizer)
