"""
Frontend Client Integration Tests for Monte Carlo API

This module simulates how the frontend JavaScript/TypeScript application
would interact with our Monte Carlo API endpoints. It implements the exact
client patterns that would be used in the actual frontend code.

Usage:
    pytest -xvs tests/e2e/test_frontend_client_integration.py
"""
import os
import json
import uuid
import pytest
import asyncio
import httpx
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants that match the frontend environment
API_BASE_URL = "http://localhost:8000/api/v1"
MONTE_CARLO_ENDPOINT = f"{API_BASE_URL}/monte-carlo"
SCENARIO_ENDPOINT = f"{MONTE_CARLO_ENDPOINT}/scenarios"
SIMULATION_ENDPOINT = f"{MONTE_CARLO_ENDPOINT}/simulations"

# Test user credentials - would be stored in frontend state management
TEST_USER_EMAIL = "test_user@example.com"
TEST_USER_PASSWORD = "SecureTestPassword123!"


class FrontendClient:
    """
    Simulates the frontend client code that would interact with our API.
    This matches exactly how the NextJS/React frontend would be structured.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def login(self, email: str, password: str) -> bool:
        """Simulates frontend login flow with Supabase."""
        from app.services.supabase_service import SupabaseService
        
        try:
            supabase = SupabaseService()
            user_data = await supabase.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            self.token = user_data.session.access_token
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            # Fallback to service key for testing
            self.token = os.environ.get("SUPABASE_SERVICE_KEY", "")
            return bool(self.token)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get auth headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def create_scenario(self, scenario_data: Dict) -> Dict:
        """Create a new scenario - matches frontend client code."""
        response = await self.client.post(
            f"{SCENARIO_ENDPOINT}",
            json=scenario_data,
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to create scenario: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def get_scenarios(self, filter_type: Optional[str] = None) -> List[Dict]:
        """
        Get list of scenarios - matches frontend client code.
        
        Args:
            filter_type: Optional scenario type to filter by
        """
        url = f"{SCENARIO_ENDPOINT}"
        if filter_type:
            url += f"?type={filter_type}"
            
        response = await self.client.get(
            url,
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to get scenarios: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def get_scenario(self, scenario_id: str) -> Dict:
        """Get a specific scenario by ID - matches frontend client code."""
        response = await self.client.get(
            f"{SCENARIO_ENDPOINT}/{scenario_id}",
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to get scenario {scenario_id}: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def update_scenario(self, scenario_id: str, scenario_data: Dict) -> Dict:
        """Update an existing scenario - matches frontend client code."""
        response = await self.client.put(
            f"{SCENARIO_ENDPOINT}/{scenario_id}",
            json=scenario_data,
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to update scenario {scenario_id}: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a scenario - matches frontend client code."""
        response = await self.client.delete(
            f"{SCENARIO_ENDPOINT}/{scenario_id}",
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to delete scenario {scenario_id}: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return True
    
    async def run_simulation_with_scenario(
        self,
        simulation_data: Dict,
        scenario_id: str,
        run_async: bool = False,
        use_cache: bool = True
    ) -> Dict:
        """
        Run a simulation with a scenario - matches frontend client code.
        
        Args:
            simulation_data: The simulation request data
            scenario_id: ID of the scenario to apply
            run_async: Whether to run asynchronously
            use_cache: Whether to use caching
        """
        url = f"{SIMULATION_ENDPOINT}/with-scenario?scenario_id={scenario_id}&run_async={str(run_async).lower()}&use_cache={str(use_cache).lower()}"
        
        response = await self.client.post(
            url,
            json=simulation_data,
            headers=self._get_headers()
        )
        
        expected_code = 202 if run_async else 200
        if response.status_code != expected_code:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to run simulation: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def get_simulation(self, simulation_id: str) -> Dict:
        """Get a specific simulation by ID - matches frontend client code."""
        response = await self.client.get(
            f"{SIMULATION_ENDPOINT}/{simulation_id}",
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to get simulation {simulation_id}: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    async def poll_simulation_until_complete(
        self,
        simulation_id: str,
        max_wait_seconds: int = 180,
        poll_interval_seconds: int = 5
    ) -> Dict:
        """
        Poll a simulation until it completes - matches frontend polling pattern.
        
        Args:
            simulation_id: ID of the simulation to poll
            max_wait_seconds: Maximum time to wait
            poll_interval_seconds: Time between polls
        """
        total_wait = 0
        
        while total_wait < max_wait_seconds:
            try:
                simulation = await self.get_simulation(simulation_id)
                
                if simulation["status"] == "COMPLETED":
                    logger.info(f"Simulation {simulation_id} completed successfully")
                    return simulation
                elif simulation["status"] == "FAILED":
                    error_msg = simulation.get("error", "Unknown error")
                    logger.error(f"Simulation {simulation_id} failed: {error_msg}")
                    raise Exception(f"Simulation failed: {error_msg}")
                
                logger.info(f"Simulation {simulation_id} still running, status: {simulation['status']}")
                await asyncio.sleep(poll_interval_seconds)
                total_wait += poll_interval_seconds
                
            except Exception as e:
                if "API Error (404)" in str(e):
                    logger.error(f"Simulation {simulation_id} not found")
                    raise
                logger.error(f"Error polling simulation: {e}")
                await asyncio.sleep(poll_interval_seconds)
                total_wait += poll_interval_seconds
        
        raise TimeoutError(f"Simulation {simulation_id} did not complete within {max_wait_seconds} seconds")
    
    async def compare_scenarios(
        self,
        base_simulation: Dict,
        scenario_ids: List[str],
        metrics: List[str],
        percentiles: List[float]
    ) -> Dict:
        """
        Compare scenarios - matches frontend client code.
        
        Args:
            base_simulation: Base simulation request data
            scenario_ids: List of scenario IDs to compare
            metrics: List of metrics to compare
            percentiles: List of percentiles to include
        """
        comparison_data = {
            "base_request": base_simulation,
            "scenario_ids": scenario_ids,
            "metrics_to_compare": metrics,
            "percentiles_to_compare": percentiles
        }
        
        response = await self.client.post(
            f"{SCENARIO_ENDPOINT}/compare",
            json=comparison_data,
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            logger.error(f"Failed to compare scenarios: {error_msg}")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response.json()
    
    def _extract_error_message(self, response) -> str:
        """Extract error message from response - matches frontend error handling."""
        try:
            data = response.json()
            if "detail" in data:
                return data["detail"]
            elif "message" in data:
                return data["message"]
            return json.dumps(data)
        except Exception:
            return response.text
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Test fixtures

@pytest.fixture
async def frontend_client():
    """Create and authenticate a frontend client."""
    client = FrontendClient(API_BASE_URL)
    await client.login(TEST_USER_EMAIL, TEST_USER_PASSWORD)
    yield client
    await client.close()

@pytest.fixture
def sample_simulation_request():
    """Create a sample simulation request."""
    return {
        "name": f"Frontend Test Simulation {uuid.uuid4().hex[:8]}",
        "description": "Test simulation created via frontend client",
        "asset_class": "mortgage",
        "projection_months": 36,
        "num_simulations": 100,  # Small number for faster tests
        "percentiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "variables": [
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
        "correlation_matrix": {
            "variables": ["interest_rate", "prepayment_rate", "default_rate"],
            "correlations": {
                "interest_rate:prepayment_rate": 0.3,
                "interest_rate:default_rate": 0.2,
                "prepayment_rate:default_rate": -0.1
            }
        },
        "asset_parameters": {
            "principal": 1000000,
            "rate": 0.05,
            "term_months": 360,
            "payment_frequency": "monthly"
        },
        "include_detailed_paths": False
    }

@pytest.fixture
async def test_scenario(frontend_client):
    """Create a test scenario for frontend testing."""
    scenario_data = {
        "id": str(uuid.uuid4()),
        "name": f"Frontend Test Scenario {uuid.uuid4().hex[:8]}",
        "type": "stress_test",
        "description": "Test scenario for frontend integration testing",
        "user_id": TEST_USER_EMAIL,
        "parameters": {
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
                "tags": ["stress_test", "high_rate", "frontend_test"]
            }
        }
    }
    
    # Create the scenario
    created = await frontend_client.create_scenario(scenario_data)
    assert created["id"] == scenario_data["id"]
    
    yield scenario_data
    
    # Clean up
    try:
        await frontend_client.delete_scenario(scenario_data["id"])
    except Exception as e:
        logger.warning(f"Failed to clean up test scenario: {e}")


# Tests that match frontend usage patterns

@pytest.mark.asyncio
async def test_frontend_scenario_crud(frontend_client):
    """Test Create, Read, Update, Delete operations for scenarios."""
    # Create
    scenario_id = str(uuid.uuid4())
    scenario_data = {
        "id": scenario_id,
        "name": f"Frontend CRUD Test {uuid.uuid4().hex[:8]}",
        "type": "baseline",
        "description": "Test CRUD operations via frontend client",
        "user_id": TEST_USER_EMAIL,
        "parameters": {
            "risk_factor_modifiers": {
                "interest_rate": {
                    "mean_shift": 0.01
                }
            }
        }
    }
    
    created = await frontend_client.create_scenario(scenario_data)
    assert created["id"] == scenario_id
    
    # Read
    retrieved = await frontend_client.get_scenario(scenario_id)
    assert retrieved["id"] == scenario_id
    assert retrieved["name"] == scenario_data["name"]
    
    # List
    scenarios = await frontend_client.get_scenarios()
    assert any(s["id"] == scenario_id for s in scenarios)
    
    # Update
    updated_data = scenario_data.copy()
    updated_data["name"] = f"Updated {scenario_data['name']}"
    updated_data["parameters"]["risk_factor_modifiers"]["interest_rate"]["mean_shift"] = 0.02
    
    updated = await frontend_client.update_scenario(scenario_id, updated_data)
    assert updated["name"] == updated_data["name"]
    assert updated["parameters"]["risk_factor_modifiers"]["interest_rate"]["mean_shift"] == 0.02
    
    # Delete
    deleted = await frontend_client.delete_scenario(scenario_id)
    assert deleted is True
    
    # Verify deletion
    with pytest.raises(Exception) as excinfo:
        await frontend_client.get_scenario(scenario_id)
    assert "API Error (404)" in str(excinfo.value)

@pytest.mark.asyncio
async def test_frontend_run_simulation(frontend_client, sample_simulation_request, test_scenario):
    """Test running a simulation with a scenario."""
    # Run simulation synchronously (as frontend would for small simulations)
    result = await frontend_client.run_simulation_with_scenario(
        sample_simulation_request,
        test_scenario["id"],
        run_async=False,
        use_cache=True
    )
    
    assert result["status"] == "COMPLETED"
    assert "result" in result
    assert "metadata" in result
    assert "scenario" in result["metadata"]
    assert result["metadata"]["scenario"]["id"] == test_scenario["id"]
    
    # Verify the simulation was stored in the database
    simulation_id = result["id"]
    retrieved = await frontend_client.get_simulation(simulation_id)
    assert retrieved["id"] == simulation_id
    
    # Run the same simulation again - should use cache
    start_time = time.time()
    cached_result = await frontend_client.run_simulation_with_scenario(
        sample_simulation_request,
        test_scenario["id"],
        run_async=False,
        use_cache=True
    )
    cache_time = time.time() - start_time
    
    # Verify it's the same result
    assert cached_result["id"] == result["id"]
    
    logger.info(f"Cache retrieval time: {cache_time:.2f} seconds")

@pytest.mark.asyncio
async def test_frontend_async_simulation(frontend_client, sample_simulation_request, test_scenario):
    """Test running an async simulation with polling - matches frontend pattern."""
    # Run simulation asynchronously (as frontend would for large simulations)
    result = await frontend_client.run_simulation_with_scenario(
        sample_simulation_request,
        test_scenario["id"],
        run_async=True,
        use_cache=False  # Force recalculation to test async
    )
    
    assert "id" in result
    assert result["status"] in ["PENDING", "PROCESSING"]
    
    # Poll for completion - this matches exactly how frontend would handle it
    simulation_id = result["id"]
    try:
        completed = await frontend_client.poll_simulation_until_complete(
            simulation_id,
            max_wait_seconds=120,
            poll_interval_seconds=5
        )
        
        assert completed["status"] == "COMPLETED"
        assert "result" in completed
        assert "metadata" in completed
        assert "scenario" in completed["metadata"]
        assert completed["metadata"]["scenario"]["id"] == test_scenario["id"]
        
    except TimeoutError:
        # This would be handled by showing a timeout message in the UI
        pytest.fail("Simulation timed out - would show error in frontend")
    except Exception as e:
        # This would be handled by showing an error message in the UI
        pytest.fail(f"Simulation failed - would show error in frontend: {e}")

@pytest.mark.asyncio
async def test_frontend_scenario_comparison(frontend_client, sample_simulation_request, test_scenario):
    """Test scenario comparison workflow - matches frontend pattern."""
    # Create a second scenario for comparison
    second_scenario_id = str(uuid.uuid4())
    second_scenario = {
        "id": second_scenario_id,
        "name": f"Comparison Scenario {uuid.uuid4().hex[:8]}",
        "type": "optimistic",
        "description": "Second scenario for comparison testing",
        "user_id": TEST_USER_EMAIL,
        "parameters": {
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
        }
    }
    
    await frontend_client.create_scenario(second_scenario)
    
    try:
        # Run comparison - matches frontend chart comparison flow
        comparison_result = await frontend_client.compare_scenarios(
            sample_simulation_request,
            [test_scenario["id"], second_scenario_id],
            ["npv", "irr", "default_rate", "prepayment_rate"],
            [0.05, 0.5, 0.95]
        )
        
        assert "scenario_results" in comparison_result
        assert len(comparison_result["scenario_results"]) == 2
        assert "comparison_metrics" in comparison_result
        
        # Verify each scenario is represented
        scenario_ids = [res["scenario_id"] for res in comparison_result["scenario_results"]]
        assert test_scenario["id"] in scenario_ids
        assert second_scenario_id in scenario_ids
        
        # Check metrics format - frontend would use this for chart rendering
        assert "comparison_metrics" in comparison_result
        for metric in ["npv", "irr", "default_rate", "prepayment_rate"]:
            if metric in comparison_result["comparison_metrics"]:
                metric_data = comparison_result["comparison_metrics"][metric]
                assert "data" in metric_data
                assert "type" in metric_data
                
                # This matches how frontend would validate data for charts
                for scenario_result in metric_data["data"]:
                    assert "scenario_id" in scenario_result
                    assert "name" in scenario_result
                    assert "values" in scenario_result
                    
                    for percentile_key in ["p05", "p50", "p95"]:
                        if percentile_key in scenario_result["values"]:
                            assert isinstance(scenario_result["values"][percentile_key], (int, float))
        
    finally:
        # Clean up
        try:
            await frontend_client.delete_scenario(second_scenario_id)
        except Exception as e:
            logger.warning(f"Failed to clean up second test scenario: {e}")

@pytest.mark.asyncio
async def test_frontend_error_handling(frontend_client, sample_simulation_request):
    """Test frontend error handling patterns."""
    # Test invalid scenario ID error handling
    invalid_id = str(uuid.uuid4())
    
    # This matches frontend error handling pattern
    try:
        await frontend_client.run_simulation_with_scenario(
            sample_simulation_request,
            invalid_id,
            run_async=False
        )
        pytest.fail("Expected exception was not raised")
    except Exception as e:
        # Frontend would show this error in UI
        assert "not found" in str(e).lower() or "404" in str(e)
        logger.info(f"Correctly handled missing scenario error: {e}")
    
    # Test validation error handling
    invalid_request = sample_simulation_request.copy()
    invalid_request["variables"] = [
        {
            "name": "interest_rate",
            "distribution": "invalid_distribution",  # Invalid distribution type
            "parameters": {"mean": 0.05, "std_dev": 0.01}
        }
    ]
    
    # Create a valid scenario for testing
    scenario_id = str(uuid.uuid4())
    scenario = {
        "id": scenario_id,
        "name": f"Error Test Scenario {uuid.uuid4().hex[:8]}",
        "type": "test",
        "description": "Scenario for error testing",
        "user_id": TEST_USER_EMAIL,
        "parameters": {}
    }
    
    await frontend_client.create_scenario(scenario)
    
    try:
        # This matches frontend validation error handling pattern
        try:
            await frontend_client.run_simulation_with_scenario(
                invalid_request,
                scenario_id,
                run_async=False
            )
            pytest.fail("Expected validation exception was not raised")
        except Exception as e:
            # Frontend would show this validation error in UI
            assert "invalid" in str(e).lower() or "validation" in str(e).lower() or "422" in str(e)
            logger.info(f"Correctly handled validation error: {e}")
    finally:
        # Clean up
        try:
            await frontend_client.delete_scenario(scenario_id)
        except Exception as e:
            logger.warning(f"Failed to clean up error test scenario: {e}")

# Main entry point for manual testing
if __name__ == "__main__":
    async def main():
        client = FrontendClient(API_BASE_URL)
        try:
            if await client.login(TEST_USER_EMAIL, TEST_USER_PASSWORD):
                scenarios = await client.get_scenarios()
                print(f"Found {len(scenarios)} scenarios")
            else:
                print("Login failed")
        finally:
            await client.close()
    
    asyncio.run(main())
