"""
API tests for the Asset Stress Testing endpoints
"""
import json
import uuid
from datetime import date, datetime
from typing import Dict, Any
import pytest
from httpx import AsyncClient
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    ResidentialMortgage, AutoLoan, AssetPoolMetrics
)
from app.services.asset_handlers.stress_testing import AssetStressTester
from app.api.v1.asset_classes.stress_endpoints import StressTestRequest

# Test data fixtures
@pytest.fixture
def sample_mortgage_asset():
    """Return a sample residential mortgage asset"""
    return ResidentialMortgage(
        asset_id="RM001",
        asset_class=AssetClass.RESIDENTIAL_MORTGAGE,
        balance=250000.0,
        rate=0.0375,
        term=360,
        age=12,
        original_balance=275000.0,
        rate_type="fixed",
        prepayment_speed=0.07,
        default_probability=0.02,
        recovery_rate=0.65,
        loss_severity=0.35,
        status="performing"
    )

@pytest.fixture
def sample_auto_loan_asset():
    """Return a sample auto loan asset"""
    return AutoLoan(
        asset_id="AL001",
        asset_class=AssetClass.AUTO_LOAN,
        balance=35000.0,
        rate=0.0425,
        term=60,
        age=10,
        original_balance=40000.0,
        rate_type="fixed",
        prepayment_speed=0.05,
        default_probability=0.03,
        recovery_rate=0.50,
        loss_severity=0.50,
        status="performing",
        vehicle_type="SUV",
        vehicle_age=2,
        new_used="used"
    )

@pytest.fixture
def sample_asset_pool(sample_mortgage_asset, sample_auto_loan_asset):
    """Return a sample asset pool with mixed assets"""
    return AssetPool(
        pool_id="POOL001",
        pool_name="Test Mixed Pool",
        assets=[sample_mortgage_asset, sample_auto_loan_asset],
        cutoff_date=date.today(),
        metadata={
            "source": "unit test",
            "description": "Test pool for stress testing"
        }
    )

@pytest.fixture
def sample_analysis_request(sample_asset_pool):
    """Return a sample analysis request"""
    return AssetPoolAnalysisRequest(
        pool=sample_asset_pool,
        analysis_date=date.today(),
        discount_rate=0.05,
        projection_periods=60,
        include_cashflows=True
    )

@pytest.fixture
def sample_stress_test_request(sample_analysis_request):
    """Return a sample stress test request"""
    return StressTestRequest(
        request=sample_analysis_request,
        scenario_names=["base", "rate_shock_up", "credit_crisis"],
        run_parallel=False,
        include_cashflows=False,
        generate_report=True
    )

@pytest.fixture
def sample_analysis_response():
    """Return a sample analysis response"""
    return AssetPoolAnalysisResponse(
        pool_name="Test Mixed Pool",
        analysis_date=date.today(),
        status="success",
        execution_time=1.25,
        metrics=AssetPoolMetrics(
            npv=273621.45,
            irr=0.0612,
            total_principal=285000.0,
            total_interest=42318.55,
            duration=3.2,
            weighted_average_life=4.7
        ),
        cashflows=[],
        analytics={},
        cache_hit=False
    )

@pytest.mark.asyncio
class TestStressTestingAPI:
    """Tests for the Stress Testing API endpoints"""
    
    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_run_stress_tests_endpoint(
        self, 
        mock_run_tests, 
        client: AsyncClient, 
        sample_stress_test_request, 
        sample_analysis_response
    ):
        """Test the run stress tests endpoint returns correct results"""
        # Arrange
        token = "mock_token"  # This would be a valid token in production
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock the stress test results
        mock_result = {
            "base": sample_analysis_response,
            "rate_shock_up": sample_analysis_response,
            "credit_crisis": sample_analysis_response
        }
        mock_run_tests.return_value = mock_result
        
        # Act
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run",
            json=sample_stress_test_request.model_dump(),
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert len(result) == 3
        assert "base" in result
        assert "rate_shock_up" in result
        assert "credit_crisis" in result
        
        # Verify the AssetStressTester was called correctly
        mock_run_tests.assert_called_once()
        args, kwargs = mock_run_tests.call_args
        assert kwargs["scenario_names"] == sample_stress_test_request.scenario_names
        
    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_run_stress_tests_with_empty_pool(
        self, 
        mock_run_tests, 
        client: AsyncClient, 
        sample_stress_test_request
    ):
        """Test validation for empty asset pool"""
        # Arrange
        token = "mock_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Empty the pool
        sample_stress_test_request.request.pool.assets = []
        
        # Act
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run",
            json=sample_stress_test_request.model_dump(),
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Asset pool must contain at least one asset" in response.json()["detail"]
        
        # Verify the AssetStressTester was not called
        mock_run_tests.assert_not_called()
        
    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_run_stress_tests_error_handling(
        self, 
        mock_run_tests, 
        client: AsyncClient, 
        sample_stress_test_request
    ):
        """Test error handling in stress testing endpoint"""
        # Arrange
        token = "mock_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Simulate an error in the stress tester
        mock_run_tests.side_effect = Exception("Test error in stress testing")
        
        # Act
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run",
            json=sample_stress_test_request.model_dump(),
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test error in stress testing" in response.json()["detail"]
        
    @patch('app.api.v1.asset_classes.stress_endpoints.manager.send_task_update')
    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_async_stress_test_endpoint(
        self,
        mock_run_tests,
        mock_ws_send,
        client: AsyncClient,
        sample_stress_test_request,
        sample_analysis_response
    ):
        """Test the async stress test endpoint"""
        # Arrange
        token = "mock_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock UUID for predictable task ID
        with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            # Act
            response = await client.post(
                "/api/v1/asset-classes/stress-testing/async-run",
                json=sample_stress_test_request.model_dump(),
                headers=headers
            )
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            
            result = response.json()
            assert result["task_id"] == "12345678-1234-5678-1234-567812345678"
            assert result["status"] == "queued"
            assert "websocket_url" in result
            assert result["pool_name"] == sample_stress_test_request.request.pool.pool_name
            
            # Verify that the background task was added
            # Note: We can't directly test the background task execution in this unit test
            
    async def test_get_stress_scenarios_endpoint(self, client: AsyncClient):
        """Test the get stress scenarios endpoint"""
        # Arrange
        token = "mock_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await client.get(
            "/api/v1/asset-classes/stress-testing/scenarios",
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert "base" in result
        assert "credit_crisis" in result
        assert "rate_shock_up" in result
        
        # Check scenario structure
        for scenario_name, scenario in result.items():
            assert "name" in scenario
            assert "description" in scenario
            assert "market_factors" in scenario

    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_run_with_cache_control(
        self,
        mock_run_tests,
        client: AsyncClient,
        sample_stress_test_request,
        sample_analysis_response
    ):
        """Test cache control parameters are correctly passed"""
        # Arrange
        token = "mock_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock the stress test results
        mock_result = {
            "base": sample_analysis_response
        }
        mock_run_tests.return_value = mock_result
        
        # Act - With cache disabled
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run?use_cache=false",
            json=sample_stress_test_request.model_dump(),
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        
        # Verify cache parameter was passed correctly
        args, kwargs = mock_run_tests.call_args
        assert kwargs["use_cache"] is False
        
        # Act - With cache enabled (default)
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run",
            json=sample_stress_test_request.model_dump(),
            headers=headers
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        
        # Verify cache parameter was passed correctly
        args, kwargs = mock_run_tests.call_args
        assert kwargs["use_cache"] is True
        
    @patch('app.api.v1.asset_classes.stress_endpoints.AssetStressTester.run_stress_tests')
    async def test_unauthorized_access(
        self,
        mock_run_tests,
        client: AsyncClient,
        sample_stress_test_request
    ):
        """Test unauthorized access is correctly handled"""
        # Arrange - No auth header
        
        # Act
        response = await client.post(
            "/api/v1/asset-classes/stress-testing/run",
            json=sample_stress_test_request.model_dump()
        )
        
        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        mock_run_tests.assert_not_called()
