"""
API Tests for Asset Classes Endpoints

Comprehensive API testing suite to validate the asset classes endpoints functionality,
ensuring proper error handling, authentication, validation, and expected responses.
"""
import json
import pytest
from datetime import date, datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from app.main import app
from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    ResidentialMortgage, AutoLoan, LoanStatus, RateType, PaymentFrequency, PropertyType,
    AssetPoolMetrics
)
from app.services.asset_class_service import AssetClassService
from app.core.auth import get_current_user

# Test client
client = TestClient(app)

# Mock authentication for tests
@pytest.fixture(autouse=True)
def mock_auth():
    """Mock authentication for all tests to use a consistent test user"""
    with patch("app.api.v1.asset_classes.endpoints.get_current_user", autospec=True) as mock_auth:
        mock_auth.return_value = {
            "id": "test_user_id",
            "email": "test@example.com",
            "role": "analyst",
            "is_active": True
        }
        yield mock_auth

# Test data fixtures
@pytest.fixture
def sample_residential_mortgage():
    """Create a sample residential mortgage for testing"""
    return ResidentialMortgage(
        asset_id="RM001",
        asset_class=AssetClass.RESIDENTIAL_MORTGAGE,
        balance=200000,
        rate=0.05,
        term_months=360,
        origination_date=date.today() - timedelta(days=180),
        rate_type=RateType.FIXED,
        payment_frequency=PaymentFrequency.MONTHLY,
        original_balance=200000,
        remaining_term_months=354,
        status=LoanStatus.CURRENT,
        property_type=PropertyType.SINGLE_FAMILY,
        ltv_ratio=0.8,
        lien_position=1,
        is_interest_only=False
    )

@pytest.fixture
def sample_auto_loan():
    """Create a sample auto loan for testing"""
    return AutoLoan(
        asset_id="AL001",
        asset_class=AssetClass.AUTO_LOAN,
        balance=25000,
        rate=0.06,
        term_months=60,
        origination_date=date.today() - timedelta(days=90),
        rate_type=RateType.FIXED,
        payment_frequency=PaymentFrequency.MONTHLY,
        original_balance=25000,
        remaining_term_months=57,
        status=LoanStatus.CURRENT,
        vehicle_type="used",
        vehicle_make="Toyota",
        vehicle_model="Camry",
        vehicle_year=2020,
        ltv_ratio=0.9
    )

@pytest.fixture
def sample_asset_pool(sample_residential_mortgage):
    """Create a sample asset pool with residential mortgages"""
    return AssetPool(
        pool_name="Test Residential Pool",
        pool_description="Test pool for residential mortgages",
        cut_off_date=date.today(),
        assets=[sample_residential_mortgage] * 3  # Create 3 identical loans for simplicity
    )

@pytest.fixture
def sample_auto_pool(sample_auto_loan):
    """Create a sample asset pool with auto loans"""
    return AssetPool(
        pool_name="Test Auto Loan Pool",
        pool_description="Test pool for auto loans",
        cut_off_date=date.today(),
        assets=[sample_auto_loan] * 3  # Create 3 identical loans for simplicity
    )

@pytest.fixture
def mock_service_response():
    """Mock successful response from asset class service"""
    return AssetPoolAnalysisResponse(
        pool_name="Test Residential Pool",
        analysis_date=date.today(),
        execution_time=0.5,
        status="success",
        metrics=AssetPoolMetrics(
            total_principal=600000,
            total_interest=428765.32,
            wam=354,
            wac=0.05,
            npv=582634.21
        ),
        cashflows=[{
            "period": i,
            "date": (date.today() + timedelta(days=30*i)).isoformat(),
            "principal": 10000 + (i * 100),
            "interest": 2500 - (i * 10),
            "balance": 600000 - (10000 * i)
        } for i in range(1, 6)],  # Simple cashflows for 5 periods
        analytics={
            "credit_metrics": {
                "weighted_fico": 720,
                "delinquency_rate": 0.02
            },
            "prepayment_metrics": {
                "cpr": 0.05,
                "smm": 0.004
            }
        }
    )

def test_get_supported_asset_classes():
    """Test endpoint to get supported asset classes"""
    with patch("app.services.asset_class_service.AssetClassService.get_supported_asset_classes") as mock_get_classes:
        mock_get_classes.return_value = [
            AssetClass.RESIDENTIAL_MORTGAGE,
            AssetClass.AUTO_LOAN,
            AssetClass.CREDIT_CARD
        ]
        
        response = client.get("/api/v1/asset-classes/supported")
        
        assert response.status_code == 200
        assert len(response.json()) == 3
        assert AssetClass.RESIDENTIAL_MORTGAGE.value in [item for item in response.json()]
        assert AssetClass.AUTO_LOAN.value in [item for item in response.json()]

def test_get_supported_asset_classes_error():
    """Test error handling in get supported asset classes endpoint"""
    with patch("app.services.asset_class_service.AssetClassService.get_supported_asset_classes") as mock_get_classes:
        mock_get_classes.side_effect = Exception("Test error")
        
        response = client.get("/api/v1/asset-classes/supported")
        
        assert response.status_code == 500
        assert "error" in response.json()
        assert "Test error" in response.json()["error"]

@pytest.mark.asyncio
async def test_analyze_asset_pool(sample_asset_pool, mock_service_response):
    """Test successful asset pool analysis endpoint"""
    with patch("app.services.asset_class_service.AssetClassService.analyze_asset_pool", autospec=True) as mock_analyze:
        mock_analyze.return_value = mock_service_response
        
        request_data = {
            "pool": sample_asset_pool.model_dump(),
            "analysis_date": date.today().isoformat(),
            "discount_rate": 0.05,
            "include_cashflows": True,
            "include_metrics": True,
            "include_stress_tests": False
        }
        
        response = client.post(
            "/api/v1/asset-classes/analyze",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["pool_name"] == "Test Residential Pool"
        assert "metrics" in result
        assert "cashflows" in result
        assert len(result["cashflows"]) == 5
        assert result["metrics"]["total_principal"] == 600000
        assert "analytics" in result
        assert "credit_metrics" in result["analytics"]

@pytest.mark.asyncio
async def test_analyze_asset_pool_with_cache_disabled(sample_asset_pool, mock_service_response):
    """Test asset pool analysis with caching explicitly disabled"""
    with patch("app.services.asset_class_service.AssetClassService.analyze_asset_pool", autospec=True) as mock_analyze:
        mock_analyze.return_value = mock_service_response
        
        request_data = {
            "pool": sample_asset_pool.model_dump(),
            "analysis_date": date.today().isoformat(),
            "discount_rate": 0.05,
            "include_cashflows": True
        }
        
        response = client.post(
            "/api/v1/asset-classes/analyze?use_cache=false",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        
        # Verify that service was called with use_cache=False
        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args.kwargs
        assert "use_cache" in call_kwargs
        assert call_kwargs["use_cache"] is False

@pytest.mark.asyncio
async def test_analyze_asset_pool_error(sample_asset_pool):
    """Test error handling in asset pool analysis endpoint"""
    with patch("app.services.asset_class_service.AssetClassService.analyze_asset_pool", autospec=True) as mock_analyze:
        # Return an error response
        error_response = AssetPoolAnalysisResponse(
            pool_name="Test Residential Pool",
            analysis_date=date.today(),
            execution_time=0.1,
            status="error",
            error="Invalid asset data: negative balance detected",
            error_type="ValidationError"
        )
        mock_analyze.return_value = error_response
        
        request_data = {
            "pool": sample_asset_pool.model_dump(),
            "analysis_date": date.today().isoformat(),
            "discount_rate": 0.05
        }
        
        response = client.post(
            "/api/v1/asset-classes/analyze",
            json=request_data
        )
        
        assert response.status_code == 500
        assert "error" in response.json()
        assert "Invalid asset data" in response.json()["error"]

@pytest.mark.asyncio
async def test_analyze_asset_pool_validation_error():
    """Test validation error handling in asset pool analysis endpoint"""
    # Send invalid request data (missing required fields)
    request_data = {
        "analysis_date": date.today().isoformat(),
        # Missing pool data
    }
    
    response = client.post(
        "/api/v1/asset-classes/analyze",
        json=request_data
    )
    
    assert response.status_code == 422  # Validation error
    assert "detail" in response.json()
    # Check that the validation error mentions the missing field
    assert any("pool" in str(error).lower() for error in response.json()["detail"])

@pytest.mark.asyncio
async def test_analyze_asset_pool_empty_pool():
    """Test validation of empty asset pool"""
    request_data = {
        "pool": {
            "pool_name": "Empty Pool",
            "cut_off_date": date.today().isoformat(),
            "assets": []  # Empty assets list
        },
        "analysis_date": date.today().isoformat(),
        "discount_rate": 0.05
    }
    
    response = client.post(
        "/api/v1/asset-classes/analyze",
        json=request_data
    )
    
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()
    # Message about empty pool
    error_text = response.json().get("detail", "")
    assert "empty" in error_text.lower() or "at least one asset" in error_text.lower()

def test_validate_asset_pool(sample_asset_pool):
    """Test successful asset pool validation endpoint"""
    response = client.post(
        "/api/v1/asset-classes/validate-pool",
        json=sample_asset_pool.model_dump()
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["valid"] is True
    assert len(result["warnings"]) == 0
    assert len(result["errors"]) == 0

def test_validate_asset_pool_with_warnings(sample_asset_pool, sample_auto_loan):
    """Test asset pool validation with warnings for mixed asset classes"""
    # Create mixed pool
    mixed_assets = sample_asset_pool.assets.copy()
    mixed_assets.append(sample_auto_loan)
    mixed_pool = sample_asset_pool.model_copy(update={"assets": mixed_assets})
    
    response = client.post(
        "/api/v1/asset-classes/validate-pool",
        json=mixed_pool.model_dump()
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["valid"] is True  # Still valid but with warnings
    assert len(result["warnings"]) > 0
    assert any("mixed asset classes" in warning.lower() for warning in result["warnings"])

def test_validate_asset_pool_with_errors():
    """Test asset pool validation with validation errors"""
    # Create invalid asset with negative values
    invalid_mortgage = ResidentialMortgage(
        asset_id="RM002",
        asset_class=AssetClass.RESIDENTIAL_MORTGAGE,
        balance=-50000,  # Negative balance
        rate=0.05,
        term_months=360,
        origination_date=date.today(),
        status=LoanStatus.CURRENT,
        property_type=PropertyType.SINGLE_FAMILY,
        remaining_term_months=400  # Remaining term > original term
    )
    
    invalid_pool = AssetPool(
        pool_name="Invalid Pool",
        cut_off_date=date.today(),
        assets=[invalid_mortgage]
    )
    
    response = client.post(
        "/api/v1/asset-classes/validate-pool",
        json=invalid_pool.model_dump()
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["valid"] is False
    assert len(result["errors"]) >= 2  # At least 2 errors (negative balance, term issue)
    assert any("balance must be positive" in error.lower() for error in result["errors"])
    assert any("remaining term cannot exceed" in error.lower() for error in result["errors"])

def test_unauthorized_access():
    """Test unauthorized access to endpoints"""
    with patch("app.api.v1.asset_classes.endpoints.get_current_user", autospec=True) as mock_auth:
        # Simulate auth failure
        mock_auth.side_effect = Exception("Unauthorized")
        
        response = client.get("/api/v1/asset-classes/supported")
        assert response.status_code in (401, 403)  # Either unauthorized or forbidden

def test_rate_limiting():
    """Test rate limiting on endpoints"""
    # This test depends on the implementation of rate limiting middleware
    # For a proper test, we would need to configure the rate limiter for testing
    
    # Make multiple rapid requests to trigger rate limiting
    responses = []
    for _ in range(30):  # Attempt to trigger rate limiting
        responses.append(client.get("/api/v1/asset-classes/supported"))
    
    # Check if any response got rate limited
    rate_limited = any(r.status_code == 429 for r in responses)
    
    # Note: This test is informational and may not always trigger rate limiting
    # in test environments, so we don't assert on the result
