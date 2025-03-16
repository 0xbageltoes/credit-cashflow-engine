"""
Unit tests for the Asset Class Service

Comprehensive test suite for the asset class service, ensuring proper functionality
and error handling across different asset classes and scenarios.
"""
import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import date, datetime, timedelta
import uuid

from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    ResidentialMortgage, AutoLoan, LoanStatus, RateType, PaymentFrequency, PropertyType
)
from app.services.asset_class_service import AssetClassService
from app.services.asset_handlers.factory import AssetHandlerFactory
from app.services.asset_handlers.residential_mortgage import ResidentialMortgageHandler
from app.services.asset_handlers.auto_loan import AutoLoanHandler
from app.core.cache_service import CacheService

# Test data fixtures
@pytest.fixture
def sample_residential_mortgage():
    """Create a sample residential mortgage for testing"""
    return ResidentialMortgage(
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
def sample_analysis_request(sample_asset_pool):
    """Create a sample analysis request"""
    return AssetPoolAnalysisRequest(
        pool=sample_asset_pool,
        analysis_date=date.today(),
        discount_rate=0.05,
        include_cashflows=True,
        include_metrics=True,
        include_stress_tests=False
    )

@pytest.fixture
def mock_cache():
    """Create a mock cache service for testing"""
    mock_cache = AsyncMock(spec=CacheService)
    mock_cache.get.return_value = None  # Default to cache miss
    mock_cache.set.return_value = True  # Default to successful cache set
    return mock_cache

@pytest.fixture
def mock_residential_handler():
    """Create a mock residential mortgage handler"""
    handler = MagicMock(spec=ResidentialMortgageHandler)
    handler.analyze_pool.return_value = AssetPoolAnalysisResponse(
        pool_name="Test Residential Pool",
        analysis_date=date.today(),
        execution_time=0.5,
        status="success",
        metrics=MagicMock(),
        cashflows=[MagicMock()] * 5,
        stress_tests=None
    )
    return handler

@pytest.fixture
def mock_handler_factory(mock_residential_handler):
    """Create a mock handler factory for testing"""
    factory = MagicMock(spec=AssetHandlerFactory)
    factory.is_supported.return_value = True
    factory.get_handler.return_value = mock_residential_handler
    factory.supported_asset_classes.return_value = [
        AssetClass.RESIDENTIAL_MORTGAGE,
        AssetClass.AUTO_LOAN
    ]
    return factory

@pytest.mark.asyncio
async def test_analyze_asset_pool_cache_hit(
    sample_analysis_request,
    mock_cache,
    mock_handler_factory
):
    """Test that analyzing an asset pool returns cached result when available"""
    # Set up cache hit
    cached_response = AssetPoolAnalysisResponse(
        pool_name="Test Residential Pool",
        analysis_date=date.today(),
        execution_time=0.1,
        status="success",
        metrics=MagicMock(),
        cashflows=[MagicMock()] * 3
    )
    mock_cache.get.return_value = json.dumps(cached_response.model_dump())
    
    # Create service with mocked dependencies
    service = AssetClassService(cache_service=mock_cache)
    service.handler_factory = mock_handler_factory
    
    # Call the method
    result = await service.analyze_asset_pool(
        request=sample_analysis_request,
        user_id="test_user",
        use_cache=True
    )
    
    # Verify results
    assert result is not None
    assert result.status == "success"
    assert result.pool_name == "Test Residential Pool"
    
    # Verify that cache was checked but handler wasn't called
    mock_cache.get.assert_called_once()
    mock_handler_factory.get_handler.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_asset_pool_cache_miss(
    sample_analysis_request,
    mock_cache,
    mock_handler_factory,
    mock_residential_handler
):
    """Test that analyzing an asset pool calls the handler on cache miss"""
    # Set up cache miss
    mock_cache.get.return_value = None
    
    # Create service with mocked dependencies
    service = AssetClassService(cache_service=mock_cache)
    service.handler_factory = mock_handler_factory
    
    # Call the method
    result = await service.analyze_asset_pool(
        request=sample_analysis_request,
        user_id="test_user",
        use_cache=True
    )
    
    # Verify results
    assert result is not None
    assert result.status == "success"
    
    # Verify that cache was checked and handler was called
    mock_cache.get.assert_called_once()
    mock_handler_factory.get_handler.assert_called_once()
    mock_residential_handler.analyze_pool.assert_called_once_with(sample_analysis_request)
    
    # Verify that result was cached
    mock_cache.set.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_asset_pool_no_cache(
    sample_analysis_request,
    mock_cache,
    mock_handler_factory,
    mock_residential_handler
):
    """Test that analyzing an asset pool bypasses cache when use_cache=False"""
    # Create service with mocked dependencies
    service = AssetClassService(cache_service=mock_cache)
    service.handler_factory = mock_handler_factory
    
    # Call the method with caching disabled
    result = await service.analyze_asset_pool(
        request=sample_analysis_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Verify results
    assert result is not None
    assert result.status == "success"
    
    # Verify that cache was not checked but handler was called
    mock_cache.get.assert_not_called()
    mock_handler_factory.get_handler.assert_called_once()
    mock_residential_handler.analyze_pool.assert_called_once_with(sample_analysis_request)
    
    # Verify that result was not cached
    mock_cache.set.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_asset_pool_unsupported_class(
    sample_analysis_request,
    mock_cache,
    mock_handler_factory
):
    """Test that analyzing an asset pool falls back to generic analysis for unsupported classes"""
    # Set up factory to report unsupported class
    mock_handler_factory.is_supported.return_value = False
    
    # Create service with mocked dependencies
    service = AssetClassService(cache_service=mock_cache)
    service.handler_factory = mock_handler_factory
    
    # Add a spy for the generic analysis method
    with patch.object(service, '_analyze_generic_pool') as mock_generic_analysis:
        mock_generic_analysis.return_value = AssetPoolAnalysisResponse(
            pool_name="Test Pool",
            analysis_date=date.today(),
            execution_time=0.2,
            status="success"
        )
        
        # Call the method
        result = await service.analyze_asset_pool(
            request=sample_analysis_request,
            user_id="test_user",
            use_cache=False
        )
    
    # Verify results
    assert result is not None
    assert result.status == "success"
    
    # Verify that generic analysis was used
    mock_handler_factory.get_handler.assert_not_called()
    mock_generic_analysis.assert_called_once_with(sample_analysis_request)

@pytest.mark.asyncio
async def test_analyze_asset_pool_error_handling(
    sample_analysis_request,
    mock_cache,
    mock_handler_factory
):
    """Test that analyzing an asset pool handles errors gracefully"""
    # Set up handler to raise an exception
    mock_handler = mock_handler_factory.get_handler.return_value
    mock_handler.analyze_pool.side_effect = ValueError("Test error")
    
    # Create service with mocked dependencies
    service = AssetClassService(cache_service=mock_cache)
    service.handler_factory = mock_handler_factory
    
    # Call the method
    result = await service.analyze_asset_pool(
        request=sample_analysis_request,
        user_id="test_user",
        use_cache=False
    )
    
    # Verify error response
    assert result is not None
    assert result.status == "error"
    assert "Test error" in result.error
    assert result.error_type == "ValueError"

def test_determine_predominant_asset_class(sample_asset_pool, sample_auto_loan):
    """Test determining the predominant asset class in a pool"""
    # Create service
    service = AssetClassService()
    
    # Test with homogeneous pool
    asset_class = service._determine_predominant_asset_class(sample_asset_pool)
    assert asset_class == AssetClass.RESIDENTIAL_MORTGAGE
    
    # Test with mixed pool, with auto loan having higher balance
    high_balance_auto = sample_auto_loan
    high_balance_auto.balance = 1000000  # Much higher than mortgage
    
    mixed_pool = AssetPool(
        pool_name="Mixed Pool",
        cut_off_date=date.today(),
        assets=[sample_asset_pool.assets[0], high_balance_auto]
    )
    
    asset_class = service._determine_predominant_asset_class(mixed_pool)
    assert asset_class == AssetClass.AUTO_LOAN  # Should choose based on balance

def test_get_supported_asset_classes(mock_handler_factory):
    """Test getting supported asset classes"""
    # Create service with mock factory
    service = AssetClassService()
    service.handler_factory = mock_handler_factory
    
    # Get supported classes
    supported_classes = service.get_supported_asset_classes()
    
    # Verify results
    assert AssetClass.RESIDENTIAL_MORTGAGE in supported_classes
    assert AssetClass.AUTO_LOAN in supported_classes
    assert len(supported_classes) == 2

def test_generate_cache_key():
    """Test generating a deterministic cache key"""
    # Create a request
    request = AssetPoolAnalysisRequest(
        pool=AssetPool(
            pool_name="Test Pool",
            cut_off_date=date.today(),
            assets=[]
        ),
        analysis_date=date.today(),
        discount_rate=0.05
    )
    
    # Create service and generate key
    service = AssetClassService()
    key1 = service.generate_cache_key(request, "user1")
    
    # Generate again with same parameters
    key2 = service.generate_cache_key(request, "user1")
    
    # Generate with different user
    key3 = service.generate_cache_key(request, "user2")
    
    # Verify results
    assert key1 == key2  # Same parameters should yield same key
    assert key1 != key3  # Different user should yield different key
    assert "test_pool" in key1.lower()  # Key should include pool name
    assert "user1" in key1  # Key should include user ID

@pytest.mark.asyncio
async def test_cache_result(mock_cache):
    """Test caching a result"""
    # Create a result
    result = AssetPoolAnalysisResponse(
        pool_name="Test Pool",
        analysis_date=date.today(),
        execution_time=0.5,
        status="success"
    )
    
    # Create service and cache result
    service = AssetClassService(cache_service=mock_cache)
    success = await service._cache_result("test_key", result)
    
    # Verify results
    assert success is True
    mock_cache.set.assert_called_once()
    
    # Test with Redis failure
    mock_cache.set.reset_mock()
    mock_cache.set.side_effect = Exception("Redis error")
    
    success = await service._cache_result("test_key", result)
    
    # Verify graceful failure
    assert success is False
    mock_cache.set.assert_called_once()

@pytest.mark.asyncio
async def test_get_cached_result(mock_cache):
    """Test retrieving a cached result"""
    # Create a cached result
    cached_result = AssetPoolAnalysisResponse(
        pool_name="Test Pool",
        analysis_date=date.today(),
        execution_time=0.5,
        status="success"
    )
    mock_cache.get.return_value = json.dumps(cached_result.model_dump())
    
    # Create service and get cached result
    service = AssetClassService(cache_service=mock_cache)
    result = await service._get_cached_result("test_key")
    
    # Verify results
    assert result is not None
    assert result.pool_name == "Test Pool"
    assert result.status == "success"
    mock_cache.get.assert_called_once_with("test_key")
    
    # Test with Redis failure
    mock_cache.get.reset_mock()
    mock_cache.get.side_effect = Exception("Redis error")
    
    result = await service._get_cached_result("test_key")
    
    # Verify graceful failure
    assert result is None
    mock_cache.get.assert_called_once_with("test_key")

def test_analyze_generic_pool():
    """Test generic pool analysis for unsupported asset classes"""
    # Create a request with a simple pool
    mortgage = ResidentialMortgage(
        balance=100000,
        rate=0.04,
        term_months=360,
        origination_date=date.today(),
        status=LoanStatus.CURRENT,
        property_type=PropertyType.SINGLE_FAMILY
    )
    
    pool = AssetPool(
        pool_name="Generic Test Pool",
        cut_off_date=date.today(),
        assets=[mortgage]
    )
    
    request = AssetPoolAnalysisRequest(
        pool=pool,
        analysis_date=date.today(),
        discount_rate=0.05,
        include_cashflows=True
    )
    
    # Create service and analyze
    service = AssetClassService()
    result = service._analyze_generic_pool(request)
    
    # Verify results
    assert result is not None
    assert result.status == "success"
    assert result.pool_name == "Generic Test Pool"
    assert result.metrics is not None
    assert result.metrics.total_principal > 0
    assert result.metrics.total_interest > 0
    assert result.metrics.npv > 0
    assert result.cashflows is not None
    assert len(result.cashflows) > 0
