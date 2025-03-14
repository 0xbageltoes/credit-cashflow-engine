"""
Unit tests for the Asset Stress Testing module
"""
import pytest
import json
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from app.services.asset_handlers.stress_testing import AssetStressTester
from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    ResidentialMortgage, AutoLoan, AssetPoolMetrics, AssetPoolStressTest
)
from app.core.cache import RedisCache

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

@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache"""
    redis_cache = AsyncMock(spec=RedisCache)
    redis_cache.get = AsyncMock(return_value=None)
    redis_cache.set = AsyncMock(return_value=True)
    redis_cache.delete = AsyncMock(return_value=True)
    return redis_cache

class TestAssetStressTester:
    """Tests for the AssetStressTester class"""
    
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_initialization(self, mock_factory, mock_redis_cache):
        """Test proper initialization of the stress tester"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.return_value = AsyncMock()
        
        # Act
        tester = AssetStressTester()
        
        # Assert
        assert tester is not None
        assert hasattr(tester, 'default_scenarios')
        assert len(tester.default_scenarios) >= 7  # Should have at least 7 default scenarios
        assert "base" in tester.default_scenarios
        assert "credit_crisis" in tester.default_scenarios
        
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_redis_cache_initialization_error(self, mock_factory, mock_redis_cache):
        """Test handling of Redis cache initialization error"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.side_effect = Exception("Connection error")
        
        # Act
        tester = AssetStressTester()
        
        # Assert
        assert tester.redis_cache is None  # Should handle error and set redis_cache to None
        assert hasattr(tester, 'handler_factory')
        
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_apply_scenario_factors(self, mock_factory, mock_redis_cache):
        """Test correct application of scenario factors to assets"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.return_value = AsyncMock()
        tester = AssetStressTester()
        
        request = sample_analysis_request()
        
        # Act - Apply rate shock
        market_factors = {"interest_rate_shock": 0.02}
        modified_request = tester._apply_scenario_factors(request, market_factors)
        
        # Assert
        assert modified_request.pool.assets[0].rate == request.pool.assets[0].rate  # Fixed rate should remain unchanged
        assert "market_factors" in modified_request.pool.metadata
        assert modified_request.pool.metadata["market_factors"] == market_factors
        
        # Act - Test with floating rate asset
        request.pool.assets[0].rate_type = "floating"
        modified_request = tester._apply_scenario_factors(request, market_factors)
        
        # Assert
        assert modified_request.pool.assets[0].rate == request.pool.assets[0].rate + market_factors["interest_rate_shock"]
        
        # Act - Test discount rate shock
        market_factors = {"discount_rate_shock": 0.01}
        modified_request = tester._apply_scenario_factors(request, market_factors)
        
        # Assert
        assert modified_request.discount_rate == request.discount_rate + market_factors["discount_rate_shock"]
    
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_determine_predominant_asset_class(self, mock_factory, mock_redis_cache):
        """Test correct identification of predominant asset class"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.return_value = AsyncMock()
        tester = AssetStressTester()
        
        pool = sample_asset_pool()
        
        # Act
        asset_class = tester._determine_predominant_asset_class(pool)
        
        # Assert
        assert asset_class == AssetClass.RESIDENTIAL_MORTGAGE  # Mortgage has higher balance
        
        # Act - Change balances to make auto loan predominant
        pool.assets[0].balance = 10000  # Mortgage
        pool.assets[1].balance = 35000  # Auto loan
        asset_class = tester._determine_predominant_asset_class(pool)
        
        # Assert
        assert asset_class == AssetClass.AUTO_LOAN
        
        # Act - Test with empty pool
        empty_pool = AssetPool(
            pool_id="EMPTY",
            pool_name="Empty Pool",
            assets=[],
            cutoff_date=date.today()
        )
        
        # Assert
        with pytest.raises(ValueError, match="Asset pool is empty"):
            tester._determine_predominant_asset_class(empty_pool)
    
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_generate_cache_key(self, mock_factory, mock_redis_cache):
        """Test cache key generation is deterministic"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.return_value = AsyncMock()
        tester = AssetStressTester()
        
        request = sample_analysis_request()
        user_id = "test_user"
        scenario_name = "rate_shock_up"
        
        # Act
        key1 = tester._generate_cache_key(request, user_id, scenario_name)
        key2 = tester._generate_cache_key(request, user_id, scenario_name)
        
        # Assert
        assert key1 == key2  # Keys should be identical for identical inputs
        assert f"user_{user_id}" in key1
        assert f"pool_{request.pool.pool_name.lower().replace(' ', '_')}" in key1
        assert f"scenario_{scenario_name}" in key1
        assert f"discount_{request.discount_rate}" in key1
        
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_cache_result(self, mock_factory, mock_redis_cache, sample_analysis_response):
        """Test proper caching of analysis results"""
        # Arrange
        mock_factory.return_value = MagicMock()
        cache_mock = AsyncMock(spec=RedisCache)
        cache_mock.set = AsyncMock(return_value=True)
        mock_redis_cache.return_value = cache_mock
        
        tester = AssetStressTester()
        tester.redis_cache = cache_mock
        
        cache_key = "test_cache_key"
        result = sample_analysis_response
        
        # Act
        success = await tester._cache_result(cache_key, result)
        
        # Assert
        assert success is True
        cache_mock.set.assert_called_once()
        
        # Act - Test with cache exception
        cache_mock.set.side_effect = Exception("Cache error")
        success = await tester._cache_result(cache_key, result)
        
        # Assert
        assert success is False
        
        # Act - Test with no redis cache
        tester.redis_cache = None
        success = await tester._cache_result(cache_key, result)
        
        # Assert
        assert success is False
    
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_get_cached_result(self, mock_factory, mock_redis_cache, sample_analysis_response):
        """Test proper retrieval of cached analysis results"""
        # Arrange
        mock_factory.return_value = MagicMock()
        cache_mock = AsyncMock(spec=RedisCache)
        mock_redis_cache.return_value = cache_mock
        
        tester = AssetStressTester()
        tester.redis_cache = cache_mock
        
        cache_key = "test_cache_key"
        result_json = json.dumps(sample_analysis_response.model_dump())
        
        # Act - Test with cache hit
        cache_mock.get = AsyncMock(return_value=result_json)
        cached_result = await tester._get_cached_result(cache_key)
        
        # Assert
        assert cached_result is not None
        assert cached_result.pool_name == sample_analysis_response.pool_name
        assert cached_result.metrics.npv == sample_analysis_response.metrics.npv
        cache_mock.get.assert_called_once_with(cache_key)
        
        # Act - Test with cache miss
        cache_mock.get = AsyncMock(return_value=None)
        cached_result = await tester._get_cached_result(cache_key)
        
        # Assert
        assert cached_result is None
        
        # Act - Test with invalid JSON
        cache_mock.get = AsyncMock(return_value="{invalid json}")
        cache_mock.delete = AsyncMock(return_value=True)
        cached_result = await tester._get_cached_result(cache_key)
        
        # Assert
        assert cached_result is None
        cache_mock.delete.assert_called_once_with(cache_key)
        
        # Act - Test with no redis cache
        tester.redis_cache = None
        cached_result = await tester._get_cached_result(cache_key)
        
        # Assert
        assert cached_result is None

    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_run_scenario_analysis(self, mock_factory, mock_redis_cache, sample_analysis_request, sample_analysis_response):
        """Test running scenario analysis with proper handler selection"""
        # Arrange
        handler_mock = AsyncMock()
        handler_mock.analyze_pool = AsyncMock(return_value=sample_analysis_response)
        
        factory_mock = MagicMock()
        factory_mock.is_supported = MagicMock(return_value=True)
        factory_mock.get_handler = MagicMock(return_value=handler_mock)
        
        mock_factory.return_value = factory_mock
        mock_redis_cache.return_value = AsyncMock()
        
        tester = AssetStressTester()
        tester.handler_factory = factory_mock
        
        scenario = {
            "name": "Test Scenario",
            "description": "Scenario for testing",
            "market_factors": {"interest_rate_shock": 0.01}
        }
        
        # Act
        result = await tester._run_scenario_analysis(sample_analysis_request, scenario)
        
        # Assert
        assert result.status == "success"
        assert result.analytics.get("scenario").get("name") == scenario["name"]
        factory_mock.is_supported.assert_called_once()
        factory_mock.get_handler.assert_called_once()
        handler_mock.analyze_pool.assert_called_once_with(sample_analysis_request)
        
        # Act - Test with exception
        handler_mock.analyze_pool.side_effect = Exception("Analysis error")
        result = await tester._run_scenario_analysis(sample_analysis_request, scenario)
        
        # Assert
        assert result.status == "error"
        assert "error" in result.model_dump()
        assert "Analysis error" in result.error
    
    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_run_stress_tests(self, mock_factory, mock_redis_cache, sample_analysis_request, sample_analysis_response):
        """Test running a full set of stress tests"""
        # Arrange
        handler_mock = AsyncMock()
        handler_mock.analyze_pool = AsyncMock(return_value=sample_analysis_response)
        
        factory_mock = MagicMock()
        factory_mock.is_supported = MagicMock(return_value=True)
        factory_mock.get_handler = MagicMock(return_value=handler_mock)
        
        cache_mock = AsyncMock(spec=RedisCache)
        cache_mock.get = AsyncMock(return_value=None)
        cache_mock.set = AsyncMock(return_value=True)
        
        mock_factory.return_value = factory_mock
        mock_redis_cache.return_value = cache_mock
        
        tester = AssetStressTester()
        tester.handler_factory = factory_mock
        tester.redis_cache = cache_mock
        
        # Mock method to simplify test
        original_run_scenario = tester._run_scenario_analysis
        tester._run_scenario_analysis = AsyncMock(return_value=sample_analysis_response)
        
        # Act - Run with default scenarios
        results = await tester.run_stress_tests(
            request=sample_analysis_request,
            user_id="test_user",
            use_cache=True,
            parallel=False
        )
        
        # Assert
        assert len(results) == len(tester.default_scenarios)
        assert "base" in results
        assert results["base"].status == "success"
        
        # Act - Run with specified scenarios
        results = await tester.run_stress_tests(
            request=sample_analysis_request,
            user_id="test_user",
            scenario_names=["base", "credit_crisis"],
            use_cache=True,
            parallel=False
        )
        
        # Assert
        assert len(results) == 2
        assert "base" in results
        assert "credit_crisis" in results
        
        # Act - Run with custom scenarios
        custom_scenarios = {
            "custom_scenario": {
                "name": "Custom Scenario",
                "description": "Test custom scenario",
                "market_factors": {
                    "interest_rate_shock": 0.05,
                    "default_multiplier": 2.0
                }
            }
        }
        
        results = await tester.run_stress_tests(
            request=sample_analysis_request,
            user_id="test_user",
            scenario_names=["base"],
            custom_scenarios=custom_scenarios,
            use_cache=True,
            parallel=False
        )
        
        # Assert
        assert len(results) == 2
        assert "base" in results
        assert "custom_scenario" in results
        
        # Restore original method
        tester._run_scenario_analysis = original_run_scenario

    @pytest.mark.asyncio
    @patch('app.services.asset_handlers.stress_testing.RedisCache')
    @patch('app.services.asset_handlers.stress_testing.AssetHandlerFactory')
    async def test_generate_stress_test_report(self, mock_factory, mock_redis_cache, sample_analysis_response):
        """Test generation of comprehensive stress test report"""
        # Arrange
        mock_factory.return_value = MagicMock()
        mock_redis_cache.return_value = AsyncMock()
        
        tester = AssetStressTester()
        
        # Create results dictionary
        base_result = sample_analysis_response
        
        scenario_result = sample_analysis_response.model_copy(deep=True)
        scenario_result.metrics.npv = base_result.metrics.npv * 0.9  # 10% reduction
        
        results = {
            "base": base_result,
            "credit_crisis": scenario_result
        }
        
        # Act
        report = await tester.generate_stress_test_report(results)
        
        # Assert
        assert report is not None
        assert "pool_name" in report
        assert "scenarios_count" in report
        assert report["scenarios_count"] == 2
        assert "base_case" in report
        assert "scenarios" in report
        assert "credit_crisis" in report["scenarios"]
        assert "metrics" in report["scenarios"]["credit_crisis"]
        assert "npv_change_percent" in report["scenarios"]["credit_crisis"]["metrics"]
        
        # Act - Test with missing base case
        results_without_base = {"credit_crisis": scenario_result}
        report = await tester.generate_stress_test_report(results_without_base)
        
        # Assert
        assert "status" in report
        assert report["status"] == "error"
        assert "error" in report
        
        # Act - Test with failed base case
        failed_base = sample_analysis_response.model_copy(deep=True)
        failed_base.status = "error"
        failed_base.error = "Base case failed"
        
        results_with_failed_base = {
            "base": failed_base,
            "credit_crisis": scenario_result
        }
        
        report = await tester.generate_stress_test_report(results_with_failed_base)
        
        # Assert
        assert "status" in report
        assert report["status"] == "error"
