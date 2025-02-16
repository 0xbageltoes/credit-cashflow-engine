"""Test module for the cashflow engine"""
import pytest
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.cashflow import CashflowService
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowProjection,
    MonteCarloConfig,
    EconomicFactors,
    StressTestScenario,
    MonteCarloResults,
    LoanData
)
from app.models.analytics import AnalyticsResult
from app.core.redis_cache import RedisCache
from app.services.analytics import AnalyticsService
from .test_data import (
    get_sample_loan,
    get_interest_only_loan,
    get_balloon_loan,
    get_hybrid_rate_loan,
    get_economic_factors,
    get_monte_carlo_config,
    get_stress_test_scenario
)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def mock_supabase():
    """Mock Supabase client"""
    mock = MagicMock()
    mock.table = MagicMock()
    mock.table().select = MagicMock()
    mock.table().select().single = MagicMock()
    mock.table().select().single().execute = AsyncMock()
    mock.table().select().single().execute.return_value = MagicMock(data={
        "market_rate": 0.045,
        "inflation_rate": 0.02,
        "unemployment_rate": 0.05,
        "gdp_growth": 0.025,
        "house_price_appreciation": 0.03,
        "month": datetime.now().month
    })
    return mock

@pytest.fixture(scope="function")
async def mock_redis():
    """Mock Redis cache"""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    return mock

@pytest.fixture(scope="function")
async def mock_analytics():
    """Mock AnalyticsService"""
    mock = MagicMock()
    mock.calculate_metrics = AsyncMock(return_value=AnalyticsResult(
        npv=100000.0,
        irr=0.06,
        dscr=1.25,
        ltv=0.75
    ))
    return mock

@pytest.fixture(scope="function")
async def cashflow_service(mock_supabase, mock_redis, mock_analytics):
    """Create an instance of CashflowService for testing."""
    service = CashflowService()
    service.supabase = mock_supabase
    service.cache = mock_redis
    service.analytics = mock_analytics
    return service

@pytest.mark.asyncio
class TestCashflowCalculations:
    """Test basic cashflow calculations"""
    
    async def test_basic_loan_calculation(self, cashflow_service):
        """Test basic loan amortization calculation"""
        loan = get_sample_loan()
        request = CashflowForecastRequest(loans=[loan])
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        assert response is not None
        assert len(response.projections) == loan.term_months
        assert response.summary_metrics["total_principal"] == pytest.approx(loan.principal)
        
        # Verify payment consistency
        total_payments = sum(p.total_payment for p in response.projections)
        assert total_payments > loan.principal  # Should include interest
        
        # Verify final balance is zero
        assert response.projections[-1].remaining_balance == pytest.approx(0, abs=0.01)

    async def test_interest_only_loan(self, cashflow_service):
        """Test interest-only period calculations"""
        loan = get_interest_only_loan()
        request = CashflowForecastRequest(loans=[loan])
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        # Check interest-only period payments
        for i in range(loan.interest_only_periods):
            payment = response.projections[i]
            assert payment.is_interest_only
            assert payment.principal == pytest.approx(0, abs=0.01)
            assert payment.interest == pytest.approx(
                loan.principal * loan.interest_rate / 12,
                abs=0.01
            )

    async def test_balloon_payment(self, cashflow_service):
        """Test balloon payment calculation"""
        loan = get_balloon_loan()
        request = CashflowForecastRequest(loans=[loan])
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        # Verify balloon payment
        final_payment = response.projections[-1]
        assert final_payment.is_balloon
        assert final_payment.principal == pytest.approx(loan.balloon_payment, abs=0.01)

    async def test_hybrid_rate_loan(self, cashflow_service):
        """Test hybrid rate loan calculations"""
        loan = get_hybrid_rate_loan()
        economic_factors = get_economic_factors()
        
        request = CashflowForecastRequest(
            loans=[loan],
            economic_factors=economic_factors
        )
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        # Verify rate adjustments
        for proj in response.projections:
            assert proj.rate >= loan.rate_floor
            assert proj.rate <= loan.rate_cap

@pytest.mark.asyncio
class TestMonteCarloSimulation:
    """Test Monte Carlo simulation features"""
    
    async def test_monte_carlo_simulation(self, cashflow_service):
        """Test Monte Carlo simulation results"""
        loan = get_sample_loan()
        monte_carlo_config = get_monte_carlo_config()
        economic_factors = get_economic_factors()
        
        request = CashflowForecastRequest(
            loans=[loan],
            monte_carlo_config=monte_carlo_config,
            economic_factors=economic_factors,
            run_monte_carlo=True
        )
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        assert response.monte_carlo_results is not None
        assert len(response.monte_carlo_results.npv_distribution) == monte_carlo_config.num_simulations
        
        # Verify confidence intervals
        ci = response.monte_carlo_results.confidence_intervals
        assert "npv" in ci
        assert "95" in ci["npv"]
        assert len(ci["npv"]["95"]) == 2
        assert ci["npv"]["95"][0] < ci["npv"]["95"][1]

    async def test_stress_scenarios(self, cashflow_service):
        """Test stress testing scenarios"""
        loan = get_sample_loan()
        monte_carlo_config = get_monte_carlo_config()
        economic_factors = get_economic_factors()
        stress_scenario = get_stress_test_scenario()
        
        request = CashflowForecastRequest(
            loans=[loan],
            monte_carlo_config=monte_carlo_config,
            economic_factors=economic_factors,
            run_monte_carlo=True
        )
        response = await cashflow_service.generate_forecast(request, "test_user")
        
        assert response.monte_carlo_results.stress_test_results is not None
        assert "High Interest Rate" in response.monte_carlo_results.stress_test_results

@pytest.mark.asyncio
class TestBatchProcessing:
    """Test batch processing capabilities"""
    
    async def test_batch_forecast(self, cashflow_service):
        """Test batch processing of multiple loans"""
        loans = [
            get_sample_loan(),
            get_interest_only_loan(),
            get_balloon_loan()
        ]
        requests = [CashflowForecastRequest(loans=[loan]) for loan in loans]
        
        # Process forecasts in parallel
        responses = await asyncio.gather(*[
            cashflow_service.generate_forecast(req, "test_user")
            for req in requests
        ])
        
        assert len(responses) == len(loans)
        for response in responses:
            assert response is not None
            assert response.summary_metrics is not None

@pytest.mark.asyncio
class TestEconomicFactors:
    """Test economic factor adjustments"""
    
    async def test_economic_factor_impacts(self, cashflow_service):
        """Test impact of economic factors on forecasts"""
        loan = get_sample_loan()
        economic_factors = get_economic_factors()
        
        # Base case without economic factors
        base_request = CashflowForecastRequest(loans=[loan])
        base_response = await cashflow_service.generate_forecast(base_request, "test_user")
        
        # Case with economic factors
        factor_request = CashflowForecastRequest(
            loans=[loan],
            economic_factors=economic_factors
        )
        factor_response = await cashflow_service.generate_forecast(factor_request, "test_user")
        
        # Verify economic factors impact the results
        assert base_response.summary_metrics["npv"] != factor_response.summary_metrics["npv"]
