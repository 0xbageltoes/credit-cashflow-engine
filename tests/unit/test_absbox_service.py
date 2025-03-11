"""
Tests for the AbsBox service integration
"""
import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from app.services.absbox_service import AbsBoxService
from app.models.structured_products import (
    StructuredDealRequest,
    LoanPoolConfig,
    LoanConfig,
    WaterfallConfig,
    AccountConfig,
    BondConfig,
    WaterfallAction,
    ScenarioConfig,
    DefaultCurveConfig
)

# Test data helpers
def get_test_loan_config():
    """Get a test loan configuration"""
    loan_date = date.today() - timedelta(days=30)
    return LoanConfig(
        balance=100000.0,
        rate=0.05,
        term=360,
        start_date=loan_date,
        rate_type="fixed",
        payment_frequency="Monthly"
    )

def get_test_pool_config():
    """Get a test loan pool configuration"""
    return LoanPoolConfig(
        pool_name="Test Pool",
        loans=[get_test_loan_config(), get_test_loan_config()]
    )

def get_test_waterfall_config():
    """Get a test waterfall configuration"""
    return WaterfallConfig(
        start_date=date.today(),
        accounts=[
            AccountConfig(name="ReserveFund", initial_balance=10000.0)
        ],
        bonds=[
            BondConfig(name="ClassA", balance=150000.0, rate=0.04),
            BondConfig(name="ClassB", balance=30000.0, rate=0.05)
        ],
        actions=[
            WaterfallAction(source="CollectedInterest", target="ClassA", amount="Interest"),
            WaterfallAction(source="CollectedInterest", target="ClassB", amount="Interest"),
            WaterfallAction(source="CollectedPrincipal", target="ClassA", amount="OutstandingPrincipal"),
            WaterfallAction(source="CollectedPrincipal", target="ClassB", amount="OutstandingPrincipal")
        ]
    )

def get_test_scenario_config():
    """Get a test scenario configuration"""
    return ScenarioConfig(
        name="Base Case",
        default_curve=DefaultCurveConfig(
            vector=[0.01, 0.02, 0.03, 0.02, 0.01]
        )
    )

def get_test_deal_request():
    """Get a test deal request"""
    return StructuredDealRequest(
        deal_name="Test Deal",
        pool=get_test_pool_config(),
        waterfall=get_test_waterfall_config(),
        scenario=get_test_scenario_config()
    )

# Tests
@pytest.fixture
def mock_engine():
    """Mock the AbsBox engine"""
    with patch("absbox.local.engine.LiqEngine") as mock:
        # Create a mock result object
        mock_result = MagicMock()
        mock_result.runTime.return_value = 0.5
        mock_result.bondFlow.return_value = pd.DataFrame({
            "ClassA": [100.0, 100.0],
            "ClassB": [50.0, 50.0]
        })
        mock_result.poolFlow.return_value = pd.DataFrame({
            "principal": [150.0, 150.0], 
            "interest": [50.0, 50.0]
        })
        mock_result.poolStats.return_value = pd.Series({"totalPrincipal": 300.0, "totalInterest": 100.0})
        mock_result.bondMetrics.return_value = pd.DataFrame()
        mock_result.poolMetrics.return_value = pd.DataFrame()
        
        # Make the mock engine return the mock result
        engine_instance = mock.return_value
        engine_instance.runDeal.return_value = mock_result
        engine_instance.runPool.return_value = mock_result
        
        yield engine_instance

def test_absbox_service_initialization():
    """Test that the AbsBox service initializes correctly"""
    service = AbsBoxService()
    assert service is not None
    assert service.engine is not None

def test_create_loan_pool():
    """Test creating a loan pool"""
    service = AbsBoxService()
    pool_config = get_test_pool_config()
    
    # Create the pool
    pool = service.create_loan_pool(pool_config)
    
    # Check the pool
    assert pool is not None
    assert len(pool.assets) == 2
    assert pool.assets[0].balance == 100000.0
    assert pool.assets[0].rate == 0.05

def test_create_waterfall():
    """Test creating a waterfall"""
    service = AbsBoxService()
    waterfall_config = get_test_waterfall_config()
    
    # Create the waterfall
    waterfall = service.create_waterfall(waterfall_config)
    
    # Check the waterfall
    assert waterfall is not None
    assert len(waterfall.bonds) == 2
    assert waterfall.bonds[0].balance == 150000.0
    assert waterfall.bonds[1].rate == 0.05

def test_create_assumption():
    """Test creating assumptions"""
    service = AbsBoxService()
    scenario_config = get_test_scenario_config()
    
    # Create the assumption
    assumption = service.create_assumption(scenario_config)
    
    # Check the assumption
    assert assumption is not None
    assert assumption.defaultCurve is not None
    assert assumption.rateCurve is not None

def test_analyze_deal(mock_engine):
    """Test analyzing a deal"""
    service = AbsBoxService()
    deal_request = get_test_deal_request()
    
    # Analyze the deal
    result = service.analyze_deal(deal_request)
    
    # Check that the engine was called correctly
    assert mock_engine.runDeal.called
    
    # Check the result
    assert result.deal_name == "Test Deal"
    assert result.status == "success"
    assert result.execution_time == 0.5
    assert len(result.bond_cashflows) > 0
    assert len(result.pool_cashflows) > 0

def test_run_scenario_analysis(mock_engine):
    """Test running scenario analysis"""
    service = AbsBoxService()
    deal_request = get_test_deal_request()
    
    # Create several scenarios
    scenarios = [
        ScenarioConfig(
            name="Low Default",
            default_curve=DefaultCurveConfig(vector=[0.01, 0.01, 0.01])
        ),
        ScenarioConfig(
            name="High Default",
            default_curve=DefaultCurveConfig(vector=[0.05, 0.05, 0.05])
        )
    ]
    
    # Run scenario analysis
    results = service.run_scenario_analysis(deal_request, scenarios)
    
    # Check that the engine was called for each scenario
    assert mock_engine.runDeal.call_count == 2
    
    # Check the results
    assert len(results) == 2
    assert results[0].scenario_name == "Low Default"
    assert results[1].scenario_name == "High Default"
    assert results[0].status == "success"
    assert results[1].status == "success"

def test_health_check():
    """Test health check functionality"""
    service = AbsBoxService()
    
    # Run health check
    health = service.health_check()
    
    # Check the result
    assert health is not None
    assert "absbox_version" in health
    assert "engine_type" in health
