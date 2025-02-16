"""Test data for credit-cashflow-engine tests"""
from datetime import datetime
from app.models.cashflow import (
    LoanData,
    MonteCarloConfig,
    EconomicFactors,
    StressTestScenario
)
from typing import Dict

def get_sample_loan() -> LoanData:
    """Get a sample loan for testing"""
    return LoanData(
        principal=100000.0,
        interest_rate=0.05,
        term_months=360,
        start_date=datetime.now().strftime("%Y-%m-%d"),
        prepayment_assumption=0.02,
        rate_type="fixed",
        balloon_payment=None,
        interest_only_periods=0,
        rate_spread=None,
        rate_cap=None,
        rate_floor=None
    )

def get_interest_only_loan() -> LoanData:
    """Get an interest-only loan for testing"""
    return LoanData(
        principal=200000.0,
        interest_rate=0.06,
        term_months=240,
        start_date=datetime.now().strftime("%Y-%m-%d"),
        interest_only_periods=60,
        prepayment_assumption=0.01,
        rate_type="fixed",
        balloon_payment=None,
        rate_spread=None,
        rate_cap=None,
        rate_floor=None
    )

def get_balloon_loan() -> LoanData:
    """Get a balloon payment loan for testing"""
    return LoanData(
        principal=300000.0,
        interest_rate=0.045,
        term_months=120,
        start_date=datetime.now().strftime("%Y-%m-%d"),
        balloon_payment=250000.0,
        prepayment_assumption=0.015,
        rate_type="fixed",
        interest_only_periods=0,
        rate_spread=None,
        rate_cap=None,
        rate_floor=None
    )

def get_hybrid_rate_loan() -> LoanData:
    """Get a hybrid rate loan for testing"""
    return LoanData(
        principal=400000.0,
        interest_rate=0.04,
        term_months=360,
        start_date=datetime.now().strftime("%Y-%m-%d"),
        rate_type="hybrid",
        rate_spread=0.02,
        rate_cap=0.08,
        rate_floor=0.03,
        prepayment_assumption=0.02,
        balloon_payment=None,
        interest_only_periods=0
    )

def get_economic_factors() -> Dict:
    """Get sample economic factors"""
    return {
        "market_rate": 0.045,
        "inflation_rate": 0.02,
        "unemployment_rate": 0.05,
        "gdp_growth": 0.025,
        "house_price_appreciation": 0.03,
        "month": 6
    }

def get_monte_carlo_config() -> MonteCarloConfig:
    """Get Monte Carlo simulation config"""
    return MonteCarloConfig(
        num_simulations=100,  # Reduced for faster tests
        default_prob=0.02,
        prepay_prob=0.05,
        rate_volatility=0.01,
        correlation_matrix={
            "rate": {"default": 0.3, "prepay": -0.2},
            "default": {"prepay": -0.1}
        },
        stress_scenarios=[get_stress_test_scenario()]
    )

def get_stress_test_scenario() -> StressTestScenario:
    """Get stress test scenario"""
    return StressTestScenario(
        name="High Interest Rate",
        description="High interest rate stress scenario",
        rate_shock=0.02,
        default_multiplier=2.0,
        prepay_multiplier=0.5,
        economic_factors={
            "market_rate": 0.07,
            "inflation_rate": 0.04,
            "unemployment_rate": 0.08,
            "gdp_growth": -0.02,
            "house_price_appreciation": -0.05,
            "month": 6
        }
    )
