"""Cashflow models for the credit-cashflow-engine"""
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class LoanData(BaseModel):
    """Model representing loan data for cashflow projections"""
    principal: float = Field(gt=0, description="Principal amount of the loan")
    interest_rate: float = Field(ge=0, le=1, description="Annual interest rate as decimal")
    term_months: int = Field(gt=0, le=600, description="Loan term in months")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    prepayment_assumption: Optional[float] = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Annual prepayment rate as decimal"
    )
    interest_only_periods: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of interest-only periods"
    )
    balloon_payment: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional balloon payment amount"
    )
    rate_type: str = Field(
        default="fixed",
        pattern="^(fixed|floating|hybrid)$",
        description="Interest rate type: fixed, floating, or hybrid"
    )
    rate_spread: Optional[float] = Field(
        default=None,
        ge=0,
        description="Spread over reference rate for floating/hybrid rates"
    )
    rate_cap: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Maximum interest rate for floating/hybrid rates"
    )
    rate_floor: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Minimum interest rate for floating/hybrid rates"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "principal": 100000.0,
                "interest_rate": 0.05,
                "term_months": 360,
                "start_date": "2025-01-01",
                "prepayment_assumption": 0.02,
                "interest_only_periods": 0,
                "balloon_payment": None,
                "rate_type": "fixed",
                "rate_spread": None,
                "rate_cap": None,
                "rate_floor": None
            }
        }
    }

    @field_validator('start_date')
    def validate_start_date(cls, v):
        """Validate start_date is in YYYY-MM-DD format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("start_date must be in YYYY-MM-DD format")

class EconomicFactors(BaseModel):
    """Model representing economic factors for scenario analysis"""
    market_rate: float = Field(default=0.045, ge=0, le=1, description="Market interest rate")
    inflation_rate: float = Field(default=0.02, ge=-0.1, le=0.5, description="Inflation rate")
    unemployment_rate: float = Field(default=0.05, ge=0, le=1, description="Unemployment rate")
    gdp_growth: float = Field(default=0.025, ge=-0.5, le=0.5, description="GDP growth rate")
    house_price_appreciation: float = Field(default=0.03, ge=-0.5, le=0.5, description="House price appreciation")
    month: int = Field(default=6, ge=1, le=12, description="Current month (1-12)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "market_rate": 0.045,
                "inflation_rate": 0.02,
                "unemployment_rate": 0.05,
                "gdp_growth": 0.025,
                "house_price_appreciation": 0.03,
                "month": 6
            }
        }
    }

class StressTestScenario(BaseModel):
    name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    rate_shock: float = Field(default=0, description="Interest rate shock")
    default_multiplier: float = Field(default=1, ge=0, description="Default probability multiplier")
    prepay_multiplier: float = Field(default=1, ge=0, description="Prepayment multiplier")
    economic_factors: Optional[EconomicFactors] = Field(None, description="Economic factors override")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Stress Test Scenario",
                "description": "Example stress test scenario",
                "rate_shock": 0.01,
                "default_multiplier": 1.5,
                "prepay_multiplier": 1.2,
                "economic_factors": {
                    "market_rate": 0.045,
                    "inflation_rate": 0.02,
                    "unemployment_rate": 0.05,
                    "gdp_growth": 0.025,
                    "house_price_appreciation": 0.03,
                    "month": 6
                }
            }
        }
    }

class MonteCarloConfig(BaseModel):
    num_simulations: int = Field(
        default=1000,
        gt=0,
        le=10000,
        description="Number of Monte Carlo simulations to run"
    )
    default_prob: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Base probability of default"
    )
    prepay_prob: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Base probability of prepayment"
    )
    rate_volatility: float = Field(
        default=0.01,
        gt=0,
        le=1,
        description="Interest rate volatility"
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Correlation matrix between risk factors"
    )
    stress_scenarios: List[StressTestScenario] = Field(
        default_factory=list,
        max_length=10,
        description="List of stress test scenarios to run"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_simulations": 1000,
                "default_prob": 0.02,
                "prepay_prob": 0.05,
                "rate_volatility": 0.01,
                "correlation_matrix": {},
                "stress_scenarios": [],
                "seed": None
            }
        }
    }

class CashflowForecastRequest(BaseModel):
    loans: List[LoanData] = Field(..., min_length=1, max_length=1000)
    discount_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Discount rate for NPV calculations"
    )
    run_monte_carlo: bool = Field(
        default=True,
        description="Whether to run Monte Carlo simulation"
    )
    monte_carlo_config: Optional[MonteCarloConfig] = Field(
        default_factory=MonteCarloConfig,
        description="Monte Carlo simulation configuration"
    )
    scenario_name: Optional[str] = Field(
        default=None,
        description="Optional scenario name"
    )
    assumptions: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional assumptions for the model"
    )
    economic_factors: Optional[EconomicFactors] = Field(
        default=None,
        description="Economic factors for scenario analysis"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "loans": [],
                "discount_rate": 0.05,
                "run_monte_carlo": True,
                "monte_carlo_config": {},
                "scenario_name": None,
                "assumptions": {},
                "economic_factors": None
            }
        }
    }

class BatchForecastRequest(BaseModel):
    forecasts: List[CashflowForecastRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of forecast requests to process in batch"
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process forecasts in parallel"
    )
    chunk_size: int = Field(
        default=10,
        gt=0,
        le=50,
        description="Size of parallel processing chunks"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "forecasts": [],
                "parallel": True,
                "chunk_size": 10
            }
        }
    }

class CashflowProjection(BaseModel):
    """Model representing a single cashflow projection period"""
    period: int = Field(ge=0, description="Payment period number")
    date: str = Field(..., description="Payment date in YYYY-MM-DD format")
    principal: float = Field(ge=0, description="Principal payment amount")
    interest: float = Field(ge=0, description="Interest payment amount")
    total_payment: float = Field(ge=0, description="Total payment amount")
    remaining_balance: float = Field(ge=0, description="Remaining loan balance")
    is_interest_only: bool = Field(default=False, description="Whether this is an interest-only payment")
    is_balloon: bool = Field(default=False, description="Whether this is a balloon payment")
    rate: float = Field(ge=0, le=1, description="Effective interest rate for the period")

    model_config = {
        "json_schema_extra": {
            "example": {
                "period": 1,
                "date": "2025-01-01",
                "principal": 1000.0,
                "interest": 50.0,
                "total_payment": 1050.0,
                "remaining_balance": 99000.0,
                "is_interest_only": False,
                "is_balloon": False,
                "rate": 0.05
            }
        }
    }

    @field_validator('date')
    def validate_date(cls, v):
        """Validate date is in YYYY-MM-DD format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")

class MonteCarloResults(BaseModel):
    """Results from Monte Carlo simulation"""
    npv_distribution: List[float] = Field(
        ...,
        description="Distribution of NPV values from simulations"
    )
    default_scenarios: List[Dict[str, Any]] = Field(
        ...,
        description="Default scenarios from simulation"
    )
    prepayment_scenarios: List[Dict[str, Any]] = Field(
        ...,
        description="Prepayment scenarios from simulation"
    )
    rate_scenarios: List[Dict[str, Any]] = Field(
        ...,
        description="Interest rate scenarios from simulation"
    )
    confidence_intervals: Dict[str, Dict[str, List[float]]] = Field(
        ...,
        description="Confidence intervals for key metrics"
    )
    stress_test_results: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Results from stress test scenarios"
    )
    var_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Value at Risk metrics"
    )
    sensitivity_analysis: Optional[Dict[str, float]] = Field(
        None,
        description="Sensitivity analysis results"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "npv_distribution": [95000.0, 98000.0, 100000.0, 102000.0, 105000.0],
                "default_scenarios": [{"period": 1, "probability": 0.02}],
                "prepayment_scenarios": [{"period": 1, "probability": 0.05}],
                "rate_scenarios": [{"period": 1, "rate": 0.05}],
                "confidence_intervals": {
                    "npv": {"95": [95000.0, 105000.0]},
                    "irr": {"95": [0.05, 0.07]}
                },
                "stress_test_results": {
                    "recession": {"npv": 90000.0, "irr": 0.04}
                },
                "var_metrics": {"95": 95000.0},
                "sensitivity_analysis": {"rate": -0.5, "default": -0.3}
            }
        }
    }

class CashflowForecastResponse(BaseModel):
    """Response model for cashflow forecast requests"""
    scenario_id: Optional[str] = Field(default=None, description="ID of the saved scenario")
    projections: List[CashflowProjection] = Field(..., description="List of cashflow projections")
    summary_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "total_principal": 0.0,
            "total_interest": 0.0,
            "total_payments": 0.0,
            "npv": 0.0,
            "irr": 0.0,
            "duration": 0.0,
            "convexity": 0.0,
            "interest_coverage_ratio": 0.0,
            "debt_service_coverage": 0.0,
            "weighted_average_life": 0.0
        },
        description="Summary metrics for the forecast"
    )
    monte_carlo_results: Optional[MonteCarloResults] = Field(
        default=None,
        description="Monte Carlo simulation results"
    )
    economic_scenario: Optional[EconomicFactors] = Field(
        default=None,
        description="Economic scenario used for the forecast"
    )
    computation_time: float = Field(
        ...,
        description="Time taken to compute the forecast in seconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "scenario_id": "abc123",
                "projections": [],
                "summary_metrics": {
                    "total_principal": 100000.0,
                    "total_interest": 50000.0,
                    "total_payments": 150000.0,
                    "npv": 95000.0,
                    "irr": 0.06,
                    "duration": 5.0,
                    "convexity": 0.5,
                    "interest_coverage_ratio": 1.25,
                    "debt_service_coverage": 1.25,
                    "weighted_average_life": 7.5
                },
                "computation_time": 1.5
            }
        }
    }

class ScenarioSaveRequest(BaseModel):
    name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    forecast_request: CashflowForecastRequest
    tags: List[str] = Field(
        default_factory=list,
        max_length=10
    )
    is_template: bool = Field(
        default=False,
        description="Whether this scenario should be saved as a template"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Example Scenario",
                "description": "Example scenario description",
                "forecast_request": {},
                "tags": [],
                "is_template": False
            }
        }
    }

class ScenarioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    tags: List[str]
    forecast_request: CashflowForecastRequest
    is_template: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "abc123",
                "name": "Example Scenario",
                "description": "Example scenario description",
                "created_at": "2025-01-01",
                "updated_at": "2025-01-01",
                "tags": [],
                "forecast_request": {},
                "is_template": False
            }
        }
    }
