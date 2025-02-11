from pydantic import BaseModel, Field, validator, constr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
import re

class LoanData(BaseModel):
    principal: float = Field(gt=0, description="Principal amount of the loan")
    interest_rate: float = Field(ge=0, le=1, description="Annual interest rate as decimal")
    term_months: int = Field(gt=0, le=600, description="Loan term in months")
    start_date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$', description="Start date in YYYY-MM-DD format")
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
    
    @validator('start_date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('start_date must be in YYYY-MM-DD format')

class EconomicFactors(BaseModel):
    market_rate: float = Field(ge=0, le=1, description="Market interest rate")
    inflation_rate: float = Field(ge=-0.1, le=0.5, description="Inflation rate")
    unemployment_rate: float = Field(ge=0, le=1, description="Unemployment rate")
    gdp_growth: float = Field(ge=-0.5, le=0.5, description="GDP growth rate")
    house_price_appreciation: float = Field(ge=-0.5, le=0.5, description="House price appreciation")
    month: int = Field(ge=1, le=12, description="Current month (1-12)")

class StressTestScenario(BaseModel):
    name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    rate_shock: float = Field(default=0, description="Interest rate shock")
    default_multiplier: float = Field(default=1, ge=0, description="Default probability multiplier")
    prepay_multiplier: float = Field(default=1, ge=0, description="Prepayment multiplier")
    economic_factors: Optional[EconomicFactors] = Field(None, description="Economic factors override")

class MonteCarloConfig(BaseModel):
    num_simulations: int = Field(
        default=1000,
        gt=0,
        le=10000,
        description="Number of Monte Carlo simulations"
    )
    default_prob: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Annual default probability"
    )
    prepay_prob: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Base prepayment probability"
    )
    rate_volatility: float = Field(
        default=0.01,
        ge=0,
        le=0.5,
        description="Interest rate volatility"
    )
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Correlation matrix between risk factors"
    )
    stress_scenarios: Optional[List[StressTestScenario]] = Field(
        default=None,
        description="List of predefined stress scenarios"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

class CashflowForecastRequest(BaseModel):
    loans: List[LoanData] = Field(..., min_items=1, max_items=1000)
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
    scenario_name: Optional[constr(min_length=1, max_length=100)] = Field(
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

class BatchForecastRequest(BaseModel):
    forecasts: List[CashflowForecastRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
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

class CashflowProjection(BaseModel):
    period: int = Field(ge=0, description="Payment period number")
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$', description="Payment date")
    principal: float = Field(ge=0, description="Principal payment amount")
    interest: float = Field(ge=0, description="Interest payment amount")
    total_payment: float = Field(ge=0, description="Total payment amount")
    remaining_balance: float = Field(ge=0, description="Remaining loan balance")
    is_interest_only: bool = Field(default=False, description="Whether this is an interest-only payment")
    is_balloon: bool = Field(default=False, description="Whether this is a balloon payment")
    rate: float = Field(ge=0, le=1, description="Effective interest rate for the period")

class MonteCarloResults(BaseModel):
    npv_distribution: List[float]
    default_scenarios: List[Dict[str, Any]]
    prepayment_scenarios: List[Dict[str, Any]]
    rate_scenarios: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Dict[str, float]]
    stress_test_results: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
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

class CashflowForecastResponse(BaseModel):
    scenario_id: Optional[str]
    projections: List[CashflowProjection]
    summary_metrics: Dict[str, Any] = Field(
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
        }
    )
    monte_carlo_results: Optional[MonteCarloResults]
    stress_test_results: Optional[Dict[str, Dict[str, float]]]
    economic_scenario: Optional[EconomicFactors]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    computation_time: float = Field(ge=0, description="Computation time in seconds")

class ScenarioSaveRequest(BaseModel):
    name: constr(min_length=1, max_length=100)
    description: Optional[constr(max_length=500)]
    forecast_request: CashflowForecastRequest
    tags: List[constr(min_length=1, max_length=50)] = Field(
        default_factory=list,
        max_items=10
    )
    is_template: bool = Field(
        default=False,
        description="Whether this scenario should be saved as a template"
    )

class ScenarioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    tags: List[str]
    forecast_request: CashflowForecastRequest
    is_template: bool = Field(default=False)
