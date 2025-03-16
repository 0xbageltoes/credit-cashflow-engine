"""Cashflow models for the credit-cashflow-engine"""
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class LoanData(BaseModel):
    """Model representing loan data for cashflow projections"""
    loan_id: str = Field(..., description="Unique loan identifier")
    principal: float = Field(gt=0, description="Principal amount of the loan")
    interest_rate: float = Field(ge=0, le=1, description="Annual interest rate as decimal")
    term_months: int = Field(gt=0, le=600, description="Loan term in months")
    origination_date: date = Field(..., description="Origination date of the loan")
    first_payment_date: Optional[date] = Field(default=None, description="Date of first payment")
    loan_type: str = Field(
        default="fixed",
        pattern="^(fixed|floating|hybrid)$",
        description="Interest rate type: fixed, floating, or hybrid"
    )
    prepayment_penalty: Optional[float] = Field(default=0.0, description="Prepayment penalty as percentage")
    default_probability: Optional[float] = Field(default=0.0, description="Annual probability of default")
    recovery_rate: Optional[float] = Field(default=1.0, description="Recovery rate in case of default")
    prepayment_model: Optional[str] = Field(default="CPR", description="Prepayment model (CPR, PSA, SMM)")
    prepayment_rate: Optional[float] = Field(default=0.0, description="Annual prepayment rate")
    prepayment_multiplier: Optional[float] = Field(default=1.0, description="Prepayment rate multiplier")
    payment_frequency: str = Field(default="monthly", description="Payment frequency (monthly, quarterly, etc.)")
    loan_purpose: Optional[str] = Field(default=None, description="Purpose of the loan")
    maturity_date: Optional[date] = Field(default=None, description="Maturity date of the loan")
    amortization_type: str = Field(default="level_payment", description="Amortization type (level_payment, interest_only)")
    interest_only_period: Optional[int] = Field(default=0, description="Number of interest-only periods")
    balloon_payment: Optional[float] = Field(default=0.0, description="Optional balloon payment amount")
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, description="Custom fields for loan data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_id": "loan-123",
                "principal": 100000.0,
                "interest_rate": 0.05,
                "term_months": 360,
                "origination_date": "2023-01-01",
                "loan_type": "fixed",
                "payment_frequency": "monthly",
                "amortization_type": "level_payment"
            }
        }
    }

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
    """Model representing a stress test scenario for cashflow analysis"""
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
    """Configuration for Monte Carlo simulation in cashflow forecasting"""
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
    """Request model for cashflow forecasting with enhanced analytics"""
    loan_data: LoanData
    discount_rate: Optional[float] = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Discount rate for NPV calculations"
    )
    prepayment_model: Optional[str] = Field(
        default="CPR",
        description="Prepayment model to use (CPR, PSA, SMM, custom)"
    )
    default_model: Optional[str] = Field(
        default="CDR",
        description="Default model to use (CDR, MDR, custom)"
    )
    run_monte_carlo: Optional[bool] = Field(
        default=False,
        description="Whether to run Monte Carlo simulation"
    )
    monte_carlo_iterations: Optional[int] = Field(
        default=1000,
        description="Number of Monte Carlo iterations"
    )
    include_analytics: Optional[bool] = Field(
        default=True,
        description="Whether to include advanced analytics metrics"
    )
    calculate_sensitivity: Optional[bool] = Field(
        default=False,
        description="Whether to calculate sensitivity analysis"
    )
    monte_carlo_config: Optional[MonteCarloConfig] = Field(
        default=None,
        description="Monte Carlo simulation configuration"
    )
    economic_factors: Optional[EconomicFactors] = Field(
        default=None,
        description="Economic factors for the forecast"
    )
    stress_scenarios: Optional[List[StressTestScenario]] = Field(
        default=None,
        description="Stress scenarios to run"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_data": {
                    "loan_id": "loan-123",
                    "principal": 100000.0,
                    "interest_rate": 0.05,
                    "term_months": 360,
                    "origination_date": "2023-01-01",
                    "loan_type": "fixed",
                    "payment_frequency": "monthly",
                    "amortization_type": "level_payment"
                },
                "discount_rate": 0.05,
                "run_monte_carlo": False,
                "include_analytics": True
            }
        }
    }

class BatchLoanRequest(BaseModel):
    """Request model for processing multiple loans in a single batch"""
    loans: List[LoanData] = Field(
        ...,
        min_length=1, 
        max_length=100,
        description="List of loans to process in the batch"
    )
    process_in_parallel: bool = Field(
        default=True,
        description="Whether to process loans in parallel for improved performance"
    )
    include_detailed_projections: bool = Field(
        default=True,
        description="Whether to include detailed period-by-period projections for each loan"
    )
    summary_metrics: bool = Field(
        default=True,
        description="Whether to include summary metrics for each loan"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "loans": [
                    {
                        "loan_id": "loan-123",
                        "principal": 200000.0,
                        "interest_rate": 0.045,
                        "term_months": 360,
                        "origination_date": "2023-01-15",
                        "loan_type": "fixed",
                        "payment_frequency": "monthly",
                        "amortization_type": "level_payment"
                    },
                    {
                        "loan_id": "loan-456",
                        "principal": 150000.0,
                        "interest_rate": 0.039,
                        "term_months": 240,
                        "origination_date": "2023-02-01",
                        "loan_type": "fixed",
                        "payment_frequency": "monthly",
                        "amortization_type": "level_payment"
                    }
                ],
                "process_in_parallel": True,
                "include_detailed_projections": True,
                "summary_metrics": True
            }
        }
    }

class BatchForecastRequest(BaseModel):
    """Batch forecast request for multiple loans"""
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
    """Individual period projection in a cashflow forecast"""
    period: int
    date: date
    principal: float
    interest: float
    total_payment: float
    remaining_balance: float
    default_amount: Optional[float] = 0.0
    prepayment_amount: Optional[float] = 0.0

class MonteCarloResults(BaseModel):
    """Monte Carlo simulation results for cashflow forecasting"""
    iterations: int
    mean_npv: float
    median_npv: float
    std_dev_npv: float
    percentile_5: float
    percentile_95: float
    detailed_results: Optional[List[Dict[str, float]]] = None
    simulation_time: Optional[float] = None
    convergence_achieved: Optional[bool] = None
    value_at_risk: Optional[Dict[str, float]] = None
    distribution_metrics: Optional[Dict[str, float]] = None

class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis results for cashflow metrics"""
    base_case: Dict[str, float] = Field(..., description="Base case metric values")
    rate_sensitivity: Dict[str, Dict[str, float]] = Field(
        ..., description="Effect of rate changes on metrics"
    )
    default_sensitivity: Dict[str, Dict[str, float]] = Field(
        ..., description="Effect of default rate changes on metrics"
    )
    prepay_sensitivity: Dict[str, Dict[str, float]] = Field(
        ..., description="Effect of prepayment rate changes on metrics"
    )
    recovery_sensitivity: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Effect of recovery rate changes on metrics"
    )
    combined_scenarios: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Effect of combined parameter changes"
    )

class MetricDefinition(BaseModel):
    """Definition of a financial metric for cashflow analysis"""
    name: str
    display_name: str
    description: str
    unit: str
    format_spec: Optional[str] = None
    category: Optional[str] = None

class CashflowSummaryMetrics(BaseModel):
    """Summary metrics for a cashflow forecast"""
    npv: float = Field(..., description="Net Present Value")
    irr: float = Field(..., description="Internal Rate of Return")
    yield_to_maturity: float = Field(..., description="Yield to Maturity")
    duration: float = Field(..., description="Macaulay Duration")
    modified_duration: float = Field(..., description="Modified Duration")
    convexity: float = Field(..., description="Convexity")
    wal: float = Field(..., description="Weighted Average Life")
    total_principal: float = Field(..., description="Total Principal Payments")
    total_interest: float = Field(..., description="Total Interest Payments")
    total_cashflow: float = Field(..., description="Total Cashflow")
    average_life: float = Field(..., description="Average Life")
    total_default: Optional[float] = Field(0.0, description="Total Default Amount")
    total_prepayment: Optional[float] = Field(0.0, description="Total Prepayment Amount")
    time_to_recovery: Optional[float] = Field(None, description="Expected Time to Recovery")
    break_even_rate: Optional[float] = Field(None, description="Break-even Interest Rate")
    effective_duration: Optional[float] = Field(None, description="Effective Duration")
    key_rate_durations: Optional[Dict[str, float]] = Field(None, description="Key Rate Durations")
    spread_duration: Optional[float] = Field(None, description="Spread Duration")
    enterprise_value: Optional[float] = Field(None, description="Enterprise Value")
    risk_adjusted_return: Optional[float] = Field(None, description="Risk-Adjusted Return on Capital")
    profitability_index: Optional[float] = Field(None, description="Profitability Index")

class CashflowForecastResponse(BaseModel):
    """Response model for a cashflow forecast with enhanced analytics"""
    loan_id: str
    projections: List[CashflowProjection]
    summary_metrics: Dict[str, Any]
    monte_carlo_results: Optional[MonteCarloResults] = None
    sensitivity_analysis: Optional[SensitivityAnalysis] = None
    metrics_definitions: Optional[Dict[str, MetricDefinition]] = None
    calculation_time: Optional[float] = None
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    version: str = Field(default="1.0.0")
    warning_messages: Optional[List[str]] = None
    error_messages: Optional[List[str]] = None
    
    @field_validator('projections')
    def validate_projections(cls, v):
        """Validate that projections are properly sequenced and balanced"""
        if not v:
            return v
            
        # Check if periods are sequential
        periods = [p.period for p in v]
        if sorted(periods) != periods:
            raise ValueError("Projection periods must be sequential")
            
        # Check if there are duplicate periods
        if len(periods) != len(set(periods)):
            raise ValueError("Duplicate period numbers in projections")
            
        # Check if final balance is close to zero
        if v and abs(v[-1].remaining_balance) > 0.01:
            # Add a warning instead of raising an error
            if hasattr(cls, 'warning_messages') and cls.warning_messages is not None:
                cls.warning_messages.append("Final balance is not zero")
            
        return v
    
    @field_validator('summary_metrics')
    def validate_summary_metrics(cls, v):
        """Validate that required summary metrics are present"""
        required_metrics = ['npv', 'irr', 'wal']
        missing = [metric for metric in required_metrics if metric.lower() not in {k.lower() for k in v.keys()}]
        
        if missing:
            raise ValueError(f"Missing required metrics: {', '.join(missing)}")
            
        return v

class ScenarioSaveRequest(BaseModel):
    """Request model for saving a cashflow scenario"""
    name: str
    description: Optional[str] = ""
    forecast_request: CashflowForecastRequest
    tags: Optional[List[str]] = []
    is_template: Optional[bool] = False

class ScenarioResponse(BaseModel):
    """Response model for a saved cashflow scenario"""
    id: str
    user_id: str
    name: str
    description: Optional[str] = ""
    forecast_config: Dict[str, Any]
    tags: List[str]
    is_template: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
