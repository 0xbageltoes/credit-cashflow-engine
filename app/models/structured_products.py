"""
Data models for structured finance products and AbsBox integration
"""
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class LoanStatus(str, Enum):
    CURRENT = "current"
    DELINQUENT = "delinquent"
    DEFAULT = "defaulted"
    PREPAID = "prepaid"

class RateType(str, Enum):
    FIXED = "fixed"
    FLOATING = "floating"

class PaymentFrequency(str, Enum):
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly" 
    SEMI_ANNUAL = "SemiAnnual"
    ANNUAL = "Annual"

class LoanConfig(BaseModel):
    """Configuration for a single loan"""
    balance: float
    rate: float
    term: int
    start_date: date
    rate_type: str = Field(default="fixed")
    payment_frequency: str = Field(default="Monthly")
    original_balance: Optional[float] = None
    original_rate: Optional[float] = None
    remaining_term: Optional[int] = None
    payment_delay: Optional[int] = None
    status: Optional[str] = None
    # Floating rate specific fields
    margin: Optional[float] = None
    index: Optional[str] = None
    reset_frequency: Optional[int] = None
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

class LoanPoolConfig(BaseModel):
    """Configuration for a loan pool"""
    loans: List[LoanConfig]
    pool_name: Optional[str] = None
    cut_off_date: Optional[date] = None
    metadata: Optional[Dict[str, Any]] = None

class AccountConfig(BaseModel):
    """Configuration for a waterfall account"""
    name: str
    initial_balance: Optional[float] = Field(default=0.0)
    interest_rate: Optional[float] = Field(default=0.0)
    account_type: Optional[str] = None

class BondConfig(BaseModel):
    """Configuration for a bond/note in the waterfall"""
    name: str
    balance: float
    rate: float
    rate_type: Optional[str] = Field(default="Fixed")
    payment_frequency: Optional[str] = Field(default="Monthly")
    original_balance: Optional[float] = None
    original_rate: Optional[float] = None
    start_date: Optional[date] = None
    seniority: Optional[int] = None

class WaterfallAction(BaseModel):
    """A single action/rule in the waterfall"""
    source: str
    target: str
    amount: Union[str, float]
    trigger: Optional[str] = None
    tag: Optional[str] = None

class WaterfallConfig(BaseModel):
    """Configuration for a deal waterfall"""
    start_date: date
    accounts: List[AccountConfig]
    bonds: List[BondConfig]
    actions: List[WaterfallAction]
    end_date: Optional[date] = None
    call_features: Optional[Dict[str, Any]] = None

class DefaultCurveConfig(BaseModel):
    """Configuration for default curve"""
    vector: List[float]
    recovery_vector: Optional[List[float]] = None
    lag: Optional[int] = Field(default=0)
    recovery_lag: Optional[int] = Field(default=6)

class PrepaymentCurveConfig(BaseModel):
    """Configuration for prepayment curve"""
    vector: List[float]
    type: Optional[str] = Field(default="vector")
    scaling_factor: Optional[float] = Field(default=1.0)
    seasoning_vector: Optional[List[float]] = None
    lag: Optional[int] = Field(default=0)

class RateCurveConfig(BaseModel):
    """Configuration for interest rate curve"""
    vector: List[float]
    dates: Optional[List[date]] = None
    curve_type: Optional[str] = Field(default="flat")

class ScenarioConfig(BaseModel):
    """Configuration for a scenario analysis"""
    name: Optional[str] = None
    default_curve: Optional[DefaultCurveConfig] = None
    interest_rate_curve: Optional[RateCurveConfig] = None
    prepayment_curve: Optional[PrepaymentCurveConfig] = None
    delinquency_curve: Optional[Dict[str, Any]] = None

class StructuredDealRequest(BaseModel):
    """Request model for structured deal analysis"""
    deal_name: str
    pool_config: LoanPoolConfig
    waterfall_config: WaterfallConfig
    scenario_config: Optional[ScenarioConfig] = None
    pricing_date: Optional[date] = None
    run_type: Optional[str] = Field(default="cashflow")
    metadata: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    """Result of a scenario analysis"""
    scenario_name: str
    bond_metrics: Dict[str, Any]
    pool_metrics: Dict[str, Any]
    execution_time: float
    status: str
    error: Optional[str] = None

class StructuredDealResponse(BaseModel):
    """Response model for structured deal analysis"""
    deal_name: str
    execution_time: float
    bond_cashflows: List[Dict[str, Any]]
    pool_cashflows: List[Dict[str, Any]]
    pool_statistics: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str
    error: Optional[str] = None
    error_type: Optional[str] = None

# Enhanced models for the AbsBoxServiceEnhanced class
class LoanData(BaseModel):
    """Detailed loan data model with additional metadata"""
    loan_id: str
    original_balance: float
    current_balance: float
    interest_rate: float
    original_term: int
    remaining_term: int
    payment_frequency: str
    origination_date: date
    maturity_date: date
    status: str = Field(default="current")
    delinquency_status: Optional[str] = None
    days_delinquent: Optional[int] = None
    default_date: Optional[date] = None
    prepayment_date: Optional[date] = None
    recovery_amount: Optional[float] = None
    recovery_date: Optional[date] = None
    last_payment_date: Optional[date] = None
    next_payment_date: Optional[date] = None
    payment_amount: Optional[float] = None
    property_type: Optional[str] = None
    property_value: Optional[float] = None
    ltv_ratio: Optional[float] = None
    dti_ratio: Optional[float] = None
    fico_score: Optional[int] = None
    geography: Optional[str] = None
    industry: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "L12345",
                "original_balance": 250000.0,
                "current_balance": 240000.0,
                "interest_rate": 0.045,
                "original_term": 360,
                "remaining_term": 350,
                "payment_frequency": "Monthly",
                "origination_date": "2023-01-15",
                "maturity_date": "2053-01-15",
                "status": "current",
                "payment_amount": 1267.85,
                "ltv_ratio": 0.8,
                "fico_score": 720
            }
        }

class CashflowPeriod(BaseModel):
    """Single period in a cashflow projection"""
    period: int
    date: date
    beginning_balance: float
    payment: float
    interest: float
    principal: float
    prepayment: float
    default: float
    recovery: float
    ending_balance: float
    scheduled_payment: Optional[float] = None
    scheduled_interest: Optional[float] = None
    scheduled_principal: Optional[float] = None
    delinquent_amount: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None

class CashflowProjection(BaseModel):
    """Cashflow projection model for loan or bond"""
    name: str
    cashflow_type: str  # "loan", "bond", "pool", etc.
    periods: List[CashflowPeriod]
    start_date: date
    end_date: date
    original_balance: float
    current_balance: float
    interest_rate: float
    term: int
    remaining_term: int
    cdr: Optional[float] = None  # Conditional Default Rate
    cpr: Optional[float] = None  # Conditional Prepayment Rate
    severity: Optional[float] = None  # Loss Severity
    total_interest: Optional[float] = None
    total_principal: Optional[float] = None
    total_prepayment: Optional[float] = None
    total_default: Optional[float] = None
    total_recovery: Optional[float] = None
    net_cashflow: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Pool_A",
                "cashflow_type": "pool",
                "start_date": "2023-01-15",
                "end_date": "2053-01-15",
                "original_balance": 10000000.0,
                "current_balance": 9850000.0,
                "interest_rate": 0.045,
                "term": 360,
                "remaining_term": 350,
                "cdr": 0.01,
                "cpr": 0.05,
                "severity": 0.35,
                "total_interest": 5457825.87,
                "total_principal": 9850000.0,
                "net_cashflow": 15307825.87
            }
        }

class EnhancedAnalyticsResult(BaseModel):
    """Enhanced analytics result with detailed metrics"""
    scenario_name: str
    run_date: datetime = Field(default_factory=datetime.now)
    bond_metrics: Dict[str, Dict[str, Any]]
    pool_metrics: Dict[str, Any]
    waterfall_metrics: Dict[str, Any]
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    stress_tests: Optional[Dict[str, Any]] = None
    execution_time: float
    status: str
    calculation_warnings: Optional[List[str]] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "scenario_name": "Base Case",
                "run_date": "2023-06-15T14:30:25",
                "bond_metrics": {
                    "Class A": {
                        "yield": 0.042,
                        "duration": 5.2,
                        "wac": 0.045,
                        "wal": 6.1,
                        "price": 99.5
                    }
                },
                "pool_metrics": {
                    "wac": 0.045,
                    "wal": 8.2,
                    "cpr": 0.05,
                    "cdr": 0.01
                },
                "execution_time": 0.45,
                "status": "success"
            }
        }

class CashflowForecastResponse(BaseModel):
    """Enhanced response model for cashflow forecasts with multiple projections"""
    deal_name: str
    run_date: datetime = Field(default_factory=datetime.now)
    pool_cashflows: CashflowProjection
    bond_cashflows: Dict[str, CashflowProjection]
    waterfall_results: Dict[str, List[Dict[str, Any]]]
    scenario_parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    execution_time: float
    engine_version: str
    status: str
    cache_hit: Optional[bool] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "deal_name": "Example RMBS 2023-1",
                "run_date": "2023-06-15T14:30:25",
                "scenario_parameters": {
                    "cpr": 0.05,
                    "cdr": 0.01,
                    "severity": 0.35
                },
                "metrics": {
                    "bond_yields": {
                        "Class A": 0.042,
                        "Class B": 0.052
                    },
                    "pool_metrics": {
                        "wac": 0.045,
                        "wal": 8.2
                    }
                },
                "execution_time": 0.75,
                "engine_version": "1.0.3",
                "status": "success"
            }
        }
