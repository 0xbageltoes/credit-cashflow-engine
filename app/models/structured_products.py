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
