"""
Data models for different asset classes in structured finance products

This module defines the Pydantic models for various asset classes supported
by the credit cashflow engine, including:
- Residential Mortgages
- Auto Loans
- Consumer Credit
- Commercial Loans
- CLOs/CDOs

Each asset class has specific attributes and validation requirements.
"""
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import date, datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid

# Common enums
class LoanStatus(str, Enum):
    CURRENT = "current"
    DELINQUENT = "delinquent"
    DEFAULT = "defaulted"
    PREPAID = "prepaid"

class RateType(str, Enum):
    FIXED = "fixed"
    FLOATING = "floating"
    HYBRID = "hybrid"
    STEP = "step"
    
class PaymentFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly" 
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    
class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"
    MULTI_FAMILY = "multi_family"
    CONDOMINIUM = "condominium"
    COOPERATIVE = "cooperative"
    TOWNHOUSE = "townhouse"
    MIXED_USE = "mixed_use"
    COMMERCIAL = "commercial"
    
class VehicleType(str, Enum):
    NEW = "new"
    USED = "used"
    
class ConsumerLoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    EDUCATION = "education"
    OTHER = "other"
    
class CommercialPropertyType(str, Enum):
    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    MULTIFAMILY = "multifamily"
    HOSPITALITY = "hospitality"
    MIXED_USE = "mixed_use"
    HEALTHCARE = "healthcare"
    OTHER = "other"
    
class AssetClass(str, Enum):
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    AUTO_LOAN = "auto_loan"
    CONSUMER_CREDIT = "consumer_credit"
    COMMERCIAL_LOAN = "commercial_loan"
    CLO_CDO = "clo_cdo"

# Base Asset Model
class BaseAsset(BaseModel):
    """Base model for all asset classes"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    balance: float = Field(..., gt=0)
    rate: float = Field(..., ge=0)
    term_months: int = Field(..., gt=0)
    origination_date: date
    rate_type: RateType = Field(default=RateType.FIXED)
    payment_frequency: PaymentFrequency = Field(default=PaymentFrequency.MONTHLY)
    original_balance: Optional[float] = None
    remaining_term_months: Optional[int] = None
    status: LoanStatus = Field(default=LoanStatus.CURRENT)
    metadata: Optional[Dict[str, Any]] = None
    asset_class: AssetClass
    
    @validator('original_balance', always=True)
    def set_original_balance(cls, v, values):
        """Set original balance to current balance if not provided"""
        if v is None and 'balance' in values:
            return values['balance']
        return v
        
    @validator('remaining_term_months', always=True)
    def set_remaining_term(cls, v, values):
        """Set remaining term to full term if not provided"""
        if v is None and 'term_months' in values:
            return values['term_months']
        return v
    
    @root_validator(skip_on_failure=True)
    def check_remaining_term(cls, values):
        """Validate that remaining term is not greater than original term"""
        if (values.get('remaining_term_months') is not None and 
            values.get('term_months') is not None and
            values.get('remaining_term_months') > values.get('term_months')):
            raise ValueError("Remaining term cannot be greater than original term")
        return values

# Residential Mortgage
class ResidentialMortgage(BaseAsset):
    """Model for residential mortgage loans"""
    asset_class: Literal[AssetClass.RESIDENTIAL_MORTGAGE] = AssetClass.RESIDENTIAL_MORTGAGE
    property_type: PropertyType = Field(default=PropertyType.SINGLE_FAMILY)
    ltv_ratio: Optional[float] = Field(None, ge=0, le=1.5)  # Allow for underwater loans with LTV > 1
    appraisal_value: Optional[float] = Field(None, gt=0)
    lien_position: Optional[int] = Field(1, ge=1)
    is_interest_only: Optional[bool] = False
    interest_only_period_months: Optional[int] = None
    balloon_payment: Optional[float] = None
    prepayment_penalty: Optional[bool] = False
    prepayment_penalty_term: Optional[int] = None
    has_pmi: Optional[bool] = False
    pmi_rate: Optional[float] = None
    location: Optional[Dict[str, Any]] = None  # For geolocation data
    
    @validator('interest_only_period_months')
    def validate_io_period(cls, v, values):
        """Validate interest-only period is less than term"""
        if v is not None and 'term_months' in values and v > values['term_months']:
            raise ValueError("Interest-only period cannot exceed loan term")
        return v

# Auto Loan
class AutoLoan(BaseAsset):
    """Model for auto loans"""
    asset_class: Literal[AssetClass.AUTO_LOAN] = AssetClass.AUTO_LOAN
    vehicle_type: VehicleType = Field(default=VehicleType.NEW)
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_year: Optional[int] = None
    ltv_ratio: Optional[float] = Field(None, ge=0, le=1.5)
    initial_depreciation_rate: Optional[float] = Field(None, ge=0, le=1)
    subsequent_depreciation_rate: Optional[float] = Field(None, ge=0, le=1)

# Consumer Credit
class ConsumerCredit(BaseAsset):
    """Model for consumer credit loans"""
    asset_class: Literal[AssetClass.CONSUMER_CREDIT] = AssetClass.CONSUMER_CREDIT
    loan_purpose: ConsumerLoanPurpose = Field(default=ConsumerLoanPurpose.OTHER)
    is_secured: bool = False
    collateral_type: Optional[str] = None
    collateral_value: Optional[float] = None
    fico_score: Optional[int] = Field(None, ge=300, le=850)
    debt_to_income: Optional[float] = Field(None, ge=0, le=1)

# Commercial Loan
class CommercialLoan(BaseAsset):
    """Model for commercial loans"""
    asset_class: Literal[AssetClass.COMMERCIAL_LOAN] = AssetClass.COMMERCIAL_LOAN
    property_type: Optional[CommercialPropertyType] = None
    ltv_ratio: Optional[float] = Field(None, ge=0, le=1.5)
    dscr: Optional[float] = Field(None, gt=0)  # Debt Service Coverage Ratio
    noi: Optional[float] = None  # Net Operating Income
    amortization_months: Optional[int] = Field(None, gt=0)
    is_interest_only: Optional[bool] = False
    interest_only_period_months: Optional[int] = None
    balloon_payment: Optional[float] = None
    prepayment_penalty: Optional[bool] = False
    prepayment_penalty_term: Optional[int] = None
    location: Optional[Dict[str, Any]] = None  # For geolocation data

# CLO/CDO Model
class CLOCDOTranche(BaseModel):
    """Model for CLO/CDO tranches"""
    name: str
    balance: float = Field(..., gt=0)
    rate: float = Field(..., ge=0)
    seniority: int = Field(..., ge=1)
    rate_type: RateType = Field(default=RateType.FLOATING)
    spread: Optional[float] = None
    index: Optional[str] = None
    attachment_point: Optional[float] = Field(None, ge=0, le=1)
    detachment_point: Optional[float] = Field(None, ge=0, le=1)
    
    @validator('detachment_point')
    def validate_attachment_detachment(cls, v, values):
        """Validate attachment and detachment points"""
        if (v is not None and 
            'attachment_point' in values and 
            values['attachment_point'] is not None and
            v <= values['attachment_point']):
            raise ValueError("Detachment point must be greater than attachment point")
        return v

class CLOCDO(BaseAsset):
    """Model for CLO/CDO structures"""
    asset_class: Literal[AssetClass.CLO_CDO] = AssetClass.CLO_CDO
    collateral_pool_count: int
    collateral_pool_balance: float = Field(..., gt=0)
    collateral_pool_warf: Optional[float] = None  # Weighted Average Rating Factor
    collateral_pool_wac: Optional[float] = None  # Weighted Average Coupon
    tranches: List[CLOCDOTranche]
    reinvestment_period_months: Optional[int] = None
    
    @validator('tranches')
    def validate_tranches(cls, v):
        """Validate at least one tranche exists"""
        if not v:
            raise ValueError("At least one tranche must be defined")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_tranche_total(cls, values):
        """Validate that tranche balance total doesn't exceed collateral pool balance"""
        if 'tranches' in values and 'collateral_pool_balance' in values:
            total_tranche_balance = sum(t.balance for t in values['tranches'])
            if total_tranche_balance > values['collateral_pool_balance'] * 1.01:  # Allow 1% margin for rounding
                raise ValueError("Total tranche balance cannot exceed collateral pool balance")
        return values

# Asset Pool Models
class AssetPool(BaseModel):
    """Model for a pool of assets"""
    pool_name: str
    pool_description: Optional[str] = None
    cut_off_date: date
    assets: List[Union[ResidentialMortgage, AutoLoan, ConsumerCredit, CommercialLoan, CLOCDO]]
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('assets')
    def validate_assets(cls, v):
        """Validate at least one asset exists"""
        if not v:
            raise ValueError("At least one asset must be defined in the pool")
        return v
    
    def total_balance(self) -> float:
        """Calculate total pool balance"""
        return sum(asset.balance for asset in self.assets)
    
    def weighted_average_rate(self) -> float:
        """Calculate weighted average interest rate"""
        total_balance = self.total_balance()
        if total_balance == 0:
            return 0
        return sum(asset.balance * asset.rate for asset in self.assets) / total_balance
    
    def weighted_average_term(self) -> float:
        """Calculate weighted average term in months"""
        total_balance = self.total_balance()
        if total_balance == 0:
            return 0
        return sum(asset.balance * asset.term_months for asset in self.assets) / total_balance

# API Request/Response Models
class AssetPoolAnalysisRequest(BaseModel):
    """Request model for asset pool analysis"""
    pool: AssetPool
    analysis_date: Optional[date] = None
    pricing_date: Optional[date] = None
    discount_rate: Optional[float] = None
    include_cashflows: bool = True
    include_metrics: bool = True
    include_stress_tests: bool = False
    scenario_name: Optional[str] = None
    calibration_params: Optional[Dict[str, Any]] = None  # For custom calibration parameters
    
    @validator('analysis_date', always=True)
    def set_analysis_date(cls, v):
        """Set analysis date to current date if not provided"""
        return v or date.today()
        
    @validator('pricing_date', always=True)
    def set_pricing_date(cls, v, values):
        """Set pricing date to analysis date if not provided"""
        if v is None and 'analysis_date' in values:
            return values['analysis_date']
        return v

class AssetPoolCashflow(BaseModel):
    """Model for asset pool cashflow projections"""
    period: int
    date: date
    scheduled_principal: float
    scheduled_interest: float
    prepayment: float
    default: float
    recovery: float
    loss: float
    balance: float
    
class AssetPoolMetrics(BaseModel):
    """Model for asset pool analysis metrics"""
    total_principal: float
    total_interest: float
    total_cashflow: float
    npv: float
    irr: Optional[float] = None
    duration: Optional[float] = None
    modified_duration: Optional[float] = None
    convexity: Optional[float] = None
    weighted_average_life: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    
class AssetPoolStressTest(BaseModel):
    """Model for asset pool stress test results"""
    scenario_name: str
    description: Optional[str] = None
    npv: float
    npv_change: float
    npv_change_percent: float
    
class AssetPoolAnalysisResponse(BaseModel):
    """Response model for asset pool analysis"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pool_name: str
    analysis_date: date
    execution_time: float
    status: str  # "success", "error", "partial"
    error: Optional[str] = None
    error_type: Optional[str] = None
    metrics: Optional[AssetPoolMetrics] = None
    cashflows: Optional[List[Union[AssetPoolCashflow, Dict[str, Any]]]] = None
    stress_tests: Optional[List[AssetPoolStressTest]] = None
    analytics: Optional[Dict[str, Any]] = None  # Additional analytics specific to asset class
    cache_hit: Optional[bool] = None  # Indicates if result came from cache
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "f8d7a982-f22d-4e34-9d7b-c04e435e6276",
                "pool_name": "Example Residential Pool",
                "analysis_date": "2023-09-01",
                "execution_time": 0.453,
                "status": "success",
                "metrics": {
                    "total_principal": 1000000,
                    "total_interest": 450000,
                    "total_cashflow": 1450000,
                    "npv": 980000,
                    "irr": 0.055,
                    "duration": 4.2,
                    "weighted_average_life": 5.1
                }
            }
        }
