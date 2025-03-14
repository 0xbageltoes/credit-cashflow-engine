"""
Specialized Asset Models

This module contains enhanced Pydantic models for specialized asset classes:
- Commercial Loans
- Consumer Credit
- CLO/CDO

These models extend the base asset class models with additional specialized fields
and validation.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import date
from enum import Enum
import uuid
from pydantic import BaseModel, Field, validator

from app.models.asset_classes import AssetClass, BaseAsset

# ENUMS FOR ASSET TYPE SPECIFIC PROPERTIES

class CommercialPropertyType(str, Enum):
    """Commercial real estate property types"""
    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    MULTIFAMILY = "multifamily"
    HOTEL = "hotel"
    HEALTHCARE = "healthcare"
    MIXED_USE = "mixed_use"
    SELF_STORAGE = "self_storage"
    OTHER = "other"

class LoanRateType(str, Enum):
    """Loan rate types"""
    FIXED = "fixed"
    FLOATING = "floating"
    HYBRID = "hybrid"

class ConsumerLoanType(str, Enum):
    """Consumer loan types"""
    CREDIT_CARD = "credit_card"
    AUTO_LOAN = "auto_loan"
    PERSONAL_LOAN = "personal_loan"
    STUDENT_LOAN = "student_loan"
    INSTALLMENT_LOAN = "installment_loan"
    MEDICAL_DEBT = "medical_debt"
    OTHER = "other"

class CLOCDOType(str, Enum):
    """CLO/CDO types"""
    CLO = "clo"
    CDO = "cdo"
    CBO = "cbo"
    CSO = "cso"
    OTHER = "other"

# ENHANCED MODELS FOR SPECIALIZED ASSET CLASSES

class CommercialLoanExtended(BaseAsset):
    """
    Extended commercial loan model with detailed properties
    
    Comprehensive model for commercial loans with property-specific
    attributes, loan terms, and risk metrics.
    """
    asset_class: AssetClass = AssetClass.COMMERCIAL_LOAN
    property_type: Optional[CommercialPropertyType] = None
    property_address: Optional[str] = None
    property_value: Optional[float] = None
    
    # Loan terms
    rate_type: LoanRateType = LoanRateType.FIXED
    spread: Optional[float] = None
    index: Optional[str] = None
    original_balance: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    remaining_term_months: int = Field(..., gt=0)
    amortization_months: Optional[int] = None
    origination_date: Optional[date] = None
    maturity_date: Optional[date] = None
    
    # Specialized fields
    is_interest_only: bool = False
    interest_only_period_months: Optional[int] = None
    balloon_payment: Optional[float] = None
    
    # Risk metrics
    ltv_ratio: Optional[float] = None
    dscr: Optional[float] = None
    debt_yield: Optional[float] = None
    
    # Default and recovery
    default_probability: Optional[float] = None
    recovery_rate: Optional[float] = None
    
    @validator('ltv_ratio')
    def validate_ltv_ratio(cls, v):
        """Validate LTV ratio is in reasonable range"""
        if v is not None and (v <= 0 or v > 1.5):
            raise ValueError('LTV ratio should be between 0 and 1.5')
        return v
    
    @validator('dscr')
    def validate_dscr(cls, v):
        """Validate DSCR is in reasonable range"""
        if v is not None and v <= 0:
            raise ValueError('DSCR must be positive')
        return v

class ConsumerCreditExtended(BaseAsset):
    """
    Extended consumer credit model with detailed properties
    
    Comprehensive model for consumer credit with borrower information,
    loan terms, and credit metrics.
    """
    asset_class: AssetClass = AssetClass.CONSUMER_CREDIT
    loan_type: Optional[ConsumerLoanType] = None
    
    # Loan terms
    rate_type: LoanRateType = LoanRateType.FIXED
    original_balance: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    remaining_term_months: int = Field(..., gt=0)
    origination_date: Optional[date] = None
    maturity_date: Optional[date] = None
    
    # Borrower information
    fico_score: Optional[int] = None
    dti_ratio: Optional[float] = None
    income: Optional[float] = None
    
    # Payment history
    payment_frequency: Optional[str] = "monthly"
    months_on_book: Optional[int] = None
    times_30_dpd: Optional[int] = None
    times_60_dpd: Optional[int] = None
    times_90_dpd: Optional[int] = None
    
    # Default and loss metrics
    default_probability: Optional[float] = None
    loss_given_default: Optional[float] = None
    recovery_rate: Optional[float] = None
    
    @validator('fico_score')
    def validate_fico_score(cls, v):
        """Validate FICO score is in valid range"""
        if v is not None and (v < 300 or v > 850):
            raise ValueError('FICO score should be between 300 and 850')
        return v
    
    @validator('dti_ratio')
    def validate_dti_ratio(cls, v):
        """Validate DTI ratio is in reasonable range"""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('DTI ratio should be between 0 and 1')
        return v

class CLOCDOTranche(BaseModel):
    """
    Model for a CLO/CDO tranche
    
    Represents a tranche in a structured product with key attributes
    like attachment points, seniority, and interest rate.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    balance: float = Field(..., gt=0)
    rate: float = Field(..., ge=0)
    rate_type: LoanRateType = LoanRateType.FLOATING
    spread: Optional[float] = None
    index: Optional[str] = None
    
    # Tranche structure
    seniority: Optional[int] = None
    attachment_point: Optional[float] = None
    detachment_point: Optional[float] = None
    
    # Ratings
    rating: Optional[str] = None
    rating_agency: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Class A",
                "balance": 80000000,
                "rate": 0.05,
                "rate_type": "floating",
                "spread": 0.02,
                "index": "LIBOR",
                "seniority": 1,
                "attachment_point": 0.2,
                "detachment_point": 1.0,
                "rating": "AAA",
                "rating_agency": "S&P"
            }
        }

class CLOCDOExtended(BaseAsset):
    """
    Extended CLO/CDO model with detailed properties
    
    Comprehensive model for CLO/CDO structured products with collateral pool
    information, tranches, and deal-specific attributes.
    """
    asset_class: AssetClass = AssetClass.CLO_CDO
    clo_cdo_type: CLOCDOType = CLOCDOType.CLO
    
    # Collateral pool information
    collateral_pool_balance: float = Field(..., gt=0)
    collateral_pool_wac: Optional[float] = None  # Weighted average coupon
    collateral_pool_warf: Optional[float] = None  # Weighted average rating factor
    collateral_pool_asset_count: Optional[int] = None
    collateral_pool_concentration: Optional[Dict[str, float]] = None
    
    # Deal information
    closing_date: Optional[date] = None
    effective_date: Optional[date] = None
    manager: Optional[str] = None
    trustee: Optional[str] = None
    
    # Reinvestment period
    reinvestment_period_months: Optional[int] = None
    reinvestment_end_date: Optional[date] = None
    
    # Tranches
    tranches: List[CLOCDOTranche] = Field(..., min_items=1)
    
    # Performance metrics
    overcollateralization_ratio: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    
    @validator('tranches')
    def validate_tranches(cls, tranches):
        """Validate that tranches are properly defined"""
        if not tranches:
            raise ValueError('At least one tranche must be defined')
        
        # Check for duplicate seniority
        seniority_values = [t.seniority for t in tranches if t.seniority is not None]
        if len(seniority_values) != len(set(seniority_values)):
            raise ValueError('Tranches cannot have duplicate seniority values')
        
        return tranches

# Request and Response models for specialized asset classes

class SpecializedAssetPoolAnalysisRequest(BaseModel):
    """Request model for specialized asset pool analysis"""
    asset_class: AssetClass
    pool_id: Optional[str] = None
    assets: List[Union[CommercialLoanExtended, ConsumerCreditExtended, CLOCDOExtended]]
    analysis_date: date
    parameters: Optional[Dict[str, Any]] = None
    include_cashflows: bool = True
    include_stress_tests: bool = False

class SpecializedAssetAnalysisResponse(BaseModel):
    """Response model for specialized asset analysis"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_class: AssetClass
    analysis_date: date
    execution_time: float
    status: str
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    cashflows: Optional[List[Dict[str, Any]]] = None
    stress_tests: Optional[List[Dict[str, Any]]] = None
    class_specific_analytics: Optional[Dict[str, Any]] = None
