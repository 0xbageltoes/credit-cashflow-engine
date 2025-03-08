"""
Core data models for the credit-cashflow-engine
"""
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LoanRequest(BaseModel):
    """
    Base model for loan requests used in cashflow calculations
    """
    principal: float = Field(gt=0, description="Principal amount of the loan")
    rate: float = Field(ge=0, le=1, description="Annual interest rate as decimal")
    term: int = Field(gt=0, le=600, description="Loan term in months")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
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
    prepayment_rate: Optional[float] = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Annual prepayment rate as decimal"
    )

    @field_validator('start_date')
    def validate_date_format(cls, v):
        """Validate the date is in the correct format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("start_date must be in YYYY-MM-DD format")

    @property
    def cache_key(self) -> str:
        """Generate a cache key for this loan request"""
        return f"loan:{self.principal}:{self.rate}:{self.term}:{self.start_date}"


class RateType(str, Enum):
    """
    Enumeration of interest rate types
    """
    FIXED = "fixed"
    FLOATING = "floating"
    HYBRID = "hybrid"


class PaymentFrequency(str, Enum):
    """
    Enumeration of payment frequencies
    """
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi-annual"
    ANNUAL = "annual"


class AdvancedLoanRequest(LoanRequest):
    """
    Advanced loan request model with additional parameters
    """
    rate_type: RateType = Field(
        default=RateType.FIXED,
        description="Interest rate type: fixed, floating, or hybrid"
    )
    payment_frequency: PaymentFrequency = Field(
        default=PaymentFrequency.MONTHLY,
        description="Payment frequency"
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
    rate_spread: Optional[float] = Field(
        default=None,
        ge=0,
        description="Spread over reference rate for floating/hybrid rates"
    )
    reference_rate_name: Optional[str] = Field(
        default=None,
        description="Name of the reference rate (e.g., SOFR, LIBOR)"
    )


class ApiResponse(BaseModel):
    """
    Standard API response model with status and optional data/error
    """
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(default=None, description="Response data if successful")
    error: Optional[str] = Field(default=None, description="Error message if not successful")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = Field(default=None, description="Unique request ID for tracking")
