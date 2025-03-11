"""Analytics models for credit-cashflow-engine"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from datetime import date

class AnalyticsResult(BaseModel):
    """Analytics result model containing key financial metrics"""
    npv: float = Field(..., description="Net Present Value")
    irr: float = Field(..., description="Internal Rate of Return")
    dscr: float = Field(..., description="Debt Service Coverage Ratio")
    ltv: float = Field(..., description="Loan to Value Ratio")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "npv": 100000.0,
                "irr": 0.06,
                "dscr": 1.25,
                "ltv": 0.75
            }
        }
    }

class EnhancedAnalyticsRequest(BaseModel):
    """Request model for enhanced analytics calculations"""
    principal: float = Field(..., description="Loan principal amount")
    interest_rate: float = Field(..., description="Annual interest rate (decimal)")
    term_months: int = Field(..., description="Loan term in months")
    start_date: date = Field(..., description="Loan start date")
    loan_id: Optional[str] = Field(None, description="Loan ID (if existing loan)")
    prepayment_rate: Optional[float] = Field(None, description="Annual prepayment rate (decimal)")
    default_rate: Optional[float] = Field(None, description="Annual default rate (decimal)")
    recovery_rate: Optional[float] = Field(0.6, description="Recovery rate in case of default (decimal)")
    discount_rate: Optional[float] = Field(None, description="Discount rate for NPV calculations (decimal)")
    rate_type: Optional[str] = Field("fixed", description="Interest rate type (fixed, floating, hybrid)")
    balloon_payment: Optional[float] = Field(None, description="Balloon payment amount")
    interest_only_periods: Optional[int] = Field(0, description="Number of interest-only periods")
    payment_frequency: Optional[str] = Field("monthly", description="Payment frequency (monthly, quarterly, annual)")
    market_rate: Optional[float] = Field(None, description="Current market rate for spread calculations")
    stress_scenarios: Optional[Dict[str, float]] = Field(None, description="Stress scenario parameters")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "principal": 100000.0,
                "interest_rate": 0.05,
                "term_months": 360,
                "start_date": "2023-01-01",
                "prepayment_rate": 0.05,
                "default_rate": 0.01,
                "recovery_rate": 0.6,
                "discount_rate": 0.04,
                "rate_type": "fixed",
                "payment_frequency": "monthly"
            }
        }
    }

class EnhancedAnalyticsResult(BaseModel):
    """Enhanced analytics result model with additional metrics from absbox"""
    npv: float = Field(..., description="Net Present Value")
    irr: float = Field(..., description="Internal Rate of Return")
    yield_value: float = Field(..., description="Yield to Maturity")
    duration: float = Field(..., description="Modified Duration")
    macaulay_duration: float = Field(..., description="Macaulay Duration")
    convexity: float = Field(..., description="Convexity")
    discount_margin: Optional[float] = Field(None, description="Discount Margin (DM)")
    z_spread: Optional[float] = Field(None, description="Z-Spread")
    e_spread: Optional[float] = Field(None, description="E-Spread (option-adjusted spread)")
    weighted_average_life: float = Field(..., description="Weighted Average Life (WAL)")
    debt_service_coverage: Optional[float] = Field(None, description="Debt Service Coverage Ratio")
    interest_coverage_ratio: Optional[float] = Field(None, description="Interest Coverage Ratio")
    sensitivity_metrics: Optional[Dict[str, float]] = Field(None, description="Sensitivity metrics")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "npv": 100000.0,
                "irr": 0.06,
                "yield_value": 0.055,
                "duration": 4.25,
                "macaulay_duration": 4.50,
                "convexity": 0.35,
                "discount_margin": 0.02,
                "z_spread": 0.015,
                "e_spread": 0.018,
                "weighted_average_life": 5.2,
                "debt_service_coverage": 1.25,
                "interest_coverage_ratio": 2.1,
                "sensitivity_metrics": {
                    "rate_up_1pct": -4.25,
                    "rate_down_1pct": 4.50
                }
            }
        }
    }

class RiskMetrics(BaseModel):
    """Model for risk metrics calculation"""
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    var_99: float = Field(..., description="Value at Risk (99% confidence)")
    expected_shortfall: float = Field(..., description="Expected Shortfall (CVaR)")
    stress_loss: float = Field(..., description="Potential loss under stress scenario")
    volatility: float = Field(..., description="Volatility of returns")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "var_95": 10000.0,
                "var_99": 15000.0,
                "expected_shortfall": 18000.0,
                "stress_loss": 25000.0,
                "volatility": 0.15
            }
        }
    }

class SensitivityAnalysis(BaseModel):
    """Model for sensitivity analysis results"""
    rate_sensitivity: Dict[str, float] = Field(..., description="Interest rate sensitivity")
    prepayment_sensitivity: Dict[str, float] = Field(..., description="Prepayment rate sensitivity")
    default_sensitivity: Dict[str, float] = Field(..., description="Default rate sensitivity")
    recovery_sensitivity: Optional[Dict[str, float]] = Field(None, description="Recovery rate sensitivity")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "rate_sensitivity": {
                    "up_1pct": -5.0,
                    "down_1pct": 5.5
                },
                "prepayment_sensitivity": {
                    "up_50pct": -2.0,
                    "down_50pct": 1.8
                },
                "default_sensitivity": {
                    "up_50pct": -3.0,
                    "down_50pct": 2.5
                }
            }
        }
    }

class AnalyticsResponse(BaseModel):
    """Generic response model for analytics endpoints"""
    status: str = Field(..., description="Status of the calculation (success, error)")
    execution_time: float = Field(..., description="Execution time in seconds")
    metrics: Optional[Union[EnhancedAnalyticsResult, RiskMetrics, SensitivityAnalysis]] = Field(None, description="Analytics metrics")
    error: Optional[str] = Field(None, description="Error message if status is error")
    cache_hit: Optional[bool] = Field(None, description="Whether the result was retrieved from cache")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "execution_time": 0.45,
                "metrics": {
                    "npv": 100000.0,
                    "irr": 0.06,
                    "yield_value": 0.055
                },
                "cache_hit": True
            }
        }
    }
