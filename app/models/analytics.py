"""Analytics models for credit-cashflow-engine"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
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

class CashflowProjection(BaseModel):
    """Model for projected cashflows"""
    cashflow_id: Optional[str] = Field(None, description="Unique identifier for this cashflow projection")
    dates: List[str] = Field(..., description="Array of cashflow dates in ISO format")
    amounts: List[float] = Field(..., description="Array of cashflow amounts")
    principal: Optional[List[float]] = Field(None, description="Array of principal portions")
    interest: Optional[List[float]] = Field(None, description="Array of interest portions")
    default: Optional[List[float]] = Field(None, description="Array of default amounts")
    recovery: Optional[List[float]] = Field(None, description="Array of recovery amounts")
    prepayment: Optional[List[float]] = Field(None, description="Array of prepayment amounts")
    ending_balance: Optional[List[float]] = Field(None, description="Array of ending balances")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "dates": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "amounts": [1000.0, 1000.0, 1000.0],
                "principal": [800.0, 805.0, 810.0],
                "interest": [200.0, 195.0, 190.0]
            }
        }
    }

class EconomicFactors(BaseModel):
    """Model for economic factors that can influence cashflows"""
    inflation_rate: Optional[float] = Field(0.0, description="Annual inflation rate as decimal")
    gdp_growth: Optional[float] = Field(0.0, description="Annual GDP growth rate as decimal")
    unemployment_rate: Optional[float] = Field(0.0, description="Unemployment rate as decimal")
    housing_price_index: Optional[float] = Field(0.0, description="Housing price index change as decimal")
    interest_rate_environment: Optional[str] = Field("neutral", description="Interest rate environment (rising, falling, neutral)")
    consumer_confidence: Optional[float] = Field(0.0, description="Consumer confidence index change as decimal")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix between factors")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "inflation_rate": 0.02,
                "gdp_growth": 0.025,
                "unemployment_rate": 0.04,
                "housing_price_index": 0.03,
                "interest_rate_environment": "rising",
                "consumer_confidence": 0.01
            }
        }
    }

class StatisticalOutputs(BaseModel):
    """Statistical outputs from Monte Carlo simulations"""
    mean: float = Field(..., description="Mean value of the simulation results")
    median: float = Field(..., description="Median value of the simulation results")
    std_dev: float = Field(..., description="Standard deviation of the simulation results")
    min_value: float = Field(..., description="Minimum value in the simulation results")
    max_value: float = Field(..., description="Maximum value in the simulation results")
    percentiles: Dict[str, float] = Field(..., description="Key percentiles of the distribution (e.g., 1%, 5%, 95%, 99%)")
    skewness: Optional[float] = Field(None, description="Skewness of the distribution")
    kurtosis: Optional[float] = Field(None, description="Kurtosis of the distribution")
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Confidence intervals for key metrics")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "mean": 1000000.0,
                "median": 980000.0,
                "std_dev": 150000.0,
                "min_value": 500000.0,
                "max_value": 1500000.0,
                "percentiles": {
                    "1": 600000.0,
                    "5": 750000.0,
                    "95": 1250000.0,
                    "99": 1400000.0
                },
                "skewness": 0.2,
                "kurtosis": 3.1
            }
        }
    }

class MonteCarloSimulationParameters(BaseModel):
    """Parameters for Monte Carlo simulation"""
    num_iterations: int = Field(1000, description="Number of iterations/scenarios to simulate")
    time_horizon: int = Field(120, description="Time horizon in months")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix for random variables")
    cpv_distributions: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Credit performance variable distributions")
    economic_scenarios: Optional[List[Dict[str, Any]]] = Field(None, description="Predefined economic scenarios to run")
    output_percentiles: Optional[List[int]] = Field([1, 5, 10, 25, 50, 75, 90, 95, 99], description="Percentiles to include in output")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "num_iterations": 1000,
                "time_horizon": 120,
                "seed": 42,
                "output_percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
                "cpv_distributions": {
                    "default_rate": {
                        "distribution": "beta",
                        "alpha": 2.0,
                        "beta": 10.0
                    },
                    "prepayment_rate": {
                        "distribution": "normal",
                        "mean": 0.05,
                        "std_dev": 0.02
                    }
                }
            }
        }
    }

class MonteCarloResult(BaseModel):
    """Results from a Monte Carlo simulation"""
    simulation_id: str = Field(..., description="Unique identifier for this simulation")
    num_iterations: int = Field(..., description="Number of iterations performed")
    time_horizon: int = Field(..., description="Time horizon in months")
    calculation_time: float = Field(..., description="Calculation time in seconds")
    npv_stats: StatisticalOutputs = Field(..., description="NPV statistical results")
    irr_stats: Optional[StatisticalOutputs] = Field(None, description="IRR statistical results")
    duration_stats: Optional[StatisticalOutputs] = Field(None, description="Duration statistical results")
    default_stats: Optional[StatisticalOutputs] = Field(None, description="Default statistical results")
    prepayment_stats: Optional[StatisticalOutputs] = Field(None, description="Prepayment statistical results")
    loss_distribution: Optional[Dict[str, float]] = Field(None, description="Loss distribution at key points")
    worst_case_cashflows: Optional[CashflowProjection] = Field(None, description="Cashflows for worst-case scenario")
    best_case_cashflows: Optional[CashflowProjection] = Field(None, description="Cashflows for best-case scenario")
    percentile_cashflows: Optional[Dict[str, CashflowProjection]] = Field(None, description="Cashflows at various percentiles")
    error: Optional[str] = Field(None, description="Error message if simulation failed")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "simulation_id": "sim_20230101_123456",
                "num_iterations": 1000,
                "time_horizon": 120,
                "calculation_time": 45.6,
                "npv_stats": {
                    "mean": 1000000.0,
                    "median": 980000.0,
                    "std_dev": 150000.0,
                    "min_value": 500000.0,
                    "max_value": 1500000.0,
                    "percentiles": {
                        "1": 600000.0,
                        "5": 750000.0,
                        "95": 1250000.0,
                        "99": 1400000.0
                    }
                }
            }
        }
    }

class RiskMetrics(BaseModel):
    """Comprehensive risk metrics for fixed income securities"""
    convexity: float = Field(0.0, description="Convexity measure")
    yield_to_maturity: float = Field(0.0, description="Yield to maturity")
    average_life: float = Field(0.0, description="Weighted average life")
    volatility: float = Field(0.0, description="Volatility of returns")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    var_99: Optional[float] = Field(None, description="Value at Risk (99% confidence)")
    expected_shortfall: Optional[float] = Field(None, description="Expected Shortfall (CVaR)")
    stress_loss: Optional[float] = Field(None, description="Potential loss under stress scenario")
    option_adjusted_spread: Optional[float] = Field(None, description="Option-adjusted spread")
    z_spread: Optional[float] = Field(None, description="Z-spread")
    effective_duration: Optional[float] = Field(None, description="Effective duration")
    key_rate_durations: Optional[Dict[str, float]] = Field(None, description="Key rate durations")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "convexity": 0.35,
                "yield_to_maturity": 0.055,
                "average_life": 5.2,
                "volatility": 0.15,
                "var_95": 10000.0,
                "var_99": 15000.0,
                "expected_shortfall": 18000.0
            }
        }
    }

class SensitivityAnalysisResult(BaseModel):
    """Results from sensitivity analysis across multiple parameters"""
    base_case: Dict[str, Any] = Field(..., description="Base case metrics")
    sensitivity_results: Dict[str, Dict[str, Any]] = Field(..., description="Sensitivity results by parameter")
    calculation_time: float = Field(0.0, description="Calculation time in seconds")
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "base_case": {
                    "npv": 1000000.0,
                    "irr": 0.055,
                    "duration": 4.5
                },
                "sensitivity_results": {
                    "discount_rate": {
                        "0.03": {
                            "metrics": {
                                "npv": 1100000.0,
                                "irr": 0.055,
                                "duration": 4.7
                            },
                            "diff_from_base": {
                                "npv": 100000.0,
                                "irr": 0.0,
                                "duration": 0.2
                            }
                        },
                        "0.05": {
                            "metrics": {
                                "npv": 900000.0,
                                "irr": 0.055,
                                "duration": 4.3
                            },
                            "diff_from_base": {
                                "npv": -100000.0,
                                "irr": 0.0,
                                "duration": -0.2
                            }
                        }
                    }
                },
                "calculation_time": 1.25
            }
        }
    }

class AnalyticsRequest(BaseModel):
    """Request model for analytics calculations"""
    cashflows: CashflowProjection = Field(..., description="Cashflow projections to analyze")
    discount_rate: float = Field(..., description="Discount rate for calculations")
    economic_factors: Optional[EconomicFactors] = Field(None, description="Economic factors to consider")
    monte_carlo_params: Optional[MonteCarloSimulationParameters] = Field(None, description="Monte Carlo simulation parameters")
    sensitivity_analysis: Optional[Dict[str, List[float]]] = Field(None, description="Parameters to use for sensitivity analysis")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "cashflows": {
                    "dates": ["2023-01-01", "2023-02-01", "2023-03-01"],
                    "amounts": [1000.0, 1000.0, 1000.0]
                },
                "discount_rate": 0.04,
                "economic_factors": {
                    "inflation_rate": 0.02,
                    "gdp_growth": 0.025
                }
            }
        }
    }
