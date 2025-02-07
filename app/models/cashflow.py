from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class LoanData(BaseModel):
    principal: float
    interest_rate: float
    term_months: int
    start_date: str
    prepayment_assumption: Optional[float] = 0.0

class CashflowForecastRequest(BaseModel):
    loans: List[LoanData]
    discount_rate: Optional[float] = 0.05
    run_monte_carlo: Optional[bool] = True
    monte_carlo_config: Optional[Dict] = {
        "num_simulations": 1000,
        "default_prob": 0.02,
        "prepay_prob": 0.05,
        "rate_volatility": 0.01
    }
    scenario_name: Optional[str] = None
    assumptions: Dict[str, float] = Field(default_factory=dict)

class CashflowProjection(BaseModel):
    period: int
    date: str
    principal: float
    interest: float
    total_payment: float
    remaining_balance: float

class CashflowForecastResponse(BaseModel):
    scenario_id: Optional[str]
    projections: List[CashflowProjection]
    summary_metrics: Dict[str, Any] = {
        "total_principal": 0.0,
        "total_interest": 0.0,
        "total_payments": 0.0,
        "npv": 0.0,
        "irr": 0.0,
        "duration": 0.0,
        "convexity": 0.0,
        "monte_carlo_results": None
    }
    monte_carlo_results: Optional[Dict[str, List[float]]]

class ScenarioSaveRequest(BaseModel):
    name: str
    description: Optional[str]
    forecast_request: CashflowForecastRequest
    tags: List[str] = Field(default_factory=list)

class ScenarioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    tags: List[str]
    forecast_request: CashflowForecastRequest
