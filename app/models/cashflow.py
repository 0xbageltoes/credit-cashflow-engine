from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class LoanData(BaseModel):
    principal: float
    interest_rate: float
    term_months: int
    start_date: str
    prepayment_assumption: Optional[float] = 0.0

class CashflowForecastRequest(BaseModel):
    loans: List[LoanData]
    scenario_name: Optional[str] = None
    monte_carlo_sims: Optional[int] = None
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
    summary_metrics: Dict[str, float]
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
