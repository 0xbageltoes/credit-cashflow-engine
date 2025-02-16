"""Analytics models for credit-cashflow-engine"""
from pydantic import BaseModel, Field

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
