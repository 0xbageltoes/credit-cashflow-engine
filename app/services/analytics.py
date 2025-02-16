"""Analytics service for calculating financial metrics"""
from typing import List, Dict, Optional
import numpy as np
import numpy_financial as npf
from app.models.analytics import AnalyticsResult
from app.models.cashflow import CashflowProjection

class AnalyticsService:
    """Service for calculating financial metrics from cashflow projections"""

    async def calculate_metrics(
        self,
        projections: List[CashflowProjection],
        discount_rate: float = 0.05
    ) -> AnalyticsResult:
        """Calculate key financial metrics from cashflow projections"""
        
        # Extract cashflows
        cashflows = [-projections[0].total_payment]  # Initial investment
        for proj in projections:
            cashflows.append(proj.total_payment)
        
        # Convert to numpy array for calculations
        cashflows = np.array(cashflows)
        
        # Calculate NPV
        npv = npf.npv(discount_rate / 12, cashflows)
        
        # Calculate IRR
        try:
            irr = npf.irr(cashflows) * 12  # Annualize monthly IRR
        except:
            irr = 0.0
        
        # Calculate DSCR (Debt Service Coverage Ratio)
        total_payments = sum(p.total_payment for p in projections)
        total_income = sum(p.total_payment + p.remaining_balance for p in projections)
        dscr = total_income / total_payments if total_payments > 0 else 0
        
        # Calculate LTV (Loan to Value)
        initial_balance = projections[0].remaining_balance
        property_value = initial_balance * 1.25  # Assuming 80% LTV at origination
        ltv = initial_balance / property_value if property_value > 0 else 1
        
        return AnalyticsResult(
            npv=float(npv),
            irr=float(irr),
            dscr=float(dscr),
            ltv=float(ltv)
        )
