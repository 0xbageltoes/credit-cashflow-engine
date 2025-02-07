import numpy as np
import numpy_financial as npf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowForecastResponse,
    CashflowProjection,
    ScenarioSaveRequest,
    ScenarioResponse
)
from app.core.config import settings
from supabase import create_client, Client
from functools import lru_cache

class CashflowService:
    def __init__(self):
        # Initialize Supabase client with service role key
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY  # Use service role key instead of anon key
        )

    async def generate_forecast(
        self,
        request: CashflowForecastRequest,
        user_id: str
    ) -> CashflowForecastResponse:
        """
        Generate cash flow projections for a set of loans
        """
        # Initialize arrays for vectorized calculations
        projections = []
        total_principal = 0
        total_interest = 0

        for loan in request.loans:
            # Calculate monthly payment using numpy financial functions
            rate = float(loan.interest_rate) / 12  # Monthly rate
            pmt = npf.pmt(rate, int(loan.term_months), -float(loan.principal))
            
            # Generate amortization schedule
            periods = np.arange(1, int(loan.term_months) + 1)
            start_date = pd.to_datetime(loan.start_date)
            dates = pd.date_range(
                start=start_date,
                periods=int(loan.term_months),
                freq='M'
            )
            
            # Calculate remaining balance for each period
            remaining_balance = npf.ppmt(
                rate,
                periods,
                int(loan.term_months),
                -float(loan.principal)
            ).cumsum()
            
            # Calculate principal and interest components
            principal_payments = npf.ppmt(
                rate,
                periods,
                int(loan.term_months),
                -float(loan.principal)
            )
            interest_payments = npf.ipmt(
                rate,
                periods,
                int(loan.term_months),
                -float(loan.principal)
            )
            
            # Apply prepayment assumptions if specified
            if float(loan.prepayment_assumption) > 0:
                prepayment_factor = (1 - float(loan.prepayment_assumption)) ** periods
                principal_payments *= prepayment_factor
                interest_payments *= prepayment_factor
                remaining_balance *= prepayment_factor

            # Create projections
            for i in range(len(periods)):
                projections.append(
                    CashflowProjection(
                        period=int(periods[i]),
                        date=dates[i].isoformat(),  # Convert to ISO format string
                        principal=float(abs(principal_payments[i])),
                        interest=float(abs(interest_payments[i])),
                        total_payment=float(abs(principal_payments[i] + interest_payments[i])),
                        remaining_balance=float(abs(remaining_balance[i]))
                    )
                )
                
            total_principal += float(loan.principal)
            total_interest += abs(interest_payments.sum())

        # Calculate summary metrics
        summary_metrics = {
            "total_principal": float(total_principal),
            "total_interest": float(total_interest),
            "total_payments": float(total_principal + total_interest),
            "weighted_avg_rate": float(sum(float(l.interest_rate) * float(l.principal) for l in request.loans) / total_principal)
        }

        # Run Monte Carlo simulation if requested
        monte_carlo_results = None
        if request.monte_carlo_sims:
            monte_carlo_results = self._run_monte_carlo(
                request.loans,
                request.monte_carlo_sims,
                request.assumptions
            )

        try:
            # Save forecast to database
            forecast_id = await self._save_forecast(request, projections, user_id)
        except Exception as e:
            print(f"Error saving forecast: {str(e)}")
            forecast_id = None

        return CashflowForecastResponse(
            scenario_id=forecast_id,
            projections=projections,
            summary_metrics=summary_metrics,
            monte_carlo_results=monte_carlo_results
        )

    def _run_monte_carlo(
        self,
        loans: List,
        num_sims: int,
        assumptions: Dict
    ) -> Dict[str, List[float]]:
        """
        Run Monte Carlo simulation for risk analysis
        """
        # Implement Monte Carlo simulation logic here
        # This is a placeholder that would need to be implemented based on specific requirements
        return {
            "total_payments": [0] * num_sims,
            "npv": [0] * num_sims
        }

    async def _save_forecast(
        self,
        request: CashflowForecastRequest,
        projections: List[CashflowProjection],
        user_id: str
    ) -> str:
        """
        Save forecast results to Supabase
        """
        data = {
            "user_id": user_id,
            "request": request.model_dump(),
            "projections": [p.model_dump() for p in projections],
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = self.supabase.table("forecast_runs").insert(data).execute()
        return result.data[0]["id"]

    async def save_scenario(
        self,
        scenario: ScenarioSaveRequest,
        user_id: str
    ) -> Dict:
        """
        Save a scenario to Supabase
        """
        data = {
            "user_id": user_id,
            "name": scenario.name,
            "description": scenario.description,
            "forecast_request": scenario.forecast_request.model_dump(),
            "tags": scenario.tags,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = self.supabase.table("cashflow_scenarios").insert(data).execute()
        return result.data[0]

    async def load_scenarios(
        self,
        user_id: str
    ) -> List[ScenarioResponse]:
        """
        Load saved scenarios for a user
        """
        result = self.supabase.table("cashflow_scenarios") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()
            
        return [ScenarioResponse(**scenario) for scenario in result.data]

    async def get_forecast_history(
        self,
        user_id: str
    ) -> List[Dict]:
        """
        Get forecast history for a user
        """
        result = self.supabase.table("forecast_runs") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
            
        return result.data
