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
from app.services.analytics import AnalyticsService, AnalyticsResult
from app.core.cache import cache_response, RedisManager

class CashflowService:
    def __init__(self):
        # Initialize Supabase client with service role key
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        self.analytics = AnalyticsService()
        self.cache = RedisManager()

    def _vectorized_loan_calculations(self, loans: List[Dict]) -> tuple:
        """Vectorized calculation of loan amortization schedules"""
        # Convert loan parameters to numpy arrays
        principals = np.array([float(loan.principal) for loan in loans])
        rates = np.array([float(loan.interest_rate) / 12 for loan in loans])
        terms = np.array([int(loan.term_months) for loan in loans])
        prepay_rates = np.array([float(loan.prepayment_assumption) for loan in loans])
        
        # Calculate monthly payments
        pmts = npf.pmt(rates[:, np.newaxis], terms[:, np.newaxis], -principals[:, np.newaxis])
        
        # Generate periods array for each loan
        max_term = max(terms)
        periods = np.arange(1, max_term + 1)
        periods_matrix = np.broadcast_to(periods, (len(loans), len(periods)))
        
        # Calculate components for all loans at once
        principal_payments = np.zeros((len(loans), max_term))
        interest_payments = np.zeros((len(loans), max_term))
        remaining_balance = np.zeros((len(loans), max_term))
        
        for i in range(len(loans)):
            term = terms[i]
            principal = principals[i]
            rate = rates[i]
            
            # Calculate for valid periods only
            valid_periods = periods[:term]
            principal_payments[i, :term] = npf.ppmt(rate, valid_periods, term, -principal)
            interest_payments[i, :term] = npf.ipmt(rate, valid_periods, term, -principal)
            remaining_balance[i, :term] = principal * (1 + rate) ** valid_periods - \
                                        pmts[i] * ((1 + rate) ** valid_periods - 1) / rate
            
            # Apply prepayment assumptions
            if prepay_rates[i] > 0:
                prepay_factor = (1 - prepay_rates[i]) ** valid_periods
                principal_payments[i, :term] *= prepay_factor
                interest_payments[i, :term] *= prepay_factor
                remaining_balance[i, :term] *= prepay_factor
        
        return principal_payments, interest_payments, remaining_balance, periods_matrix

    async def generate_forecast(
        self,
        request: CashflowForecastRequest,
        user_id: str
    ) -> CashflowForecastResponse:
        """Generate cash flow projections for a set of loans using vectorized operations"""
        # Vectorized calculations
        principal_payments, interest_payments, remaining_balance, periods = \
            self._vectorized_loan_calculations(request.loans)
        
        # Combine results into projections
        projections = []
        total_principal = 0
        total_interest = 0
        
        for loan_idx, loan in enumerate(request.loans):
            start_date = pd.to_datetime(loan.start_date)
            dates = pd.date_range(
                start=start_date,
                periods=int(loan.term_months),
                freq='M'
            )
            
            for period_idx in range(int(loan.term_months)):
                projections.append(
                    CashflowProjection(
                        period=int(periods[loan_idx, period_idx]),
                        date=dates[period_idx].isoformat(),
                        principal=float(abs(principal_payments[loan_idx, period_idx])),
                        interest=float(abs(interest_payments[loan_idx, period_idx])),
                        total_payment=float(abs(principal_payments[loan_idx, period_idx] + 
                                             interest_payments[loan_idx, period_idx])),
                        remaining_balance=float(abs(remaining_balance[loan_idx, period_idx]))
                    )
                )
            
            total_principal += float(loan.principal)
            total_interest += abs(interest_payments[loan_idx, :int(loan.term_months)].sum())

        # Calculate analytics
        combined_cashflows = np.sum(principal_payments + interest_payments, axis=0)
        analytics_result = self.analytics.analyze_cashflows(
            combined_cashflows,
            discount_rate=0.05,
            run_monte_carlo=True
        )

        # Prepare response with enhanced metrics
        await self._save_forecast(
            request,
            projections,
            user_id,
            analytics_result
        )
        return CashflowForecastResponse(
            projections=projections,
            summary_metrics={
                "total_principal": float(total_principal),
                "total_interest": float(total_interest),
                "total_payments": float(total_principal + total_interest),
                "npv": float(analytics_result.npv),
                "irr": float(analytics_result.irr),
                "duration": float(analytics_result.duration),
                "convexity": float(analytics_result.convexity),
                "monte_carlo_results": analytics_result.monte_carlo_results
            }
        )

    async def _batch_save_projections(
        self,
        projections: List[CashflowProjection],
        forecast_id: str,
        user_id: str
    ) -> None:
        """Save projections in batches for better performance"""
        batch = []
        for proj in projections:
            batch.append({
                "forecast_id": forecast_id,
                "user_id": user_id,
                "period": proj.period,
                "date": proj.date,
                "principal": proj.principal,
                "interest": proj.interest,
                "total_payment": proj.total_payment,
                "remaining_balance": proj.remaining_balance
            })
            
            if len(batch) >= settings.BATCH_SIZE:
                await self._save_batch(batch)
                batch = []
        
        # Save remaining items
        if batch:
            await self._save_batch(batch)

    async def _save_batch(self, batch: List[Dict]) -> None:
        """Save a batch of records to Supabase"""
        try:
            await self.supabase.table("forecast_projections").insert(batch).execute()
        except Exception as e:
            print(f"Error saving batch: {str(e)}")
            raise

    @cache_response(ttl=3600)  # Cache for 1 hour
    async def get_forecast_by_id(self, forecast_id: str, user_id: str) -> Optional[Dict]:
        """Get forecast by ID with caching"""
        try:
            response = await self.supabase.table("forecast_runs") \
                .select("*") \
                .eq("id", forecast_id) \
                .eq("user_id", user_id) \
                .single() \
                .execute()
            return response.data
        except Exception as e:
            print(f"Error fetching forecast: {str(e)}")
            return None

    async def _save_forecast(
        self,
        request: CashflowForecastRequest,
        projections: List[CashflowProjection],
        user_id: str,
        analytics_result: AnalyticsResult
    ) -> str:
        """Save forecast with optimized batch operations"""
        try:
            # Save forecast run
            forecast_data = {
                "user_id": user_id,
                "scenario_name": request.scenario_name or "Default Scenario",
                "total_principal": sum(float(loan.principal) for loan in request.loans),
                "total_interest": analytics_result.total_interest,
                "npv": analytics_result.npv,
                "irr": analytics_result.irr,
                "duration": analytics_result.duration,
                "convexity": analytics_result.convexity,
                "monte_carlo_results": analytics_result.monte_carlo_results
            }
            
            response = await self.supabase.table("forecast_runs").insert(forecast_data).execute()
            forecast_id = response.data[0]["id"]
            
            # Save projections in batches
            await self._batch_save_projections(projections, forecast_id, user_id)
            
            # Cache the forecast
            cache_key = f"forecast:{forecast_id}"
            self.cache.set(cache_key, {
                "forecast": forecast_data,
                "projections": [proj.dict() for proj in projections]
            })
            
            return forecast_id
            
        except Exception as e:
            print(f"Error saving forecast: {str(e)}")
            raise

    @cache_response(ttl=3600)
    async def list_forecasts(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 10
    ) -> Dict:
        """List forecasts with pagination and caching"""
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Get total count
            count_response = await self.supabase.table("forecast_runs") \
                .select("id", count="exact") \
                .eq("user_id", user_id) \
                .execute()
            total_count = count_response.count
            
            # Get paginated results
            response = await self.supabase.table("forecast_runs") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .range(offset, offset + page_size - 1) \
                .execute()
                
            return {
                "results": response.data,
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size
            }
            
        except Exception as e:
            print(f"Error listing forecasts: {str(e)}")
            raise

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
