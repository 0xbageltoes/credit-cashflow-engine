"""Supabase client and database operations"""
import os
from typing import Dict, List, Optional
from datetime import datetime
from supabase import create_client, Client

from app.models.cashflow import (
    LoanData,
    CashflowProjection,
    MonteCarloResults,
    CashflowForecastResponse
)

class SupabaseClient:
    """Supabase client wrapper for database operations"""
    def __init__(self):
        url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Using service role key for tests
        if not url or not key:
            raise ValueError("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables must be set")
        
        self.client: Client = create_client(url, key)

    def create_loan(self, user_id: str, loan: LoanData) -> Dict:
        """Create a new loan record"""
        loan_data = loan.model_dump()
        loan_data["user_id"] = user_id
        
        result = self.client.table("loans").insert(loan_data).execute()
        self._log_audit(user_id, "create", "loan", result.data[0]["id"], loan_data)
        return result.data[0]
    
    def get_loan(self, user_id: str, loan_id: str) -> Optional[Dict]:
        """Get a loan record by ID"""
        result = self.client.table("loans").select("*").eq("id", loan_id).eq("user_id", user_id).execute()
        return result.data[0] if result.data else None
    
    def list_loans(self, user_id: str) -> List[Dict]:
        """List all loans for a user"""
        result = self.client.table("loans").select("*").eq("user_id", user_id).execute()
        return result.data
    
    def update_loan(self, user_id: str, loan_id: str, loan: LoanData) -> Dict:
        """Update a loan record"""
        loan_data = loan.model_dump()
        result = self.client.table("loans").update(loan_data).eq("id", loan_id).eq("user_id", user_id).execute()
        self._log_audit(user_id, "update", "loan", loan_id, loan_data)
        return result.data[0]
    
    def delete_loan(self, user_id: str, loan_id: str) -> None:
        """Delete a loan record"""
        self.client.table("loans").delete().eq("id", loan_id).eq("user_id", user_id).execute()
        self._log_audit(user_id, "delete", "loan", loan_id, None)
    
    def save_cashflow_projections(self, user_id: str, loan_id: str, response: CashflowForecastResponse) -> None:
        """Save cashflow projections and Monte Carlo results"""
        forecast_data = {
            "user_id": user_id,
            "request": {
                "loan_id": loan_id,
                "summary_metrics": response.summary_metrics,
                "computation_time": response.computation_time
            },
            "projections": {
                "cashflows": [proj.model_dump() for proj in response.projections],
                "monte_carlo_results": response.monte_carlo_results.model_dump() if response.monte_carlo_results else None
            }
        }
        
        self.client.table("forecast_runs").insert(forecast_data).execute()
    
    def get_cashflow_projections(self, user_id: str, loan_id: str) -> List[Dict]:
        """Get cashflow projections for a loan"""
        result = self.client.table("forecast_runs").select("projections").eq("user_id", user_id).execute()
        if not result.data:
            return []
        
        # Get the latest forecast run for this loan
        latest_run = result.data[-1]  # Assuming ordered by created_at
        return latest_run["projections"]["cashflows"]
    
    def get_monte_carlo_results(self, user_id: str, loan_id: str) -> Optional[Dict]:
        """Get Monte Carlo results for a loan"""
        result = self.client.table("forecast_runs").select("projections").eq("user_id", user_id).execute()
        if not result.data:
            return None
        
        # Get the latest forecast run for this loan
        latest_run = result.data[-1]  # Assuming ordered by created_at
        return latest_run["projections"]["monte_carlo_results"]
    
    def _log_audit(self, user_id: str, action: str, entity_type: str, entity_id: str, changes: Optional[Dict]) -> None:
        """Log an audit entry"""
        audit_data = {
            "user_id": user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "changes": changes,
            "created_at": datetime.now().isoformat()
        }
        self.client.table("audit_log").insert(audit_data).execute()
