"""Supabase client and database operations"""
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from supabase import create_client, Client

from app.models.cashflow import (
    LoanData,
    CashflowProjection,
    MonteCarloResults,
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse
)
from app.models.analytics import EnhancedAnalyticsResult
from app.models.structured_products import StructuredDealRequest, StructuredDealResponse

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
    
    def save_cashflow_projections(self, user_id: str, loan_id: str, response: CashflowForecastResponse) -> Dict:
        """Save cashflow projections and Monte Carlo results with enhanced analytics metrics"""
        # Create forecast run with enhanced metrics
        forecast_data = {
            "user_id": user_id,
            "scenario_name": f"Loan Analysis {loan_id[:8]}",  # Ensuring non-empty valid string
            "total_principal": float(response.summary_metrics["total_principal"]),
            "total_interest": float(response.summary_metrics["total_interest"]),
            "npv": float(response.summary_metrics["npv"]),
            "irr": float(response.summary_metrics["irr"]),
            "duration": float(response.summary_metrics["duration"]),
            "convexity": float(response.summary_metrics["convexity"]),
            "weighted_average_life": float(response.summary_metrics.get("weighted_average_life", 0.0)),
            "yield_value": float(response.summary_metrics.get("yield_value", 0.0)),
            "spread_value": float(response.summary_metrics.get("spread_value", 0.0)),
            "macaulay_duration": float(response.summary_metrics.get("macaulay_duration", 0.0)),
            "modified_duration": float(response.summary_metrics.get("modified_duration", 0.0)),
            "discount_margin": float(response.summary_metrics.get("discount_margin", 0.0)),
            "debt_service_coverage": float(response.summary_metrics.get("debt_service_coverage", 0.0)),
            "monte_carlo_results": response.monte_carlo_results.model_dump() if response.monte_carlo_results else None
        }
        
        result = self.client.table("forecast_runs").insert(forecast_data).execute()
        forecast_id = result.data[0]["id"]
        
        # Save projections
        for projection in response.projections:
            projection_data = {
                "forecast_id": forecast_id,
                "user_id": user_id,
                "period": projection.period,
                "date": projection.date,
                "principal": projection.principal,
                "interest": projection.interest,
                "total_payment": projection.total_payment,
                "remaining_balance": projection.remaining_balance,
                "default_amount": 0.0,  # Add default values for new fields
                "prepayment_amount": 0.0  # Add default values for new fields
            }
            self.client.table("forecast_projections").insert(projection_data).execute()
        
        return result.data[0]
    
    def get_cashflow_projections(self, user_id: str, forecast_id: Optional[str] = None) -> List[Dict]:
        """Get cashflow projections for a specific forecast or latest forecast"""
        # If forecast_id is provided, use it, otherwise get the latest
        if not forecast_id:
            forecast_result = self.client.table("forecast_runs").select("id").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            if not forecast_result.data:
                return []
            
            forecast_id = forecast_result.data[0]["id"]
        
        # Get all projections for this forecast
        result = self.client.table("forecast_projections").select("*").eq("forecast_id", forecast_id).eq("user_id", user_id).execute()
        return result.data
    
    def get_forecast_run(self, user_id: str, forecast_id: Optional[str] = None) -> Optional[Dict]:
        """Get a specific forecast run or the latest one"""
        query = self.client.table("forecast_runs").select("*").eq("user_id", user_id)
        
        if forecast_id:
            query = query.eq("id", forecast_id)
        else:
            query = query.order("created_at", desc=True).limit(1)
            
        result = query.execute()
        return result.data[0] if result.data else None
    
    def list_forecast_runs(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all forecast runs for a user"""
        result = self.client.table("forecast_runs") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        return result.data
    
    def save_cashflow_scenario(self, user_id: str, scenario: ScenarioSaveRequest) -> Dict:
        """Save a cashflow scenario"""
        scenario_data = {
            "user_id": user_id,
            "name": scenario.name,
            "description": scenario.description,
            "forecast_config": scenario.forecast_request.model_dump(),
            "tags": scenario.tags,
            "is_template": scenario.is_template
        }
        
        result = self.client.table("cashflow_scenarios").insert(scenario_data).execute()
        self._log_audit(user_id, "create", "scenario", result.data[0]["id"], scenario_data)
        return result.data[0]
    
    def update_cashflow_scenario(self, user_id: str, scenario_id: str, scenario: ScenarioSaveRequest) -> Dict:
        """Update a cashflow scenario"""
        scenario_data = {
            "name": scenario.name,
            "description": scenario.description,
            "forecast_config": scenario.forecast_request.model_dump(),
            "tags": scenario.tags,
            "is_template": scenario.is_template,
            "updated_at": datetime.now().isoformat()
        }
        
        result = self.client.table("cashflow_scenarios").update(scenario_data).eq("id", scenario_id).eq("user_id", user_id).execute()
        self._log_audit(user_id, "update", "scenario", scenario_id, scenario_data)
        return result.data[0]
    
    def get_cashflow_scenario(self, user_id: str, scenario_id: str) -> Optional[Dict]:
        """Get a specific cashflow scenario"""
        result = self.client.table("cashflow_scenarios").select("*").eq("id", scenario_id).eq("user_id", user_id).execute()
        return result.data[0] if result.data else None
    
    def list_cashflow_scenarios(self, user_id: str, limit: int = 100, offset: int = 0, is_template: Optional[bool] = None) -> List[Dict]:
        """List cashflow scenarios for a user"""
        query = self.client.table("cashflow_scenarios") \
            .select("*") \
            .eq("user_id", user_id)
            
        if is_template is not None:
            query = query.eq("is_template", is_template)
            
        result = query.order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
            
        return result.data
    
    def delete_cashflow_scenario(self, user_id: str, scenario_id: str) -> None:
        """Delete a cashflow scenario"""
        self.client.table("cashflow_scenarios").delete().eq("id", scenario_id).eq("user_id", user_id).execute()
        self._log_audit(user_id, "delete", "scenario", scenario_id, None)
    
    def get_monte_carlo_results(self, user_id: str, loan_id: str) -> Optional[Dict]:
        """Get Monte Carlo results for a loan"""
        result = self.client.table("forecast_runs").select("monte_carlo_results").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if not result.data:
            return None
        
        return result.data[0]["monte_carlo_results"]
    
    def save_enhanced_analytics(self, user_id: str, loan_id: str, analytics: EnhancedAnalyticsResult) -> Dict:
        """Save enhanced analytics results for a loan"""
        analytics_data = analytics.model_dump()
        analytics_data["user_id"] = user_id
        analytics_data["loan_id"] = loan_id
        
        # Convert all numeric values to float to ensure PostgreSQL compatibility
        for key, value in analytics_data.items():
            if isinstance(value, (int, float)) and key != "user_id" and key != "loan_id":
                analytics_data[key] = float(value)
        
        result = self.client.table("enhanced_analytics_results").insert(analytics_data).execute()
        self._log_audit(user_id, "create", "enhanced_analytics", result.data[0]["id"], analytics_data)
        return result.data[0]
    
    def get_enhanced_analytics(self, user_id: str, loan_id: str) -> Optional[Dict]:
        """Get enhanced analytics results for a loan"""
        result = self.client.table("enhanced_analytics_results") \
            .select("*") \
            .eq("loan_id", loan_id) \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        return result.data[0] if result.data else None
    
    def list_enhanced_analytics(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all enhanced analytics results for a user"""
        result = self.client.table("enhanced_analytics_results") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        return result.data
    
    def save_structured_deal_result(self, user_id: str, deal_result: StructuredDealResponse) -> Dict:
        """Save structured deal analysis results"""
        deal_data = {
            "user_id": user_id,
            "deal_name": deal_result.deal_name,
            "execution_time": float(deal_result.execution_time),
            "bond_cashflows": deal_result.bond_cashflows,
            "pool_cashflows": deal_result.pool_cashflows,
            "pool_statistics": deal_result.pool_statistics,
            "metrics": deal_result.metrics,
            "status": deal_result.status,
            "error": deal_result.error,
            "error_type": deal_result.error_type
        }
        
        result = self.client.table("structured_deal_results").insert(deal_data).execute()
        self._log_audit(user_id, "create", "structured_deal", result.data[0]["id"], deal_data)
        return result.data[0]
    
    def get_structured_deal_result(self, user_id: str, deal_id: str) -> Optional[Dict]:
        """Get a specific structured deal analysis result"""
        result = self.client.table("structured_deal_results") \
            .select("*") \
            .eq("id", deal_id) \
            .eq("user_id", user_id) \
            .execute()
        return result.data[0] if result.data else None
    
    def list_structured_deal_results(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all structured deal analysis results for a user"""
        result = self.client.table("structured_deal_results") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        return result.data
    
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
