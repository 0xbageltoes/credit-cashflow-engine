from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_current_user
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse,
    LoanData
)
from app.services.cashflow import CashflowService
from app.database.supabase import SupabaseClient
from typing import List
from uuid import UUID

cashflow_router = APIRouter()

# Dependency to get Supabase client
async def get_supabase():
    return SupabaseClient()

@cashflow_router.post("/loans", response_model=dict)
async def create_loan(
    loan: LoanData,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Create a new loan"""
    try:
        return await supabase.create_loan(current_user["id"], loan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/loans", response_model=List[dict])
async def list_loans(
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """List all loans for current user"""
    try:
        return await supabase.list_loans(current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/loans/{loan_id}", response_model=dict)
async def get_loan(
    loan_id: UUID,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Get a specific loan"""
    try:
        loan = await supabase.get_loan(current_user["id"], str(loan_id))
        if not loan:
            raise HTTPException(status_code=404, detail="Loan not found")
        return loan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.put("/loans/{loan_id}", response_model=dict)
async def update_loan(
    loan_id: UUID,
    loan: LoanData,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Update a loan"""
    try:
        return await supabase.update_loan(current_user["id"], str(loan_id), loan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.delete("/loans/{loan_id}")
async def delete_loan(
    loan_id: UUID,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Delete a loan"""
    try:
        await supabase.delete_loan(current_user["id"], str(loan_id))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.post("/forecast", response_model=CashflowForecastResponse)
async def create_forecast(
    request: CashflowForecastRequest,
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends(),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Generate a cash flow forecast and save results"""
    try:
        # Generate forecast
        result = await cashflow_service.generate_forecast(request, current_user["id"])
        
        # Save results to database if loan_id is provided
        if hasattr(request, 'loan_id') and request.loan_id:
            await supabase.save_cashflow_projections(
                current_user["id"],
                str(request.loan_id),
                result
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/forecast/{loan_id}/projections", response_model=List[dict])
async def get_projections(
    loan_id: UUID,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Get saved projections for a loan"""
    try:
        return await supabase.get_cashflow_projections(current_user["id"], str(loan_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/forecast/{loan_id}/monte-carlo", response_model=dict)
async def get_monte_carlo(
    loan_id: UUID,
    current_user: dict = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Get Monte Carlo results for a loan"""
    try:
        results = await supabase.get_monte_carlo_results(current_user["id"], str(loan_id))
        if not results:
            raise HTTPException(status_code=404, detail="Monte Carlo results not found")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.post("/scenario/save")
async def save_scenario(
    scenario: ScenarioSaveRequest,
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """Save a forecasting scenario"""
    try:
        return await cashflow_service.save_scenario(scenario, current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/scenario/load", response_model=List[ScenarioResponse])
async def load_scenarios(
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """Load saved scenarios for the current user"""
    try:
        return await cashflow_service.load_scenarios(current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cashflow_router.get("/history")
async def get_history(
    current_user: dict = Depends(get_current_user),
    cashflow_service: CashflowService = Depends()
):
    """Get forecast history for the current user"""
    try:
        return await cashflow_service.get_forecast_history(current_user["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
