from fastapi import APIRouter, Depends, HTTPException, Header, Body, Query
from fastapi.responses import JSONResponse

from app.database.supabase import SupabaseClient
from app.services.absbox_service import AbsBoxService
from app.models.cashflow import (
    LoanData, 
    CashflowForecastRequest, 
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse
)
from app.services.auth import get_user_id_from_token

router = APIRouter()

@router.post("/forecasts/", response_model=CashflowForecastResponse)
async def create_cashflow_forecast(
    request: CashflowForecastRequest,
    authorization: str = Header(...),
):
    """Generate cashflow projections for a loan."""
    user_id = get_user_id_from_token(authorization)
    
    # Initialize services
    db = SupabaseClient()
    absbox_service = AbsBoxService()
    
    try:
        # Generate cashflow projections
        response = absbox_service.generate_cashflow_forecast(request)
        
        # Save to database
        db.save_cashflow_projections(user_id, request.loan_data.loan_id, response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cashflow projection: {str(e)}")

@router.get("/forecasts/")
async def list_forecast_runs(
    authorization: str = Header(...),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all forecast runs for a user."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        forecasts = db.list_forecast_runs(user_id, limit, offset)
        return JSONResponse(content={"forecasts": forecasts, "count": len(forecasts)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing forecasts: {str(e)}")

@router.get("/forecasts/{forecast_id}")
async def get_forecast_run(
    forecast_id: str,
    authorization: str = Header(...),
):
    """Get a specific forecast run."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        forecast = db.get_forecast_run(user_id, forecast_id)
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return JSONResponse(content=forecast)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving forecast: {str(e)}")

@router.get("/forecasts/{forecast_id}/projections")
async def get_forecast_projections(
    forecast_id: str,
    authorization: str = Header(...),
):
    """Get cashflow projections for a specific forecast."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        projections = db.get_cashflow_projections(user_id, forecast_id)
        return JSONResponse(content={"projections": projections, "count": len(projections)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving projections: {str(e)}")

@router.post("/scenarios/", response_model=dict)
async def create_scenario(
    scenario: ScenarioSaveRequest,
    authorization: str = Header(...),
):
    """Save a cashflow scenario."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        result = db.save_cashflow_scenario(user_id, scenario)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving scenario: {str(e)}")

@router.put("/scenarios/{scenario_id}", response_model=dict)
async def update_scenario(
    scenario_id: str,
    scenario: ScenarioSaveRequest,
    authorization: str = Header(...),
):
    """Update a cashflow scenario."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        # Check if scenario exists
        existing = db.get_cashflow_scenario(user_id, scenario_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Scenario not found")
            
        result = db.update_cashflow_scenario(user_id, scenario_id, scenario)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating scenario: {str(e)}")

@router.get("/scenarios/")
async def list_scenarios(
    authorization: str = Header(...),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    is_template: Optional[bool] = Query(None),
):
    """List all cashflow scenarios for a user."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        scenarios = db.list_cashflow_scenarios(user_id, limit, offset, is_template)
        return JSONResponse(content={"scenarios": scenarios, "count": len(scenarios)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing scenarios: {str(e)}")

@router.get("/scenarios/{scenario_id}")
async def get_scenario(
    scenario_id: str,
    authorization: str = Header(...),
):
    """Get a specific cashflow scenario."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        scenario = db.get_cashflow_scenario(user_id, scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return JSONResponse(content=scenario)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scenario: {str(e)}")

@router.delete("/scenarios/{scenario_id}")
async def delete_scenario(
    scenario_id: str,
    authorization: str = Header(...),
):
    """Delete a cashflow scenario."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        # Check if scenario exists
        existing = db.get_cashflow_scenario(user_id, scenario_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Scenario not found")
            
        db.delete_cashflow_scenario(user_id, scenario_id)
        return JSONResponse(content={"message": "Scenario deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting scenario: {str(e)}")

@router.post("/scenarios/{scenario_id}/run", response_model=CashflowForecastResponse)
async def run_scenario(
    scenario_id: str,
    authorization: str = Header(...),
):
    """Run a saved cashflow scenario."""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    absbox_service = AbsBoxService()
    
    try:
        # Get the scenario
        scenario = db.get_cashflow_scenario(user_id, scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        # Create a forecast request from the scenario
        forecast_config = scenario.get("forecast_config", {})
        request = CashflowForecastRequest(**forecast_config)
        
        # Generate cashflow projections
        response = absbox_service.generate_cashflow_forecast(request)
        
        # Save to database with reference to scenario
        db.save_cashflow_projections(user_id, request.loan_data.loan_id, response)
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running scenario: {str(e)}")

@router.post("/loans", response_model=dict)
async def create_loan(
    loan: LoanData,
    authorization: str = Header(...),
):
    """Create a new loan"""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        return db.create_loan(user_id, loan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loans", response_model=dict)
async def list_loans(
    authorization: str = Header(...),
):
    """List all loans for current user"""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        return db.list_loans(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loans/{loan_id}", response_model=dict)
async def get_loan(
    loan_id: str,
    authorization: str = Header(...),
):
    """Get a specific loan"""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        loan = db.get_loan(user_id, loan_id)
        if not loan:
            raise HTTPException(status_code=404, detail="Loan not found")
        return loan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/loans/{loan_id}", response_model=dict)
async def update_loan(
    loan_id: str,
    loan: LoanData,
    authorization: str = Header(...),
):
    """Update a loan"""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        return db.update_loan(user_id, loan_id, loan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/loans/{loan_id}")
async def delete_loan(
    loan_id: str,
    authorization: str = Header(...),
):
    """Delete a loan"""
    user_id = get_user_id_from_token(authorization)
    
    db = SupabaseClient()
    try:
        db.delete_loan(user_id, loan_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
