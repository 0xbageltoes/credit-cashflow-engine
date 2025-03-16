"""API endpoints for structured finance products using AbsBox"""
import logging
from typing import Dict, List, Any, Optional
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from app.services.absbox_service import AbsBoxService
from app.models.structured_products import (
    StructuredDealRequest,
    StructuredDealResponse,
    ScenarioConfig,
    AnalysisResult
)
from app.core.monitoring import REQUEST_COUNT, CALCULATION_TIME
from app.api.deps import get_current_user
from app.core.cache_service import CacheService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", summary="Check structured finance engine health")
async def check_health():
    """Check health of AbsBox and Hastructure engine"""
    service = AbsBoxService()
    return service.health_check()

@router.post("/deals/analyze", 
            response_model=StructuredDealResponse, 
            summary="Analyze structured finance deal")
async def analyze_deal(
    deal_request: StructuredDealRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze a structured finance deal using AbsBox
    
    This endpoint takes a structured deal configuration including:
    - Loan pool details
    - Waterfall structure
    - Assumptions and scenarios
    
    It returns detailed cashflow projections, metrics and statistics.
    """
    try:
        REQUEST_COUNT.labels(endpoint="/api/v1/structured-products/deals/analyze").inc()
        
        logger.info(f"Processing structured deal analysis for {deal_request.deal_name}")
        service = AbsBoxService()
        result = service.analyze_deal(deal_request)
        
        if result.status == "error":
            logger.error(f"Error analyzing deal: {result.error}")
            return JSONResponse(
                status_code=422,
                content={"detail": result.error}
            )
            
        return result
    except Exception as e:
        logger.exception(f"Error processing structured deal analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@router.post("/deals/scenarios", 
            response_model=List[AnalysisResult], 
            summary="Run scenario analysis on deal structure")
async def run_scenarios(
    deal_request: StructuredDealRequest,
    scenarios: List[ScenarioConfig] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Run multiple scenarios on a deal structure
    
    This endpoint allows you to test a single deal structure against multiple
    scenarios, such as different default rates, interest rate environments,
    or prepayment assumptions.
    
    Returns summary metrics for each scenario for comparison.
    """
    try:
        REQUEST_COUNT.labels(endpoint="/api/v1/structured-products/deals/scenarios").inc()
        
        service = AbsBoxService()
        logger.info(f"Running {len(scenarios)} scenarios on deal {deal_request.deal_name}")
        results = service.run_scenario_analysis(deal_request, scenarios)
            
        return results
    except Exception as e:
        logger.exception(f"Error running scenario analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario analysis error: {str(e)}")

@router.get("/metrics", summary="Get structured finance calculation metrics")
async def get_metrics(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get metrics about structured finance calculations
    
    Returns timing metrics for various calculations and operations
    """
    metrics = {}
    
    # Collect calculation time metrics
    for name, metric in CALCULATION_TIME._metrics.items():
        if name.startswith("absbox_"):
            metrics[name] = {
                "count": metric._count,
                "sum": metric._sum,
                "avg": metric._sum / metric._count if metric._count > 0 else 0
            }
    
    return {
        "calculation_metrics": metrics
    }
