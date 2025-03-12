"""API endpoints for cashflow calculations"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user
from app.models.cashflow import LoanData, BatchForecastRequest, CashflowForecastResponse
from app.services.cashflow import CashflowService
from app.core.monitoring import request_counter, CALCULATION_TIME
from app.models.analytics import EnhancedAnalyticsRequest

# Import the enhanced absbox service for advanced analytics
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/calculate", 
            response_model=CashflowForecastResponse, 
            summary="Calculate loan cashflows")
async def calculate_cashflow(
    loan_data: LoanData,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate detailed cashflows for a single loan
    
    This endpoint returns period-by-period cashflow projections including:
    - Principal and interest payments
    - Remaining balance
    - Cumulative metrics
    """
    try:
        request_counter.labels(endpoint="/api/v1/cashflow/calculate").inc()
        
        cashflow_service = CashflowService()
        result = cashflow_service.calculate_loan_cashflow(loan_data)
        
        logger.info(f"Calculated cashflow for loan with principal {loan_data.principal} and term {loan_data.term_months}")
        return result
    except Exception as e:
        logger.exception(f"Error calculating cashflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

@router.post("/calculate-batch", 
            response_model=List[CashflowForecastResponse], 
            summary="Calculate cashflows for multiple loans")
async def calculate_batch(
    batch_request: BatchForecastRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate cashflows for multiple loans in a single request
    
    This endpoint is optimized for batch processing of multiple loan scenarios.
    """
    try:
        request_counter.labels(endpoint="/api/v1/cashflow/calculate-batch").inc()
        
        cashflow_service = CashflowService()
        results = cashflow_service.calculate_batch(batch_request)
        
        logger.info(f"Calculated cashflows for {len(batch_request.loans)} loans")
        return results
    except Exception as e:
        logger.exception(f"Error calculating batch cashflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch calculation error: {str(e)}")

@router.post("/enhanced-metrics", 
            summary="Calculate enhanced financial metrics for a loan")
async def calculate_enhanced_metrics(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate enhanced financial metrics for a loan using AbsBox library
    
    This endpoint utilizes the advanced financial analytics capabilities of
    the AbsBox library to calculate metrics such as:
    - Net Present Value (NPV)
    - Internal Rate of Return (IRR)
    - Yield-to-Maturity
    - Macaulay Duration
    - Convexity
    - And more
    """
    try:
        request_counter.labels(endpoint="/api/v1/cashflow/enhanced-metrics").inc()
        
        service = AbsBoxServiceEnhanced()
        result = service.calculate_enhanced_metrics(request)
        
        if result.status == "error":
            logger.error(f"Error calculating enhanced metrics: {result.error}")
            return JSONResponse(
                status_code=422,
                content={"detail": result.error}
            )
        
        logger.info(f"Calculated enhanced metrics for loan with principal {request.principal}")
        return result
    except Exception as e:
        logger.exception(f"Error calculating enhanced metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

@router.post("/sensitivity", 
            summary="Perform sensitivity analysis on a loan")
async def sensitivity_analysis(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Perform sensitivity analysis on a loan
    
    This endpoint analyzes how financial metrics change when varying:
    - Interest rates
    - Prepayment rates
    - Default rates
    """
    try:
        request_counter.labels(endpoint="/api/v1/cashflow/sensitivity").inc()
        
        service = AbsBoxServiceEnhanced()
        result = service.perform_sensitivity_analysis(request)
        
        if result.status == "error":
            logger.error(f"Error performing sensitivity analysis: {result.error}")
            return JSONResponse(
                status_code=422,
                content={"detail": result.error}
            )
        
        logger.info(f"Performed sensitivity analysis for loan")
        return result
    except Exception as e:
        logger.exception(f"Error performing sensitivity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@router.post("/risk-metrics", 
            summary="Calculate risk metrics for a loan")
async def calculate_risk_metrics(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate risk metrics for a loan
    
    Returns key risk indicators such as:
    - Value at Risk (VaR)
    - Expected shortfall (ES)
    - Drawdown metrics
    - Stress test results
    """
    try:
        request_counter.labels(endpoint="/api/v1/cashflow/risk-metrics").inc()
        
        service = AbsBoxServiceEnhanced()
        result = service.calculate_risk_metrics(request)
        
        if result.status == "error":
            logger.error(f"Error calculating risk metrics: {result.error}")
            return JSONResponse(
                status_code=422,
                content={"detail": result.error}
            )
        
        logger.info(f"Calculated risk metrics for loan")
        return result
    except Exception as e:
        logger.exception(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk calculation error: {str(e)}")
