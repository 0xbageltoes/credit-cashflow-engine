"""
Enhanced Analytics API Endpoints

This module provides endpoints for enhanced analytics using the AbsBox library.
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import time

from app.database.supabase import SupabaseClient
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.models.cashflow import (
    LoanData, 
    CashflowForecastRequest, 
    CashflowForecastResponse,
    ScenarioSaveRequest,
    ScenarioResponse,
    BatchForecastRequest
)
from app.models.analytics import (
    EnhancedAnalyticsRequest,
    EnhancedAnalyticsResult, 
    RiskMetrics, 
    SensitivityAnalysis,
    AnalyticsResponse
)
from app.models.structured_products import (
    StructuredDealRequest, 
    StructuredDealResponse,
    LoanPoolConfig
)
from app.api.deps import get_current_user, get_absbox_service, get_database_client

router = APIRouter()

@router.post("/enhanced-analytics/", response_model=AnalyticsResponse)
async def calculate_enhanced_analytics(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    db: SupabaseClient = Depends(get_database_client)
):
    """
    Calculate enhanced analytics metrics for a loan using the AbsBox library.
    
    This endpoint provides advanced financial metrics beyond basic cashflow forecasting,
    including duration, convexity, yield metrics, and other risk measures.
    """
    start_time = time.time()
    user_id = current_user["id"]
    
    try:
        # First check if we have this result in cache
        cache_key = f"enhanced_analytics:{user_id}:{request.loan_id or 'new_loan'}:{hash(str(request.dict()))}"
        cached_result = absbox_service.get_from_cache(cache_key)
        
        if cached_result:
            execution_time = time.time() - start_time
            return AnalyticsResponse(
                status="success",
                execution_time=execution_time,
                metrics=cached_result,
                cache_hit=True
            )
            
        # Use AbsBox to generate enhanced analytics
        metrics = absbox_service.calculate_enhanced_metrics(request)
        
        # Save to database if this is an existing loan
        if request.loan_id:
            forecast_id = db.save_enhanced_analytics(user_id, request.loan_id, metrics)
        
        # Save to cache
        absbox_service.save_to_cache(cache_key, metrics)
        
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="success",
            execution_time=execution_time,
            metrics=metrics,
            cache_hit=False
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="error",
            execution_time=execution_time,
            error=str(e)
        )

@router.post("/risk-metrics/", response_model=AnalyticsResponse)
async def calculate_risk_metrics(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    db: SupabaseClient = Depends(get_database_client)
):
    """
    Calculate risk metrics for a loan using the AbsBox library.
    
    This endpoint provides risk measures including Value at Risk (VaR),
    Expected Shortfall, and stress test results.
    """
    start_time = time.time()
    user_id = current_user["id"]
    
    try:
        # First check if we have this result in cache
        cache_key = f"risk_metrics:{user_id}:{request.loan_id or 'new_loan'}:{hash(str(request.dict()))}"
        cached_result = absbox_service.get_from_cache(cache_key)
        
        if cached_result:
            execution_time = time.time() - start_time
            return AnalyticsResponse(
                status="success",
                execution_time=execution_time,
                metrics=cached_result,
                cache_hit=True
            )
            
        # Use AbsBox to generate risk metrics
        risk_metrics = absbox_service.calculate_risk_metrics(request)
        
        # Save to cache
        absbox_service.save_to_cache(cache_key, risk_metrics)
        
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="success",
            execution_time=execution_time,
            metrics=risk_metrics,
            cache_hit=False
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="error",
            execution_time=execution_time,
            error=str(e)
        )

@router.post("/sensitivity-analysis/", response_model=AnalyticsResponse)
async def calculate_sensitivity(
    request: EnhancedAnalyticsRequest,
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service)
):
    """
    Calculate sensitivity analysis for a loan using the AbsBox library.
    
    This endpoint provides sensitivity measures for various risk factors including
    interest rates, prepayment speeds, and default rates.
    """
    start_time = time.time()
    user_id = current_user["id"]
    
    try:
        # First check if we have this result in cache
        cache_key = f"sensitivity:{user_id}:{request.loan_id or 'new_loan'}:{hash(str(request.dict()))}"
        cached_result = absbox_service.get_from_cache(cache_key)
        
        if cached_result:
            execution_time = time.time() - start_time
            return AnalyticsResponse(
                status="success",
                execution_time=execution_time,
                metrics=cached_result,
                cache_hit=True
            )
            
        # Use AbsBox to generate sensitivity analysis
        sensitivity = absbox_service.calculate_sensitivity(request)
        
        # Save to cache
        absbox_service.save_to_cache(cache_key, sensitivity)
        
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="success",
            execution_time=execution_time,
            metrics=sensitivity,
            cache_hit=False
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return AnalyticsResponse(
            status="error",
            execution_time=execution_time,
            error=str(e)
        )

@router.post("/structured-deal/analyze/", response_model=StructuredDealResponse)
async def analyze_structured_deal(
    deal: StructuredDealRequest,
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    db: SupabaseClient = Depends(get_database_client)
):
    """
    Analyze a structured finance deal using the AbsBox library.
    
    This endpoint provides comprehensive analysis for structured deals including
    bond cashflows, pool performance, and key metrics.
    """
    user_id = current_user["id"]
    
    try:
        # Use AbsBox to analyze the structured deal
        result = absbox_service.analyze_structured_deal(deal)
        
        # Save to database
        deal_id = db.save_structured_deal_analysis(user_id, deal, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing structured deal: {str(e)}")

@router.get("/absbox/health/", response_model=Dict[str, Any])
async def check_absbox_health(
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service)
):
    """
    Check the health of the AbsBox service.
    
    This endpoint provides diagnostic information about the AbsBox service,
    including cache status and engine connectivity.
    """
    try:
        health_info = absbox_service.check_health()
        return {
            "status": "ok",
            "version": health_info.get("version", "unknown"),
            "cache_status": health_info.get("cache_status", "unknown"),
            "memory_usage": health_info.get("memory_usage", "unknown"),
            "uptime": health_info.get("uptime", "unknown")
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/absbox/cache/clear/", response_model=Dict[str, Any])
async def clear_absbox_cache(
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    pattern: Optional[str] = None
):
    """
    Clear the AbsBox service cache.
    
    This endpoint clears cached results from the AbsBox service.
    An optional pattern can be provided to selectively clear cache entries.
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required to clear cache"
        )
    
    try:
        cleared_count = absbox_service.clear_cache(pattern)
        return {
            "status": "ok",
            "cleared_keys": cleared_count,
            "pattern": pattern or "*"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/absbox/metrics/", response_model=Dict[str, Any])
async def get_absbox_metrics(
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service)
):
    """
    Get usage metrics for the AbsBox service.
    
    This endpoint provides performance and usage metrics for the AbsBox service.
    """
    try:
        metrics = absbox_service.get_metrics()
        return {
            "status": "ok",
            "metrics": metrics
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/samples/structured-deal/", response_model=StructuredDealRequest)
async def create_sample_deal(
    current_user: Dict = Depends(get_current_user),
    complexity: str = Query("medium", regex="^(simple|medium|complex)$")
):
    """
    Create a sample structured deal for testing.
    
    This endpoint generates a sample structured deal configuration with varying complexity.
    Options: simple, medium, complex
    """
    try:
        absbox_service = AbsBoxServiceEnhanced()
        sample_deal = absbox_service.create_sample_deal(complexity)
        return sample_deal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating sample deal: {str(e)}")

@router.post("/batch-analytics/", response_model=List[AnalyticsResponse])
async def batch_analytics(
    requests: List[EnhancedAnalyticsRequest],
    current_user: Dict = Depends(get_current_user),
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    db: SupabaseClient = Depends(get_database_client)
):
    """
    Calculate enhanced analytics for multiple loans in batch.
    
    This endpoint allows efficient processing of multiple loan analytics requests
    in a single API call.
    """
    user_id = current_user["id"]
    results = []
    
    for req in requests:
        start_time = time.time()
        try:
            # First check if we have this result in cache
            cache_key = f"enhanced_analytics:{user_id}:{req.loan_id or 'new_loan'}:{hash(str(req.dict()))}"
            cached_result = absbox_service.get_from_cache(cache_key)
            
            if cached_result:
                execution_time = time.time() - start_time
                results.append(AnalyticsResponse(
                    status="success",
                    execution_time=execution_time,
                    metrics=cached_result,
                    cache_hit=True
                ))
                continue
                
            # Use AbsBox to generate enhanced analytics
            metrics = absbox_service.calculate_enhanced_metrics(req)
            
            # Save to database if this is an existing loan
            if req.loan_id:
                forecast_id = db.save_enhanced_analytics(user_id, req.loan_id, metrics)
            
            # Save to cache
            absbox_service.save_to_cache(cache_key, metrics)
            
            execution_time = time.time() - start_time
            results.append(AnalyticsResponse(
                status="success",
                execution_time=execution_time,
                metrics=metrics,
                cache_hit=False
            ))
        except Exception as e:
            execution_time = time.time() - start_time
            results.append(AnalyticsResponse(
                status="error",
                execution_time=execution_time,
                error=str(e)
            ))
    
    return results
