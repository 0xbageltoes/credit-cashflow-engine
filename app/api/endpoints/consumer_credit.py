"""
Consumer Credit API Endpoints

Production-quality API endpoints for consumer credit analysis operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import date

from app.models.asset_classes import (
    AssetClass, ConsumerCredit, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.asset_handlers.consumer_credit import ConsumerCreditHandler
from app.api.deps import get_consumer_credit_handler
from app.core.monitoring import CalculationTracker

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/consumer-credit",
    tags=["consumer credit"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze", response_model=AssetPoolAnalysisResponse, status_code=200)
async def analyze_consumer_credit_pool(
    request: AssetPoolAnalysisRequest,
    background_tasks: BackgroundTasks,
    handler: ConsumerCreditHandler = Depends(get_consumer_credit_handler)
):
    """
    Analyze a consumer credit pool.
    
    Performs detailed analysis of consumer credit assets using AbsBox engine,
    providing cashflows, metrics and optional stress tests.
    
    Args:
        request: Analysis request containing pool details and parameters
        background_tasks: FastAPI background tasks for async processing
        handler: The ConsumerCreditHandler dependency
        
    Returns:
        Detailed analysis result with metrics, cashflows, and optional stress tests
    """
    try:
        logger.info(f"Received request to analyze consumer credit pool: {request.pool.pool_name}")
        
        # Validate that pool contains consumer credit assets
        consumer_assets = [
            asset for asset in request.pool.assets 
            if asset.asset_class == AssetClass.CONSUMER_CREDIT
        ]
        
        if not consumer_assets:
            raise HTTPException(
                status_code=400,
                detail="Pool must contain at least one consumer credit asset"
            )
        
        # For performance, we can run the analysis in the background for large pools
        if len(consumer_assets) > 100 and not request.include_cashflows:
            # This would be implemented for async processing
            # background_tasks.add_task(handler.analyze_pool, request)
            # return {"status": "processing", "message": "Analysis started in background"}
            pass
        
        # Run analysis directly
        with CalculationTracker("consumer_credit_api_analyze"):
            result = handler.analyze_pool(request)
            
            if result.status == "error":
                logger.error(f"Error analyzing consumer credit pool: {result.error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {result.error}"
                )
            
            return result
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error analyzing consumer credit pool: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/loan-types", response_model=List[str])
async def get_loan_types():
    """
    Get the list of valid consumer credit loan types.
    
    Returns a list of available consumer credit loan types
    that can be used when creating or updating loans.
    
    Returns:
        List of loan type values
    """
    try:
        # This would normally come from a model enum or database
        loan_types = [
            "credit_card", 
            "auto_loan", 
            "personal_loan", 
            "student_loan", 
            "installment_loan", 
            "medical_debt", 
            "other"
        ]
        
        return loan_types
    except Exception as e:
        logger.exception(f"Error retrieving consumer loan types: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve consumer loan types: {str(e)}"
        )

@router.post("/validate", response_model=Dict[str, Any])
async def validate_consumer_credit(
    loan: ConsumerCredit
):
    """
    Validate a consumer credit model.
    
    Checks if the consumer credit data is valid and provides data quality insights.
    
    Args:
        loan: Consumer credit data to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {"valid": True, "issues": []}
        
        # Validate loan data
        if loan.balance <= 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Loan balance must be positive")
        
        if loan.rate < 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Interest rate cannot be negative")
        
        if loan.term_months <= 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Term months must be positive")
        
        if loan.remaining_term_months > loan.term_months:
            validation_results["valid"] = False
            validation_results["issues"].append("Remaining term cannot exceed original term")
        
        if loan.fico_score and (loan.fico_score < 300 or loan.fico_score > 850):
            validation_results["valid"] = False
            validation_results["issues"].append("FICO score should be between 300 and 850")
        
        # Data quality checks
        if not loan.loan_type:
            validation_results["issues"].append("Loan type is recommended")
        
        if not loan.fico_score:
            validation_results["issues"].append("FICO score is recommended")
            
        if not loan.dti_ratio:
            validation_results["issues"].append("DTI ratio is recommended")
        
        return validation_results
    
    except Exception as e:
        logger.exception(f"Error validating consumer credit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )

@router.get("/risk-assessment", response_model=Dict[str, Any])
async def get_consumer_credit_risk_assessment(
    fico_score: int = Query(..., description="FICO Credit Score"),
    dti_ratio: float = Query(..., description="Debt-to-Income Ratio"),
    delinquency_history: int = Query(0, description="Number of past delinquencies"),
    loan_type: str = Query(..., description="Type of consumer loan")
):
    """
    Get risk assessment for consumer credit based on key metrics.
    
    Provides a risk score and evaluation for consumer credit based on
    FICO score, DTI ratio, delinquency history, and loan type.
    
    Args:
        fico_score: FICO credit score
        dti_ratio: Debt-to-income ratio
        delinquency_history: Number of past delinquencies
        loan_type: Type of consumer loan
        
    Returns:
        Risk assessment details including score and category
    """
    try:
        # Validate input
        if fico_score < 300 or fico_score > 850:
            raise HTTPException(
                status_code=400,
                detail="FICO score must be between 300 and 850"
            )
        
        if dti_ratio < 0 or dti_ratio > 1:
            raise HTTPException(
                status_code=400,
                detail="DTI ratio must be between 0 and 1"
            )
        
        if delinquency_history < 0:
            raise HTTPException(
                status_code=400,
                detail="Delinquency history cannot be negative"
            )
        
        # Simple risk scoring algorithm
        # FICO score component (0-100)
        fico_component = min(100, max(0, (fico_score - 300) / 5.5))
        
        # DTI component (0-100, lower is better)
        dti_component = min(100, max(0, 100 - (dti_ratio * 100)))
        
        # Delinquency component (0-100, lower is better)
        delinquency_component = min(100, max(0, 100 - (delinquency_history * 20)))
        
        # Adjust weightings based on loan type
        if loan_type == "credit_card":
            weights = {"fico": 0.5, "dti": 0.3, "delinquency": 0.2}
        elif loan_type == "auto_loan":
            weights = {"fico": 0.4, "dti": 0.4, "delinquency": 0.2}
        elif loan_type == "personal_loan":
            weights = {"fico": 0.45, "dti": 0.35, "delinquency": 0.2}
        elif loan_type == "student_loan":
            weights = {"fico": 0.35, "dti": 0.45, "delinquency": 0.2}
        else:
            weights = {"fico": 0.4, "dti": 0.4, "delinquency": 0.2}
        
        # Calculate weighted risk score (0-100, higher is better)
        risk_score = (
            fico_component * weights["fico"] +
            dti_component * weights["dti"] +
            delinquency_component * weights["delinquency"]
        )
        
        # Determine risk category
        if risk_score >= 80:
            risk_category = "Very Low Risk"
            default_probability = 0.005
        elif risk_score >= 70:
            risk_category = "Low Risk"
            default_probability = 0.02
        elif risk_score >= 60:
            risk_category = "Moderate Risk"
            default_probability = 0.05
        elif risk_score >= 50:
            risk_category = "Medium Risk"
            default_probability = 0.10
        elif risk_score >= 40:
            risk_category = "Elevated Risk"
            default_probability = 0.15
        elif risk_score >= 30:
            risk_category = "High Risk"
            default_probability = 0.25
        else:
            risk_category = "Very High Risk"
            default_probability = 0.40
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
            "default_probability": default_probability,
            "components": {
                "fico_score": {
                    "value": fico_score,
                    "component_score": round(fico_component, 2),
                    "weight": weights["fico"]
                },
                "dti_ratio": {
                    "value": dti_ratio,
                    "component_score": round(dti_component, 2),
                    "weight": weights["dti"]
                },
                "delinquency_history": {
                    "value": delinquency_history,
                    "component_score": round(delinquency_component, 2),
                    "weight": weights["delinquency"]
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error calculating consumer credit risk assessment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate risk assessment: {str(e)}"
        )

@router.get("/metrics/{pool_id}", response_model=AssetPoolMetrics)
async def get_consumer_credit_pool_metrics(
    pool_id: str,
    analysis_date: Optional[date] = Query(None),
    discount_rate: Optional[float] = Query(0.08),
    handler: ConsumerCreditHandler = Depends(get_consumer_credit_handler)
):
    """
    Get metrics for a specific consumer credit pool.
    
    Retrieves stored or calculates metrics for a consumer credit pool by ID.
    
    Args:
        pool_id: Unique identifier for the loan pool
        analysis_date: Date to use for analysis (defaults to current date)
        discount_rate: Rate for discounting cashflows (defaults to 8%)
        handler: The ConsumerCreditHandler dependency
        
    Returns:
        Pool metrics including NPV, duration, and weighted average life
    """
    try:
        # This would typically query a database to retrieve the pool by ID
        # For now, return an error as this is a placeholder
        raise HTTPException(
            status_code=501,
            detail="This endpoint is a placeholder and will be implemented when database integration is complete"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving consumer credit pool metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )
