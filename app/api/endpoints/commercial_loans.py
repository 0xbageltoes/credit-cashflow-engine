"""
Commercial Loans API Endpoints

Production-quality API endpoints for commercial loan analysis operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import logging
from datetime import date

from app.models.asset_classes import (
    AssetClass, CommercialLoan, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.asset_handlers.commercial_loan import CommercialLoanHandler
from app.api.deps import get_commercial_loan_handler
from app.core.monitoring import CalculationTracker

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/commercial-loans",
    tags=["commercial loans"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze", response_model=AssetPoolAnalysisResponse, status_code=200)
async def analyze_commercial_loan_pool(
    request: AssetPoolAnalysisRequest,
    background_tasks: BackgroundTasks,
    handler: CommercialLoanHandler = Depends(get_commercial_loan_handler)
):
    """
    Analyze a commercial loan pool.
    
    Performs detailed analysis of commercial loan assets using AbsBox engine,
    providing cashflows, metrics and optional stress tests.
    
    Args:
        request: Analysis request containing pool details and parameters
        background_tasks: FastAPI background tasks for async processing
        handler: The CommercialLoanHandler dependency
        
    Returns:
        Detailed analysis result with metrics, cashflows, and optional stress tests
    """
    try:
        logger.info(f"Received request to analyze commercial loan pool: {request.pool.pool_name}")
        
        # Validate that pool contains commercial loans
        commercial_loans = [
            asset for asset in request.pool.assets 
            if asset.asset_class == AssetClass.COMMERCIAL_LOAN
        ]
        
        if not commercial_loans:
            raise HTTPException(
                status_code=400,
                detail="Pool must contain at least one commercial loan asset"
            )
        
        # For performance, we can run the analysis in the background for large pools
        if len(commercial_loans) > 100 and not request.include_cashflows:
            # This would be implemented for async processing
            # background_tasks.add_task(handler.analyze_pool, request)
            # return {"status": "processing", "message": "Analysis started in background"}
            pass
        
        # Run analysis directly
        with CalculationTracker("commercial_loan_api_analyze"):
            result = handler.analyze_pool(request)
            
            if result.status == "error":
                logger.error(f"Error analyzing commercial loan pool: {result.error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {result.error}"
                )
            
            return result
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error analyzing commercial loan pool: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/property-types", response_model=List[str])
async def get_property_types():
    """
    Get the list of valid commercial property types.
    
    Returns a list of available property types for commercial loans
    that can be used when creating or updating loans.
    
    Returns:
        List of property type values
    """
    try:
        # This would normally come from a model enum or database
        property_types = [
            "office", 
            "retail", 
            "industrial", 
            "multifamily", 
            "hotel", 
            "healthcare", 
            "mixed_use", 
            "self_storage", 
            "other"
        ]
        
        return property_types
    except Exception as e:
        logger.exception(f"Error retrieving property types: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve property types: {str(e)}"
        )

@router.post("/validate", response_model=dict)
async def validate_commercial_loan(
    loan: CommercialLoan
):
    """
    Validate a commercial loan model.
    
    Checks if the commercial loan data is valid and provides data quality insights.
    
    Args:
        loan: Commercial loan data to validate
        
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
        
        if loan.ltv_ratio and (loan.ltv_ratio <= 0 or loan.ltv_ratio > 1.5):
            validation_results["valid"] = False
            validation_results["issues"].append("LTV ratio should be between 0 and 1.5")
        
        if loan.dscr and loan.dscr <= 0:
            validation_results["valid"] = False
            validation_results["issues"].append("DSCR must be positive")
        
        # Data quality checks
        if not loan.property_type:
            validation_results["issues"].append("Property type is recommended")
        
        if not loan.ltv_ratio:
            validation_results["issues"].append("LTV ratio is recommended")
            
        if not loan.dscr:
            validation_results["issues"].append("DSCR is recommended")
        
        return validation_results
    
    except Exception as e:
        logger.exception(f"Error validating commercial loan: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )

@router.get("/metrics/{pool_id}", response_model=AssetPoolMetrics)
async def get_commercial_loan_pool_metrics(
    pool_id: str,
    analysis_date: Optional[date] = Query(None),
    discount_rate: Optional[float] = Query(0.06),
    handler: CommercialLoanHandler = Depends(get_commercial_loan_handler)
):
    """
    Get metrics for a specific commercial loan pool.
    
    Retrieves stored or calculates metrics for a commercial loan pool by ID.
    
    Args:
        pool_id: Unique identifier for the loan pool
        analysis_date: Date to use for analysis (defaults to current date)
        discount_rate: Rate for discounting cashflows (defaults to 6%)
        handler: The CommercialLoanHandler dependency
        
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
        logger.exception(f"Error retrieving commercial loan pool metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )
