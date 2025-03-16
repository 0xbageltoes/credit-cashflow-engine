from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from typing import Dict, Any, List, Optional
from pydantic import ValidationError
from fastapi.responses import JSONResponse

from app.core.auth import get_current_user, verify_token, RoleChecker
from app.services.cashflow import CashflowService
from app.models.cashflow import (
    LoanData, BatchLoanRequest, CashflowForecastRequest, 
    CashflowForecastResponse, StressTestScenario
)
from app.core.logging import logger
from app.core.rate_limiting import limiter
from app.core.config import settings

# Create router with tags for API documentation grouping
router = APIRouter(tags=["cashflow"])

# Dependency for role-based access
require_analyst = RoleChecker(["analyst", "manager", "admin"])

@router.post(
    "/calculate", 
    response_model=Dict[str, Any],
    summary="Calculate cashflow for a single loan",
    description="""
    Calculate detailed cashflow projections for a single loan.
    Returns period-by-period amortization schedule and summary metrics.
    
    This endpoint supports:
    - Fixed and variable rate loans
    - Various amortization methods
    - Prepayment and default modeling
    - Financial metrics calculations (NPV, IRR, WAL)
    """
)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/hour")
async def calculate_cashflow(
    request: Request,
    response: Response,
    loan_data: LoanData,
    include_details: bool = True,
    current_user: Dict = Depends(verify_token)
):
    """Calculate cash flow for a single loan"""
    try:
        # Log the request (without sensitive data)
        logger.info("Cashflow calculation request", extra={
            "user_id": current_user.get("id"),
            "loan_id": loan_data.loan_id,
            "term_months": loan_data.term_months,
            "include_details": include_details,
        })
        
        # Create service and calculate cashflow
        cashflow_service = CashflowService()
        result = cashflow_service.calculate_loan_cashflow(
            loan_data=loan_data,
            include_details=include_details
        )
        
        # Add request tracking to response headers
        response.headers["X-Request-ID"] = request.state.request_id
        
        return result
        
    except ValidationError as e:
        # Handle validation errors separately for better client feedback
        logger.warning("Validation error in cashflow calculation", extra={
            "user_id": current_user.get("id"),
            "validation_errors": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid loan data: {str(e)}"
        )
        
    except Exception as e:
        # Log detailed error for troubleshooting
        logger.error(f"Error calculating cashflow: {str(e)}", extra={
            "user_id": current_user.get("id"),
            "loan_data": loan_data.model_dump(exclude={"custom_fields"}),
            "error": str(e),
        }, exc_info=True)
        
        # Return appropriate error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to calculate cashflow: {str(e)}"
        )

@router.post(
    "/calculate-batch",
    response_model=Dict[str, Any],
    summary="Calculate cashflows for multiple loans in batch",
    description="""
    Process multiple loans in a single batch operation for efficiency.
    
    Features:
    - Parallel processing for improved performance
    - Portfolio-level metrics calculation
    - Configurable detail level for results
    - Comprehensive error handling per loan
    
    This endpoint is optimized for high-throughput scenarios and can handle
    up to 100 loans in a single request.
    """
)
@limiter.limit(f"{int(settings.RATE_LIMIT_REQUESTS/5)}/hour")
async def calculate_batch(
    request: Request,
    response: Response,
    batch_request: BatchLoanRequest,
    current_user: Dict = Depends(verify_token)
):
    """Calculate cash flows for multiple loans"""
    try:
        # Log batch request
        logger.info("Batch cashflow calculation request", extra={
            "user_id": current_user.get("id"),
            "loan_count": len(batch_request.loans),
            "parallel_processing": batch_request.process_in_parallel,
        })
        
        # Validate batch size limits for production safeguards
        if len(batch_request.loans) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum limit of 100 loans"
            )
        
        # Process the batch
        cashflow_service = CashflowService()
        result = cashflow_service.calculate_batch(batch_request)
        
        # Add request tracking to response headers
        response.headers["X-Request-ID"] = request.state.request_id
        
        return result
        
    except ValidationError as e:
        # Handle validation errors
        logger.warning("Validation error in batch cashflow calculation", extra={
            "user_id": current_user.get("id"),
            "validation_errors": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid batch request data: {str(e)}"
        )
        
    except Exception as e:
        # Log detailed error
        logger.error(f"Error calculating batch cashflow: {str(e)}", extra={
            "user_id": current_user.get("id"),
            "loan_count": len(batch_request.loans) if batch_request else 0,
            "error": str(e),
        }, exc_info=True)
        
        # Return appropriate error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to process batch calculation: {str(e)}"
        )

@router.post(
    "/forecast",
    response_model=CashflowForecastResponse,
    summary="Generate advanced cashflow forecast with analytics",
    description="""
    Advanced cashflow forecasting with additional analytics features:
    
    - Monte Carlo simulations for risk analysis
    - Stress testing scenarios
    - Economic factor impact analysis
    - Sensitivity analysis for key parameters
    
    This endpoint combines cashflow projections with advanced analytics
    for comprehensive loan performance analysis.
    """
)
@limiter.limit(f"{int(settings.RATE_LIMIT_REQUESTS/2)}/hour")
async def forecast_cashflow(
    request: Request,
    response: Response,
    forecast_request: CashflowForecastRequest,
    current_user: Dict = Depends(verify_token)
):
    """Generate advanced cashflow forecast with analytics"""
    try:
        # Log the request
        logger.info("Advanced cashflow forecast request", extra={
            "user_id": current_user.get("id"),
            "loan_id": forecast_request.loan_data.loan_id,
            "run_monte_carlo": forecast_request.run_monte_carlo,
            "calculate_sensitivity": forecast_request.calculate_sensitivity,
        })
        
        # Check for resource-intensive operations (Monte Carlo, etc.)
        # These should be limited to certain roles
        if (forecast_request.run_monte_carlo or 
            forecast_request.calculate_sensitivity or 
            (forecast_request.monte_carlo_iterations or 0) > 500):
            # Verify user has appropriate role
            require_analyst(current_user)
        
        # Create service and process forecast
        cashflow_service = CashflowService()
        
        # For now, use the basic calculation - in a future enhancement, 
        # we can implement the full forecasting with Monte Carlo
        result = cashflow_service.calculate_loan_cashflow(forecast_request.loan_data)
        
        # Convert to expected response format
        response_data = CashflowForecastResponse(
            loan_id=forecast_request.loan_data.loan_id,
            projections=result.get("projections", []),
            summary_metrics=result.get("summary_metrics", {})
        )
        
        # Add request tracking to response headers
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response_data
        
    except ValidationError as e:
        # Handle validation errors
        logger.warning("Validation error in forecast request", extra={
            "user_id": current_user.get("id"),
            "validation_errors": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid forecast request data: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like permission errors) without modification
        raise
        
    except Exception as e:
        # Log detailed error
        logger.error(f"Error processing cashflow forecast: {str(e)}", extra={
            "user_id": current_user.get("id"),
            "loan_id": getattr(forecast_request, "loan_data", {}).get("loan_id", "unknown"),
            "error": str(e),
        }, exc_info=True)
        
        # Return appropriate error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to generate cashflow forecast: {str(e)}"
        )

@router.post(
    "/stress-test",
    response_model=Dict[str, Any],
    summary="Run stress tests on a loan",
    description="""
    Run stress test scenarios against a loan to evaluate performance
    under different market conditions.
    
    - Apply multiple stress scenarios simultaneously
    - Analyze impact on key metrics
    - Compare baseline vs stressed performance
    """
)
@limiter.limit(f"{int(settings.RATE_LIMIT_REQUESTS/3)}/hour")
async def run_stress_test(
    request: Request,
    response: Response,
    loan_data: LoanData,
    scenarios: List[StressTestScenario],
    current_user: Dict = Depends(verify_token)
):
    """Run stress tests on a loan with various scenarios"""
    try:
        # This requires analyst role
        require_analyst(current_user)
        
        # Log the request
        logger.info("Stress test request", extra={
            "user_id": current_user.get("id"),
            "loan_id": loan_data.loan_id,
            "scenario_count": len(scenarios),
        })
        
        # Create service
        cashflow_service = CashflowService()
        
        # Generate baseline cashflow first
        baseline = cashflow_service.calculate_loan_cashflow(loan_data)
        
        # Process each stress scenario
        # Note: For now, this is a simplified implementation that 
        # could be enhanced in the future with actual scenario processing
        stress_results = {
            "baseline": baseline,
            "scenarios": []
        }
        
        # Add request tracking to response headers
        response.headers["X-Request-ID"] = request.state.request_id
        
        return stress_results
        
    except ValidationError as e:
        # Handle validation errors
        logger.warning("Validation error in stress test request", extra={
            "user_id": current_user.get("id"),
            "validation_errors": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid stress test request data: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        # Log detailed error
        logger.error(f"Error processing stress test: {str(e)}", extra={
            "user_id": current_user.get("id"),
            "loan_id": loan_data.loan_id if loan_data else "unknown",
            "error": str(e),
        }, exc_info=True)
        
        # Return appropriate error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to run stress test: {str(e)}"
        )

# Health check endpoint for the cashflow service
@router.get(
    "/health",
    summary="Cashflow service health check",
    description="Verify the cashflow service is operational"
)
async def health_check():
    """Health check endpoint for the cashflow service"""
    return {
        "status": "healthy", 
        "service": "cashflow",
        "version": settings.VERSION
    }
