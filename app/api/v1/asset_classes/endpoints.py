"""
Asset Classes API Endpoints

Production-ready implementation of API endpoints for analyzing different asset classes
in structured finance with comprehensive error handling, validation, and Redis caching.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uuid

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.monitoring import CalculationTracker
from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolMetrics, AssetPoolCashflow
)
from app.services.asset_class_service import AssetClassService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/asset-classes",
    tags=["asset-classes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
    }
)

# Initialize service
asset_class_service = AssetClassService()

@router.get(
    "/supported",
    summary="Get supported asset classes",
    description="Returns a list of asset classes supported by the system",
    response_model=List[AssetClass]
)
async def get_supported_asset_classes(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get supported asset classes
    
    Args:
        current_user: The authenticated user
        
    Returns:
        List of supported asset classes
    """
    try:
        supported_classes = asset_class_service.get_supported_asset_classes()
        return supported_classes
    except Exception as e:
        logger.exception(f"Error getting supported asset classes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving supported asset classes: {str(e)}"
        )

@router.post(
    "/analyze",
    summary="Analyze asset pool",
    description="Analyzes a pool of assets with specified parameters",
    response_model=AssetPoolAnalysisResponse
)
async def analyze_asset_pool(
    request: AssetPoolAnalysisRequest,
    use_cache: Optional[bool] = True,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze an asset pool
    
    Args:
        request: The analysis request details
        use_cache: Whether to use caching (default: True)
        current_user: The authenticated user
        
    Returns:
        Analysis results
    """
    request_id = str(uuid.uuid4())
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Request {request_id}: Asset pool analysis for user {user_id}, pool: {request.pool.pool_name}")
    
    try:
        with CalculationTracker(f"analyze_asset_pool_{request_id}"):
            # Perform analysis with comprehensive logging
            start_time = time.time()
            
            # Validate pool has assets
            if not request.pool.assets:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Asset pool must contain at least one asset"
                )
            
            # Ensure we have a valid analysis date
            if not request.analysis_date:
                logger.warning(f"Request {request_id}: No analysis date provided, using current date")
            
            # Perform the analysis
            result = await asset_class_service.analyze_asset_pool(
                request=request,
                user_id=user_id,
                use_cache=use_cache
            )
            
            logger.info(f"Request {request_id}: Analysis completed in {time.time() - start_time:.2f}s with status: {result.status}")
            
            # Check for errors
            if result.status == "error":
                logger.error(f"Request {request_id}: Analysis failed with error: {result.error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Analysis failed: {result.error}"
                )
            
            return result
            
    except ValidationError as e:
        logger.error(f"Request {request_id}: Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Request {request_id}: Unexpected error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during analysis: {str(e)}"
        )

@router.post(
    "/validate-pool",
    summary="Validate asset pool",
    description="Validates an asset pool for consistency and completeness",
    response_model=Dict[str, Any]
)
async def validate_asset_pool(
    pool: AssetPool,
    current_user: Dict = Depends(get_current_user)
):
    """
    Validate an asset pool
    
    Args:
        pool: The asset pool to validate
        current_user: The authenticated user
        
    Returns:
        Validation results
    """
    request_id = str(uuid.uuid4())
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Request {request_id}: Asset pool validation for user {user_id}, pool: {pool.pool_name}")
    
    try:
        # Perform basic validation
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for empty pool
        if not pool.assets:
            validation_results["valid"] = False
            validation_results["errors"].append("Asset pool must contain at least one asset")
            return validation_results
        
        # Check for consistency of asset classes
        asset_classes = set(asset.asset_class for asset in pool.assets)
        if len(asset_classes) > 1:
            validation_results["warnings"].append(
                f"Mixed asset classes in pool: {', '.join(str(ac) for ac in asset_classes)}"
            )
        
        # Check supported asset classes
        supported_classes = asset_class_service.get_supported_asset_classes()
        unsupported_classes = [ac for ac in asset_classes if ac not in supported_classes]
        if unsupported_classes:
            validation_results["warnings"].append(
                f"Pool contains unsupported asset classes: {', '.join(str(ac) for ac in unsupported_classes)}"
            )
        
        # Check for invalid values
        for i, asset in enumerate(pool.assets):
            # Check for negative values
            if asset.balance <= 0:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Asset {i}: Balance must be positive")
            
            if asset.term_months <= 0:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Asset {i}: Term must be positive")
            
            if asset.rate < 0:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Asset {i}: Rate cannot be negative")
            
            # Check remaining term vs. original term
            if asset.remaining_term_months is not None and asset.remaining_term_months > asset.term_months:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Asset {i}: Remaining term cannot exceed original term")
            
        logger.info(f"Request {request_id}: Validation completed with result: {validation_results['valid']}")
        
        return validation_results
            
    except ValidationError as e:
        logger.error(f"Request {request_id}: Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Request {request_id}: Unexpected error during validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during validation: {str(e)}"
        )
