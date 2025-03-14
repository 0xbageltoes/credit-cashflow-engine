"""
CLO/CDO API Endpoints

Production-quality API endpoints for CLO/CDO structured product analysis operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import date

from app.models.asset_classes import (
    AssetClass, CLOCDO, CLOCDOTranche, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.asset_handlers.clo_cdo import CLOCDOHandler
from app.api.deps import get_clo_cdo_handler
from app.core.monitoring import CalculationTracker

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/clo-cdo",
    tags=["clo cdo"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze", response_model=AssetPoolAnalysisResponse, status_code=200)
async def analyze_clo_cdo_structure(
    request: AssetPoolAnalysisRequest,
    background_tasks: BackgroundTasks,
    handler: CLOCDOHandler = Depends(get_clo_cdo_handler)
):
    """
    Analyze a CLO/CDO structure.
    
    Performs detailed analysis of CLO/CDO structured products using AbsBox engine,
    providing cashflows, metrics, tranche analysis and optional stress tests.
    
    Args:
        request: Analysis request containing pool details and parameters
        background_tasks: FastAPI background tasks for async processing
        handler: The CLOCDOHandler dependency
        
    Returns:
        Detailed analysis result with metrics, cashflows, and optional stress tests
    """
    try:
        logger.info(f"Received request to analyze CLO/CDO structure: {request.pool.pool_name}")
        
        # Validate that pool contains CLO/CDO assets
        clo_cdo_assets = [
            asset for asset in request.pool.assets 
            if asset.asset_class == AssetClass.CLO_CDO
        ]
        
        if not clo_cdo_assets:
            raise HTTPException(
                status_code=400,
                detail="Pool must contain at least one CLO/CDO asset"
            )
        
        # Run analysis
        with CalculationTracker("clo_cdo_api_analyze"):
            result = handler.analyze_pool(request)
            
            if result.status == "error":
                logger.error(f"Error analyzing CLO/CDO structure: {result.error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {result.error}"
                )
            
            return result
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error analyzing CLO/CDO structure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post("/validate", response_model=Dict[str, Any])
async def validate_clo_cdo(
    clo_cdo: CLOCDO
):
    """
    Validate a CLO/CDO model.
    
    Checks if the CLO/CDO data is valid and provides data quality insights.
    
    Args:
        clo_cdo: CLO/CDO data to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {"valid": True, "issues": [], "tranche_issues": {}}
        
        # Validate CLO/CDO data
        if clo_cdo.collateral_pool_balance <= 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Collateral pool balance must be positive")
        
        if clo_cdo.collateral_pool_wac is not None and clo_cdo.collateral_pool_wac < 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Weighted average coupon cannot be negative")
        
        # Check if tranches exist
        if not clo_cdo.tranches or len(clo_cdo.tranches) == 0:
            validation_results["valid"] = False
            validation_results["issues"].append("CLO/CDO must have at least one tranche")
        
        # Calculate total tranche balance and validate individual tranches
        total_tranche_balance = 0
        
        for i, tranche in enumerate(clo_cdo.tranches):
            tranche_issues = []
            
            # Validate tranche data
            if tranche.balance <= 0:
                validation_results["valid"] = False
                tranche_issues.append("Tranche balance must be positive")
            
            if tranche.rate < 0:
                validation_results["valid"] = False
                tranche_issues.append("Interest rate cannot be negative")
            
            if tranche.attachment_point is not None and tranche.detachment_point is not None:
                if tranche.attachment_point < 0 or tranche.attachment_point > 1:
                    validation_results["valid"] = False
                    tranche_issues.append("Attachment point must be between 0 and 1")
                
                if tranche.detachment_point < 0 or tranche.detachment_point > 1:
                    validation_results["valid"] = False
                    tranche_issues.append("Detachment point must be between 0 and 1")
                
                if tranche.attachment_point >= tranche.detachment_point:
                    validation_results["valid"] = False
                    tranche_issues.append("Attachment point must be less than detachment point")
            
            # Add tranche balance to total
            total_tranche_balance += tranche.balance
            
            # Add any issues to the tranche-specific issues
            if tranche_issues:
                validation_results["tranche_issues"][f"tranche_{i}"] = tranche_issues
        
        # Check if total tranche balance exceeds collateral balance (overcollateralization)
        if total_tranche_balance > clo_cdo.collateral_pool_balance:
            validation_results["valid"] = False
            validation_results["issues"].append("Total tranche balance exceeds collateral pool balance")
        
        # Data quality checks
        if clo_cdo.collateral_pool_wac is None:
            validation_results["issues"].append("Weighted average coupon is recommended")
        
        if clo_cdo.collateral_pool_warf is None:
            validation_results["issues"].append("Weighted average rating factor is recommended")
            
        # Check for attachment/detachment point coverage
        attachment_points = [t.attachment_point for t in clo_cdo.tranches if t.attachment_point is not None]
        detachment_points = [t.detachment_point for t in clo_cdo.tranches if t.detachment_point is not None]
        
        if attachment_points and detachment_points:
            # Check for gaps in coverage
            sorted_points = sorted([(p, 'a') for p in attachment_points] + [(p, 'd') for p in detachment_points])
            current_level = 0
            
            for point, point_type in sorted_points:
                if point_type == 'a':  # Attachment point
                    if point > current_level and current_level > 0:
                        validation_results["issues"].append("Gap detected in tranche coverage")
                    current_level += 1
                else:  # Detachment point
                    current_level -= 1
            
            # Check if last detachment point is 1.0
            if max(detachment_points) < 1.0:
                validation_results["issues"].append("Highest detachment point does not reach 100%")
        
        return validation_results
    
    except Exception as e:
        logger.exception(f"Error validating CLO/CDO structure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )

@router.post("/tranche-analysis", response_model=Dict[str, Any])
async def analyze_tranches(
    clo_cdo: CLOCDO,
    stress_scenarios: Optional[bool] = Query(False, description="Include stress scenarios")
):
    """
    Analyze CLO/CDO tranches for risk and return.
    
    Provides detailed analysis of each tranche including expected returns,
    credit enhancement, and risk metrics.
    
    Args:
        clo_cdo: CLO/CDO structure to analyze
        stress_scenarios: Whether to include stress scenario analysis
        
    Returns:
        Detailed tranche analysis
    """
    try:
        # Validate basic CLO/CDO data
        if clo_cdo.collateral_pool_balance <= 0:
            raise HTTPException(
                status_code=400,
                detail="Collateral pool balance must be positive"
            )
        
        if not clo_cdo.tranches or len(clo_cdo.tranches) == 0:
            raise HTTPException(
                status_code=400,
                detail="CLO/CDO must have at least one tranche"
            )
        
        # Calculate total tranche balance
        total_tranche_balance = sum(t.balance for t in clo_cdo.tranches)
        
        # Calculate overcollateralization ratio
        oc_ratio = clo_cdo.collateral_pool_balance / total_tranche_balance if total_tranche_balance > 0 else None
        
        # Calculate weighted average cost of capital
        wacc = sum(t.balance * t.rate for t in clo_cdo.tranches) / total_tranche_balance if total_tranche_balance > 0 else None
        
        # Calculate excess spread
        excess_spread = (clo_cdo.collateral_pool_wac or 0) - wacc if wacc else None
        
        # Analyze each tranche
        tranche_analysis = []
        
        # Sort tranches by seniority
        sorted_tranches = sorted(clo_cdo.tranches, key=lambda t: t.seniority if t.seniority is not None else 999)
        
        for i, tranche in enumerate(sorted_tranches):
            # Calculate tranche metrics
            tranche_pct = tranche.balance / total_tranche_balance if total_tranche_balance > 0 else 0
            
            # Calculate credit enhancement
            credit_enhancement = tranche.attachment_point if tranche.attachment_point is not None else None
            
            # Calculate estimated loss rate based on position
            if tranche.attachment_point is not None and tranche.detachment_point is not None:
                # Simple loss rate calculation based on position
                # Assumes higher tranches have lower loss rates
                loss_rate = max(0, 0.10 - (tranche.attachment_point * 0.20))
            else:
                # Estimate loss rate based on seniority
                if tranche.seniority == 1:  # Most senior
                    loss_rate = 0.001  # 0.1%
                elif tranche.seniority == 2:
                    loss_rate = 0.005  # 0.5%
                elif tranche.seniority == 3:
                    loss_rate = 0.01  # 1%
                elif tranche.seniority == 4:
                    loss_rate = 0.03  # 3%
                elif tranche.seniority == 5:
                    loss_rate = 0.08  # 8%
                else:
                    loss_rate = 0.15  # 15%
            
            # Calculate expected return after losses
            expected_return = tranche.rate - loss_rate
            
            # Calculate return on equity (for equity tranche)
            roe = None
            if i == len(sorted_tranches) - 1:  # Last/equity tranche
                if excess_spread and excess_spread > 0:
                    roe = excess_spread * (total_tranche_balance / tranche.balance)
            
            tranche_info = {
                "name": tranche.name,
                "balance": tranche.balance,
                "rate": tranche.rate,
                "percent_of_deal": tranche_pct,
                "seniority": tranche.seniority,
                "attachment_point": tranche.attachment_point,
                "detachment_point": tranche.detachment_point,
                "credit_enhancement": credit_enhancement,
                "expected_loss_rate": loss_rate,
                "expected_return": expected_return
            }
            
            if roe is not None:
                tranche_info["return_on_equity"] = roe
            
            tranche_analysis.append(tranche_info)
        
        # Build response
        result = {
            "tranches": tranche_analysis,
            "deal_metrics": {
                "overcollateralization_ratio": oc_ratio,
                "weighted_average_cost": wacc,
                "excess_spread": excess_spread,
                "total_balance": total_tranche_balance
            }
        }
        
        # Add stress scenarios if requested
        if stress_scenarios:
            # Define stress scenarios
            scenarios = [
                {
                    "name": "high_defaults",
                    "description": "High default scenario (2x default rate)",
                    "default_multiplier": 2.0,
                    "impact": {}
                },
                {
                    "name": "low_recovery",
                    "description": "Low recovery scenario",
                    "recovery_multiplier": 0.7,
                    "impact": {}
                },
                {
                    "name": "severe_stress",
                    "description": "Severe stress scenario (3x defaults, low recovery)",
                    "default_multiplier": 3.0,
                    "recovery_multiplier": 0.5,
                    "impact": {}
                }
            ]
            
            # Calculate impact of each scenario on each tranche
            for scenario in scenarios:
                for i, tranche in enumerate(tranche_analysis):
                    default_multiplier = scenario.get("default_multiplier", 1.0)
                    
                    # Calculate stressed loss rate - increases impact on lower tranches more
                    base_loss_rate = tranche["expected_loss_rate"]
                    
                    # The effect is exponential on lower tranches
                    if tranche["attachment_point"] is not None and tranche["attachment_point"] < 0.2:
                        # Junior tranches experience higher losses in stress scenarios
                        stressed_loss_rate = base_loss_rate * (default_multiplier ** 2)
                    elif tranche["attachment_point"] is not None and tranche["attachment_point"] < 0.4:
                        # Mezzanine tranches
                        stressed_loss_rate = base_loss_rate * default_multiplier * 1.5
                    else:
                        # Senior tranches
                        stressed_loss_rate = base_loss_rate * default_multiplier
                    
                    # Cap loss rate
                    stressed_loss_rate = min(1.0, stressed_loss_rate)
                    
                    # Calculate return impact
                    return_impact = base_loss_rate - stressed_loss_rate
                    
                    scenario["impact"][tranche["name"]] = {
                        "stressed_loss_rate": stressed_loss_rate,
                        "return_impact": return_impact,
                        "expected_return": tranche["rate"] - stressed_loss_rate
                    }
            
            result["stress_scenarios"] = scenarios
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error analyzing CLO/CDO tranches: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Tranche analysis failed: {str(e)}"
        )

@router.get("/waterfall-scenarios", response_model=Dict[str, Any])
async def run_waterfall_scenarios(
    pool_id: str,
    handler: CLOCDOHandler = Depends(get_clo_cdo_handler)
):
    """
    Run multiple waterfall scenarios on a CLO/CDO structure.
    
    Analyzes a CLO/CDO structure under different economic scenarios
    to evaluate performance across market conditions.
    
    Args:
        pool_id: Unique identifier for the CLO/CDO structure
        handler: The CLOCDOHandler dependency
        
    Returns:
        Scenario analysis results for the structure
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
        logger.exception(f"Error running CLO/CDO waterfall scenarios: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run waterfall scenarios: {str(e)}"
        )

@router.get("/deal-comparison", response_model=Dict[str, Any])
async def compare_clo_cdo_deals(
    deal_ids: List[str] = Query(..., description="List of CLO/CDO deal IDs to compare"),
    metrics: List[str] = Query(["oc_ratio", "excess_spread", "weighted_average_life"], 
                             description="Metrics to include in comparison")
):
    """
    Compare multiple CLO/CDO deals.
    
    Provides a side-by-side comparison of key metrics for multiple
    CLO/CDO structures to facilitate investment decisions.
    
    Args:
        deal_ids: List of CLO/CDO deal IDs to compare
        metrics: List of metrics to include in the comparison
        
    Returns:
        Comparative analysis of selected deals
    """
    try:
        # This would typically query a database to retrieve the deals
        # For now, return an error as this is a placeholder
        raise HTTPException(
            status_code=501,
            detail="This endpoint is a placeholder and will be implemented when database integration is complete"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error comparing CLO/CDO deals: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare deals: {str(e)}"
        )
