"""
Optimized Monte Carlo simulation API endpoints

This module defines FastAPI endpoints for running memory-efficient
Monte Carlo simulations with proper correlation modeling and performance
optimization.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.users import User
from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult
)
from app.services.monte_carlo_optimized import OptimizedMonteCarloService
from app.worker.monte_carlo_tasks import run_optimized_monte_carlo_simulation
from app.core.logging import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

class EconomicFactorsModel(BaseModel):
    """Model for economic factors"""
    market_rate: float = Field(0.04, description="Current market interest rate")
    inflation_rate: float = Field(0.02, description="Current inflation rate")
    unemployment_rate: float = Field(0.042, description="Current unemployment rate")
    gdp_growth: float = Field(0.025, description="Current GDP growth rate")
    house_price_index_growth: float = Field(0.03, description="Current house price index growth rate")
    
    class Config:
        schema_extra = {
            "example": {
                "market_rate": 0.04,
                "inflation_rate": 0.02,
                "unemployment_rate": 0.042,
                "gdp_growth": 0.025,
                "house_price_index_growth": 0.03
            }
        }

class VolatilitiesModel(BaseModel):
    """Model for factor volatilities"""
    market_rate: float = Field(0.1, description="Volatility of market interest rate")
    inflation_rate: float = Field(0.05, description="Volatility of inflation rate")
    unemployment_rate: float = Field(0.15, description="Volatility of unemployment rate")
    gdp_growth: float = Field(0.2, description="Volatility of GDP growth rate")
    house_price_index_growth: float = Field(0.12, description="Volatility of house price index growth rate")
    
    class Config:
        schema_extra = {
            "example": {
                "market_rate": 0.1,
                "inflation_rate": 0.05,
                "unemployment_rate": 0.15,
                "gdp_growth": 0.2,
                "house_price_index_growth": 0.12
            }
        }

class OptimizedSimulationRequest(BaseModel):
    """Request model for optimized Monte Carlo simulation"""
    name: str = Field(..., description="Name of the simulation")
    description: Optional[str] = Field(None, description="Description of the simulation")
    loan_data: Dict[str, Any] = Field(..., description="Loan data dictionary")
    base_economic_factors: EconomicFactorsModel = Field(..., description="Base economic factors")
    volatilities: Optional[VolatilitiesModel] = Field(None, description="Volatilities for economic factors")
    correlation_matrix: Optional[List[List[float]]] = Field(None, description="Correlation matrix for economic factors")
    num_scenarios: int = Field(1000, description="Number of scenarios to run", gt=0, le=100000)
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Optimized Mortgage Portfolio Simulation",
                "description": "Monte Carlo simulation for mortgage portfolio with correlation modeling",
                "loan_data": {
                    "principal": 500000,
                    "interest_rate": 0.045,
                    "term_months": 360,
                    "origination_date": "2024-01-01",
                    "loan_type": "mortgage",
                    "credit_score": 720,
                    "ltv_ratio": 0.8,
                    "dti_ratio": 0.36
                },
                "base_economic_factors": {
                    "market_rate": 0.04,
                    "inflation_rate": 0.02,
                    "unemployment_rate": 0.042,
                    "gdp_growth": 0.025,
                    "house_price_index_growth": 0.03
                },
                "num_scenarios": 1000
            }
        }

@router.post("/monte-carlo/optimized", response_model=Dict[str, Any])
async def run_optimized_simulation(
    request: OptimizedSimulationRequest,
    current_user: User = Depends(get_current_user),
    run_async: bool = Query(True, description="Run the simulation asynchronously using Celery")
):
    """
    Run an optimized Monte Carlo simulation with proper correlation modeling
    
    This endpoint provides access to the memory-efficient and performance-optimized
    Monte Carlo simulation implementation. It properly handles correlations between
    economic factors and uses batched processing for improved memory efficiency.
    
    If run_async is True, the simulation will be queued for processing by a worker.
    If run_async is False, the simulation will be processed immediately.
    """
    try:
        logger.info(f"Creating optimized Monte Carlo simulation: {request.name}")
        
        service = OptimizedMonteCarloService()
        user_id = current_user.id
        
        # Convert correlation matrix from list of lists to numpy array if provided
        correlation_matrix = None
        if request.correlation_matrix:
            import numpy as np
            correlation_matrix = np.array(request.correlation_matrix)
        
        # Convert models to dictionaries
        base_factors = request.base_economic_factors.dict()
        volatilities = request.volatilities.dict() if request.volatilities else None
        
        if run_async:
            # Queue as a Celery task
            task = run_optimized_monte_carlo_simulation.delay(
                request.dict(),
                user_id,
                correlation_matrix=correlation_matrix.tolist() if correlation_matrix is not None else None
            )
            
            return {
                "status": "queued",
                "message": "Optimized simulation has been queued for processing",
                "task_id": task.id,
                "simulation_id": None  # Will be generated by the worker
            }
        else:
            # Run synchronously
            result = await service.run_simulation(
                loan_data=request.loan_data,
                base_economic_factors=base_factors,
                num_scenarios=request.num_scenarios,
                correlation_matrix=correlation_matrix,
                volatilities=volatilities,
                batch_size=request.batch_size,
                seed=request.seed
            )
            
            return {
                "status": "completed",
                "message": "Optimized simulation completed successfully",
                "task_id": None,
                "simulation_id": f"opt_{user_id}_{int(result['execution_info']['execution_time_seconds'])}",
                "result": result
            }
    except Exception as e:
        logger.error(f"Error creating optimized Monte Carlo simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating optimized simulation: {str(e)}")

@router.post("/monte-carlo/optimized/scenarios", response_model=Dict[str, Any])
async def generate_correlated_scenarios(
    base_factors: EconomicFactorsModel = Body(..., description="Base economic factors"),
    volatilities: Optional[VolatilitiesModel] = Body(None, description="Volatilities for economic factors"),
    correlation_matrix: Optional[List[List[float]]] = Body(None, description="Correlation matrix for economic factors"),
    num_scenarios: int = Query(100, description="Number of scenarios to generate", gt=0, le=10000),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
    current_user: User = Depends(get_current_user)
):
    """
    Generate correlated economic scenarios for Monte Carlo simulation
    
    This endpoint provides access to the scenario generation functionality
    of the optimized Monte Carlo implementation. It creates economic scenarios
    with proper correlation structure between factors.
    """
    try:
        logger.info(f"Generating correlated scenarios for user {current_user.id}")
        
        service = OptimizedMonteCarloService()
        
        # Convert correlation matrix from list of lists to numpy array if provided
        corr_matrix = None
        if correlation_matrix:
            import numpy as np
            corr_matrix = np.array(correlation_matrix)
        
        # Generate scenarios
        scenarios = service.generate_scenarios(
            base_factors=base_factors.dict(),
            num_scenarios=num_scenarios,
            correlation_matrix=corr_matrix,
            volatilities=volatilities.dict() if volatilities else None,
            seed=seed
        )
        
        return {
            "status": "success",
            "num_scenarios": num_scenarios,
            "scenarios": scenarios
        }
    except Exception as e:
        logger.error(f"Error generating correlated scenarios: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating scenarios: {str(e)}")
"""
