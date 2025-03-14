"""
Monte Carlo Simulation API Endpoints

This module provides API endpoints for running Monte Carlo simulations
for financial analytics. It includes endpoints for:
- Creating and running simulations
- Retrieving simulation results
- Managing saved simulations
- Retrieving simulation statistics

The module implements production-ready endpoints with proper error handling,
validation, and Redis caching integration.
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, status, Body
from fastapi.responses import JSONResponse

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.monitoring import CalculationTracker
from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult,
    SavedSimulation,
    ScenarioDefinition,
    SimulationStatus
)
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.supabase_service import SupabaseService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/monte-carlo",
    tags=["monte-carlo"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
    }
)

# Initialize services
monte_carlo_service = MonteCarloSimulationService()
supabase_service = SupabaseService()

# Background task for running simulations
async def _run_simulation_task(
    request: MonteCarloSimulationRequest,
    user_id: str,
    simulation_id: str,
    use_cache: bool = True
):
    """
    Background task for running a Monte Carlo simulation
    
    Args:
        request: The simulation request
        user_id: The user ID
        simulation_id: The simulation ID
        use_cache: Whether to use caching
    """
    try:
        logger.info(f"Starting background simulation task for simulation {simulation_id}")
        
        # Run the simulation
        result = await monte_carlo_service.run_simulation(
            request=request,
            user_id=user_id,
            use_cache=use_cache
        )
        
        # Update the simulation status in the database
        await supabase_service.update_item(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": result.dict(),
                "updated_at": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Completed background simulation task for simulation {simulation_id}")
    
    except Exception as e:
        logger.error(f"Error in background simulation task for simulation {simulation_id}: {str(e)}")
        
        # Update the simulation status to failed
        await supabase_service.update_item(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": {
                    "status": SimulationStatus.FAILED,
                    "error": str(e)
                },
                "updated_at": datetime.now().isoformat()
            }
        )

@router.post(
    "/simulations",
    summary="Create a new Monte Carlo simulation",
    description="Create and optionally run a new Monte Carlo simulation",
    response_model=SavedSimulation
)
async def create_simulation(
    request: MonteCarloSimulationRequest,
    background_tasks: BackgroundTasks,
    run_async: bool = Query(True, description="Run the simulation asynchronously"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new Monte Carlo simulation
    
    Args:
        request: The simulation request
        background_tasks: FastAPI background tasks
        run_async: Whether to run the simulation asynchronously
        use_cache: Whether to use cached results
        current_user: The authenticated user
        
    Returns:
        The created simulation
    """
    user_id = current_user.get("id", "anonymous")
    simulation_id = str(uuid.uuid4())
    
    logger.info(f"Creating Monte Carlo simulation {simulation_id} for user {user_id}")
    
    try:
        # Create a saved simulation record
        saved_simulation = SavedSimulation(
            id=simulation_id,
            user_id=user_id,
            name=request.name,
            description=request.description,
            request=request,
            result={
                "status": SimulationStatus.PENDING,
                "start_time": datetime.now().isoformat()
            } if run_async else None
        )
        
        # Save to database
        await supabase_service.create_item(
            table="monte_carlo_simulations",
            data=saved_simulation.dict()
        )
        
        # If requested to run asynchronously, queue the task
        if run_async:
            background_tasks.add_task(
                _run_simulation_task,
                request=request,
                user_id=user_id,
                simulation_id=simulation_id,
                use_cache=use_cache
            )
            
            logger.info(f"Queued asynchronous simulation task for simulation {simulation_id}")
            
            return saved_simulation
        
        # Otherwise, run synchronously
        logger.info(f"Running synchronous simulation for simulation {simulation_id}")
        
        with CalculationTracker(f"monte_carlo_simulation_{simulation_id}"):
            start_time = time.time()
            
            result = await monte_carlo_service.run_simulation(
                request=request,
                user_id=user_id,
                use_cache=use_cache
            )
            
            logger.info(f"Completed synchronous simulation {simulation_id} in {time.time() - start_time:.2f}s")
        
        # Update the saved simulation with the result
        saved_simulation.result = result
        saved_simulation.updated_at = datetime.now()
        
        # Update in database
        await supabase_service.update_item(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": result.dict(),
                "updated_at": saved_simulation.updated_at.isoformat()
            }
        )
        
        return saved_simulation
    
    except Exception as e:
        logger.error(f"Error creating Monte Carlo simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating Monte Carlo simulation: {str(e)}"
        )

@router.get(
    "/simulations/{simulation_id}",
    summary="Get a Monte Carlo simulation",
    description="Get a Monte Carlo simulation by ID",
    response_model=SavedSimulation
)
async def get_simulation(
    simulation_id: str = Path(..., description="The simulation ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get a Monte Carlo simulation by ID
    
    Args:
        simulation_id: The simulation ID
        current_user: The authenticated user
        
    Returns:
        The simulation
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Getting Monte Carlo simulation {simulation_id} for user {user_id}")
    
    try:
        # Get from database
        result = await supabase_service.get_item(
            table="monte_carlo_simulations",
            id=simulation_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
        
        # Check ownership
        if result.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this simulation"
            )
        
        # Convert to SavedSimulation
        return SavedSimulation(**result)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error getting Monte Carlo simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Monte Carlo simulation: {str(e)}"
        )

@router.get(
    "/simulations",
    summary="List Monte Carlo simulations",
    description="List Monte Carlo simulations for the current user",
    response_model=List[SavedSimulation]
)
async def list_simulations(
    limit: int = Query(20, description="Maximum number of simulations to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: Dict = Depends(get_current_user)
):
    """
    List Monte Carlo simulations for the current user
    
    Args:
        limit: Maximum number of simulations to return
        offset: Offset for pagination
        current_user: The authenticated user
        
    Returns:
        List of simulations
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Listing Monte Carlo simulations for user {user_id}")
    
    try:
        # Get from database
        results = await supabase_service.query_items(
            table="monte_carlo_simulations",
            query={
                "user_id": user_id
            },
            limit=limit,
            offset=offset,
            order_by="created_at",
            ascending=False
        )
        
        # Convert to SavedSimulation objects
        return [SavedSimulation(**result) for result in results]
    
    except Exception as e:
        logger.error(f"Error listing Monte Carlo simulations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing Monte Carlo simulations: {str(e)}"
        )

@router.delete(
    "/simulations/{simulation_id}",
    summary="Delete a Monte Carlo simulation",
    description="Delete a Monte Carlo simulation by ID",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_simulation(
    simulation_id: str = Path(..., description="The simulation ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a Monte Carlo simulation by ID
    
    Args:
        simulation_id: The simulation ID
        current_user: The authenticated user
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Deleting Monte Carlo simulation {simulation_id} for user {user_id}")
    
    try:
        # Get from database first to check ownership
        simulation = await supabase_service.get_item(
            table="monte_carlo_simulations",
            id=simulation_id
        )
        
        if not simulation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
        
        # Check ownership
        if simulation.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to delete this simulation"
            )
        
        # Delete from database
        await supabase_service.delete_item(
            table="monte_carlo_simulations",
            id=simulation_id
        )
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting Monte Carlo simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting Monte Carlo simulation: {str(e)}"
        )

@router.post(
    "/simulations/{simulation_id}/run",
    summary="Run a Monte Carlo simulation",
    description="Run a previously created Monte Carlo simulation",
    response_model=SavedSimulation
)
async def run_simulation(
    background_tasks: BackgroundTasks,
    simulation_id: str = Path(..., description="The simulation ID"),
    run_async: bool = Query(True, description="Run the simulation asynchronously"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Run a previously created Monte Carlo simulation
    
    Args:
        background_tasks: FastAPI background tasks
        simulation_id: The simulation ID
        run_async: Whether to run the simulation asynchronously
        use_cache: Whether to use cached results
        current_user: The authenticated user
        
    Returns:
        The updated simulation
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Running Monte Carlo simulation {simulation_id} for user {user_id}")
    
    try:
        # Get from database
        simulation_data = await supabase_service.get_item(
            table="monte_carlo_simulations",
            id=simulation_id
        )
        
        if not simulation_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
        
        # Check ownership
        if simulation_data.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to run this simulation"
            )
        
        # Convert to SavedSimulation
        saved_simulation = SavedSimulation(**simulation_data)
        
        # Check if already running
        if (saved_simulation.result and 
            saved_simulation.result.status == SimulationStatus.RUNNING):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Simulation is already running"
            )
        
        # Update status to pending
        saved_simulation.result = MonteCarloSimulationResult(
            id=str(uuid.uuid4()),
            name=saved_simulation.name,
            description=saved_simulation.description,
            status=SimulationStatus.PENDING,
            start_time=datetime.now(),
            num_simulations=saved_simulation.request.num_simulations,
            num_completed=0,
            summary_statistics={},
            percentiles={}
        )
        
        saved_simulation.updated_at = datetime.now()
        
        # Update in database
        await supabase_service.update_item(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": saved_simulation.result.dict(),
                "updated_at": saved_simulation.updated_at.isoformat()
            }
        )
        
        # If requested to run asynchronously, queue the task
        if run_async:
            background_tasks.add_task(
                _run_simulation_task,
                request=saved_simulation.request,
                user_id=user_id,
                simulation_id=simulation_id,
                use_cache=use_cache
            )
            
            logger.info(f"Queued asynchronous simulation task for simulation {simulation_id}")
            
            return saved_simulation
        
        # Otherwise, run synchronously
        logger.info(f"Running synchronous simulation for simulation {simulation_id}")
        
        with CalculationTracker(f"monte_carlo_simulation_{simulation_id}"):
            start_time = time.time()
            
            result = await monte_carlo_service.run_simulation(
                request=saved_simulation.request,
                user_id=user_id,
                use_cache=use_cache
            )
            
            logger.info(f"Completed synchronous simulation {simulation_id} in {time.time() - start_time:.2f}s")
        
        # Update the saved simulation with the result
        saved_simulation.result = result
        saved_simulation.updated_at = datetime.now()
        
        # Update in database
        await supabase_service.update_item(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": result.dict(),
                "updated_at": saved_simulation.updated_at.isoformat()
            }
        )
        
        return saved_simulation
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running Monte Carlo simulation: {str(e)}"
        )

@router.post(
    "/simulations/with-scenario",
    summary="Run a Monte Carlo simulation with a specific scenario applied",
    description="Run a Monte Carlo simulation with a specific scenario applied",
    response_model=Dict[str, Any]
)
async def run_simulation_with_scenario(
    request: MonteCarloSimulationRequest,
    scenario_id: str = Query(..., description="ID of the scenario to apply"),
    background_tasks: BackgroundTasks = None,
    run_async: bool = Query(True, description="Run the simulation asynchronously"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Run a Monte Carlo simulation with a specific scenario applied
    
    Args:
        request: The simulation request
        scenario_id: ID of the scenario to apply
        background_tasks: FastAPI background tasks
        run_async: Whether to run the simulation asynchronously
        use_cache: Whether to use cached results
        current_user: The authenticated user
        
    Returns:
        The simulation with scenario applied
    """
    try:
        user_id = current_user["id"]
        
        # Generate a simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Log the request
        logger.info(f"Running Monte Carlo simulation {simulation_id} with scenario {scenario_id} for user {user_id}")
        
        # Initialize services
        monte_carlo_service = MonteCarloSimulationService()
        supabase_service = SupabaseService()
        
        # Check if the scenario exists
        scenario = await supabase_service.get_scenario(scenario_id, user_id)
        if not scenario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario {scenario_id} not found"
            )
        
        # Create initial response
        response = {
            "id": simulation_id,
            "status": SimulationStatus.PENDING,
            "message": "Simulation has been created",
            "created_at": datetime.now().isoformat(),
            "scenario_id": scenario_id,
            "scenario_name": scenario.get("name", "Unknown")
        }
        
        if run_async:
            # Create initial simulation record
            simulation_record = {
                "id": simulation_id,
                "user_id": user_id,
                "request": request.dict(),
                "scenario_id": scenario_id,
                "result": {
                    "id": simulation_id,
                    "name": request.name,
                    "description": request.description,
                    "status": SimulationStatus.PENDING,
                    "start_time": datetime.now().isoformat(),
                    "num_simulations": request.num_simulations,
                    "num_completed": 0,
                    "scenario_id": scenario_id
                },
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert the initial record
            await supabase_service.insert_db(
                table="monte_carlo_simulations",
                data=simulation_record
            )
            
            # Run the simulation in the background
            if background_tasks:
                # Run as FastAPI background task
                background_tasks.add_task(
                    _run_simulation_with_scenario_task,
                    request,
                    user_id,
                    simulation_id,
                    scenario_id,
                    use_cache
                )
                response["status"] = SimulationStatus.PENDING
                response["message"] = "Simulation has been queued for processing"
            else:
                # Run as Celery task
                from app.worker.monte_carlo_tasks import run_monte_carlo_simulation
                
                task = run_monte_carlo_simulation.delay(
                    request.dict(),
                    user_id,
                    simulation_id,
                    scenario_id,
                    use_cache
                )
                
                response["status"] = SimulationStatus.PENDING
                response["message"] = "Simulation has been queued for processing"
                response["task_id"] = task.id
        else:
            # Run synchronously
            try:
                with CalculationTracker(f"monte_carlo_with_scenario_{simulation_id}"):
                    result = await monte_carlo_service.run_simulation_with_scenario(
                        request,
                        scenario_id,
                        user_id,
                        use_cache=use_cache
                    )
                
                # Save the simulation
                saved_simulation = {
                    "id": simulation_id,
                    "user_id": user_id,
                    "name": request.name,
                    "description": request.description,
                    "request": request.dict(),
                    "result": result.dict(),
                    "scenario_id": scenario_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                await supabase_service.create_simulation(saved_simulation)
                
                response["status"] = SimulationStatus.COMPLETED
                response["message"] = "Simulation has been completed"
                response["result"] = result.dict()
            except Exception as e:
                logger.error(f"Error running Monte Carlo simulation with scenario: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error running simulation: {str(e)}"
                )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Monte Carlo simulation with scenario: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating simulation: {str(e)}"
        )

async def _run_simulation_with_scenario_task(
    request: MonteCarloSimulationRequest,
    user_id: str,
    simulation_id: str,
    scenario_id: str,
    use_cache: bool = True
):
    """
    Background task for running a Monte Carlo simulation with scenario
    
    Args:
        request: The simulation request
        user_id: The user ID
        simulation_id: The simulation ID
        scenario_id: The scenario ID
        use_cache: Whether to use caching
    """
    try:
        # Initialize services
        monte_carlo_service = MonteCarloSimulationService()
        supabase_service = SupabaseService()
        
        # Update simulation status
        simulation = await supabase_service.get_item(
            table="monte_carlo_simulations",
            id=simulation_id
        )
        
        if simulation:
            # Update the status
            result = simulation.get("result", {})
            result["status"] = SimulationStatus.RUNNING
            result["start_time"] = datetime.now().isoformat()
            
            # Save the updated simulation
            await supabase_service.update_item(
                table="monte_carlo_simulations",
                id=simulation_id,
                data={
                    "result": result,
                    "updated_at": datetime.now().isoformat()
                }
            )
        
        # Run the simulation with the scenario
        with CalculationTracker(f"monte_carlo_with_scenario_{simulation_id}"):
            result = await monte_carlo_service.run_simulation_with_scenario(
                request,
                scenario_id,
                user_id,
                use_cache=use_cache
            )
        
        # Save the simulation
        saved_simulation = {
            "id": simulation_id,
            "user_id": user_id,
            "name": request.name,
            "description": request.description,
            "request": request.dict(),
            "result": result.dict(),
            "scenario_id": scenario_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        await supabase_service.create_simulation(saved_simulation)
        
        # Update the simulation status
        if simulation:
            await supabase_service.update_item(
                table="monte_carlo_simulations",
                id=simulation_id,
                data={
                    "result": result.dict(),
                    "updated_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat()
                }
            )
        
        logger.info(f"Completed Monte Carlo simulation {simulation_id} with scenario {scenario_id}")
    
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation {simulation_id} with scenario {scenario_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update simulation status
        try:
            simulation = await supabase_service.get_item(
                table="monte_carlo_simulations",
                id=simulation_id
            )
            
            if simulation:
                # Update the status
                result = simulation.get("result", {})
                result["status"] = SimulationStatus.FAILED
                result["error"] = str(e)
                result["end_time"] = datetime.now().isoformat()
                
                # Save the updated simulation
                await supabase_service.update_item(
                    table="monte_carlo_simulations",
                    id=simulation_id,
                    data={
                        "result": result,
                        "updated_at": datetime.now().isoformat()
                    }
                )
        except Exception as update_error:
            logger.error(f"Error updating failed simulation status: {str(update_error)}")

@router.post(
    "/scenarios/compare",
    summary="Compare Monte Carlo simulations with different scenarios",
    description="Compare Monte Carlo simulations with different scenarios",
    response_model=Dict[str, Any]
)
async def compare_scenarios(
    base_request: MonteCarloSimulationRequest,
    scenario_ids: List[str] = Body(..., description="List of scenario IDs to compare"),
    metrics_to_compare: List[str] = Body(None, description="List of metrics to compare"),
    percentiles_to_compare: List[float] = Body(None, description="List of percentiles to compare"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Compare Monte Carlo simulations with different scenarios
    
    Args:
        base_request: The base simulation request
        scenario_ids: List of scenario IDs to compare
        metrics_to_compare: List of metrics to include in comparison
        percentiles_to_compare: List of percentiles to include in comparison
        current_user: The authenticated user
        
    Returns:
        Comparison results
    """
    try:
        user_id = current_user["id"]
        
        # Initialize services
        monte_carlo_service = MonteCarloSimulationService()
        supabase_service = SupabaseService()
        
        # Verify that all scenarios exist
        for scenario_id in scenario_ids:
            scenario = await supabase_service.get_scenario(scenario_id, user_id)
            if not scenario:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenario {scenario_id} not found"
                )
        
        # Log the request
        logger.info(f"Comparing scenarios for user {user_id}: {', '.join(scenario_ids)}")
        
        # Run the comparison with error handling
        try:
            with CalculationTracker(f"monte_carlo_scenario_comparison_{user_id}"):
                comparison_result = await monte_carlo_service.compare_scenarios(
                    base_request,
                    scenario_ids,
                    user_id,
                    metrics_to_compare,
                    percentiles_to_compare
                )
            
            return comparison_result
        except Exception as e:
            logger.error(f"Error comparing scenarios: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error comparing scenarios: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scenario comparison: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in scenario comparison: {str(e)}"
        )

@router.post(
    "/scenarios",
    summary="Create a scenario definition",
    description="Create a new scenario definition for Monte Carlo simulations",
    response_model=ScenarioDefinition
)
async def create_scenario(
    scenario: ScenarioDefinition,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new scenario definition
    
    Args:
        scenario: The scenario definition
        current_user: The authenticated user
        
    Returns:
        The created scenario
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Creating scenario definition for user {user_id}")
    
    try:
        # Add user ID to scenario data
        scenario_data = scenario.dict()
        scenario_data["user_id"] = user_id
        
        # Save to database
        result = await supabase_service.create_item(
            table="monte_carlo_scenarios",
            data=scenario_data
        )
        
        # Convert to ScenarioDefinition
        return ScenarioDefinition(**result)
    
    except Exception as e:
        logger.error(f"Error creating scenario definition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating scenario definition: {str(e)}"
        )

@router.get(
    "/scenarios",
    summary="List scenario definitions",
    description="List scenario definitions for the current user",
    response_model=List[ScenarioDefinition]
)
async def list_scenarios(
    limit: int = Query(20, description="Maximum number of scenarios to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: Dict = Depends(get_current_user)
):
    """
    List scenario definitions for the current user
    
    Args:
        limit: Maximum number of scenarios to return
        offset: Offset for pagination
        current_user: The authenticated user
        
    Returns:
        List of scenarios
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Listing scenario definitions for user {user_id}")
    
    try:
        # Get from database
        results = await supabase_service.query_items(
            table="monte_carlo_scenarios",
            query={
                "user_id": user_id
            },
            limit=limit,
            offset=offset,
            order_by="created_at",
            ascending=False
        )
        
        # Convert to ScenarioDefinition objects
        return [ScenarioDefinition(**result) for result in results]
    
    except Exception as e:
        logger.error(f"Error listing scenario definitions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing scenario definitions: {str(e)}"
        )

@router.get(
    "/scenarios/{scenario_id}",
    summary="Get a scenario definition",
    description="Get a scenario definition by ID",
    response_model=ScenarioDefinition
)
async def get_scenario(
    scenario_id: str = Path(..., description="The scenario ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get a scenario definition by ID
    
    Args:
        scenario_id: The scenario ID
        current_user: The authenticated user
        
    Returns:
        The scenario
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Getting scenario definition {scenario_id} for user {user_id}")
    
    try:
        # Get from database
        result = await supabase_service.get_item(
            table="monte_carlo_scenarios",
            id=scenario_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario {scenario_id} not found"
            )
        
        # Check ownership
        if result.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this scenario"
            )
        
        # Convert to ScenarioDefinition
        return ScenarioDefinition(**result)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error getting scenario definition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting scenario definition: {str(e)}"
        )

@router.put(
    "/scenarios/{scenario_id}",
    summary="Update a scenario definition",
    description="Update a scenario definition by ID",
    response_model=ScenarioDefinition
)
async def update_scenario(
    scenario: ScenarioDefinition,
    scenario_id: str = Path(..., description="The scenario ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Update a scenario definition by ID
    
    Args:
        scenario: The updated scenario definition
        scenario_id: The scenario ID
        current_user: The authenticated user
        
    Returns:
        The updated scenario
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Updating scenario definition {scenario_id} for user {user_id}")
    
    try:
        # Get from database first to check ownership
        existing_scenario = await supabase_service.get_item(
            table="monte_carlo_scenarios",
            id=scenario_id
        )
        
        if not existing_scenario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario {scenario_id} not found"
            )
        
        # Check ownership
        if existing_scenario.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to update this scenario"
            )
        
        # Update scenario data
        scenario_data = scenario.dict()
        scenario_data["user_id"] = user_id
        scenario_data["updated_at"] = datetime.now().isoformat()
        
        # Update in database
        result = await supabase_service.update_item(
            table="monte_carlo_scenarios",
            id=scenario_id,
            data=scenario_data
        )
        
        # Convert to ScenarioDefinition
        return ScenarioDefinition(**result)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error updating scenario definition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating scenario definition: {str(e)}"
        )

@router.delete(
    "/scenarios/{scenario_id}",
    summary="Delete a scenario definition",
    description="Delete a scenario definition by ID",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_scenario(
    scenario_id: str = Path(..., description="The scenario ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a scenario definition by ID
    
    Args:
        scenario_id: The scenario ID
        current_user: The authenticated user
    """
    user_id = current_user.get("id", "anonymous")
    
    logger.info(f"Deleting scenario definition {scenario_id} for user {user_id}")
    
    try:
        # Get from database first to check ownership
        scenario = await supabase_service.get_item(
            table="monte_carlo_scenarios",
            id=scenario_id
        )
        
        if not scenario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario {scenario_id} not found"
            )
        
        # Check ownership
        if scenario.get("user_id") != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to delete this scenario"
            )
        
        # Delete from database
        await supabase_service.delete_item(
            table="monte_carlo_scenarios",
            id=scenario_id
        )
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting scenario definition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting scenario definition: {str(e)}"
        )
