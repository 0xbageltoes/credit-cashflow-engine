"""
Monte Carlo Simulation Celery Tasks

This module contains Celery tasks for running Monte Carlo simulations in the background.
It includes tasks for:
- Running Monte Carlo simulations
- Updating simulation status in Supabase
- Cleaning up expired simulations

The tasks are designed to integrate with the Monte Carlo simulation service and
provide robust error handling and monitoring.
"""
import logging
import time
from datetime import datetime, timedelta
import json
import traceback
import httpx
from typing import Dict, Any, List, Optional

from celery import shared_task
from celery.signals import task_prerun, task_success, task_failure
from celery.utils.log import get_task_logger

from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResult,
    SimulationStatus
)
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.services.supabase_service import SupabaseService
from app.core.config import settings
from app.core.monitoring import CalculationTracker
from app.api.v1.websockets.task_status import (
    broadcast_simulation_progress,
    broadcast_simulation_completion,
    broadcast_simulation_error
)

# Setup logging
logger = get_task_logger(__name__)

# Initialize services
monte_carlo_service = MonteCarloSimulationService()
supabase_service = SupabaseService()

# Create a base Task class with error handling
class MonteCarloTask:
    """Base task class with error handling and WebSocket integration"""
    
    # Set max retries and retry delay
    max_retries = 3
    default_retry_delay = 60  # 1 minute
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Extract simulation_id if available
        simulation_id = kwargs.get("simulation_id")
        if not simulation_id and args and len(args) > 1:
            simulation_id = args[1]
        
        if simulation_id:
            # Update simulation status in database
            try:
                # Get the current simulation record
                simulation = supabase_service.get_item_sync(
                    table="monte_carlo_simulations",
                    id=simulation_id
                )
                
                if simulation:
                    # Update the status
                    result = simulation.get("result", {})
                    result["status"] = SimulationStatus.FAILED
                    result["error"] = str(exc)
                    result["end_time"] = datetime.now().isoformat()
                    
                    # Save the updated simulation
                    supabase_service.update_item_sync(
                        table="monte_carlo_simulations",
                        id=simulation_id,
                        data={"result": result}
                    )
                    
                    # Broadcast error via WebSocket
                    self._async_broadcast_error(simulation_id, str(exc))
            
            except Exception as e:
                logger.error(f"Error updating failed simulation {simulation_id}: {str(e)}")
    
    def _async_broadcast_progress(self, simulation_id: str, status: str, progress: int, total: int):
        """Broadcast progress update via WebSocket"""
        # Create a background task to send the WebSocket update
        if settings.WEBSOCKET_ENABLED:
            try:
                # Make an async HTTP request to the local API endpoint
                # This allows us to broadcast from a sync context
                url = f"{settings.INTERNAL_API_BASE_URL}/api/v1/internal/websocket/broadcast-progress"
                payload = {
                    "simulation_id": simulation_id,
                    "status": status,
                    "progress": progress,
                    "total": total,
                    "api_key": settings.INTERNAL_API_KEY
                }
                
                with httpx.Client(timeout=5.0) as client:
                    client.post(url, json=payload)
            
            except Exception as e:
                logger.error(f"Error broadcasting progress: {str(e)}")
    
    def _async_broadcast_completion(self, simulation_id: str, result: Dict[str, Any]):
        """Broadcast completion via WebSocket"""
        if settings.WEBSOCKET_ENABLED:
            try:
                # Make an async HTTP request to the local API endpoint
                url = f"{settings.INTERNAL_API_BASE_URL}/api/v1/internal/websocket/broadcast-completion"
                payload = {
                    "simulation_id": simulation_id,
                    "result": {
                        k: v for k, v in result.items() 
                        if k in ["status", "execution_time_seconds", "summary_statistics", "percentiles"]
                    },
                    "api_key": settings.INTERNAL_API_KEY
                }
                
                with httpx.Client(timeout=5.0) as client:
                    client.post(url, json=payload)
            
            except Exception as e:
                logger.error(f"Error broadcasting completion: {str(e)}")
    
    def _async_broadcast_error(self, simulation_id: str, error: str):
        """Broadcast error via WebSocket"""
        if settings.WEBSOCKET_ENABLED:
            try:
                # Make an async HTTP request to the local API endpoint
                url = f"{settings.INTERNAL_API_BASE_URL}/api/v1/internal/websocket/broadcast-error"
                payload = {
                    "simulation_id": simulation_id,
                    "error": error,
                    "api_key": settings.INTERNAL_API_KEY
                }
                
                with httpx.Client(timeout=5.0) as client:
                    client.post(url, json=payload)
            
            except Exception as e:
                logger.error(f"Error broadcasting error: {str(e)}")

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extras):
    """
    Signal handler that runs before a task is executed
    
    Args:
        sender: The task being run
        task_id: The ID of the task
        task: The task instance
        args: The positional arguments passed to the task
        kwargs: The keyword arguments passed to the task
    """
    logger.info(f"Starting task {task.name} with ID {task_id}")
    
    # Record task start in monitoring
    if task.name == "app.worker.monte_carlo_tasks.run_monte_carlo_simulation":
        # Extract simulation_id from args if available
        if args and len(args) > 1:
            simulation_id = args[1]
            CalculationTracker.start_task(f"monte_carlo_task_{simulation_id}")

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """
    Signal handler that runs when a task succeeds
    
    Args:
        sender: The task that succeeded
        result: The result of the task
    """
    logger.info(f"Task {sender.name} completed successfully")
    
    # Record task success in monitoring
    if sender.name == "app.worker.monte_carlo_tasks.run_monte_carlo_simulation":
        # Extract simulation_id from result if available
        if result and isinstance(result, dict) and "simulation_id" in result:
            simulation_id = result["simulation_id"]
            CalculationTracker.end_task(f"monte_carlo_task_{simulation_id}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extras):
    """
    Signal handler that runs when a task fails
    
    Args:
        sender: The task that failed
        task_id: The ID of the task
        exception: The exception that caused the failure
        args: The positional arguments passed to the task
        kwargs: The keyword arguments passed to the task
        traceback: The traceback of the exception
        einfo: Additional error information
    """
    logger.error(f"Task {sender.name} with ID {task_id} failed: {str(exception)}")
    
    # Record task failure in monitoring
    if sender.name == "app.worker.monte_carlo_tasks.run_monte_carlo_simulation":
        # Extract simulation_id from args if available
        if args and len(args) > 1:
            simulation_id = args[1]
            CalculationTracker.end_task(f"monte_carlo_task_{simulation_id}", success=False)
            
            # Update simulation status in database
            try:
                logger.info(f"Updating failed simulation status for {simulation_id}")
                supabase_service.update_item_sync(
                    table="monte_carlo_simulations",
                    id=simulation_id,
                    data={
                        "result": {
                            "status": SimulationStatus.FAILED,
                            "error": str(exception),
                            "end_time": datetime.now().isoformat()
                        },
                        "updated_at": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error updating failed simulation status: {str(e)}")

@shared_task(
    bind=True,
    name="run_monte_carlo_simulation",
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=600,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    acks_late=True,
    time_limit=3600,
    soft_time_limit=3000,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60}
)
def run_monte_carlo_simulation(
    self,
    request: Dict,
    user_id: str,
    simulation_id: str,
    scenario_id: str = None,
    use_cache: bool = True
):
    """
    Celery task to run a Monte Carlo simulation
    
    Args:
        self: Celery task instance
        request: The simulation request as a dict
        user_id: ID of the user running the simulation
        simulation_id: ID for this simulation
        scenario_id: Optional ID of the scenario to apply
        use_cache: Whether to use caching
        
    Returns:
        Dict with simulation result
    """
    try:
        # Convert dict to MonteCarloSimulationRequest
        from app.models.monte_carlo import MonteCarloSimulationRequest
        
        # Initialize services
        from app.services.monte_carlo_service import MonteCarloSimulationService
        from app.services.supabase_service import SupabaseService
        
        monte_carlo_service = MonteCarloSimulationService()
        supabase_service = SupabaseService()
        
        # Get the current task
        task_id = self.request.id
        
        # Set up progress callback
        def progress_callback(completed, total):
            # Calculate percentage
            percentage = int(100 * completed / total)
            
            # Get task meta
            task_meta = self.backend.get_task_meta(task_id)
            current_info = task_meta.get("result", {})
            
            # Update with new progress
            if isinstance(current_info, dict):
                current_info["num_completed"] = completed
                current_info["progress_percentage"] = percentage
                
                # Save updated meta
                self.update_state(
                    state="PROGRESS",
                    meta=current_info
                )
        
        # Set initial state
        simulation_request = MonteCarloSimulationRequest.parse_obj(request)
        self.update_state(
            state="STARTED",
            meta={
                "id": simulation_id,
                "name": simulation_request.name,
                "description": simulation_request.description,
                "status": "RUNNING",
                "start_time": datetime.now().isoformat(),
                "num_simulations": simulation_request.num_simulations,
                "num_completed": 0,
                "progress_percentage": 0,
                "scenario_id": scenario_id
            }
        )
        
        # Update the status in Supabase
        simulation = {
            "id": simulation_id,
            "user_id": user_id,
            "request": request,
            "result": {
                "id": simulation_id,
                "name": simulation_request.name,
                "description": simulation_request.description,
                "status": "RUNNING",
                "start_time": datetime.now().isoformat(),
                "num_simulations": simulation_request.num_simulations,
                "num_completed": 0,
                "scenario_id": scenario_id
            },
            "scenario_id": scenario_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Check if simulation already exists
        existing_simulation = None
        try:
            existing_simulation = supabase_service.get_item_sync(
                table="monte_carlo_simulations",
                id=simulation_id
            )
        except Exception as e:
            logger.warning(f"Error checking for existing simulation: {str(e)}")
        
        # Update or create simulation record
        try:
            if existing_simulation:
                # Update existing simulation
                supabase_service.update_item_sync(
                    table="monte_carlo_simulations",
                    id=simulation_id,
                    data={
                        "result": simulation["result"],
                        "updated_at": datetime.now().isoformat()
                    }
                )
            else:
                # Create new simulation
                supabase_service.insert_db_sync(
                    table="monte_carlo_simulations",
                    data=simulation
                )
        except Exception as e:
            logger.error(f"Error updating simulation status in database: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue execution even if DB update fails
        
        # Run the simulation (with or without scenario)
        start_time = time.time()
        result = None
        
        if scenario_id:
            # Run with scenario
            logger.info(f"Running Monte Carlo simulation {simulation_id} with scenario {scenario_id}")
            
            # Run the simulation with scenario
            result = monte_carlo_service.run_simulation_with_scenario_sync(
                simulation_request,
                scenario_id,
                user_id,
                use_cache,
                progress_callback
            )
        else:
            # Run without scenario
            logger.info(f"Running Monte Carlo simulation {simulation_id}")
            
            # Run the simulation
            result = monte_carlo_service.run_simulation_sync(
                simulation_request,
                user_id,
                use_cache,
                progress_callback
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Convert result to dict
        result_dict = result.dict()
        
        # Update the result with execution time
        result_dict["execution_time_seconds"] = execution_time
        result_dict["completed_at"] = datetime.now().isoformat()
        
        # Handle detailed paths separately if they exist (they can be large)
        detailed_paths = None
        if "detailed_paths" in result_dict and result_dict["detailed_paths"]:
            # Store the detailed paths separately or summarize
            detailed_paths = result_dict.pop("detailed_paths")
            result_dict["has_detailed_paths"] = True
        else:
            result_dict["has_detailed_paths"] = False
        
        # Save the final result to Supabase
        try:
            # Update the simulation record
            supabase_service.update_item_sync(
                table="monte_carlo_simulations",
                id=simulation_id,
                data={
                    "result": result_dict,
                    "updated_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat()
                }
            )
            
            if detailed_paths:
                # Save the detailed paths to a separate table or blob storage
                # This is implementation-specific based on how large the data is
                pass
                
            logger.info(f"Saved Monte Carlo simulation {simulation_id} result to database")
        except Exception as e:
            logger.error(f"Error saving simulation result to database: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Log the result size for debugging
            result_json = json.dumps(result_dict)
            logger.debug(f"Result size: {len(result_json)} bytes")
        
        logger.info(f"Completed Monte Carlo simulation {simulation_id} in {execution_time:.2f} seconds")
        
        # Return the result
        return {
            "id": simulation_id,
            "status": "COMPLETED",
            "message": "Simulation completed successfully",
            "execution_time_seconds": execution_time,
            "result": result_dict
        }
    
    except Exception as e:
        # Log the error
        error_msg = f"Error running Monte Carlo simulation {simulation_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Try to update the simulation status in the database
        try:
            # Initialize Supabase service
            from app.services.supabase_service import SupabaseService
            supabase_service = SupabaseService()
            
            # Try to get the existing simulation
            existing_simulation = supabase_service.get_item_sync(
                table="monte_carlo_simulations",
                id=simulation_id
            )
            
            if existing_simulation:
                # Update with error status
                result = existing_simulation.get("result", {})
                result["status"] = "FAILED"
                result["error"] = str(e)
                result["error_details"] = traceback.format_exc()
                result["end_time"] = datetime.now().isoformat()
                
                # Save the updated simulation
                supabase_service.update_item_sync(
                    table="monte_carlo_simulations",
                    id=simulation_id,
                    data={
                        "result": result,
                        "updated_at": datetime.now().isoformat()
                    }
                )
                
                logger.info(f"Updated Monte Carlo simulation {simulation_id} with failure status")
        except Exception as db_error:
            logger.error(f"Error updating simulation status after failure: {str(db_error)}")
        
        # Check if retry is beneficial
        should_retry = not isinstance(e, (ValueError, TypeError, AttributeError))
        
        if should_retry and self.request.retries < self.max_retries:
            logger.info(f"Retrying Monte Carlo simulation task (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=60)
        else:
            # Raise the error for Celery to handle
            raise

@shared_task(
    name="app.worker.monte_carlo_tasks.cleanup_expired_simulations",
    bind=True,
    time_limit=600,  # 10 minutes time limit
    soft_time_limit=540,  # 9 minutes soft time limit
)
def cleanup_expired_simulations(self):
    """
    Clean up expired simulation results to free up storage
    
    This task runs periodically to delete or archive old simulations
    based on retention policy.
    
    Args:
        self: The task instance
    """
    task_id = self.request.id
    logger.info(f"Starting cleanup of expired simulations (task ID: {task_id})")
    
    try:
        # Calculate the cutoff date for retention
        retention_days = settings.MONTE_CARLO_RETENTION_DAYS or 30
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_date_str = cutoff_date.isoformat()
        
        # Query for expired simulations
        expired_simulations = supabase_service.query_items_sync(
            table="monte_carlo_simulations",
            query={
                "created_at": {"lt": cutoff_date_str}
            },
            limit=1000  # Process in batches
        )
        
        logger.info(f"Found {len(expired_simulations)} expired simulations to clean up")
        
        # Process expired simulations
        if not expired_simulations:
            return {"status": "completed", "count": 0}
        
        archived_count = 0
        deleted_count = 0
        
        for simulation in expired_simulations:
            simulation_id = simulation.get("id")
            
            # If the simulation has results, archive them
            if settings.MONTE_CARLO_ARCHIVE_ENABLED and simulation.get("result"):
                # Archive the simulation (simplified - in a real implementation, 
                # this might store to a cheaper storage option or compress)
                try:
                    # Insert into archive table
                    supabase_service.create_item_sync(
                        table="monte_carlo_simulations_archive",
                        data={
                            "original_id": simulation_id,
                            "user_id": simulation.get("user_id"),
                            "name": simulation.get("name"),
                            "description": simulation.get("description"),
                            "result_summary": {
                                "status": simulation.get("result", {}).get("status"),
                                "summary_statistics": simulation.get("result", {}).get("summary_statistics"),
                                "percentiles": simulation.get("result", {}).get("percentiles")
                            },
                            "original_created_at": simulation.get("created_at"),
                            "archived_at": datetime.now().isoformat()
                        }
                    )
                    archived_count += 1
                except Exception as e:
                    logger.error(f"Error archiving simulation {simulation_id}: {str(e)}")
            
            # Delete the simulation from the main table
            try:
                supabase_service.delete_item_sync(
                    table="monte_carlo_simulations",
                    id=simulation_id
                )
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting simulation {simulation_id}: {str(e)}")
        
        logger.info(f"Completed cleanup: archived {archived_count}, deleted {deleted_count} simulations")
        
        return {
            "status": "completed",
            "archived_count": archived_count,
            "deleted_count": deleted_count
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up expired simulations: {str(e)}")
        logger.error(traceback.format_exc())
        raise
