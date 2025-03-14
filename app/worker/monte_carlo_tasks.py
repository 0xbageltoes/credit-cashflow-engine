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
    name="app.worker.monte_carlo_tasks.run_monte_carlo_simulation",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # Maximum of 10 minutes between retries
    retry_jitter=True,
    max_retries=3,
    time_limit=3600,  # 1 hour time limit
    soft_time_limit=3540,  # 59 minutes soft time limit
)
def run_monte_carlo_simulation(self, request_dict: dict, simulation_id: str, user_id: str, use_cache: bool = True):
    """
    Run a Monte Carlo simulation in the background
    
    Args:
        self: The task instance
        request_dict: The simulation request as a dictionary
        simulation_id: The ID of the simulation
        user_id: The ID of the user running the simulation
        use_cache: Whether to use cached results if available
        
    Returns:
        Dictionary with information about the completed simulation
    """
    task_id = self.request.id
    logger.info(f"Starting Monte Carlo simulation {simulation_id} (task ID: {task_id})")
    
    try:
        # Update simulation status in database
        supabase_service.update_item_sync(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": {
                    "status": SimulationStatus.RUNNING,
                    "start_time": datetime.now().isoformat(),
                    "task_id": task_id
                },
                "updated_at": datetime.now().isoformat()
            }
        )
        
        # Convert request_dict to MonteCarloSimulationRequest
        request = MonteCarloSimulationRequest.parse_obj(request_dict)
        
        # Record metrics
        with CalculationTracker(f"monte_carlo_simulation_{simulation_id}"):
            start_time = time.time()
            
            # Run the simulation
            result = monte_carlo_service.run_simulation_sync(
                request=request,
                user_id=user_id,
                use_cache=use_cache
            )
            
            execution_time = time.time() - start_time
        
        # Update simulation in database
        supabase_service.update_item_sync(
            table="monte_carlo_simulations",
            id=simulation_id,
            data={
                "result": result.dict(),
                "updated_at": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Completed Monte Carlo simulation {simulation_id} in {execution_time:.2f}s")
        
        # Return information about the completed simulation
        return {
            "simulation_id": simulation_id,
            "user_id": user_id,
            "status": SimulationStatus.COMPLETED,
            "execution_time": execution_time
        }
    
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation {simulation_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update simulation status in database
        try:
            supabase_service.update_item_sync(
                table="monte_carlo_simulations",
                id=simulation_id,
                data={
                    "result": {
                        "status": SimulationStatus.FAILED,
                        "error": str(e),
                        "end_time": datetime.now().isoformat()
                    },
                    "updated_at": datetime.now().isoformat()
                }
            )
        except Exception as update_error:
            logger.error(f"Error updating failed simulation status: {str(update_error)}")
        
        # Re-raise the exception for Celery's retry mechanism
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
