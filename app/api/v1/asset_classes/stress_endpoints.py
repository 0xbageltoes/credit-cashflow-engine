"""
Asset Classes Stress Testing API Endpoints

Production-ready implementation of stress testing endpoints for different asset classes
with comprehensive error handling, validation, and Redis caching integration.
"""
import logging
import time
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import uuid
import asyncio
from datetime import datetime, timedelta
import traceback

from app.core.auth import get_current_user, PermissionLevel, verify_permissions
from app.core.config import settings
from app.core.monitoring import CalculationTracker, PerformanceMetrics, StressTestMetrics, CACHE_HITS, CACHE_MISSES
from app.core.stress_testing_config import get_stress_test_settings
from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolMetrics, AssetPoolCashflow, AssetPoolStressTest
)
from app.services.asset_handlers.stress_testing import AssetStressTester
from app.core.websocket import manager
from app.core.rate_limiter import RateLimiter
from app.core.redis_manager import RedisManager

# Setup logging
logger = logging.getLogger(__name__)

# Get stress testing settings
stress_settings = get_stress_test_settings()

# Create router with proper API documentation
router = APIRouter(
    prefix="/stress-testing",
    tags=["stress-testing"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request parameters"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication credentials not provided or invalid"},
        status.HTTP_403_FORBIDDEN: {"description": "Insufficient permissions to access this resource"},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service temporarily unavailable"}
    }
)

# Initialize rate limiter for production protection
rate_limiter = RateLimiter(
    max_requests=stress_settings.STRESS_TEST_RATE_LIMIT_PER_MINUTE,
    time_window=60  # 1 minute
)

# Initialize Redis manager for results caching
redis_manager = RedisManager(
    url=settings.REDIS_URL,
    prefix="stress_test_results:",
    default_ttl=settings.STRESS_TEST_RESULTS_TTL_SECONDS or 86400
)

# Initialize a global metrics registry
stress_metrics = {}

# Request Models
class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    request: AssetPoolAnalysisRequest
    scenario_names: Optional[List[str]] = None
    custom_scenarios: Optional[Dict[str, Dict[str, Any]]] = None
    run_parallel: bool = True
    max_workers: int = Field(
        default=4, 
        ge=1, 
        le=16, 
        description="Maximum number of parallel workers (1-16)"
    )
    include_cashflows: bool = False  # Default to false to reduce response size
    generate_report: bool = True

# Response Models
class StressTestTaskResponse(BaseModel):
    """Response model for async stress test task"""
    task_id: str
    status: str
    message: str
    pool_name: str
    scenario_count: int
    websocket_url: Optional[str] = None
    estimated_completion_time: Optional[str] = None

class StressTestResults(BaseModel):
    """Response model for stress test results"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pool_name: str
    analysis_date: str
    execution_time: float
    scenario_results: Dict[str, AssetPoolAnalysisResponse]
    report: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

# Initialize service
stress_tester = AssetStressTester()

@router.post(
    "/run",
    summary="Run stress tests on asset pool",
    description="""
    Runs a series of stress tests on an asset pool with specified scenarios.
    
    This endpoint performs multiple stress simulations based on predefined or custom scenarios 
    and returns detailed results including NPV impact, duration changes, and other key metrics.
    
    For large asset pools (>1,000 assets) or many scenarios, consider using the async-run endpoint.
    """,
    response_model=Dict[str, AssetPoolAnalysisResponse],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Rate limit exceeded, try again later"
        }
    }
)
async def run_stress_tests(
    request: StressTestRequest,
    req: Request,
    use_cache: Optional[bool] = True,
    current_user: Dict = Depends(get_current_user)
):
    """
    Run stress tests on asset pool
    
    Args:
        request: The stress test request
        req: The original FastAPI request object
        use_cache: Whether to use caching (default: True)
        current_user: The authenticated user
        
    Returns:
        Stress test results by scenario
    """
    request_id = str(uuid.uuid4())
    user_id = current_user.get("id", "anonymous")
    
    # Check rate limits for production safety
    client_ip = req.client.host if req.client else "unknown"
    if not await rate_limiter.check_rate_limit(f"{user_id}_{client_ip}"):
        logger.warning(f"Request {request_id}: Rate limit exceeded for user {user_id} from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Verify user has appropriate permissions
    try:
        verify_permissions(current_user, PermissionLevel.ANALYST)
    except ValueError as e:
        logger.warning(f"Request {request_id}: Permission denied for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    
    logger.info(f"Request {request_id}: Stress tests for user {user_id}, pool: {request.request.pool.pool_name}")
    
    # Track performance for all operations
    stress_tracker = PerformanceMetrics(f"stress_test_{request_id}")
    
    try:
        with stress_tracker.track("total"):
            # Validate request within production limits
            with stress_tracker.track("validation"):
                # Validate pool has assets
                if not request.request.pool.assets:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Asset pool must contain at least one asset"
                    )
                
                # Enforce pool size limits
                if len(request.request.pool.assets) > stress_settings.STRESS_TEST_MAX_ASSETS_PER_POOL:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Asset pool exceeds maximum size limit of {stress_settings.STRESS_TEST_MAX_ASSETS_PER_POOL} assets"
                    )
                
                # Limit max workers for production stability
                request.max_workers = min(
                    max(1, request.max_workers), 
                    stress_settings.STRESS_TEST_MAX_WORKERS
                )
            
            # Configure cashflows inclusion
            if not request.include_cashflows:
                request.request.include_cashflows = False
            
            # Run stress tests
            start_time = time.time()
            
            with stress_tracker.track("stress_testing"):
                results = await stress_tester.run_stress_tests(
                    request=request.request,
                    user_id=user_id,
                    scenario_names=request.scenario_names,
                    custom_scenarios=request.custom_scenarios,
                    use_cache=use_cache,
                    parallel=request.run_parallel,
                    max_workers=request.max_workers
                )
            
            execution_time = time.time() - start_time
            logger.info(
                f"Request {request_id}: Completed {len(results)} stress tests in {execution_time:.2f}s"
            )
            
            # Generate report if requested
            if request.generate_report:
                with stress_tracker.track("report_generation"):
                    report = await stress_tester.generate_stress_test_report(
                        results=results,
                        output_format="json"
                    )
                    
                    # Add report to base case result if successful
                    if "base" in results and isinstance(report, dict) and report.get("status") != "error":
                        results["base"].analytics = results["base"].analytics or {}
                        results["base"].analytics["stress_test_report"] = report
            
            # Add performance metrics to the response
            if "base" in results:
                results["base"].analytics = results["base"].analytics or {}
                results["base"].analytics["performance_metrics"] = stress_tracker.get_metrics()
            
            return results
            
    except ValidationError as e:
        logger.error(f"Request {request_id}: Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except asyncio.TimeoutError:
        logger.error(f"Request {request_id}: Operation timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Operation timed out. Consider using the async-run endpoint for large requests."
        )
    except Exception as e:
        logger.exception(f"Request {request_id}: Unexpected error during stress testing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during stress testing: {str(e)}"
        )

@router.post(
    "/async-run",
    summary="Run stress tests asynchronously",
    description="""
    Queues stress tests to run in the background and sends updates via websocket.
    
    This endpoint is ideal for large asset pools or multiple scenarios that may take
    longer to process. The response includes a task ID and websocket URL for tracking progress.
    
    Real-time updates are provided through the websocket connection, including progress 
    and final results when completed.
    """,
    response_model=StressTestTaskResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def run_stress_tests_async(
    request: StressTestRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    use_cache: Optional[bool] = True,
    current_user: Dict = Depends(get_current_user)
):
    """
    Run stress tests asynchronously
    
    Args:
        request: The stress test request
        req: The original FastAPI request object
        background_tasks: FastAPI background tasks
        use_cache: Whether to use caching (default: True)
        current_user: The authenticated user
        
    Returns:
        Task information with websocket details
    """
    task_id = str(uuid.uuid4())
    user_id = current_user.get("id", "anonymous")
    
    # Initialize metrics tracker for this user if not exists
    if user_id not in stress_metrics:
        stress_metrics[user_id] = StressTestMetrics(user_id)
    
    # Track stress test request in metrics
    stress_metrics[user_id].track_test_request(parallel=request.run_parallel)
    stress_metrics[user_id].track_asset_count(len(request.request.pool.assets))
    stress_metrics[user_id].increment_active_tests()
    
    # Check rate limits for production safety
    client_ip = req.client.host if req.client else "unknown"
    if not await rate_limiter.check_rate_limit(f"{user_id}_{client_ip}"):
        logger.warning(f"Task {task_id}: Rate limit exceeded for user {user_id} from {client_ip}")
        
        # Log detailed rate limit information
        rate_info = await rate_limiter.get_limit_info(f"{user_id}_{client_ip}")
        logger.info(f"Rate limit details: {rate_info}")
        
        stress_metrics[user_id].track_error("rate_limit_exceeded")
        stress_metrics[user_id].decrement_active_tests()
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Verify user has appropriate permissions
    try:
        verify_permissions(current_user, PermissionLevel.ANALYST)
    except ValueError as e:
        logger.warning(f"Task {task_id}: Permission denied for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    
    logger.info(f"Task {task_id}: Queuing async stress tests for user {user_id}, pool: {request.request.pool.pool_name}")
    
    try:
        # Record initial task metadata for tracking and recovery
        task_metadata = {
            "user_id": user_id,
            "client_ip": client_ip,
            "pool_name": request.request.pool.pool_name,
            "asset_count": len(request.request.pool.assets),
            "request_time": datetime.now().isoformat(),
            "parameters": {
                "run_parallel": request.run_parallel,
                "max_workers": request.max_workers,
                "include_cashflows": request.include_cashflows,
                "generate_report": request.generate_report
            }
        }
        
        # Validate request within production limits
        # Validate pool has assets
        if not request.request.pool.assets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Asset pool must contain at least one asset"
            )
        
        # Enforce pool size limits
        if len(request.request.pool.assets) > stress_settings.STRESS_TEST_MAX_ASSETS_PER_POOL:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Asset pool exceeds maximum size limit of {stress_settings.STRESS_TEST_MAX_ASSETS_PER_POOL} assets"
            )
        
        # Limit max workers for production stability
        max_workers_original = request.max_workers
        request.max_workers = min(
            max(1, request.max_workers), 
            stress_settings.STRESS_TEST_MAX_WORKERS
        )
        
        if max_workers_original != request.max_workers:
            logger.info(f"Task {task_id}: Adjusted max_workers from {max_workers_original} to {request.max_workers}")
        
        # Configure cashflows inclusion
        if not request.include_cashflows:
            request.request.include_cashflows = False
        
        # Get scenario information for better progress tracking
        scenario_names = request.scenario_names
        if not scenario_names and not request.custom_scenarios:
            # Using default scenarios - need to get their names
            tester = AssetStressTester()
            scenario_names = list(tester.default_scenarios.keys())
        
        # If using custom scenarios, get their names
        custom_scenario_names = []
        if request.custom_scenarios:
            custom_scenario_names = list(request.custom_scenarios.keys())
        
        # Combine all scenario names for tracking
        all_scenario_names = []
        if scenario_names:
            all_scenario_names.extend(scenario_names)
        if custom_scenario_names:
            all_scenario_names.extend(custom_scenario_names)
        
        # Add scenario info to task metadata
        task_metadata["scenarios"] = {
            "default": scenario_names or [],
            "custom": custom_scenario_names or [],
            "total_count": len(all_scenario_names) or len(request.custom_scenarios or {}) or 7  # Default to 7 if empty
        }
        
        # Initialize task status in the WebSocket manager
        await manager.send_task_update(
            task_id=task_id,
            status="queued",
            message="Stress tests queued and awaiting execution",
            data=task_metadata
        )
        
        # Define background task with comprehensive progress tracking
        async def run_stress_tests_bg():
            # Track performance for all operations
            stress_tracker = PerformanceMetrics(f"async_stress_test_{task_id}")
            current_scenario = "initializing"
            scenario_count = task_metadata["scenarios"]["total_count"]
            scenario_index = 0
            
            try:
                with stress_tracker.track("total"):
                    # Send initial status
                    await manager.send_task_update(
                        task_id=task_id,
                        status="running",
                        message="Starting stress tests",
                        data={
                            "progress": 5,
                            "pool_name": request.request.pool.pool_name,
                            "start_time": datetime.now().isoformat(),
                            "scenario_count": scenario_count,
                            "current_scenario": current_scenario,
                            "completed_scenarios": 0
                        }
                    )
                    
                    # Hook for progress updates during scenario processing
                    async def scenario_progress_callback(scenario_name, status, progress_pct=None, details=None):
                        nonlocal current_scenario, scenario_index
                        
                        # Update current scenario name and index
                        if status == "starting":
                            current_scenario = scenario_name
                            scenario_index += 1
                            
                            # Track scenario metrics
                            stress_metrics[user_id].track_scenario_execution(scenario_name, success=True)
                        
                        # Calculate overall progress based on scenario index and internal progress
                        # Base progress is 5-90% (reserving 0-5% for init and 90-100% for report generation)
                        scenario_progress = progress_pct or 0
                        base_progress = 5  # Starting at 5%
                        per_scenario_progress = 85 / max(1, scenario_count)  # 85% spread across all scenarios
                        
                        # Overall progress combines completed scenarios + current scenario's progress
                        if status == "completed":
                            overall_progress = base_progress + (scenario_index * per_scenario_progress)
                        else:
                            current_scenario_progress = per_scenario_progress * (scenario_progress / 100)
                            previous_scenarios_progress = (scenario_index - 1) * per_scenario_progress
                            overall_progress = base_progress + previous_scenarios_progress + current_scenario_progress
                        
                        # Ensure progress stays within bounds
                        overall_progress = min(90, max(5, overall_progress))
                        
                        # Update task status with detailed progress information
                        await manager.send_task_update(
                            task_id=task_id,
                            status="running",
                            message=f"Processing scenario: {scenario_name} - {status}",
                            data={
                                "progress": round(overall_progress, 1),
                                "current_scenario": scenario_name,
                                "scenario_status": status,
                                "scenario_progress": scenario_progress,
                                "scenario_index": scenario_index,
                                "scenario_count": scenario_count,
                                "completed_scenarios": scenario_index - 1,
                                "scenario_details": details or {},
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    
                    # Run stress tests with progress tracking
                    with stress_tracker.track("stress_testing"):
                        start_time = time.time()
                        
                        # Update status - about to run tests
                        await scenario_progress_callback("base_case", "preparing")
                        
                        # Track worker utilization
                        if request.run_parallel:
                            stress_metrics[user_id].track_worker_utilization(
                                request.max_workers, 
                                utilization_percent=90.0  # Estimate
                            )
                        
                        # Run the stress tests with progress tracking
                        results = await stress_tester.run_stress_tests(
                            request=request.request,
                            user_id=user_id,
                            scenario_names=request.scenario_names,
                            custom_scenarios=request.custom_scenarios,
                            use_cache=use_cache,
                            parallel=request.run_parallel,
                            max_workers=request.max_workers,
                            progress_callback=scenario_progress_callback
                        )
                        
                        # Update status - generating report
                        await manager.send_task_update(
                            task_id=task_id,
                            status="running",
                            message="Generating stress test report",
                            data={
                                "progress": 90,
                                "scenarios_completed": len(results),
                                "current_scenario": "report_generation",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        # Generate report if requested
                        report = None
                        if request.generate_report:
                            with stress_tracker.track("report_generation"):
                                # Update status - generating report
                                await manager.send_task_update(
                                    task_id=task_id,
                                    status="running",
                                    message="Generating comprehensive stress test report",
                                    data={
                                        "progress": 95,
                                        "scenarios_completed": len(results),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )
                                
                                report = await stress_tester.generate_stress_test_report(
                                    results=results,
                                    output_format="json"
                                )
                        
                        # Calculate total execution time
                        execution_time = time.time() - start_time
                        
                        # Track execution metrics
                        stress_metrics[user_id].track_execution_duration(
                            duration=execution_time,
                            parallel=request.run_parallel,
                            scenario_count=len(results)
                        )
                        
                        # Prepare response
                        response = StressTestResults(
                            pool_name=request.request.pool.pool_name,
                            analysis_date=request.request.analysis_date.isoformat() if request.request.analysis_date else "",
                            execution_time=execution_time,
                            scenario_results=results,
                            report=report,
                            performance_metrics=stress_tracker.get_metrics()
                        )
                        
                        # Cache the results for future retrieval
                        try:
                            await redis_manager.set(
                                key=task_id,
                                value=response.model_dump_json(),
                                ttl=settings.STRESS_TEST_RESULTS_TTL_SECONDS or 86400  # Default to 1 day
                            )
                        except Exception as e:
                            logger.warning(f"Task {task_id}: Failed to cache results: {str(e)}")
                        
                        # Save results overview without full cashflow details to reduce size
                        results_overview = {}
                        for scenario_name, result in results.items():
                            # Extract just the summary metrics, not full cashflows
                            results_overview[scenario_name] = {
                                "npv": result.metrics.npv if result.metrics else None,
                                "duration": result.metrics.duration if result.metrics else None,
                                "yield_value": result.metrics.yield_value if result.metrics else None,
                                "irr": result.metrics.irr if result.metrics else None,
                                "status": "completed"
                            }
                        
                        # Send completed status with results overview
                        await manager.send_task_update(
                            task_id=task_id,
                            status="completed",
                            message="Stress tests completed successfully",
                            data={
                                "progress": 100,
                                "scenarios_completed": len(results),
                                "execution_time": execution_time,
                                "completion_time": datetime.now().isoformat(),
                                "results_overview": results_overview,
                                "performance_metrics": stress_tracker.get_metrics(),
                                "pool_name": request.request.pool.pool_name,
                                "scenario_count": len(results),
                                # Include metadata for the full results retrieval endpoint
                                "results_available_at": f"/api/v1/stress-testing/results/{task_id}"
                            }
                        )
                    
            except asyncio.CancelledError:
                logger.warning(f"Task {task_id}: Background task was cancelled")
                stress_metrics[user_id].track_error("cancelled")
                await manager.send_task_update(
                    task_id=task_id,
                    status="cancelled",
                    message="Task was cancelled",
                    data={
                        "pool_name": request.request.pool.pool_name,
                        "cancel_time": datetime.now().isoformat(),
                        "progress": 0,
                        "current_scenario": current_scenario,
                        "scenario_index": scenario_index,
                        "scenario_count": scenario_count,
                        "performance_metrics": stress_tracker.get_metrics()
                    }
                )
                
            except Exception as e:
                logger.exception(f"Task {task_id}: Error in background stress test: {str(e)}")
                stress_metrics[user_id].track_error(type(e).__name__)
                
                # Gather exception details for better diagnostics
                error_type = type(e).__name__
                error_traceback = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                
                # Send error status with detailed diagnostic information
                await manager.send_task_update(
                    task_id=task_id,
                    status="error",
                    message=f"Error running stress tests: {str(e)}",
                    data={
                        "error_type": error_type,
                        "error_traceback": error_traceback,
                        "pool_name": request.request.pool.pool_name,
                        "error_time": datetime.now().isoformat(),
                        "progress": -1,  # Use -1 to indicate error
                        "current_scenario": current_scenario,
                        "scenario_index": scenario_index,
                        "scenario_count": scenario_count,
                        "performance_metrics": stress_tracker.get_metrics()
                    }
                )
            finally:
                # Decrement active tests counter
                stress_metrics[user_id].decrement_active_tests()
        
        # Queue background task
        background_tasks.add_task(run_stress_tests_bg)
        
        # Calculate estimated completion time based on pool size and scenario count
        asset_count = len(request.request.pool.assets)
        scenario_count = task_metadata["scenarios"]["total_count"]
        
        # More sophisticated completion time estimation based on asset complexity
        base_time = 5  # Minimum seconds
        # Assets have different processing costs based on type/complexity
        asset_cost_factor = 0.02  # Average seconds per asset per scenario
        parallel_discount = 0.7 if request.run_parallel else 1.0  # Parallel processing discount
        worker_factor = max(1, request.max_workers) / 4  # Worker efficiency factor (normalized to 4 workers)
        report_overhead = 5 if request.generate_report else 0  # Report generation overhead
        
        # Calculate estimated processing time
        estimated_seconds = (
            base_time + 
            (asset_count * scenario_count * asset_cost_factor * parallel_discount / min(1, worker_factor)) + 
            report_overhead
        )
        
        # Apply min/max bounds for reasonable estimates
        estimated_seconds = max(10, min(900, estimated_seconds))  # Between 10s and 15min
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        # Return task information with WebSocket details
        websocket_url = f"/ws/{user_id}?task_id={task_id}"
        
        # Add task to active tasks for user
        return StressTestTaskResponse(
            task_id=task_id,
            status="queued",
            message="Stress tests queued successfully",
            pool_name=request.request.pool.pool_name,
            scenario_count=scenario_count,
            websocket_url=websocket_url,
            estimated_completion_time=estimated_completion.isoformat()
        )
            
    except ValidationError as e:
        logger.error(f"Task {task_id}: Validation error: {str(e)}")
        stress_metrics[user_id].track_error("validation_error")
        stress_metrics[user_id].decrement_active_tests()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Task {task_id}: Unexpected error queueing stress tests: {str(e)}")
        stress_metrics[user_id].track_error(type(e).__name__)
        stress_metrics[user_id].decrement_active_tests()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error queueing stress tests: {str(e)}"
        )

@router.get(
    "/results/{task_id}",
    summary="Get stress test results by task ID",
    description="""
    Retrieves the full results of a previously completed stress test by its task ID.
    
    Results are cached in Redis for efficient retrieval and include all scenario 
    details, performance metrics, and the report if it was generated.
    
    If the results are no longer available in the cache (e.g., TTL expired), 
    an appropriate error message is returned.
    """,
    response_model=StressTestResults,
    responses={
        status.HTTP_200_OK: {"description": "Successful retrieval of stress test results"},
        status.HTTP_404_NOT_FOUND: {"description": "Results not found or expired"},
        status.HTTP_403_FORBIDDEN: {"description": "Unauthorized access"},
    }
)
async def get_stress_test_results(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get stress test results by task ID
    
    Args:
        task_id: The ID of the stress test task
        current_user: The authenticated user
        
    Returns:
        Complete stress test results
    """
    user_id = current_user.get("id", "anonymous")
    logger.info(f"User {user_id} requesting stress test results for task {task_id}")
    
    # Check permissions
    try:
        verify_permissions(current_user, PermissionLevel.ANALYST)
    except ValueError as e:
        logger.warning(f"Permission denied for user {user_id} to access task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    
    # Try to retrieve cached results
    try:
        cached_results = await redis_manager.get(task_id)
        
        if cached_results:
            logger.info(f"Retrieved cached results for task {task_id}")
            
            # Track cache hit in metrics if available
            if user_id in stress_metrics:
                # Simulate cache hit for stress test results
                CACHE_HITS.labels(key_type="stress_test_results").inc()
            
            # Parse the cached results
            return StressTestResults.parse_raw(cached_results)
        
        # Track cache miss
        if user_id in stress_metrics:
            CACHE_MISSES.labels(key_type="stress_test_results").inc()
        
        # Results not found in cache
        logger.warning(f"Task {task_id}: Results not found in cache")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results for task {task_id} not found or expired. "
                   f"Results are typically available for {settings.STRESS_TEST_RESULTS_TTL_SECONDS // 3600} hours after completion."
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.exception(f"Error retrieving results for task {task_id}: {str(e)}")
        
        # Track error
        if user_id in stress_metrics:
            stress_metrics[user_id].track_error(type(e).__name__)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stress test results: {str(e)}"
        )

@router.delete(
    "/results/{task_id}",
    summary="Delete stress test results",
    description="""
    Deletes stress test results from the cache.
    
    This endpoint is useful for cleaning up old results or removing sensitive data.
    Only users with admin permissions can delete results.
    """,
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Results successfully deleted"},
        status.HTTP_404_NOT_FOUND: {"description": "Results not found"},
        status.HTTP_403_FORBIDDEN: {"description": "Unauthorized access"},
    }
)
async def delete_stress_test_results(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete stress test results by task ID
    
    Args:
        task_id: The ID of the stress test task
        current_user: The authenticated user
    """
    user_id = current_user.get("id", "anonymous")
    logger.info(f"User {user_id} requesting deletion of stress test results for task {task_id}")
    
    # Check permissions - require admin for deletion
    try:
        verify_permissions(current_user, PermissionLevel.ADMIN)
    except ValueError as e:
        logger.warning(f"Permission denied for user {user_id} to delete task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin permissions required to delete results: {str(e)}"
        )
    
    # Delete the cached results
    try:
        deleted = await redis_manager.delete(task_id)
        
        if deleted:
            logger.info(f"Deleted results for task {task_id}")
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        
        # Results not found
        logger.warning(f"Task {task_id}: Results not found for deletion")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results for task {task_id} not found"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.exception(f"Error deleting results for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting stress test results: {str(e)}"
        )
