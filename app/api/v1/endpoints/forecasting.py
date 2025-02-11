from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import uuid
import asyncio
from app.core.websocket import manager
from app.models.cashflow import CashflowForecastRequest, BatchForecastRequest
from app.tasks.forecasting import generate_forecast, run_stress_test
from app.core.auth import get_current_user
from app.core.redis_cache import RedisCache
from app.core.logging import logger

router = APIRouter()
cache = RedisCache()

@router.post("/forecast/async")
async def create_forecast(
    request: CashflowForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Queue an asynchronous forecast calculation"""
    try:
        task = generate_forecast.delay(request.dict(), current_user["id"])
        cache.set_task_status(task.id, {"status": "PENDING", "user_id": current_user["id"]})
        background_tasks.add_task(update_task_status, task.id)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}", extra={
            "user_id": current_user["id"],
            "request": request.dict()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/batch")
async def create_batch_forecast(
    request: BatchForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Queue multiple forecasts for batch processing"""
    try:
        tasks = []
        for forecast in request.forecasts:
            task = generate_forecast.delay(forecast.dict(), current_user["id"])
            tasks.append(task.id)
            cache.set_task_status(task.id, {"status": "PENDING", "user_id": current_user["id"]})
            background_tasks.add_task(update_task_status, task.id)
        
        # Store batch task group
        batch_id = str(uuid.uuid4())
        cache.set(f"batch:{batch_id}", {
            "task_ids": tasks,
            "user_id": current_user["id"],
            "total": len(tasks),
            "completed": 0
        })
        
        return {"batch_id": batch_id, "task_ids": tasks}
    except Exception as e:
        logger.error(f"Error creating batch forecast: {str(e)}", extra={
            "user_id": current_user["id"],
            "request": request.dict()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/stress-test")
async def create_stress_test(
    request: CashflowForecastRequest,
    scenarios: List[Dict],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Queue a stress test calculation"""
    try:
        task = run_stress_test.delay(request.dict(), current_user["id"], scenarios)
        cache.set_task_status(task.id, {
            "status": "PENDING",
            "user_id": current_user["id"],
            "scenario_count": len(scenarios)
        })
        background_tasks.add_task(update_task_status, task.id)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error creating stress test: {str(e)}", extra={
            "user_id": current_user["id"],
            "request": request.dict(),
            "scenarios": len(scenarios)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/{task_id}/status")
async def get_forecast_status(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get the status of a forecast calculation"""
    try:
        status = cache.get_task_status(task_id)
        if not status:
            task = generate_forecast.AsyncResult(task_id)
            status = {"status": task.status}
            cache.set_task_status(task_id, status)
        return status
    except Exception as e:
        logger.error(f"Error getting forecast status: {str(e)}", extra={
            "user_id": current_user["id"],
            "task_id": task_id
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get the status of a batch forecast"""
    try:
        batch = cache.get(f"batch:{batch_id}")
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        if batch["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to view this batch")
        
        statuses = []
        completed = 0
        for task_id in batch["task_ids"]:
            status = cache.get_task_status(task_id)
            if status and status["status"] in ["SUCCESS", "FAILURE"]:
                completed += 1
            statuses.append({"task_id": task_id, "status": status})
        
        # Update completion count
        batch["completed"] = completed
        cache.set(f"batch:{batch_id}", batch)
        
        return {
            "total": batch["total"],
            "completed": completed,
            "tasks": statuses
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}", extra={
            "user_id": current_user["id"],
            "batch_id": batch_id
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    connection_id = str(uuid.uuid4())
    try:
        await manager.connect(websocket, user_id, connection_id)
        await manager.broadcast_to_user(user_id, {"type": "connection", "status": "connected"})
        
        while True:
            data = await websocket.receive_text()
            await manager.broadcast_to_user(user_id, {"type": "message", "data": data})
            
    except WebSocketDisconnect:
        manager.disconnect(user_id, connection_id)
        cache.cleanup_websocket(connection_id)
        await manager.broadcast_to_user(user_id, {"type": "connection", "status": "disconnected"})

async def update_task_status(task_id: str):
    """Background task to update task status"""
    task = generate_forecast.AsyncResult(task_id)
    status = cache.get_task_status(task_id)
    
    if not status:
        return
    
    user_id = status.get("user_id")
    if not user_id:
        return
    
    while not task.ready():
        await asyncio.sleep(1)
    
    if task.successful():
        result = task.get()
        status["status"] = "SUCCESS"
        status["result"] = result
    else:
        status["status"] = "FAILURE"
        status["error"] = str(task.result)
    
    cache.set_task_status(task_id, status)
    
    # Send WebSocket update if user is connected
    if user_id in manager.active_connections:
        await manager.broadcast_to_user(user_id, {
            "type": "task_update",
            "task_id": task_id,
            "status": status
        })
