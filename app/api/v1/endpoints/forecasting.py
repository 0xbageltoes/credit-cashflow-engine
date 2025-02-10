from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, List
import uuid
from app.core.websocket import manager
from app.models.cashflow import CashflowForecastRequest
from app.tasks.forecasting import generate_forecast, run_stress_test
from app.core.auth import get_current_user
from app.core.cache import SQLiteCache

router = APIRouter()
cache = SQLiteCache()

# Mock user for testing
mock_user = {"id": "test-user-id", "email": "test@example.com"}

@router.post("/forecast/async")
async def create_forecast(
    request: CashflowForecastRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Queue an asynchronous forecast calculation"""
    task = generate_forecast.delay(request.dict(), current_user["id"])
    return {"task_id": task.id}

@router.post("/forecast/stress-test")
async def create_stress_test(
    request: CashflowForecastRequest,
    scenarios: List[Dict],
    current_user: Dict = Depends(get_current_user)
):
    """Queue a stress test calculation"""
    task = run_stress_test.delay(request.dict(), current_user["id"], scenarios)
    return {"task_id": task.id}

@router.get("/forecast/{task_id}/status")
async def get_forecast_status(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get the status of a forecast calculation"""
    status = cache.get(f"task_status:{task_id}")
    if not status:
        task = generate_forecast.AsyncResult(task_id)
        status = task.status
    return {"status": status}

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    connection_id = str(uuid.uuid4())
    try:
        await manager.connect(websocket, user_id, connection_id)
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "subscribe" and "task_id" in data:
                await manager.subscribe_to_task(data["task_id"], user_id)
                await websocket.send_json({
                    "type": "subscribed",
                    "task_id": data["task_id"]
                })
    except WebSocketDisconnect:
        manager.disconnect(user_id, connection_id)
