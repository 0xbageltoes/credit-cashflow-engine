"""
WebSocket API Router Package

This package contains WebSocket endpoints for real-time communication.
"""
from fastapi import APIRouter

from app.api.v1.websockets.task_status import router as task_status_router

# Create WebSocket API router
websocket_router = APIRouter()

# Include WebSocket endpoint routers
websocket_router.include_router(task_status_router)
