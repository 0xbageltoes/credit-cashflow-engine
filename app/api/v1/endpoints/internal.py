"""
Internal API Routes

These routes are for internal microservice communication and should not be
exposed publicly. They require an internal API key for authentication.
"""
import logging
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import verify_internal_api_key
from app.api.v1.websockets.task_status import (
    broadcast_simulation_progress,
    broadcast_simulation_completion,
    broadcast_simulation_error
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router for internal endpoints
router = APIRouter()

# Models
class ProgressUpdateRequest(BaseModel):
    """Request model for progress update broadcasting"""
    simulation_id: str
    status: str
    progress: int
    total: int
    api_key: str

class CompletionUpdateRequest(BaseModel):
    """Request model for completion update broadcasting"""
    simulation_id: str
    result: Dict[str, Any]
    api_key: str

class ErrorUpdateRequest(BaseModel):
    """Request model for error update broadcasting"""
    simulation_id: str
    error: str
    api_key: str

class ApiResponse(BaseModel):
    """Standard API response"""
    success: bool
    message: str

# Define API key dependency
def verify_api_key(api_key: str):
    """Verify internal API key"""
    if not verify_internal_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

@router.post(
    "/websocket/broadcast-progress",
    response_model=ApiResponse,
    description="Broadcast simulation progress update via WebSocket"
)
async def broadcast_progress(
    request: ProgressUpdateRequest = Body(...)
):
    """
    Broadcast simulation progress update via WebSocket
    
    This endpoint is for internal use only and requires an API key.
    It is used by Celery tasks to broadcast progress updates.
    
    Args:
        request: The progress update request
    
    Returns:
        API response
    """
    # Verify API key
    verify_api_key(request.api_key)
    
    try:
        # Broadcast the update
        await broadcast_simulation_progress(
            request.simulation_id,
            request.status,
            request.progress,
            request.total
        )
        
        return {
            "success": True,
            "message": "Progress update broadcast successfully"
        }
    
    except Exception as e:
        logger.error(f"Error broadcasting progress update: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error broadcasting progress update: {str(e)}"
        )

@router.post(
    "/websocket/broadcast-completion",
    response_model=ApiResponse,
    description="Broadcast simulation completion via WebSocket"
)
async def broadcast_completion(
    request: CompletionUpdateRequest = Body(...)
):
    """
    Broadcast simulation completion via WebSocket
    
    This endpoint is for internal use only and requires an API key.
    It is used by Celery tasks to broadcast completion updates.
    
    Args:
        request: The completion update request
    
    Returns:
        API response
    """
    # Verify API key
    verify_api_key(request.api_key)
    
    try:
        # Broadcast the update
        await broadcast_simulation_completion(
            request.simulation_id,
            request.result
        )
        
        return {
            "success": True,
            "message": "Completion update broadcast successfully"
        }
    
    except Exception as e:
        logger.error(f"Error broadcasting completion update: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error broadcasting completion update: {str(e)}"
        )

@router.post(
    "/websocket/broadcast-error",
    response_model=ApiResponse,
    description="Broadcast simulation error via WebSocket"
)
async def broadcast_error(
    request: ErrorUpdateRequest = Body(...)
):
    """
    Broadcast simulation error via WebSocket
    
    This endpoint is for internal use only and requires an API key.
    It is used by Celery tasks to broadcast error updates.
    
    Args:
        request: The error update request
    
    Returns:
        API response
    """
    # Verify API key
    verify_api_key(request.api_key)
    
    try:
        # Broadcast the update
        await broadcast_simulation_error(
            request.simulation_id,
            request.error
        )
        
        return {
            "success": True,
            "message": "Error update broadcast successfully"
        }
    
    except Exception as e:
        logger.error(f"Error broadcasting error update: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error broadcasting error update: {str(e)}"
        )
