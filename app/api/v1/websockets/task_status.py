"""
WebSocket API for Real-time Task Status Updates

This module provides WebSocket endpoints for real-time updates on task status,
particularly for long-running Monte Carlo simulations. It includes:
- Connection management with authentication
- Task status subscription and updates
- Progress notifications for running tasks
- Completion and error notifications

The WebSocket API is fully decoupled from frontend logic and provides
only the backend endpoints for real-time communication.
"""
import logging
import json
import asyncio
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import verify_token
from app.services.supabase_service import SupabaseService
from app.models.monte_carlo import SimulationStatus
from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Initialize services
supabase_service = SupabaseService()

# Create router
router = APIRouter()

# Connection manager for WebSocket connections
class ConnectionManager:
    """
    Manages active WebSocket connections and broadcasting of messages
    """
    def __init__(self):
        """Initialize the connection manager"""
        # Map of user_id -> list of connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # Map of simulation_id -> set of user_ids subscribed
        self.simulation_subscriptions: Dict[str, Set[str]] = {}
        
        # Map of connection_id -> user_id for quick lookup
        self.connection_map: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        """
        Connect a WebSocket client
        
        Args:
            websocket: The WebSocket connection
            user_id: The ID of the authenticated user
            connection_id: Unique ID for this connection
        """
        await websocket.accept()
        
        # Initialize user's connections list if not exists
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        # Add the connection
        self.active_connections[user_id].append(websocket)
        self.connection_map[connection_id] = user_id
        
        logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
        
        # Send a welcome message
        await self.send_personal_message(
            {
                "type": "connection_established",
                "message": "Connected to task status WebSocket",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            },
            websocket
        )
    
    async def disconnect(self, websocket: WebSocket, connection_id: str):
        """
        Disconnect a WebSocket client
        
        Args:
            websocket: The WebSocket connection
            connection_id: Unique ID for this connection
        """
        # Get the user_id for this connection
        user_id = self.connection_map.get(connection_id)
        
        if not user_id:
            logger.warning(f"Disconnect requested for unknown connection: {connection_id}")
            return
        
        # Remove the connection
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            
            # If no more connections for user, clean up
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Clean up connection map
        if connection_id in self.connection_map:
            del self.connection_map[connection_id]
        
        # Clean up any subscriptions for this user
        for simulation_id, subscribers in list(self.simulation_subscriptions.items()):
            if user_id in subscribers:
                subscribers.remove(user_id)
                if not subscribers:
                    del self.simulation_subscriptions[simulation_id]
        
        logger.info(f"WebSocket connection closed: {connection_id} for user {user_id}")
    
    async def subscribe_to_simulation(self, simulation_id: str, user_id: str, websocket: WebSocket):
        """
        Subscribe a user to updates for a specific simulation
        
        Args:
            simulation_id: The ID of the simulation to subscribe to
            user_id: The ID of the user subscribing
            websocket: The WebSocket connection
        
        Returns:
            True if subscription was successful, False otherwise
        """
        # Check if the user has access to this simulation
        try:
            simulation = await supabase_service.get_item(
                table="monte_carlo_simulations",
                id=simulation_id
            )
            
            if not simulation:
                await self.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Simulation {simulation_id} not found",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
                return False
            
            # Check if user has access to this simulation
            if simulation.get("user_id") != user_id:
                # Check if user is admin (simplified - in a real implementation,
                # this would check a proper admin flag)
                is_admin = False  # Placeholder for admin check
                
                if not is_admin:
                    await self.send_personal_message(
                        {
                            "type": "error",
                            "message": "You do not have permission to access this simulation",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )
                    return False
            
            # Initialize subscriptions for this simulation if not exists
            if simulation_id not in self.simulation_subscriptions:
                self.simulation_subscriptions[simulation_id] = set()
            
            # Add user to subscribers
            self.simulation_subscriptions[simulation_id].add(user_id)
            
            # Send confirmation
            await self.send_personal_message(
                {
                    "type": "subscription_confirmed",
                    "simulation_id": simulation_id,
                    "message": f"Subscribed to updates for simulation {simulation_id}",
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            # Send initial status
            status = simulation.get("result", {}).get("status", SimulationStatus.PENDING)
            progress = simulation.get("result", {}).get("num_completed", 0)
            total = simulation.get("request", {}).get("num_simulations", 0)
            
            await self.send_personal_message(
                {
                    "type": "simulation_status",
                    "simulation_id": simulation_id,
                    "status": status,
                    "progress": progress,
                    "total": total,
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error subscribing to simulation {simulation_id}: {str(e)}")
            
            await self.send_personal_message(
                {
                    "type": "error",
                    "message": f"Error subscribing to simulation: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            return False
    
    async def unsubscribe_from_simulation(self, simulation_id: str, user_id: str, websocket: WebSocket):
        """
        Unsubscribe a user from updates for a specific simulation
        
        Args:
            simulation_id: The ID of the simulation to unsubscribe from
            user_id: The ID of the user unsubscribing
            websocket: The WebSocket connection
        """
        # Remove user from subscribers
        if simulation_id in self.simulation_subscriptions:
            if user_id in self.simulation_subscriptions[simulation_id]:
                self.simulation_subscriptions[simulation_id].remove(user_id)
                
                # If no more subscribers, clean up
                if not self.simulation_subscriptions[simulation_id]:
                    del self.simulation_subscriptions[simulation_id]
                
                # Send confirmation
                await self.send_personal_message(
                    {
                        "type": "unsubscription_confirmed",
                        "simulation_id": simulation_id,
                        "message": f"Unsubscribed from updates for simulation {simulation_id}",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
                
                return True
        
        # If we get here, the user wasn't subscribed
        await self.send_personal_message(
            {
                "type": "error",
                "message": f"Not subscribed to simulation {simulation_id}",
                "timestamp": datetime.now().isoformat()
            },
            websocket
        )
        
        return False
    
    async def broadcast_simulation_update(self, simulation_id: str, data: Dict[str, Any]):
        """
        Broadcast a simulation update to all subscribed users
        
        Args:
            simulation_id: The ID of the simulation being updated
            data: The update data to broadcast
        """
        if simulation_id not in self.simulation_subscriptions:
            return
        
        # Add simulation_id and timestamp to data
        broadcast_data = {
            **data,
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all subscribed users
        for user_id in self.simulation_subscriptions[simulation_id]:
            if user_id in self.active_connections:
                for connection in self.active_connections[user_id]:
                    try:
                        await connection.send_text(json.dumps(broadcast_data))
                    except Exception as e:
                        logger.error(f"Error sending to WebSocket for user {user_id}: {str(e)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection
        
        Args:
            message: The message to send
            websocket: The WebSocket connection to send to
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")

# Create a global connection manager instance
manager = ConnectionManager()

# Define WebSocket endpoint
@router.websocket("/ws/task-status")
async def task_status_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for task status updates
    
    This endpoint allows clients to:
    - Receive real-time updates on task status
    - Subscribe to specific simulations
    - Receive progress notifications
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Wait for authentication message
        authentication_timeout = settings.WEBSOCKET_AUTH_TIMEOUT_SECONDS or 30
        try:
            auth_data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=authentication_timeout
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1008, reason="Authentication timeout")
            return
        
        # Verify token
        token = auth_data.get("token")
        if not token:
            await websocket.close(code=1008, reason="Missing authentication token")
            return
        
        try:
            # Verify JWT token
            user_id = verify_token(token)
            
            if not user_id:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            await websocket.close(code=1008, reason="Authentication error")
            return
        
        # Accept the connection
        await manager.connect(websocket, user_id, connection_id)
        
        # Main message handling loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                
                # Process message based on type
                message_type = data.get("type")
                
                if message_type == "subscribe":
                    simulation_id = data.get("simulation_id")
                    if simulation_id:
                        await manager.subscribe_to_simulation(simulation_id, user_id, websocket)
                    else:
                        await manager.send_personal_message(
                            {
                                "type": "error",
                                "message": "Missing simulation_id for subscription",
                                "timestamp": datetime.now().isoformat()
                            },
                            websocket
                        )
                
                elif message_type == "unsubscribe":
                    simulation_id = data.get("simulation_id")
                    if simulation_id:
                        await manager.unsubscribe_from_simulation(simulation_id, user_id, websocket)
                    else:
                        await manager.send_personal_message(
                            {
                                "type": "error",
                                "message": "Missing simulation_id for unsubscription",
                                "timestamp": datetime.now().isoformat()
                            },
                            websocket
                        )
                
                elif message_type == "ping":
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )
                
                else:
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "message": f"Unknown message type: {message_type}",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )
            
            except WebSocketDisconnect:
                await manager.disconnect(websocket, connection_id)
                break
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Error processing message: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
    
    except WebSocketDisconnect:
        if user_id:
            await manager.disconnect(websocket, connection_id)
    
    except Exception as e:
        logger.error(f"Unhandled WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Server error")
        except:
            pass

# Function to be called by Celery tasks to broadcast status updates
async def broadcast_simulation_progress(simulation_id: str, status: str, progress: int, total: int):
    """
    Broadcast a simulation progress update
    
    Args:
        simulation_id: The ID of the simulation
        status: The status of the simulation
        progress: The number of completed simulations
        total: The total number of simulations
    """
    await manager.broadcast_simulation_update(
        simulation_id,
        {
            "type": "simulation_status",
            "status": status,
            "progress": progress,
            "total": total
        }
    )

# Function to be called when a simulation completes
async def broadcast_simulation_completion(simulation_id: str, result: Dict[str, Any]):
    """
    Broadcast a simulation completion notification
    
    Args:
        simulation_id: The ID of the simulation
        result: The result of the simulation (summary data only)
    """
    await manager.broadcast_simulation_update(
        simulation_id,
        {
            "type": "simulation_completed",
            "result": {
                k: v for k, v in result.items() 
                if k in ["status", "execution_time_seconds", "summary_statistics", "percentiles"]
            }
        }
    )

# Function to be called when a simulation fails
async def broadcast_simulation_error(simulation_id: str, error: str):
    """
    Broadcast a simulation error notification
    
    Args:
        simulation_id: The ID of the simulation
        error: The error message
    """
    await manager.broadcast_simulation_update(
        simulation_id,
        {
            "type": "simulation_error",
            "error": error
        }
    )
