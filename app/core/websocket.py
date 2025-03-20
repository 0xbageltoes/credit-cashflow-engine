"""
WebSocket Manager for Real-time Communication

This module provides a WebSocket connection manager that handles connections,
broadcasting messages, and disconnections for real-time communication.
"""

from fastapi import WebSocket
from typing import Dict, List, Any, Optional, Set
import logging
import json
import asyncio
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    WebSocket connection manager for handling multiple client connections
    
    This class provides methods to:
    - Register new WebSocket connections
    - Remove connections when clients disconnect
    - Broadcast messages to all connected clients or specific users
    - Send targeted messages to specific connections
    """
    
    def __init__(self):
        """Initialize the connection manager"""
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task = None
    
    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """
        Accept a new WebSocket connection and register it
        
        Args:
            websocket: The WebSocket connection to add
            user_id: Unique identifier for the user
        """
        await websocket.accept()
        
        # Add the connection to the active connections for this user
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connection added for user {user_id}")
        
        # Start the broadcast task if not running
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_messages())
    
    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """
        Remove a WebSocket connection when a client disconnects
        
        Args:
            websocket: The WebSocket connection to remove
            user_id: Unique identifier for the user
        """
        # Remove the connection from the user's connections
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                logger.info(f"WebSocket connection removed for user {user_id}")
                
                # Remove the user entirely if no connections remain
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                    logger.info(f"User {user_id} has no more active connections")
            except ValueError:
                # Connection was already removed
                logger.warning(f"Attempted to remove non-existent connection for user {user_id}")
    
    async def broadcast(self, message: Any, user_ids: Optional[List[str]] = None) -> None:
        """
        Add a message to the broadcast queue
        
        Args:
            message: The message to broadcast (will be converted to JSON)
            user_ids: Optional list of user IDs to send the message to.
                       If None, the message will be sent to all users.
        """
        await self._message_queue.put({
            'message': message,
            'user_ids': user_ids,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _broadcast_messages(self) -> None:
        """Background task to send queued messages to clients"""
        logger.info("Starting WebSocket broadcast task")
        
        while True:
            try:
                # Get the next message from the queue
                item = await self._message_queue.get()
                message = item['message']
                user_ids = item['user_ids']
                
                # Convert to JSON string if not already a string
                if not isinstance(message, str):
                    message = json.dumps(message)
                
                # Send to specific users or all users
                if user_ids:
                    # Send to specific users
                    for user_id in user_ids:
                        await self._send_to_user(user_id, message)
                else:
                    # Send to all users
                    await self._send_to_all(message)
                
                # Mark task as done
                self._message_queue.task_done()
            except asyncio.CancelledError:
                logger.info("WebSocket broadcast task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket broadcast task: {e}")
    
    async def _send_to_user(self, user_id: str, message: str) -> None:
        """
        Send a message to a specific user
        
        Args:
            user_id: The user to send the message to
            message: The message to send
        """
        if user_id in self.active_connections:
            disconnected = []
            
            for i, websocket in enumerate(self.active_connections[user_id]):
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    # Connection is likely closed, mark for removal
                    logger.warning(f"Error sending to user {user_id}: {e}")
                    disconnected.append(i)
            
            # Remove disconnected websockets (in reverse order to preserve indices)
            for i in sorted(disconnected, reverse=True):
                del self.active_connections[user_id][i]
            
            # Remove user if no connections remain
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def _send_to_all(self, message: str) -> None:
        """
        Send a message to all connected users
        
        Args:
            message: The message to send
        """
        # Create a copy of user_ids to avoid modification during iteration
        user_ids = list(self.active_connections.keys())
        
        for user_id in user_ids:
            await self._send_to_user(user_id, message)
    
    def get_active_users(self) -> Set[str]:
        """
        Get the set of currently connected user IDs
        
        Returns:
            Set of user IDs with active connections
        """
        return set(self.active_connections.keys())
    
    def user_connection_count(self, user_id: str) -> int:
        """
        Get the number of active connections for a user
        
        Args:
            user_id: The user to check
            
        Returns:
            Number of active connections
        """
        if user_id in self.active_connections:
            return len(self.active_connections[user_id])
        return 0

# Create a global WebSocket manager instance
manager = WebSocketManager()