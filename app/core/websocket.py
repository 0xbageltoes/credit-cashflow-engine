from fastapi import WebSocket
from typing import Dict, Set, Any
import json
import asyncio
from app.core.logging import logger

class WebSocketManager:
    def __init__(self):
        # Map of user_id to set of connection_ids
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Map of task_id to set of user_ids
        self.task_subscriptions: Dict[str, Set[str]] = {}
        # Map of batch_id to set of user_ids
        self.batch_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        self.active_connections[user_id][connection_id] = websocket
        logger.info(f"WebSocket connected: user_id={user_id}, connection_id={connection_id}")
    
    def disconnect(self, user_id: str, connection_id: str):
        """Disconnect a WebSocket client"""
        if user_id in self.active_connections:
            self.active_connections[user_id].pop(connection_id, None)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: user_id={user_id}, connection_id={connection_id}")
    
    async def broadcast_to_user(self, user_id: str, message: Any):
        """Send a message to all connections of a user"""
        if user_id in self.active_connections:
            disconnected = []
            for connection_id, websocket in self.active_connections[user_id].items():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to websocket: {str(e)}")
                    disconnected.append((user_id, connection_id))
            
            # Clean up disconnected websockets
            for user_id, connection_id in disconnected:
                self.disconnect(user_id, connection_id)
    
    async def subscribe_to_task(self, task_id: str, user_id: str):
        """Subscribe a user to task updates"""
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        self.task_subscriptions[task_id].add(user_id)
        logger.info(f"User subscribed to task: user_id={user_id}, task_id={task_id}")
    
    async def subscribe_to_batch(self, batch_id: str, user_id: str):
        """Subscribe a user to batch updates"""
        if batch_id not in self.batch_subscriptions:
            self.batch_subscriptions[batch_id] = set()
        self.batch_subscriptions[batch_id].add(user_id)
        logger.info(f"User subscribed to batch: user_id={user_id}, batch_id={batch_id}")
    
    async def broadcast_task_update(self, task_id: str, status: Dict):
        """Broadcast task status update to subscribed users"""
        if task_id in self.task_subscriptions:
            message = {
                "type": "task_update",
                "task_id": task_id,
                "status": status
            }
            for user_id in self.task_subscriptions[task_id]:
                await self.broadcast_to_user(user_id, message)
    
    async def broadcast_batch_update(self, batch_id: str, status: Dict):
        """Broadcast batch status update to subscribed users"""
        if batch_id in self.batch_subscriptions:
            message = {
                "type": "batch_update",
                "batch_id": batch_id,
                "status": status
            }
            for user_id in self.batch_subscriptions[batch_id]:
                await self.broadcast_to_user(user_id, message)
    
    async def broadcast_error(self, user_id: str, error: str, context: Dict = None):
        """Broadcast error message to user"""
        message = {
            "type": "error",
            "error": error,
            "context": context or {}
        }
        await self.broadcast_to_user(user_id, message)
    
    def cleanup_task(self, task_id: str):
        """Clean up task subscriptions"""
        self.task_subscriptions.pop(task_id, None)
    
    def cleanup_batch(self, batch_id: str):
        """Clean up batch subscriptions"""
        self.batch_subscriptions.pop(batch_id, None)

manager = WebSocketManager()
