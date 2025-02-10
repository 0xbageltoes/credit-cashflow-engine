from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        # Store active connections: {user_id: {connection_id: WebSocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Store task subscriptions: {task_id: [user_ids]}
        self.task_subscribers: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        self.active_connections[user_id][connection_id] = websocket

    def disconnect(self, user_id: str, connection_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].pop(connection_id, None)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Clean up task subscriptions
        for task_id in list(self.task_subscribers.keys()):
            if user_id in self.task_subscribers[task_id]:
                self.task_subscribers[task_id].remove(user_id)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]

    async def subscribe_to_task(self, task_id: str, user_id: str):
        """Subscribe user to task updates"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = []
        if user_id not in self.task_subscribers[task_id]:
            self.task_subscribers[task_id].append(user_id)

    async def broadcast_task_update(self, task_id: str, message: Dict):
        """Broadcast task update to all subscribed users"""
        if task_id in self.task_subscribers:
            for user_id in self.task_subscribers[task_id]:
                if user_id in self.active_connections:
                    for websocket in self.active_connections[user_id].values():
                        try:
                            await websocket.send_json({
                                "task_id": task_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                **message
                            })
                        except Exception as e:
                            print(f"Error sending message to user {user_id}: {str(e)}")

    async def send_personal_message(self, message: Dict, user_id: str):
        """Send message to specific user's all connections"""
        if user_id in self.active_connections:
            for websocket in self.active_connections[user_id].values():
                try:
                    await websocket.send_json({
                        "timestamp": datetime.utcnow().isoformat(),
                        **message
                    })
                except Exception as e:
                    print(f"Error sending personal message to user {user_id}: {str(e)}")

manager = ConnectionManager()
