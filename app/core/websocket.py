"""
WebSocket Manager Module for Real-time Status Updates

Production-ready implementation of WebSocket connections management with comprehensive
error handling, connection monitoring, task status persistence, and performance tracking.
"""
from fastapi import WebSocket
from typing import Dict, Set, Any, Optional, List, Union
import json
import asyncio
import time
from datetime import datetime, timedelta
import uuid
import traceback
from contextlib import asynccontextmanager

from app.core.logging import logger
from app.core.config import settings
from app.core.monitoring import PerformanceMetrics
from app.core.redis_cache import RedisCache, RedisConfig, get_redis_client

class ConnectionHealthMonitor:
    """
    Monitors the health of WebSocket connections with automatic
    cleanup of stale connections for improved reliability.
    """
    
    def __init__(self, manager: 'WebSocketManager'):
        """Initialize the health monitor"""
        self.manager = manager
        self.last_ping_time: Dict[str, Dict[str, float]] = {}
        self.ping_interval = 30.0  # seconds
        self.heartbeat_timeout = 120.0  # seconds
        self.running = False
        self.monitor_task = None
    
    async def start(self):
        """Start the health monitoring task"""
        if self.running:
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("WebSocket health monitor started")
    
    async def stop(self):
        """Stop the health monitoring task"""
        if not self.running:
            return
            
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop for connection health"""
        try:
            while self.running:
                await self._check_connections()
                await self._send_heartbeats()
                await asyncio.sleep(15)  # Check every 15 seconds
        except asyncio.CancelledError:
            logger.info("WebSocket health monitor cancelled")
        except Exception as e:
            logger.exception(f"Error in WebSocket health monitor: {str(e)}")
    
    async def _check_connections(self):
        """Check all connections for timeouts"""
        current_time = time.time()
        stale_connections = []
        
        # Find stale connections
        for user_id, connections in self.last_ping_time.items():
            for connection_id, last_time in connections.items():
                if current_time - last_time > self.heartbeat_timeout:
                    stale_connections.append((user_id, connection_id))
        
        # Clean up stale connections
        for user_id, connection_id in stale_connections:
            logger.warning(f"Closing stale WebSocket connection: user_id={user_id}, connection_id={connection_id}")
            await self.manager.disconnect(user_id, connection_id, reason="heartbeat_timeout")
    
    async def _send_heartbeats(self):
        """Send heartbeat pings to all active connections"""
        for user_id, connections in self.manager.active_connections.items():
            for connection_id, websocket in connections.items():
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update ping time
                    if user_id not in self.last_ping_time:
                        self.last_ping_time[user_id] = {}
                    self.last_ping_time[user_id][connection_id] = time.time()
                    
                except Exception as e:
                    logger.error(f"Error sending heartbeat to websocket: {str(e)}")
                    await self.manager.disconnect(user_id, connection_id, reason="heartbeat_failed")
    
    def update_connection_time(self, user_id: str, connection_id: str):
        """Update last ping time for a connection"""
        if user_id not in self.last_ping_time:
            self.last_ping_time[user_id] = {}
        self.last_ping_time[user_id][connection_id] = time.time()
    
    def remove_connection(self, user_id: str, connection_id: str):
        """Remove connection from tracking"""
        if user_id in self.last_ping_time:
            self.last_ping_time[user_id].pop(connection_id, None)
            if not self.last_ping_time[user_id]:
                del self.last_ping_time[user_id]

class TaskStatus:
    """
    Store and manage task status information with persistence
    to Redis for reliable recovery across service restarts.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    
    def __init__(self, task_id: str, status: str = "pending", message: str = "", data: Dict = None):
        """Initialize task status"""
        self.task_id = task_id
        self.status = status
        self.message = message
        self.data = data or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def update(self, status: str, message: str = "", data: Dict = None):
        """Update task status"""
        self.status = status
        
        if message:
            self.message = message
            
        if data:
            # Update data instead of replacing
            self.data.update(data)
            
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskStatus':
        """Create from dictionary"""
        task = cls(task_id=data["task_id"])
        task.status = data["status"]
        task.message = data["message"]
        task.data = data["data"]
        task.created_at = data["created_at"]
        task.updated_at = data["updated_at"]
        return task

class WebSocketManager:
    """
    Production-ready WebSocket connection manager with comprehensive error handling,
    connection monitoring, task status persistence, and performance tracking.
    """
    def __init__(self):
        """Initialize WebSocket manager with all required components"""
        # Map of user_id to set of connection_ids
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Map of task_id to task status
        self.task_statuses: Dict[str, TaskStatus] = {}
        # Map of task_id to set of user_ids
        self.task_subscriptions: Dict[str, Set[str]] = {}
        # Map of batch_id to set of user_ids
        self.batch_subscriptions: Dict[str, Set[str]] = {}
        # Map of connection_id to metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        # Health monitor
        self.health_monitor = ConnectionHealthMonitor(self)
        # Flag for redis cache status
        self._redis_available = False
        # Redis client for persistence
        self._redis_client = None
        # Performance metrics
        self._metrics = PerformanceMetrics("websocket_manager")
        # Initialize task cleanup
        self._task_cleanup_task = None
        # Websocket stats
        self.stats = {
            "total_messages_sent": 0,
            "total_messages_failed": 0,
            "total_connections": 0,
            "current_connections": 0,
            "total_tasks": 0
        }
    
    async def initialize(self):
        """Initialize the WebSocket manager"""
        # Initialize Redis for task persistence
        try:
            redis_config = RedisConfig(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                retry_on_timeout=True,
                ssl=settings.REDIS_SSL,
                ssl_cert_reqs=None if settings.REDIS_SSL_CERT_REQS == "none" else settings.REDIS_SSL_CERT_REQS
            )
            
            self._redis_client = await get_redis_client(redis_config)
            self._redis_available = True
            logger.info("WebSocket manager Redis connected")
            
            # Load existing task statuses from Redis
            await self._load_task_statuses()
        except Exception as e:
            logger.warning(f"WebSocket manager Redis connection failed: {str(e)}. Task persistence disabled.")
            self._redis_available = False
        
        # Start health monitor
        await self.health_monitor.start()
        
        # Start task cleanup
        self._task_cleanup_task = asyncio.create_task(self._clean_old_tasks_loop())
        
        logger.info("WebSocket manager initialized")
    
    async def shutdown(self):
        """Shutdown the WebSocket manager cleanly"""
        # Stop health monitor
        await self.health_monitor.stop()
        
        # Cancel task cleanup
        if self._task_cleanup_task:
            self._task_cleanup_task.cancel()
            try:
                await self._task_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for user_id, connections in list(self.active_connections.items()):
            for connection_id, websocket in list(connections.items()):
                try:
                    await websocket.close(code=1001, reason="Server shutdown")
                except Exception as e:
                    logger.error(f"Error closing websocket: {str(e)}")
            self.active_connections[user_id] = {}
        
        self.active_connections.clear()
        logger.info("WebSocket manager shutdown complete")
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str = None):
        """Connect a new WebSocket client with proper monitoring and error handling"""
        connection_id = connection_id or str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            if user_id not in self.active_connections:
                self.active_connections[user_id] = {}
                
            self.active_connections[user_id][connection_id] = websocket
            
            # Register with health monitor
            self.health_monitor.update_connection_time(user_id, connection_id)
            
            # Store metadata for debugging
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": datetime.now().isoformat(),
                "client_info": websocket.headers.get("user-agent", ""),
                "remote_addr": websocket.client.host if websocket.client else "unknown"
            }
            
            # Update stats
            self.stats["total_connections"] += 1
            self.stats["current_connections"] += 1
            
            logger.info(f"WebSocket connected: user_id={user_id}, connection_id={connection_id}")
            
            # Send welcome message with connection ID
            await websocket.send_json({
                "type": "connected",
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat()
            })
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {str(e)}")
            traceback.print_exc()
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass
            raise
    
    async def disconnect(self, user_id: str, connection_id: str, reason: str = "client_disconnected"):
        """Disconnect a WebSocket client with cleanup"""
        # Close socket if it's still open
        if user_id in self.active_connections and connection_id in self.active_connections[user_id]:
            websocket = self.active_connections[user_id][connection_id]
            try:
                await websocket.close(code=1000, reason=reason)
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {str(e)}")
        
        # Remove from active connections
        if user_id in self.active_connections:
            self.active_connections[user_id].pop(connection_id, None)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from health monitor
        self.health_monitor.remove_connection(user_id, connection_id)
        
        # Remove metadata
        self.connection_metadata.pop(connection_id, None)
        
        # Update stats
        self.stats["current_connections"] = max(0, self.stats["current_connections"] - 1)
        
        logger.info(f"WebSocket disconnected: user_id={user_id}, connection_id={connection_id}, reason={reason}")
    
    @asynccontextmanager
    async def connection_context(self, websocket: WebSocket, user_id: str):
        """Context manager for handling WebSocket connections"""
        connection_id = None
        try:
            connection_id = await self.connect(websocket, user_id)
            yield connection_id
        finally:
            if connection_id:
                await self.disconnect(user_id, connection_id)
    
    async def broadcast_to_user(self, user_id: str, message: Any):
        """Send a message to all connections of a user with proper error handling"""
        if user_id not in self.active_connections:
            return
            
        with self._metrics.track(f"broadcast_to_user_{user_id}"):
            disconnected = []
            successful = 0
            
            message_type = message.get("type", "unknown") if isinstance(message, dict) else "raw"
            
            for connection_id, websocket in self.active_connections[user_id].items():
                try:
                    if isinstance(message, dict):
                        await websocket.send_json(message)
                    else:
                        await websocket.send_text(str(message))
                        
                    successful += 1
                    self.stats["total_messages_sent"] += 1
                    
                except Exception as e:
                    self.stats["total_messages_failed"] += 1
                    logger.error(f"Error sending message to websocket: {str(e)}")
                    disconnected.append((user_id, connection_id))
            
            # Clean up disconnected websockets
            for user_id, connection_id in disconnected:
                await self.disconnect(user_id, connection_id, reason="send_failed")
            
            logger.debug(f"Broadcast to user {user_id}: {message_type}, {successful} successful, {len(disconnected)} failed")
    
    async def subscribe_to_task(self, task_id: str, user_id: str):
        """Subscribe a user to task updates with status delivery"""
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
            
        self.task_subscriptions[task_id].add(user_id)
        logger.info(f"User subscribed to task: user_id={user_id}, task_id={task_id}")
        
        # If we already have status for this task, send it immediately
        task_status = await self.get_task_status(task_id)
        if task_status:
            await self.broadcast_to_user(user_id, {
                "type": "task_update",
                "task_id": task_id,
                "status": task_status
            })
    
    async def send_task_update(self, task_id: str, status: str, message: str = "", data: Dict = None):
        """
        Update task status and broadcast to subscribers with proper persistence
        
        Args:
            task_id: Task identifier
            status: Status string (pending, running, completed, error, cancelled)
            message: Message describing current status
            data: Additional data for the status update
        """
        with self._metrics.track(f"send_task_update_{task_id}"):
            # Create or update task status
            if task_id not in self.task_statuses:
                self.task_statuses[task_id] = TaskStatus(task_id=task_id)
                self.stats["total_tasks"] += 1
                
            # Update task status
            self.task_statuses[task_id].update(status=status, message=message, data=data)
            
            # Persist to Redis if available
            if self._redis_available and self._redis_client:
                try:
                    key = f"task_status:{task_id}"
                    # Save with TTL (24 hours)
                    await self._redis_client.set(
                        key, 
                        json.dumps(self.task_statuses[task_id].to_dict()),
                        ex=86400  # 24 hours
                    )
                except Exception as e:
                    logger.error(f"Error persisting task status to Redis: {str(e)}")
            
            # Broadcast to subscribers
            if task_id in self.task_subscriptions:
                task_status = self.task_statuses[task_id].to_dict()
                message = {
                    "type": "task_update",
                    "task_id": task_id,
                    "status": task_status
                }
                
                for user_id in self.task_subscriptions[task_id]:
                    await self.broadcast_to_user(user_id, message)
            
            # Clean up completed tasks eventually
            if status in [TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.CANCELLED]:
                logger.info(f"Task {task_id} reached terminal state: {status}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current task status with Redis fallback
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status as dictionary or None if not found
        """
        # Check in-memory cache first
        if task_id in self.task_statuses:
            return self.task_statuses[task_id].to_dict()
            
        # Try to retrieve from Redis if not found in memory
        if self._redis_available and self._redis_client:
            try:
                key = f"task_status:{task_id}"
                data = await self._redis_client.get(key)
                
                if data:
                    # Parse and cache in memory
                    task_status = TaskStatus.from_dict(json.loads(data))
                    self.task_statuses[task_id] = task_status
                    return task_status.to_dict()
            except Exception as e:
                logger.error(f"Error retrieving task status from Redis: {str(e)}")
        
        return None
    
    async def _load_task_statuses(self):
        """Load task statuses from Redis on startup"""
        if not self._redis_available or not self._redis_client:
            return
            
        try:
            # Scan for task status keys
            cursor = 0
            prefix = "task_status:"
            
            while True:
                cursor, keys = await self._redis_client.scan(cursor=cursor, match=f"{prefix}*", count=100)
                
                for key in keys:
                    try:
                        task_id = key[len(prefix):]
                        data = await self._redis_client.get(key)
                        
                        if data:
                            task_status = TaskStatus.from_dict(json.loads(data))
                            self.task_statuses[task_id] = task_status
                            
                            # Only keep non-terminal statuses or recent ones
                            if task_status.status in [TaskStatus.PENDING, TaskStatus.RUNNING] or \
                               self._is_recent_task(task_status):
                                self.task_statuses[task_id] = task_status
                    except Exception as e:
                        logger.error(f"Error loading task status {key}: {str(e)}")
                
                if cursor == 0:
                    break
            
            logger.info(f"Loaded {len(self.task_statuses)} task statuses from Redis")
            
        except Exception as e:
            logger.error(f"Error loading task statuses from Redis: {str(e)}")
    
    def _is_recent_task(self, task_status: TaskStatus) -> bool:
        """Check if task status is recent (less than 1 hour old)"""
        try:
            updated_at = datetime.fromisoformat(task_status.updated_at)
            age = datetime.now() - updated_at
            return age < timedelta(hours=1)
        except:
            return False
    
    async def _clean_old_tasks_loop(self):
        """Periodically clean up old tasks"""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self._clean_old_tasks()
        except asyncio.CancelledError:
            logger.info("Task cleanup cancelled")
        except Exception as e:
            logger.error(f"Error in task cleanup loop: {str(e)}")
    
    async def _clean_old_tasks(self):
        """Clean up old task statuses"""
        now = datetime.now()
        to_remove = []
        
        for task_id, task_status in self.task_statuses.items():
            try:
                # Keep running tasks
                if task_status.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    continue
                    
                # Remove older terminal tasks
                updated_at = datetime.fromisoformat(task_status.updated_at)
                age = now - updated_at
                
                # Keep for 24 hours if completed, 48 if error
                max_age = timedelta(hours=48 if task_status.status == TaskStatus.ERROR else 24)
                
                if age > max_age:
                    to_remove.append(task_id)
            except Exception as e:
                logger.error(f"Error checking task age for {task_id}: {str(e)}")
        
        # Remove old tasks
        for task_id in to_remove:
            self.task_statuses.pop(task_id, None)
            self.task_subscriptions.pop(task_id, None)
            
            # Remove from Redis if available
            if self._redis_available and self._redis_client:
                try:
                    key = f"task_status:{task_id}"
                    await self._redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Error removing task status from Redis: {str(e)}")
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old task statuses")
    
    async def broadcast_error(self, user_id: str, error: str, context: Dict = None):
        """Broadcast error message to user with proper error context"""
        message = {
            "type": "error",
            "error": error,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_user(user_id, message)
    
    async def broadcast_notification(self, user_id: str, message: str, notification_type: str = "info", data: Dict = None):
        """Broadcast notification message to user"""
        notification = {
            "type": "notification",
            "notification_type": notification_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_user(user_id, notification)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        stats = self.stats.copy()
        stats["active_users"] = len(self.active_connections)
        stats["active_tasks"] = sum(1 for status in self.task_statuses.values() 
                                  if status.status in [TaskStatus.PENDING, TaskStatus.RUNNING])
        stats["total_task_subscriptions"] = sum(len(subs) for subs in self.task_subscriptions.values())
        return stats

# Create singleton instance
manager = WebSocketManager()
