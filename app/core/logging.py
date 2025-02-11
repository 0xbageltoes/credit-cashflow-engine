import logging
import json
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

# Configure logging
logger = logging.getLogger("cashflow_engine")
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Get request body
        body = await self._get_request_body(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request details
        log_data = {
            "request": {
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
                "body": body
            },
            "response": {
                "status_code": response.status_code,
                "duration": duration
            }
        }
        
        # Log at appropriate level based on status code
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
        
        return response
    
    async def _get_request_body(self, request: Request) -> dict:
        """Get request body without consuming it"""
        body = {}
        
        if request.method in ["POST", "PUT", "PATCH"]:
            # Save request body state
            receive_ = request.receive
            
            async def receive() -> Message:
                message = await receive_()
                if message["type"] == "http.request":
                    body.update({
                        "content": message.get("body", b"").decode()
                    })
                return message
            
            request._receive = receive
        
        return body

class TaskLoggingMiddleware:
    """Middleware for logging Celery task execution"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, task):
        start_time = time.time()
        
        try:
            result = self.get_response(task)
            success = True
        except Exception as e:
            result = str(e)
            success = False
            raise
        finally:
            duration = time.time() - start_time
            log_data = {
                "task": {
                    "id": task.request.id,
                    "name": task.name,
                    "args": task.request.args,
                    "kwargs": task.request.kwargs,
                    "duration": duration,
                    "success": success
                }
            }
            
            if success:
                logger.info(json.dumps(log_data))
            else:
                logger.error(json.dumps(log_data))
