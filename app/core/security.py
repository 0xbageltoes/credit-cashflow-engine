from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import secrets

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove any headers that might leak information
        if "Server" in response.headers:
            del response.headers["Server"]
        
        return response

def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure password"""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:,.<>?"
    return "".join(secrets.choice(alphabet) for _ in range(length))

def sanitize_input(value: str) -> str:
    """Basic sanitization of user input to prevent common injection attacks"""
    if not value:
        return value
    
    # Replace potentially dangerous characters
    sanitized = value.replace("<", "&lt;").replace(">", "&gt;")
    sanitized = sanitized.replace("'", "&#39;").replace('"', "&quot;")
    sanitized = sanitized.replace(";", "&#59;")
    
    return sanitized

def validate_jwt_payload(payload: dict) -> bool:
    """Additional validation of JWT payload beyond basic signature verification"""
    required_fields = ["sub", "exp", "iat"]
    
    # Check that all required fields exist
    if not all(field in payload for field in required_fields):
        return False
    
    # Check that token is not too old (iat not more than 7 days ago)
    import time
    now = int(time.time())
    if now - payload.get("iat", 0) > 7 * 24 * 60 * 60:  # 7 days
        return False
    
    return True
