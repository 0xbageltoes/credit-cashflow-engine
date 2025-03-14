"""
API Router for v1 REST API endpoints

This module defines the FastAPI routers for all v1 API endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    forecasting,
    health,
    auth,
    assets,
    cashflows,
    pricing,
    reports,
    scenarios, 
    monte_carlo,
)
from app.api.v1.cashflow import router as cashflow_router
from app.api.v1.health import router as health_router
from app.api.v1.structured_products import router as structured_products_router
from app.api.v1.enhanced_analytics import router as enhanced_analytics_router
from app.api.v1.asset_classes.endpoints import router as asset_classes_router
from app.api.v1.asset_classes.stress_endpoints import router as stress_testing_router
from app.api.v1.asset_classes.specialized_assets import router as specialized_assets_router
from app.api.v1.monte_carlo import router as monte_carlo_router
from app.api.v1.websockets import websocket_router

# Create main API router
api_router = APIRouter()

# Include API endpoint routers
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
api_router.include_router(cashflow_router, prefix="/cashflow", tags=["cashflow"])
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(structured_products_router, prefix="/structured-products", tags=["structured-products"])
api_router.include_router(enhanced_analytics_router, prefix="/enhanced-analytics", tags=["enhanced-analytics"])
api_router.include_router(asset_classes_router, prefix="/asset-classes", tags=["asset-classes"])
api_router.include_router(stress_testing_router, prefix="/asset-classes", tags=["stress-testing"])
api_router.include_router(specialized_assets_router, prefix="/specialized-assets", tags=["specialized-assets"])
api_router.include_router(monte_carlo_router, prefix="/monte-carlo", tags=["monte-carlo"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# Include WebSocket router (No prefix for WebSocket endpoints)
api_router.include_router(websocket_router, tags=["websockets"])
