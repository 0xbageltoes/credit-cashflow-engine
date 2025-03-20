"""
API Router for v1 REST API endpoints

This module defines the FastAPI routers for all v1 API endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    forecasting,
    health,
    auth,
    internal,
    cashflow,
)
from app.api.v1.cashflow import router as cashflow_router
from app.api.v1.health import router as health_router
from app.api.v1.structured_products import router as structured_products_router
from app.api.endpoints.deal_library import router as deal_library_router

# Create main API router
api_router = APIRouter()

# Include API endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    forecasting.router,
    prefix="/forecasting",
    tags=["forecasting"]
)

api_router.include_router(
    internal.router,
    prefix="/internal",
    tags=["internal"],
    include_in_schema=False
)

api_router.include_router(
    cashflow.router,
    prefix="/cashflow",
    tags=["cashflow"]
)

# Include structured routers
api_router.include_router(
    cashflow_router,
    prefix="/cashflow",
    tags=["cashflow processing"]
)

api_router.include_router(
    health_router,
    prefix="/system",
    tags=["system health"]
)

api_router.include_router(
    structured_products_router,
    prefix="/structured",
    tags=["structured products"]
)

# Include deal library router
api_router.include_router(
    deal_library_router,
    prefix="/deals",
    tags=["deal library"]
)
