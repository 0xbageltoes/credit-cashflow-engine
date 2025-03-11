from fastapi import APIRouter
from app.api.v1.endpoints import forecasting
from app.api.v1.cashflow import router as cashflow_router
from app.api.v1.health import router as health_router
from app.api.v1.structured_products import router as structured_products_router
from app.api.v1.enhanced_analytics import router as enhanced_analytics_router

api_router = APIRouter()
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
api_router.include_router(cashflow_router, prefix="/cashflow", tags=["cashflow"])
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(structured_products_router, prefix="/structured-products", tags=["structured-products"])
api_router.include_router(enhanced_analytics_router, prefix="/enhanced-analytics", tags=["enhanced-analytics"])
