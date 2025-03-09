from fastapi import APIRouter
from app.api.v1.endpoints import forecasting, health, cashflow

api_router = APIRouter()
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(cashflow.router, prefix="/cashflow", tags=["cashflow"])
