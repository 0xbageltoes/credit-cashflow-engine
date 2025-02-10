from fastapi import APIRouter
from app.api.v1.endpoints import forecasting

api_router = APIRouter()
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
