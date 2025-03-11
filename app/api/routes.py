from fastapi import APIRouter
from app.api.v1.api import api_router as api_v1_router

router = APIRouter()

# Include API v1 routes
router.include_router(api_v1_router, prefix="/v1")
