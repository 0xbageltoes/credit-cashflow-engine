"""
Specialized Asset Classes API Endpoints

This module integrates all specialized asset class handlers into the API router.
It includes endpoints for consumer credit, commercial loans, and CLO/CDO products.
"""
from fastapi import APIRouter

# Import the specialized asset class routers
from app.api.endpoints.consumer_credit import router as consumer_credit_router
from app.api.endpoints.commercial_loans import router as commercial_loans_router
from app.api.endpoints.clo_cdo import router as clo_cdo_router

# Create router for specialized asset handlers
router = APIRouter()

# Include the specialized asset class routers
router.include_router(consumer_credit_router, prefix="/consumer-credit")
router.include_router(commercial_loans_router, prefix="/commercial-loans")
router.include_router(clo_cdo_router, prefix="/clo-cdo")
