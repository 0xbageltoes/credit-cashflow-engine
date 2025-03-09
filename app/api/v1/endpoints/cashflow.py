from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from app.core.auth import get_current_user
from app.services.cashflow import CashflowService
from app.models.cashflow import LoanData, BatchLoanRequest
from app.core.logging import logger

router = APIRouter()

@router.post("/calculate")
async def calculate_cashflow(
    loan_data: LoanData,
    current_user: Dict = Depends(get_current_user)
):
    """Calculate cash flow for a single loan"""
    try:
        cashflow_service = CashflowService()
        result = cashflow_service.calculate_loan_cashflow(loan_data)
        return result
    except Exception as e:
        logger.error(f"Error calculating cashflow: {str(e)}", extra={
            "user_id": current_user["id"],
            "loan_data": loan_data.dict()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-batch")
async def calculate_batch(
    batch_request: BatchLoanRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Calculate cash flows for multiple loans"""
    try:
        cashflow_service = CashflowService()
        result = cashflow_service.calculate_batch(batch_request)
        return result
    except Exception as e:
        logger.error(f"Error calculating batch cashflow: {str(e)}", extra={
            "user_id": current_user["id"],
            "loan_count": len(batch_request.loans)
        })
        raise HTTPException(status_code=500, detail=str(e))
