from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import asyncio
from celery.utils.log import get_task_logger
from app.core.celery_app import celery_app
from app.services.cashflow import CashflowService
from app.services.analytics import AnalyticsService
from app.models.cashflow import CashflowForecastRequest, CashflowProjection
from app.core.cache import RedisCache

logger = get_task_logger(__name__)

@celery_app.task(name="app.tasks.forecasting.generate_forecast", bind=True)
def generate_forecast(self, request_dict: Dict, user_id: str) -> str:
    """
    Generate cashflow forecast asynchronously
    Returns the forecast_id which can be used to fetch results
    """
    # Initialize services
    cashflow_service = CashflowService()
    analytics = AnalyticsService()
    cache = RedisCache()
    
    try:
        # Update task status
        self.update_state(state='STARTED')
        cache.set(f"task_status:{self.request.id}", "started", 3600)
        
        # Convert dict to request object
        logger.info("Converting request to CashflowForecastRequest")
        request = CashflowForecastRequest(**request_dict)
        
        # Generate forecast
        logger.info("Generating vectorized loan calculations")
        forecast_result = cashflow_service._vectorized_loan_calculations(request.loans)
        principal_payments, interest_payments, remaining_balance, periods = forecast_result
        
        # Create projections
        logger.info("Creating cashflow projections")
        projections = []
        for loan_idx, loan in enumerate(request.loans):
            start_date = pd.to_datetime(loan.start_date)
            dates = pd.date_range(
                start=start_date,
                periods=int(loan.term_months),
                freq='M'
            )
            
            for period_idx in range(int(loan.term_months)):
                projections.append(
                    CashflowProjection(
                        period=int(periods[loan_idx, period_idx]),
                        date=dates[period_idx].isoformat(),
                        principal=float(abs(principal_payments[loan_idx, period_idx])),
                        interest=float(abs(interest_payments[loan_idx, period_idx])),
                        total_payment=float(abs(principal_payments[loan_idx, period_idx] + 
                                             interest_payments[loan_idx, period_idx])),
                        remaining_balance=float(abs(remaining_balance[loan_idx, period_idx]))
                    )
                )
        
        # Run analytics
        logger.info("Running cashflow analytics")
        combined_cashflows = np.sum(principal_payments + interest_payments, axis=0)
        analytics_result = analytics.analyze_cashflows(
            combined_cashflows,
            discount_rate=request.discount_rate,
            run_monte_carlo=request.run_monte_carlo,
            monte_carlo_config=request.monte_carlo_config
        )
        
        # Save results using asyncio
        logger.info("Saving forecast results")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            forecast_id = loop.run_until_complete(
                cashflow_service._save_forecast(
                    request=request,
                    projections=projections,
                    analytics_result=analytics_result,
                    user_id=user_id
                )
            )
        finally:
            loop.close()
        
        # Update task status and cache results
        logger.info("Forecast generation completed")
        cache.set(f"task_status:{self.request.id}", "completed", 3600)
        cache.set(
            f"forecast_result:{forecast_id}",
            {
                "status": "completed",
                "forecast_id": forecast_id,
                "summary": {
                    "total_principal": float(sum(loan.principal for loan in request.loans)),
                    "total_interest": float(analytics_result.total_interest),
                    "npv": float(analytics_result.npv),
                    "irr": float(analytics_result.irr)
                }
            },
            3600
        )
        
        return forecast_id
        
    except Exception as e:
        # Log error and update status
        logger.error(f"Error in forecast generation: {str(e)}")
        if 'forecast_id' in locals():
            cache.set(f"task_status:{forecast_id}", f"failed: {str(e)}", 3600)
        raise

@celery_app.task(name="app.tasks.forecasting.run_stress_test")
def run_stress_test(
    request_dict: Dict,
    user_id: str,
    stress_scenarios: List[Dict]
) -> str:
    """
    Run stress tests on cashflow forecasts
    """
    results = []
    cache = RedisCache()
    
    try:
        base_request = CashflowForecastRequest(**request_dict)
        
        for scenario in stress_scenarios:
            # Apply stress scenario adjustments
            stressed_request = modify_request_for_scenario(base_request, scenario)
            
            # Generate stressed forecast
            forecast_id = generate_forecast(
                stressed_request.dict(),
                user_id
            )
            
            results.append({
                "scenario_name": scenario["name"],
                "forecast_id": forecast_id
            })
        
        # Save stress test results
        stress_test_id = save_stress_test_results(results, user_id)
        cache.set(f"stress_test:{stress_test_id}", results, 3600)
        
        return stress_test_id
        
    except Exception as e:
        print(f"Error in stress testing: {str(e)}")
        raise

def modify_request_for_scenario(
    request: CashflowForecastRequest,
    scenario: Dict
) -> CashflowForecastRequest:
    """
    Modify request parameters based on stress scenario
    """
    modified_request = request.copy(deep=True)
    
    # Apply rate shocks
    if "rate_shock" in scenario:
        for loan in modified_request.loans:
            loan.interest_rate *= (1 + scenario["rate_shock"])
    
    # Apply prepayment shocks
    if "prepay_shock" in scenario:
        for loan in modified_request.loans:
            loan.prepayment_assumption *= (1 + scenario["prepay_shock"])
    
    # Apply default probability adjustments
    if "default_shock" in scenario:
        modified_request.monte_carlo_config["default_prob"] *= (
            1 + scenario["default_shock"]
        )
    
    return modified_request

def save_stress_test_results(results: List[Dict], user_id: str) -> str:
    """
    Save stress test results to database
    """
    cashflow_service = CashflowService()
    return cashflow_service._save_stress_test_results(results, user_id)
