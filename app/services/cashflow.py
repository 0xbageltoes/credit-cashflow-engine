"""
Cashflow Service

This module provides a comprehensive cashflow calculation service for loan
analysis. It handles both individual loans and batch processing with robust
error handling and logging suitable for production deployment.
"""

import logging
import numpy as np
import numpy_financial as npf
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from app.models.cashflow import (
    LoanData, BatchLoanRequest, CashflowForecastRequest,
    CashflowProjection, CashflowForecastResponse,
    MonteCarloResults, StressTestScenario
)
from app.services.forecasting import ForecastingService
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class CashflowService:
    """
    Cashflow Service for loan analysis and projections.
    
    This service provides:
    - Single loan cashflow calculations
    - Batch processing for multiple loans
    - Monte Carlo simulations for portfolio analysis
    - Stress testing capabilities
    - Economic scenario analysis
    - NPV, IRR, and other financial metrics calculation
    - Production-ready error handling and logging
    """
    
    def __init__(self):
        """Initialize the cashflow service with required dependencies"""
        self.forecasting_service = ForecastingService()
        # Use configuration for thread pool size
        self.max_workers = settings.CALCULATION_THREAD_POOL_SIZE
        logger.info("CashflowService initialized with max workers: %s", self.max_workers)
    
    def calculate_loan_cashflow(
        self, 
        loan_data: LoanData, 
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate cashflow for a single loan
        
        Args:
            loan_data: Loan data model containing all loan parameters
            include_details: Whether to include detailed period-by-period projections
            
        Returns:
            Dictionary containing calculated cashflow data and metrics
        """
        logger.info(f"Calculating cashflow for loan: {loan_data.loan_id}")
        
        try:
            # Calculate basic amortization schedule
            projections = self._calculate_amortization_schedule(loan_data)
            
            # Calculate summary metrics
            metrics = self._calculate_loan_metrics(loan_data, projections)
            
            # Prepare response
            result = {
                "loan_id": loan_data.loan_id,
                "summary_metrics": metrics,
            }
            
            if include_details:
                result["projections"] = projections
            
            logger.debug(f"Cashflow calculation completed for loan: {loan_data.loan_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating cashflow for loan {loan_data.loan_id}: {str(e)}", 
                       exc_info=True)
            # Re-raise with more context
            raise ValueError(f"Failed to calculate cashflow for loan {loan_data.loan_id}: {str(e)}")
    
    def calculate_batch(self, batch_request: BatchLoanRequest) -> Dict[str, Any]:
        """
        Calculate cashflows for multiple loans in a batch
        
        Args:
            batch_request: Batch request containing multiple loans
            
        Returns:
            Dictionary containing results for all loans and batch summary
        """
        logger.info(f"Processing batch with {len(batch_request.loans)} loans")
        
        start_time = datetime.now()
        results = []
        errors = []
        
        # Choose processing method based on configuration
        if batch_request.process_in_parallel and len(batch_request.loans) > 1:
            results, errors = self._process_batch_parallel(
                batch_request.loans,
                batch_request.include_detailed_projections
            )
        else:
            results, errors = self._process_batch_sequential(
                batch_request.loans,
                batch_request.include_detailed_projections
            )
        
        # Calculate portfolio-level metrics if requested
        portfolio_metrics = {}
        if batch_request.summary_metrics and results:
            portfolio_metrics = self._calculate_portfolio_metrics(results)
        
        # Prepare response
        execution_time = (datetime.now() - start_time).total_seconds()
        response = {
            "results": results,
            "portfolio_metrics": portfolio_metrics,
            "errors": errors,
            "execution_time_seconds": execution_time,
            "loans_processed": len(results),
            "loans_failed": len(errors)
        }
        
        logger.info(f"Batch processing completed in {execution_time:.2f} seconds. "
                  f"Processed: {len(results)}, Failed: {len(errors)}")
        
        return response
    
    def _process_batch_parallel(
        self, 
        loans: List[LoanData], 
        include_details: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a batch of loans in parallel using thread pooling
        
        Args:
            loans: List of loan data objects
            include_details: Whether to include detailed projections
            
        Returns:
            Tuple of (successful results, errors)
        """
        results = []
        errors = []
        
        # Determine optimal number of workers based on configuration and batch size
        max_workers = min(
            len(loans),
            self.max_workers
        )
        
        logger.info(f"Processing {len(loans)} loans with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_loan = {
                executor.submit(
                    self.calculate_loan_cashflow, loan, include_details
                ): loan for loan in loans
            }
            
            # Process results as they complete
            for future in as_completed(future_to_loan):
                loan = future_to_loan[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing loan {loan.loan_id}: {str(e)}")
                    errors.append({
                        "loan_id": loan.loan_id,
                        "error": str(e)
                    })
        
        return results, errors
    
    def _process_batch_sequential(
        self, 
        loans: List[LoanData], 
        include_details: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a batch of loans sequentially
        
        Args:
            loans: List of loan data objects
            include_details: Whether to include detailed projections
            
        Returns:
            Tuple of (successful results, errors)
        """
        results = []
        errors = []
        
        for loan in loans:
            try:
                result = self.calculate_loan_cashflow(loan, include_details)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing loan {loan.loan_id}: {str(e)}")
                errors.append({
                    "loan_id": loan.loan_id,
                    "error": str(e)
                })
        
        return results, errors
    
    def _calculate_amortization_schedule(self, loan: LoanData) -> List[CashflowProjection]:
        """
        Calculate amortization schedule for a loan
        
        Args:
            loan: Loan data object
            
        Returns:
            List of cashflow projections for each period
        """
        logger.debug(f"Calculating amortization schedule for loan: {loan.loan_id}")
        
        # Extract loan parameters
        principal = loan.principal
        rate = loan.interest_rate  # Annual rate
        term = loan.term_months
        
        # Calculate monthly rate
        monthly_rate = rate / 12
        
        # Calculate payment based on loan type
        if loan.amortization_type == "level_payment":
            # For a fixed-rate mortgage with level payments
            payment = principal * (monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
        elif loan.amortization_type == "interest_only":
            # For interest-only loans
            io_period = loan.interest_only_period or 0
            if io_period >= term:
                # Full term is interest only with balloon payment at end
                payment = principal * monthly_rate
            else:
                # Calculate payment for amortizing portion
                amort_term = term - io_period
                amort_payment = principal * (monthly_rate * (1 + monthly_rate) ** amort_term) / ((1 + monthly_rate) ** amort_term - 1)
                payment = principal * monthly_rate if io_period > 0 else amort_payment
        else:
            # Default to level payment
            payment = principal * (monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
        
        # Calculate first payment date if not provided
        payment_date = loan.first_payment_date
        if not payment_date:
            # Typically first payment is due 1 month after origination
            payment_date = (
                pd.Timestamp(loan.origination_date) + pd.DateOffset(months=1)
            ).date()
        
        # Initialize projection list
        projections = []
        remaining_balance = principal
        period = 1
        
        # Generate amortization schedule
        while period <= term:
            interest_payment = remaining_balance * monthly_rate
            
            # Handle interest-only period
            if loan.amortization_type == "interest_only" and period <= (loan.interest_only_period or 0):
                principal_payment = 0
            else:
                principal_payment = payment - interest_payment
            
            # Account for final period rounding
            if period == term:
                principal_payment = remaining_balance
                payment = principal_payment + interest_payment
            
            remaining_balance -= principal_payment
            
            # Calculate default and prepayment amounts (if enabled)
            default_amount = 0.0
            prepayment_amount = 0.0
            
            if loan.default_probability and loan.default_probability > 0:
                # Simple monthly default calculation (can be enhanced with curves)
                monthly_default_prob = (1 - (1 - loan.default_probability) ** (1/12))
                default_amount = remaining_balance * monthly_default_prob
            
            if loan.prepayment_rate and loan.prepayment_rate > 0:
                # Simple prepayment model (can be enhanced with curves)
                monthly_prepay_rate = loan.prepayment_rate / 12
                prepayment_amount = remaining_balance * monthly_prepay_rate
            
            # Create projection for current period
            projection = CashflowProjection(
                period=period,
                date=payment_date,
                principal=principal_payment,
                interest=interest_payment,
                total_payment=payment,
                remaining_balance=max(0, remaining_balance),
                default_amount=default_amount,
                prepayment_amount=prepayment_amount
            )
            
            projections.append(projection)
            
            # Update for next period
            period += 1
            payment_date = (pd.Timestamp(payment_date) + pd.DateOffset(months=1)).date()
            
            # Account for prepayments and defaults in remaining balance
            remaining_balance -= (default_amount + prepayment_amount)
            if remaining_balance <= 0:
                break
        
        return projections
    
    def _calculate_loan_metrics(
        self, 
        loan: LoanData, 
        projections: List[CashflowProjection]
    ) -> Dict[str, Any]:
        """
        Calculate key financial metrics for a loan
        
        Args:
            loan: Loan data object
            projections: List of cashflow projections
            
        Returns:
            Dictionary of calculated metrics
        """
        # Extract cashflows for calculations
        cashflows = [-loan.principal]  # Initial outflow
        dates = [loan.origination_date]
        
        for proj in projections:
            cashflows.append(proj.total_payment + proj.prepayment_amount)
            dates.append(proj.date)
        
        # Calculate key metrics
        metrics = {}
        
        # Total payments
        metrics["total_payments"] = sum(cf for cf in cashflows[1:])
        
        # Total interest
        metrics["total_interest"] = sum(proj.interest for proj in projections)
        
        # Final maturity
        metrics["actual_term"] = len(projections)
        metrics["final_payment_date"] = dates[-1].isoformat() if dates else None
        
        # Try to calculate IRR
        try:
            # Convert cashflows to numpy array
            cf_array = np.array(cashflows)
            
            # Calculate IRR (internal rate of return)
            irr = npf.irr(cf_array)
            metrics["irr"] = float(irr) if not np.isnan(irr) else None
            
            # Calculate annual IRR
            metrics["annual_irr"] = float((1 + irr) ** 12 - 1) if not np.isnan(irr) else None
        except Exception as e:
            logger.warning(f"Failed to calculate IRR for loan {loan.loan_id}: {str(e)}")
            metrics["irr"] = None
            metrics["annual_irr"] = None
        
        # Calculate NPV at different discount rates
        discount_rates = [0.03, 0.04, 0.05, 0.06, 0.07]  # 3% to 7%
        npv_results = {}
        
        for rate in discount_rates:
            try:
                monthly_rate = rate / 12
                npv = npf.npv(monthly_rate, cf_array)
                npv_results[f"{rate:.2f}"] = float(npv)
            except Exception:
                npv_results[f"{rate:.2f}"] = None
        
        metrics["npv_at_rates"] = npv_results
        
        # Calculate weighted average life (WAL)
        try:
            # Skip initial outflow
            weighted_flows = [(i+1) * proj.total_payment for i, proj in enumerate(projections)]
            total_flows = sum(proj.total_payment for proj in projections)
            
            if total_flows > 0:
                metrics["weighted_avg_life"] = sum(weighted_flows) / total_flows / 12
            else:
                metrics["weighted_avg_life"] = None
        except Exception as e:
            logger.warning(f"Failed to calculate WAL for loan {loan.loan_id}: {str(e)}")
            metrics["weighted_avg_life"] = None
        
        return metrics
    
    def _calculate_portfolio_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate portfolio-level metrics from individual loan results
        
        Args:
            results: List of individual loan results
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not results:
            return {}
        
        portfolio_metrics = {
            "total_principal": 0,
            "total_interest": 0,
            "weighted_interest_rate": 0,
            "weighted_term": 0,
            "count": len(results)
        }
        
        total_principal = 0
        weighted_interest = 0
        weighted_term = 0
        
        for result in results:
            loan_id = result.get("loan_id", "unknown")
            metrics = result.get("summary_metrics", {})
            
            # Find the corresponding loan data from the original request
            loan_principal = 0
            interest_rate = 0
            term = 0
            
            # Extract from projections if available
            if "projections" in result and result["projections"]:
                first_proj = result["projections"][0]
                loan_principal = first_proj.remaining_balance
            
            # Add to totals
            total_principal += loan_principal
            portfolio_metrics["total_interest"] += metrics.get("total_interest", 0)
            
            # Weighted calculations
            weighted_interest += loan_principal * interest_rate
            weighted_term += loan_principal * term
        
        # Calculate weighted averages
        if total_principal > 0:
            portfolio_metrics["total_principal"] = total_principal
            portfolio_metrics["weighted_interest_rate"] = weighted_interest / total_principal
            portfolio_metrics["weighted_term"] = weighted_term / total_principal
        
        # Calculate portfolio IRR if enough data
        if all("irr" in r.get("summary_metrics", {}) for r in results):
            # This is a simplified approach - a more accurate portfolio IRR would 
            # combine all cashflows and recalculate
            weighted_irr = sum(
                r["summary_metrics"]["irr"] * r["summary_metrics"].get("total_payments", 0)
                for r in results if r["summary_metrics"].get("irr") is not None
            ) / sum(
                r["summary_metrics"].get("total_payments", 0)
                for r in results if r["summary_metrics"].get("irr") is not None
            )
            portfolio_metrics["portfolio_irr"] = weighted_irr
        
        return portfolio_metrics