"""
Residential Mortgage Asset Handler

Production-quality implementation for analyzing residential mortgage assets using
the AbsBox engine with comprehensive error handling and detailed analytics.
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
import pandas as pd
import numpy as np

# Import AbsBox libraries
import absbox as ab

# Setup logging
logger = logging.getLogger(__name__)

from app.core.monitoring import CalculationTracker
from app.models.asset_classes import (
    AssetClass, ResidentialMortgage, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

class ResidentialMortgageHandler:
    """
    Handler for residential mortgage asset analysis
    
    Production-ready implementation of residential mortgage analytics with
    comprehensive error handling, metrics, and stress testing.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None):
        """
        Initialize the residential mortgage handler
        
        Args:
            absbox_service: The AbsBox service to use (created if not provided)
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        logger.info("ResidentialMortgageHandler initialized")
    
    def analyze_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Analyze a pool of residential mortgages
        
        Args:
            request: The analysis request containing pool and parameters
            
        Returns:
            AssetPoolAnalysisResponse with detailed analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing residential mortgage pool: {request.pool.pool_name}")
            with CalculationTracker("residential_mortgage_analysis"):
                # Filter to ensure we only process residential mortgages
                residential_assets = [
                    asset for asset in request.pool.assets 
                    if asset.asset_class == AssetClass.RESIDENTIAL_MORTGAGE
                ]
                
                if not residential_assets:
                    raise ValueError("No residential mortgage assets found in pool")
                
                # Create AbsBox loan objects
                loans = self._create_absbox_loans(residential_assets)
                
                # Create pool and run analysis
                pool = self._create_absbox_pool(loans, request.pool.pool_name, 
                                              request.pool.cut_off_date)
                
                # Calculate cashflows with appropriate settings
                cashflows = self._calculate_cashflows(pool, request.analysis_date)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    cashflows, 
                    request.discount_rate or 0.05,  # Default 5% if not provided
                    request.analysis_date
                )
                
                # Run stress tests if requested
                stress_tests = None
                if request.include_stress_tests:
                    stress_tests = self._run_stress_tests(pool, 
                                                        cashflows, metrics, 
                                                        request.analysis_date,
                                                        request.discount_rate or 0.05)
                
                # Build response
                result = AssetPoolAnalysisResponse(
                    pool_name=request.pool.pool_name,
                    analysis_date=request.analysis_date,
                    execution_time=time.time() - start_time,
                    status="success",
                    metrics=metrics,
                    cashflows=cashflows if request.include_cashflows else None,
                    stress_tests=stress_tests
                )
                
                return result
                
        except Exception as e:
            logger.exception(f"Error analyzing residential mortgage pool: {str(e)}")
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date,
                execution_time=time.time() - start_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_absbox_loans(self, mortgages: List[ResidentialMortgage]) -> List[Any]:
        """
        Create AbsBox loan objects from residential mortgage models
        
        Args:
            mortgages: List of residential mortgage models
            
        Returns:
            List of AbsBox loan objects
        """
        ab_loans = []
        
        for mortgage in mortgages:
            try:
                # Handle different rate types
                if mortgage.rate_type.value == "fixed":
                    loan = ab.local.FixedRateLoan(
                        balance=mortgage.balance,
                        rate=mortgage.rate,
                        originBalance=mortgage.original_balance,
                        originRate=mortgage.rate,  # Use same rate if no original provided
                        originTerm=mortgage.term_months,
                        remainTerm=mortgage.remaining_term_months
                    )
                elif mortgage.rate_type.value == "floating":
                    loan = ab.local.FloatingRateLoan(
                        balance=mortgage.balance,
                        rate=mortgage.rate,
                        margin=0.02,  # Default if not specified
                        originBalance=mortgage.original_balance,
                        originRate=mortgage.rate,
                        originTerm=mortgage.term_months,
                        remainTerm=mortgage.remaining_term_months
                    )
                elif mortgage.rate_type.value == "hybrid":
                    # For hybrid loans, we use an advanced setup
                    loan = ab.local.HybridRateLoan(
                        balance=mortgage.balance,
                        rate=mortgage.rate,
                        margin=0.02,  # Default if not specified
                        fixedTerm=60,  # Assume 5 years fixed (common)
                        originBalance=mortgage.original_balance,
                        originRate=mortgage.rate,
                        originTerm=mortgage.term_months,
                        remainTerm=mortgage.remaining_term_months
                    )
                else:
                    # Default to fixed rate if type not recognized
                    logger.warning(f"Unrecognized rate type {mortgage.rate_type}, defaulting to fixed")
                    loan = ab.local.FixedRateLoan(
                        balance=mortgage.balance,
                        rate=mortgage.rate,
                        originBalance=mortgage.original_balance,
                        originRate=mortgage.rate,
                        originTerm=mortgage.term_months,
                        remainTerm=mortgage.remaining_term_months
                    )
                
                # Add additional properties for enhanced analysis
                if mortgage.ltv_ratio:
                    loan.ltv = mortgage.ltv_ratio
                if mortgage.is_interest_only and mortgage.interest_only_period_months:
                    loan.IOTerm = mortgage.interest_only_period_months
                
                # Add to list
                ab_loans.append(loan)
            except Exception as e:
                logger.error(f"Error creating AbsBox loan for mortgage: {str(e)}")
                # Continue with other loans - don't let one bad loan stop the whole pool
                continue
                
        if not ab_loans:
            raise ValueError("Failed to create any valid AbsBox loan objects")
            
        return ab_loans
    
    def _create_absbox_pool(self, loans: List[Any], name: str, cut_off_date: date) -> Any:
        """
        Create an AbsBox pool from loan objects
        
        Args:
            loans: List of AbsBox loan objects
            name: Name of the pool
            cut_off_date: Cut-off date for the pool
            
        Returns:
            AbsBox pool object
        """
        try:
            # Create the pool with the loans
            pool = ab.local.Pool(
                name=name,
                cutoff=cut_off_date,
                loans=loans
            )
            
            return pool
        except Exception as e:
            logger.exception(f"Error creating AbsBox pool: {str(e)}")
            raise ValueError(f"Failed to create AbsBox pool: {str(e)}")
    
    def _calculate_cashflows(self, pool: Any, analysis_date: date) -> List[AssetPoolCashflow]:
        """
        Calculate cashflows for the mortgage pool
        
        Args:
            pool: AbsBox pool object
            analysis_date: Date to start analysis from
            
        Returns:
            List of AssetPoolCashflow objects
        """
        try:
            # Create default assumptions
            default_assumption = ab.local.DefaultAssumption(
                vector=[0.005] * 120,  # Default 0.5% CDR (Constant Default Rate)
                recovery=0.65,  # 65% recovery rate (standard assumption)
                lag=6  # 6 month lag for recoveries (standard assumption)
            )
            
            prepay_assumption = ab.local.PrepayAssumption(
                vector=[0.06] * 120  # Default 6% CPR (Constant Prepayment Rate)
            )
            
            # Run the cashflow projection
            cf_engine = ab.local.CFEngine()
            cashflows = cf_engine.runCF(
                pool, 
                defaultAssumption=default_assumption,
                prepayAssumption=prepay_assumption
            )
            
            # Convert to our model
            result_cashflows = []
            for i, row in enumerate(cashflows.cf):
                # Convert the AbsBox cashflow to our model
                cf = AssetPoolCashflow(
                    period=i,
                    date=analysis_date.replace(month=analysis_date.month + i) if i == 0 else \
                         analysis_date.replace(month=analysis_date.month + i % 12, 
                                              year=analysis_date.year + i // 12),
                    scheduled_principal=row.get('scheduledPrincipal', 0.0),
                    scheduled_interest=row.get('scheduledInterest', 0.0),
                    prepayment=row.get('prepayment', 0.0),
                    default=row.get('default', 0.0),
                    recovery=row.get('recovery', 0.0),
                    loss=row.get('loss', 0.0) if 'loss' in row else (
                        row.get('default', 0.0) - row.get('recovery', 0.0)
                    ),
                    balance=row.get('balance', 0.0)
                )
                result_cashflows.append(cf)
                
            return result_cashflows
            
        except Exception as e:
            logger.exception(f"Error calculating cashflows: {str(e)}")
            raise ValueError(f"Failed to calculate cashflows: {str(e)}")
    
    def _calculate_metrics(
        self, 
        cashflows: List[AssetPoolCashflow], 
        discount_rate: float,
        analysis_date: date
    ) -> AssetPoolMetrics:
        """
        Calculate comprehensive metrics for the mortgage pool
        
        Args:
            cashflows: List of cashflow projections
            discount_rate: Discount rate for present value calculations
            analysis_date: Date of analysis
            
        Returns:
            AssetPoolMetrics object with calculated metrics
        """
        try:
            # Extract key components
            total_scheduled_principal = sum(cf.scheduled_principal for cf in cashflows)
            total_scheduled_interest = sum(cf.scheduled_interest for cf in cashflows)
            total_prepayment = sum(cf.prepayment for cf in cashflows)
            total_loss = sum(cf.loss for cf in cashflows)
            
            # Calculate total cashflow
            total_cashflow = total_scheduled_principal + total_scheduled_interest + \
                             total_prepayment - total_loss
            
            # Calculate NPV
            npv = 0.0
            for i, cf in enumerate(cashflows):
                # Monthly discount factor
                monthly_rate = (1 + discount_rate) ** (1/12) - 1
                # Cashflow for the period
                period_cf = cf.scheduled_principal + cf.scheduled_interest + \
                            cf.prepayment - cf.loss
                # Present value of this period's cashflow
                npv += period_cf / ((1 + monthly_rate) ** (i+1))
            
            # Calculate IRR
            # Using numpy's IRR function
            periods = len(cashflows)
            if periods > 0:
                # Initial outflow (negative) is the starting balance
                cashflow_array = [-cashflows[0].balance]
                # Add each period's net cashflow
                for cf in cashflows:
                    period_cf = cf.scheduled_principal + cf.scheduled_interest + \
                                cf.prepayment - cf.loss
                    cashflow_array.append(period_cf)
                    
                # Calculate IRR if we have valid cashflows
                if any(cf != 0 for cf in cashflow_array):
                    try:
                        irr = np.irr(cashflow_array)
                        # Convert to annual rate
                        annual_irr = (1 + irr) ** 12 - 1
                    except:
                        # IRR calculation might fail if no solution found
                        annual_irr = None
                else:
                    annual_irr = None
            else:
                annual_irr = None
            
            # Calculate duration
            if periods > 0 and npv > 0:
                # Macaulay duration
                duration = 0.0
                for i, cf in enumerate(cashflows):
                    period_cf = cf.scheduled_principal + cf.scheduled_interest + \
                                cf.prepayment - cf.loss
                    # Time in years
                    t = (i+1) / 12
                    # Present value of this period's cashflow
                    pv = period_cf / ((1 + discount_rate) ** t)
                    # Weighted time
                    duration += t * pv
                
                # Normalize by total NPV
                macaulay_duration = duration / npv if npv > 0 else 0
                # Modified duration
                modified_duration = macaulay_duration / (1 + discount_rate)
                
                # Convexity
                convexity = 0.0
                for i, cf in enumerate(cashflows):
                    period_cf = cf.scheduled_principal + cf.scheduled_interest + \
                                cf.prepayment - cf.loss
                    # Time in years
                    t = (i+1) / 12
                    # Present value
                    pv = period_cf / ((1 + discount_rate) ** t)
                    # Weighted squared time
                    convexity += t * (t + 1) * pv
                
                # Normalize and adjust
                convexity = convexity / (npv * (1 + discount_rate) ** 2) if npv > 0 else 0
            else:
                macaulay_duration = 0
                modified_duration = 0
                convexity = 0
            
            # Calculate WAL
            if periods > 0:
                weighted_time = 0.0
                total_principal = 0.0
                
                for i, cf in enumerate(cashflows):
                    # Time in years
                    t = (i+1) / 12
                    # Principal payments (scheduled + prepayments)
                    principal_payment = cf.scheduled_principal + cf.prepayment
                    weighted_time += t * principal_payment
                    total_principal += principal_payment
                
                wal = weighted_time / total_principal if total_principal > 0 else 0
            else:
                wal = 0
            
            # Create metrics object
            metrics = AssetPoolMetrics(
                total_principal=total_scheduled_principal + total_prepayment,
                total_interest=total_scheduled_interest,
                total_cashflow=total_cashflow,
                npv=npv,
                irr=annual_irr,
                duration=macaulay_duration,
                modified_duration=modified_duration,
                convexity=convexity,
                weighted_average_life=wal,
                yield_to_maturity=annual_irr  # Often the same as IRR for fixed rate
            )
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating metrics: {str(e)}")
            # Return minimal metrics with major fields
            return AssetPoolMetrics(
                total_principal=sum(cf.scheduled_principal for cf in cashflows),
                total_interest=sum(cf.scheduled_interest for cf in cashflows),
                total_cashflow=sum(cf.scheduled_principal + cf.scheduled_interest for cf in cashflows),
                npv=0.0
            )
    
    def _run_stress_tests(
        self,
        pool: Any,
        base_cashflows: List[AssetPoolCashflow],
        base_metrics: AssetPoolMetrics,
        analysis_date: date,
        discount_rate: float
    ) -> List[AssetPoolStressTest]:
        """
        Run stress tests on the mortgage pool
        
        Args:
            pool: AbsBox pool object
            base_cashflows: Baseline cashflow projections 
            base_metrics: Baseline metrics
            analysis_date: Date of analysis
            discount_rate: Discount rate for present value calculations
            
        Returns:
            List of AssetPoolStressTest objects with stress test results
        """
        stress_tests = []
        
        try:
            # Common stress scenarios
            scenarios = [
                {
                    "name": "High Default",
                    "description": "Double the default rate",
                    "default_vector": [0.01] * 120,  # 1% CDR
                    "prepay_vector": [0.06] * 120,   # Normal prepayment
                },
                {
                    "name": "High Prepayment",
                    "description": "Double the prepayment rate",
                    "default_vector": [0.005] * 120,  # Normal default
                    "prepay_vector": [0.12] * 120,    # 12% CPR
                },
                {
                    "name": "Severe Stress",
                    "description": "High default + low recovery + low prepayment",
                    "default_vector": [0.015] * 120,  # 1.5% CDR
                    "prepay_vector": [0.03] * 120,    # 3% CPR (half normal)
                    "recovery": 0.4,                 # 40% recovery (lower than usual)
                }
            ]
            
            # Run each scenario
            cf_engine = ab.local.CFEngine()
            
            for scenario in scenarios:
                # Create assumptions
                default_assumption = ab.local.DefaultAssumption(
                    vector=scenario["default_vector"],
                    recovery=scenario.get("recovery", 0.65),
                    lag=6
                )
                
                prepay_assumption = ab.local.PrepayAssumption(
                    vector=scenario["prepay_vector"]
                )
                
                # Run the cashflow projection with this scenario
                cashflows = cf_engine.runCF(
                    pool, 
                    defaultAssumption=default_assumption,
                    prepayAssumption=prepay_assumption
                )
                
                # Convert to our model
                scenario_cashflows = []
                for i, row in enumerate(cashflows.cf):
                    cf = AssetPoolCashflow(
                        period=i,
                        date=analysis_date.replace(month=analysis_date.month + i) if i == 0 else \
                            analysis_date.replace(month=analysis_date.month + i % 12, 
                                                year=analysis_date.year + i // 12),
                        scheduled_principal=row.get('scheduledPrincipal', 0.0),
                        scheduled_interest=row.get('scheduledInterest', 0.0),
                        prepayment=row.get('prepayment', 0.0),
                        default=row.get('default', 0.0),
                        recovery=row.get('recovery', 0.0),
                        loss=row.get('loss', 0.0) if 'loss' in row else (
                            row.get('default', 0.0) - row.get('recovery', 0.0)
                        ),
                        balance=row.get('balance', 0.0)
                    )
                    scenario_cashflows.append(cf)
                
                # Calculate metrics for this scenario
                scenario_metrics = self._calculate_metrics(
                    scenario_cashflows,
                    discount_rate,
                    analysis_date
                )
                
                # Calculate changes
                npv_change = scenario_metrics.npv - base_metrics.npv
                npv_change_percent = (npv_change / base_metrics.npv * 100) if base_metrics.npv != 0 else 0
                
                # Create stress test result
                stress_test = AssetPoolStressTest(
                    scenario_name=scenario["name"],
                    description=scenario["description"],
                    npv=scenario_metrics.npv,
                    npv_change=npv_change,
                    npv_change_percent=npv_change_percent
                )
                
                stress_tests.append(stress_test)
                
            return stress_tests
            
        except Exception as e:
            logger.exception(f"Error running stress tests: {str(e)}")
            # Return a default stress test to indicate the error
            return [
                AssetPoolStressTest(
                    scenario_name="Error",
                    description=f"Failed to run stress tests: {str(e)}",
                    npv=base_metrics.npv,
                    npv_change=0.0,
                    npv_change_percent=0.0
                )
            ]
