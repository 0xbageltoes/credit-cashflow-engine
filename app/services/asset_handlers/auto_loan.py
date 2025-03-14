"""
Auto Loan Asset Handler

Production-quality implementation for analyzing auto loan assets using
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
    AssetClass, AutoLoan, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

class AutoLoanHandler:
    """
    Handler for auto loan asset analysis
    
    Production-ready implementation of auto loan analytics with
    comprehensive error handling, metrics, and stress testing.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None):
        """
        Initialize the auto loan handler
        
        Args:
            absbox_service: The AbsBox service to use (created if not provided)
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        logger.info("AutoLoanHandler initialized")
    
    def analyze_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Analyze a pool of auto loans
        
        Args:
            request: The analysis request containing pool and parameters
            
        Returns:
            AssetPoolAnalysisResponse with detailed analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing auto loan pool: {request.pool.pool_name}")
            with CalculationTracker("auto_loan_analysis"):
                # Filter to ensure we only process auto loans
                auto_assets = [
                    asset for asset in request.pool.assets 
                    if asset.asset_class == AssetClass.AUTO_LOAN
                ]
                
                if not auto_assets:
                    raise ValueError("No auto loan assets found in pool")
                
                # Create AbsBox loan objects with auto-specific parameters
                loans = self._create_absbox_loans(auto_assets)
                
                # Create pool and run analysis
                pool = self._create_absbox_pool(loans, request.pool.pool_name, 
                                              request.pool.cut_off_date)
                
                # Calculate cashflows with appropriate settings for auto loans
                # Auto loans have different default/prepay characteristics than mortgages
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
            logger.exception(f"Error analyzing auto loan pool: {str(e)}")
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date,
                execution_time=time.time() - start_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_absbox_loans(self, auto_loans: List[AutoLoan]) -> List[Any]:
        """
        Create AbsBox loan objects from auto loan models
        
        Args:
            auto_loans: List of auto loan models
            
        Returns:
            List of AbsBox loan objects
        """
        ab_loans = []
        
        for loan in auto_loans:
            try:
                # Handle different rate types - auto loans are predominantly fixed rate
                if loan.rate_type.value in ["fixed", "step"]:
                    ab_loan = ab.local.FixedRateLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,  # Use same rate if no original provided
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                elif loan.rate_type.value == "floating":
                    ab_loan = ab.local.FloatingRateLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        margin=0.03,  # Higher margin typical for auto loans
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                else:
                    # Default to fixed rate if type not recognized
                    logger.warning(f"Unrecognized rate type {loan.rate_type}, defaulting to fixed")
                    ab_loan = ab.local.FixedRateLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                
                # Add auto-specific properties for enhanced analysis
                if loan.ltv_ratio:
                    ab_loan.ltv = loan.ltv_ratio
                
                # Add vehicle-specific properties that influence default and recovery
                ab_loan.assetInfo = {
                    "type": "auto",
                    "vehicle_type": loan.vehicle_type.value if loan.vehicle_type else "new",
                    "make": loan.vehicle_make if loan.vehicle_make else "unknown",
                    "model": loan.vehicle_model if loan.vehicle_model else "unknown",
                    "year": loan.vehicle_year if loan.vehicle_year else datetime.now().year - 3,
                    "initial_depreciation": loan.initial_depreciation_rate if loan.initial_depreciation_rate else 0.2,
                    "subsequent_depreciation": loan.subsequent_depreciation_rate if loan.subsequent_depreciation_rate else 0.1
                }
                
                # Add to list
                ab_loans.append(ab_loan)
            except Exception as e:
                logger.error(f"Error creating AbsBox loan for auto loan: {str(e)}")
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
                loans=loans,
                assetType="auto"  # Specific asset type for better default behavior
            )
            
            return pool
        except Exception as e:
            logger.exception(f"Error creating AbsBox pool: {str(e)}")
            raise ValueError(f"Failed to create AbsBox pool: {str(e)}")
    
    def _calculate_cashflows(self, pool: Any, analysis_date: date) -> List[AssetPoolCashflow]:
        """
        Calculate cashflows for the auto loan pool
        
        Args:
            pool: AbsBox pool object
            analysis_date: Date to start analysis from
            
        Returns:
            List of AssetPoolCashflow objects
        """
        try:
            # Create auto loan specific assumptions
            # Auto loans typically have higher default rates but shorter terms
            default_assumption = ab.local.DefaultAssumption(
                vector=[0.01] * 60,  # Default 1% CDR (higher than mortgages)
                recovery=0.50,  # 50% recovery rate for autos (lower than homes)
                lag=3  # 3 month lag for recoveries (faster than mortgages)
            )
            
            # Auto loan prepayments are influenced by different factors than mortgages
            prepay_assumption = ab.local.PrepayAssumption(
                vector=[0.08] * 60  # Default 8% CPR (higher than mortgages)
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
        Calculate comprehensive metrics for the auto loan pool
        
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
            
            # Create metrics object with all relevant auto loan performance metrics
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
                yield_to_maturity=annual_irr
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
        Run stress tests on the auto loan pool with auto-specific scenarios
        
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
            # Auto-specific stress scenarios
            scenarios = [
                {
                    "name": "Economic Downturn",
                    "description": "Increased defaults and depreciation",
                    "default_vector": [0.02] * 60,  # 2% CDR (double baseline)
                    "prepay_vector": [0.04] * 60,   # 4% CPR (half baseline - fewer new car sales)
                    "recovery": 0.40,              # Lower recovery due to market oversupply
                },
                {
                    "name": "Rising Interest Rates",
                    "description": "Higher rates lead to more prepayments as borrowers refinance",
                    "default_vector": [0.01] * 60,  # Normal default
                    "prepay_vector": [0.12] * 60,   # 12% CPR (higher prepayments)
                },
                {
                    "name": "Severe Recession",
                    "description": "Major economic downturn with high unemployment",
                    "default_vector": [0.03] * 60,  # 3% CDR (triple baseline)
                    "prepay_vector": [0.03] * 60,   # 3% CPR (very low - reduced sales)
                    "recovery": 0.35,              # Severely depressed recovery values
                }
            ]
            
            # Run each scenario
            cf_engine = ab.local.CFEngine()
            
            for scenario in scenarios:
                # Create assumptions
                default_assumption = ab.local.DefaultAssumption(
                    vector=scenario["default_vector"],
                    recovery=scenario.get("recovery", 0.50),
                    lag=3  # Auto loans typically have faster recovery times
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
