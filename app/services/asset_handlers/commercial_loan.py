"""
Commercial Loan Asset Handler

Production-quality implementation for analyzing commercial loan assets using
the AbsBox engine with comprehensive error handling and detailed analytics.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import date

# Import AbsBox libraries
import absbox as ab
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.cache_service import CacheService
from app.core.monitoring import CalculationTracker, CALCULATION_TIME
from app.models.asset_classes import (
    AssetClass, CommercialLoan, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

class CommercialLoanHandler:
    """
    Handler for commercial loan asset analysis
    
    Production-ready implementation of commercial loan analytics with
    comprehensive error handling, metrics, and stress testing.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None, cache_service: Optional[CacheService] = None):
        """
        Initialize the commercial loan handler
        
        Args:
            absbox_service: The AbsBox service to use (created if not provided)
            cache_service: Optional cache service for performance optimization
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        self.cache_service = cache_service or CacheService()
        logger.info("CommercialLoanHandler initialized")
    
    def analyze_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Analyze a pool of commercial loans
        
        Args:
            request: The analysis request containing pool and parameters
            
        Returns:
            AssetPoolAnalysisResponse with detailed analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing commercial loan pool: {request.pool.pool_name}")
            with CalculationTracker("commercial_loan_analysis"):
                # Filter to ensure we only process commercial loan assets
                commercial_assets = [
                    asset for asset in request.pool.assets 
                    if asset.asset_class == AssetClass.COMMERCIAL_LOAN
                ]
                
                if not commercial_assets:
                    raise ValueError("No commercial loan assets found in pool")
                
                # Create AbsBox loan objects
                loans = self._create_absbox_loans(commercial_assets)
                
                # Create pool and run analysis
                pool = self._create_absbox_pool(loans, request.pool.pool_name, 
                                             request.pool.cut_off_date)
                
                # Calculate cashflows with appropriate settings
                cashflows = self._calculate_cashflows(pool, request.analysis_date)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    cashflows, 
                    request.discount_rate or 0.06,  # Default 6% for commercial loans
                    request.analysis_date
                )
                
                # Run specialized commercial loan analytics
                analytics = self._run_commercial_loan_analytics(
                    commercial_assets,
                    cashflows
                )
                
                # Run stress tests if requested
                stress_tests = None
                if request.include_stress_tests:
                    stress_tests = self._run_stress_tests(
                        pool, 
                        cashflows, 
                        metrics, 
                        request.analysis_date,
                        request.discount_rate or 0.06
                    )
                
                # Build response
                result = AssetPoolAnalysisResponse(
                    pool_name=request.pool.pool_name,
                    analysis_date=request.analysis_date,
                    execution_time=time.time() - start_time,
                    status="success",
                    metrics=metrics,
                    cashflows=cashflows if request.include_cashflows else None,
                    stress_tests=stress_tests,
                    analytics=analytics
                )
                
                return result
                
        except Exception as e:
            logger.exception(f"Error analyzing commercial loan pool: {str(e)}")
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date,
                execution_time=time.time() - start_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_absbox_loans(self, commercial_loans: List[CommercialLoan]) -> List[Any]:
        """
        Create AbsBox loan objects from commercial loan models
        
        Args:
            commercial_loans: List of commercial loan models
            
        Returns:
            List of AbsBox loan objects
        """
        ab_loans = []
        
        for loan in commercial_loans:
            try:
                # Use specialized commercial loan type from AbsBox
                if loan.rate_type.value == "fixed":
                    ab_loan = ab.local.CommercialMortgage(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                elif loan.rate_type.value == "floating":
                    ab_loan = ab.local.FloatingRateMortgage(
                        balance=loan.balance,
                        rate=loan.rate,
                        margin=0.025,  # Typical margin for commercial loans
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                else:
                    # Default to commercial mortgage for other types
                    logger.warning(f"Unrecognized rate type {loan.rate_type}, defaulting to commercial mortgage")
                    ab_loan = ab.local.CommercialMortgage(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months
                    )
                
                # Set amortization term if different from loan term
                if loan.amortization_months and loan.amortization_months != loan.term_months:
                    ab_loan.amortTerm = loan.amortization_months
                
                # Set interest-only period if applicable
                if loan.is_interest_only and loan.interest_only_period_months:
                    ab_loan.IOTerm = loan.interest_only_period_months
                
                # Add DSCR if available
                if loan.dscr:
                    ab_loan.dscr = loan.dscr
                    # Use DSCR to influence default probability
                    if loan.dscr < 1.0:
                        ab_loan.defaultRate = 0.10  # High default risk
                    elif loan.dscr < 1.25:
                        ab_loan.defaultRate = 0.05  # Medium default risk
                    elif loan.dscr < 1.5:
                        ab_loan.defaultRate = 0.02  # Lower default risk
                    else:
                        ab_loan.defaultRate = 0.01  # Low default risk
                
                # Add LTV if available
                if loan.ltv_ratio:
                    ab_loan.ltv = loan.ltv_ratio
                    # Use LTV to influence recovery rate
                    if loan.ltv_ratio > 0.8:
                        ab_loan.recoveryRate = 0.6  # Lower recovery rate for high LTV
                    else:
                        ab_loan.recoveryRate = 0.75  # Higher recovery rate for low LTV
                
                # Add balloon payment if applicable
                if loan.balloon_payment:
                    ab_loan.balloon = loan.balloon_payment
                
                # Add to list
                ab_loans.append(ab_loan)
            except Exception as e:
                logger.error(f"Error creating AbsBox loan for commercial loan: {str(e)}")
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
            logger.info(f"Creating AbsBox pool with {len(loans)} loans")
            
            # Create pool with appropriate settings for commercial loans
            pool = ab.local.Pool(
                assets=loans,
                name=name,
                start_date=cut_off_date.strftime('%Y-%m-%d')
            )
            
            # Set commercial loan specific pool parameters
            pool.default_lag = 2  # 2 months lag for defaults
            pool.recovery_lag = 12  # 12 months to recover for commercial properties
            pool.delinquency_threshold = 90  # 90 days delinquent before default
            
            return pool
        except Exception as e:
            logger.exception(f"Error creating AbsBox pool: {str(e)}")
            raise ValueError(f"Failed to create AbsBox pool: {str(e)}")
    
    def _calculate_cashflows(self, pool: Any, analysis_date: date) -> List[AssetPoolCashflow]:
        """
        Calculate cashflows for the pool
        
        Args:
            pool: AbsBox pool object
            analysis_date: Date to start analysis from
            
        Returns:
            List of cashflow projections
        """
        try:
            logger.info("Calculating cashflows")
            # Use specialized AbsBox service to run cashflow projections
            with CALCULATION_TIME.labels(calculation_type="commercial_loan_cashflows").time():
                # Run cashflow projections with appropriate settings for commercial loans
                cf = self.absbox_service.run_cashflow_projection(
                    pool=pool,
                    start_date=analysis_date,
                    prepayment_model="CPR",
                    prepayment_rate=0.05,    # 5% annual prepayment for commercial loans
                    default_model="CDR",
                    default_rate=0.02,       # 2% annual default rate
                    recovery_rate=0.65       # 65% recovery on commercial loans
                )
                
                # Convert AbsBox cashflows to our model
                cashflows = []
                for i, row in cf.iterrows():
                    cashflow = AssetPoolCashflow(
                        period=i+1,
                        date=row.get('date') if isinstance(row.get('date'), date) else analysis_date,
                        scheduled_principal=float(row.get('scheduled_principal', 0)),
                        scheduled_interest=float(row.get('scheduled_interest', 0)),
                        prepayment=float(row.get('prepayment', 0)),
                        default=float(row.get('default', 0)),
                        recovery=float(row.get('recovery', 0)),
                        loss=float(row.get('loss', 0)),
                        balance=float(row.get('balance', 0))
                    )
                    cashflows.append(cashflow)
                
                return cashflows
        except Exception as e:
            logger.exception(f"Error calculating cashflows: {str(e)}")
            raise ValueError(f"Failed to calculate cashflows: {str(e)}")
    
    def _calculate_metrics(self, cashflows: List[AssetPoolCashflow], discount_rate: float, 
                         analysis_date: date) -> AssetPoolMetrics:
        """
        Calculate metrics for the cashflows
        
        Args:
            cashflows: List of cashflow projections
            discount_rate: Rate to discount future cashflows
            analysis_date: Date to start analysis from
            
        Returns:
            AssetPoolMetrics with calculated metrics
        """
        try:
            logger.info("Calculating commercial loan metrics")
            
            # Simplified metric calculation for brevity
            # Convert cashflows to arrays for calculation
            cf_dates = np.array([(cf.date - analysis_date).days/365 for cf in cashflows])
            cf_scheduled_principal = np.array([cf.scheduled_principal for cf in cashflows])
            cf_scheduled_interest = np.array([cf.scheduled_interest for cf in cashflows])
            cf_prepayment = np.array([cf.prepayment for cf in cashflows])
            cf_recovery = np.array([cf.recovery for cf in cashflows])
            
            # Calculate total cashflows
            total_principal = np.sum(cf_scheduled_principal) + np.sum(cf_prepayment)
            total_interest = np.sum(cf_scheduled_interest)
            total_cashflow = total_principal + total_interest + np.sum(cf_recovery)
            
            # Calculate present value
            cashflow = cf_scheduled_principal + cf_scheduled_interest + cf_prepayment + cf_recovery
            npv = np.sum(cashflow / (1 + discount_rate) ** cf_dates)
            
            # Calculate duration (Macaulay duration)
            weights = cashflow / np.sum(cashflow)
            duration = np.sum(weights * cf_dates)
            
            # Calculate weighted average life (WAL)
            weights_principal = (cf_scheduled_principal + cf_prepayment) / (total_principal)
            wal = np.sum(weights_principal * cf_dates)
            
            # Create metrics object
            metrics = AssetPoolMetrics(
                total_principal=float(total_principal),
                total_interest=float(total_interest),
                total_cashflow=float(total_cashflow),
                npv=float(npv),
                duration=float(duration),
                weighted_average_life=float(wal)
            )
            
            return metrics
        except Exception as e:
            logger.exception(f"Error calculating metrics: {str(e)}")
            # Return basic metrics if calculation fails
            return AssetPoolMetrics(
                total_principal=sum(cf.scheduled_principal for cf in cashflows),
                total_interest=sum(cf.scheduled_interest for cf in cashflows),
                total_cashflow=sum(cf.scheduled_principal + cf.scheduled_interest for cf in cashflows),
                npv=0.0  # Cannot calculate NPV if error occurs
            )
    
    def _run_commercial_loan_analytics(self, 
                                     commercial_assets: List[CommercialLoan], 
                                     cashflows: List[AssetPoolCashflow]) -> Dict[str, Any]:
        """
        Run specialized commercial loan analytics
        
        Args:
            commercial_assets: List of commercial loan assets
            cashflows: List of cashflow projections
            
        Returns:
            Dict containing specialized analytics
        """
        try:
            logger.info("Running specialized commercial loan analytics")
            
            # Property type distribution
            property_types = {}
            total_balance = sum(asset.balance for asset in commercial_assets)
            
            for asset in commercial_assets:
                if asset.property_type:
                    property_type = asset.property_type.value
                    if property_type not in property_types:
                        property_types[property_type] = 0
                    property_types[property_type] += asset.balance
            
            # Calculate property type distribution percentages
            property_distribution = {
                k: v/total_balance for k, v in property_types.items()
            } if total_balance > 0 else {}
            
            # Calculate weighted average DSCR
            weighted_dscr = 0.0
            dscr_coverage = 0.0
            
            for asset in commercial_assets:
                if asset.dscr:
                    weighted_dscr += asset.dscr * asset.balance
                    dscr_coverage += asset.balance
            
            avg_dscr = weighted_dscr / dscr_coverage if dscr_coverage > 0 else None
            
            # Calculate weighted average LTV
            weighted_ltv = 0.0
            ltv_coverage = 0.0
            
            for asset in commercial_assets:
                if asset.ltv_ratio:
                    weighted_ltv += asset.ltv_ratio * asset.balance
                    ltv_coverage += asset.balance
            
            avg_ltv = weighted_ltv / ltv_coverage if ltv_coverage > 0 else None
            
            # Calculate balloon risk
            balloon_balance = sum(asset.balloon_payment or 0 for asset in commercial_assets)
            balloon_percentage = balloon_balance / total_balance if total_balance > 0 else 0
            
            # Concentration analysis
            concentration_threshold = 0.05  # 5% of pool
            large_loans = [
                {
                    "id": asset.id,
                    "balance": asset.balance,
                    "percentage": asset.balance / total_balance if total_balance > 0 else 0,
                    "property_type": asset.property_type.value if asset.property_type else "unknown"
                }
                for asset in commercial_assets
                if asset.balance / total_balance >= concentration_threshold
            ] if total_balance > 0 else []
            
            # Create analytics report
            analytics = {
                "property_distribution": property_distribution,
                "average_metrics": {
                    "dscr": avg_dscr,
                    "ltv": avg_ltv
                },
                "balloon_risk": {
                    "balloon_balance": balloon_balance,
                    "balloon_percentage": balloon_percentage
                },
                "concentration_risk": {
                    "large_loans": large_loans,
                    "large_loan_count": len(large_loans),
                    "large_loan_percentage": sum(loan["percentage"] for loan in large_loans)
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.exception(f"Error in commercial loan analytics: {str(e)}")
            return {"error": str(e), "error_type": type(e).__name__}
    
    def _run_stress_tests(self, pool: Any, cashflows: List[AssetPoolCashflow], 
                        metrics: AssetPoolMetrics, analysis_date: date,
                        discount_rate: float) -> List[AssetPoolStressTest]:
        """
        Run stress tests on the commercial loan pool
        
        Args:
            pool: AbsBox pool object
            cashflows: List of cashflow projections
            metrics: Calculated metrics for the base case
            analysis_date: Date to start analysis from
            discount_rate: Rate to discount future cashflows
            
        Returns:
            List of stress test results
        """
        try:
            logger.info("Running commercial loan stress tests")
            
            stress_tests = []
            base_npv = metrics.npv
            
            # Define commercial loan specific stress scenarios
            scenarios = [
                {
                    "name": "cap_rate_expansion",
                    "description": "Cap rate expansion stress (property value decline)",
                    "default_multiplier": 1.5,
                    "recovery_multiplier": 0.8
                },
                {
                    "name": "vacancy_increase",
                    "description": "Increased vacancy rates",
                    "default_multiplier": 2.0,
                    "recovery_multiplier": 0.9
                },
                {
                    "name": "interest_rate_shock",
                    "description": "Interest rate shock (+200 bps)",
                    "discount_rate_delta": 0.02
                },
                {
                    "name": "severe_recession",
                    "description": "Severe recession scenario",
                    "default_multiplier": 3.0,
                    "recovery_multiplier": 0.7,
                    "discount_rate_delta": 0.01
                }
            ]
            
            # Run each stress scenario
            for scenario in scenarios:
                try:
                    # Apply stress parameters
                    stress_pool = pool  # Reuse the pool
                    
                    # Adjust discount rate if specified
                    scenario_discount_rate = discount_rate
                    if "discount_rate_delta" in scenario:
                        scenario_discount_rate += scenario["discount_rate_delta"]
                    
                    # Run stressed cashflow projection
                    stress_cf = self.absbox_service.run_cashflow_projection(
                        pool=stress_pool,
                        start_date=analysis_date,
                        prepayment_model="CPR",
                        prepayment_rate=0.05,  # Keep prepayment constant
                        default_model="CDR",
                        default_rate=0.02 * scenario.get("default_multiplier", 1.0),
                        recovery_rate=0.65 * scenario.get("recovery_multiplier", 1.0)
                    )
                    
                    # Convert to our cashflow model
                    stress_cashflows = []
                    for i, row in stress_cf.iterrows():
                        cashflow = AssetPoolCashflow(
                            period=i+1,
                            date=row.get('date') if isinstance(row.get('date'), date) else analysis_date,
                            scheduled_principal=float(row.get('scheduled_principal', 0)),
                            scheduled_interest=float(row.get('scheduled_interest', 0)),
                            prepayment=float(row.get('prepayment', 0)),
                            default=float(row.get('default', 0)),
                            recovery=float(row.get('recovery', 0)),
                            loss=float(row.get('loss', 0)),
                            balance=float(row.get('balance', 0))
                        )
                        stress_cashflows.append(cashflow)
                    
                    # Calculate NPV under stress
                    cf_dates = np.array([(cf.date - analysis_date).days/365 for cf in stress_cashflows])
                    cashflow = np.array([
                        cf.scheduled_principal + cf.scheduled_interest + cf.prepayment + cf.recovery
                        for cf in stress_cashflows
                    ])
                    stress_npv = np.sum(cashflow / (1 + scenario_discount_rate) ** cf_dates)
                    
                    # Calculate impact
                    npv_change = stress_npv - base_npv
                    npv_change_percent = npv_change / base_npv * 100 if base_npv != 0 else float('inf')
                    
                    # Create stress test result
                    stress_test = AssetPoolStressTest(
                        scenario_name=scenario["name"],
                        description=scenario["description"],
                        npv=float(stress_npv),
                        npv_change=float(npv_change),
                        npv_change_percent=float(npv_change_percent)
                    )
                    
                    stress_tests.append(stress_test)
                    
                except Exception as e:
                    logger.error(f"Error in stress scenario {scenario['name']}: {str(e)}")
                    # Include failed scenario with error details
                    stress_test = AssetPoolStressTest(
                        scenario_name=scenario["name"],
                        description=f"Error: {str(e)}",
                        npv=0.0,
                        npv_change=0.0,
                        npv_change_percent=0.0
                    )
                    stress_tests.append(stress_test)
            
            return stress_tests
            
        except Exception as e:
            logger.exception(f"Error running stress tests: {str(e)}")
            return []
