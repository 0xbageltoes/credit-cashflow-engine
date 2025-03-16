"""
Consumer Credit Asset Handler

Production-quality implementation for analyzing consumer credit assets using
the AbsBox engine with comprehensive error handling and detailed analytics.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import date
from app.core.cache_service import CacheService

# Import AbsBox libraries
import absbox as ab
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

from app.core.monitoring import CalculationTracker, CALCULATION_TIME
from app.models.asset_classes import (
    AssetClass, ConsumerCredit, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

class ConsumerCreditHandler:
    """
    Handler for consumer credit asset analysis
    
    Production-ready implementation of consumer credit analytics with
    comprehensive error handling, metrics, and stress testing.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None, cache_service: Optional[CacheService] = None):
        """
        Initialize the consumer credit handler
        
        Args:
            absbox_service: The AbsBox service to use (created if not provided)
            cache_service: Optional cache service for performance optimization
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        self.cache_service = cache_service or CacheService()
        logger.info("ConsumerCreditHandler initialized")
    
    def analyze_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Analyze a pool of consumer credit assets
        
        Args:
            request: The analysis request containing pool and parameters
            
        Returns:
            AssetPoolAnalysisResponse with detailed analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing consumer credit pool: {request.pool.pool_name}")
            with CalculationTracker("consumer_credit_analysis"):
                # Filter to ensure we only process consumer credit assets
                consumer_assets = [
                    asset for asset in request.pool.assets 
                    if asset.asset_class == AssetClass.CONSUMER_CREDIT
                ]
                
                if not consumer_assets:
                    raise ValueError("No consumer credit assets found in pool")
                
                # Create AbsBox loan objects
                loans = self._create_absbox_loans(consumer_assets)
                
                # Create pool and run analysis
                pool = self._create_absbox_pool(loans, request.pool.pool_name, 
                                              request.pool.cut_off_date)
                
                # Calculate cashflows with appropriate settings
                cashflows = self._calculate_cashflows(pool, request.analysis_date)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    cashflows, 
                    request.discount_rate or 0.07,  # Default 7% for consumer loans if not provided
                    request.analysis_date
                )
                
                # Run specialized consumer credit analytics
                analytics = self._run_consumer_credit_analytics(
                    consumer_assets, 
                    cashflows, 
                    request.analysis_date
                )
                
                # Run stress tests if requested
                stress_tests = None
                if request.include_stress_tests:
                    stress_tests = self._run_stress_tests(
                        pool, 
                        cashflows, 
                        metrics, 
                        request.analysis_date,
                        request.discount_rate or 0.07
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
            logger.exception(f"Error analyzing consumer credit pool: {str(e)}")
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date,
                execution_time=time.time() - start_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_absbox_loans(self, consumer_loans: List[ConsumerCredit]) -> List[Any]:
        """
        Create AbsBox loan objects from consumer credit models
        
        Args:
            consumer_loans: List of consumer credit models
            
        Returns:
            List of AbsBox loan objects
        """
        ab_loans = []
        
        for loan in consumer_loans:
            try:
                # Use specialized consumer credit loan type from AbsBox
                if loan.rate_type.value == "fixed":
                    ab_loan = ab.local.PersonalLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months,
                        defaultRate=0.02  # Default value, should be calibrated
                    )
                elif loan.rate_type.value == "floating":
                    ab_loan = ab.local.FloatingRateLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        margin=0.03,  # Higher margin typical for consumer loans
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months,
                        defaultRate=0.02  # Default value, should be calibrated
                    )
                else:
                    # Default to personal loan for other types
                    logger.warning(f"Unrecognized rate type {loan.rate_type}, defaulting to personal loan")
                    ab_loan = ab.local.PersonalLoan(
                        balance=loan.balance,
                        rate=loan.rate,
                        originBalance=loan.original_balance,
                        originRate=loan.rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.remaining_term_months,
                        defaultRate=0.02  # Default value, should be calibrated
                    )
                
                # Add credit risk factors if available
                if loan.fico_score:
                    # Adjust default rate based on FICO score
                    if loan.fico_score < 600:
                        ab_loan.defaultRate = 0.08  # High risk
                    elif loan.fico_score < 660:
                        ab_loan.defaultRate = 0.05  # Medium risk
                    elif loan.fico_score < 720:
                        ab_loan.defaultRate = 0.02  # Average risk
                    else:
                        ab_loan.defaultRate = 0.01  # Low risk
                
                # Add debt-to-income impact if available
                if loan.debt_to_income:
                    # Higher DTI increases default risk
                    dti_factor = min(2.0, 1.0 + loan.debt_to_income * 2)
                    ab_loan.defaultRate *= dti_factor
                
                # Add secured/unsecured status
                if loan.is_secured and loan.collateral_value:
                    ab_loan.recovery = 0.7  # 70% recovery for secured loans
                else:
                    ab_loan.recovery = 0.2  # 20% recovery for unsecured loans
                
                # Add loan to list
                ab_loans.append(ab_loan)
            except Exception as e:
                logger.error(f"Error creating AbsBox loan for consumer loan: {str(e)}")
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
            
            # Create pool with appropriate settings for consumer loans
            pool = ab.local.Pool(
                assets=loans,
                name=name,
                start_date=cut_off_date.strftime('%Y-%m-%d')
            )
            
            # Set consumer credit specific pool parameters
            pool.default_lag = 1  # 1 month lag for defaults
            pool.recovery_lag = 3  # 3 months to recover
            pool.delinquency_threshold = 60  # 60 days delinquent before default
            
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
            with CALCULATION_TIME.labels(calculation_type="consumer_credit_cashflows").time():
                # Run cashflow projections with appropriate settings for consumer loans
                cf = self.absbox_service.run_cashflow_projection(
                    pool=pool,
                    start_date=analysis_date,
                    prepayment_model="CPR",  # Constant Prepayment Rate
                    prepayment_rate=0.08,    # 8% annual prepayment for consumer loans
                    default_model="CDR",     # Constant Default Rate
                    default_rate=0.03,       # 3% annual default rate
                    recovery_rate=0.35       # 35% recovery on consumer loans
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
            logger.info("Calculating consumer loan metrics")
            with CALCULATION_TIME.labels(calculation_type="consumer_credit_metrics").time():
                # Convert cashflows to numpy arrays for efficient calculation
                cf_dates = np.array([(cf.date - analysis_date).days/365 for cf in cashflows])
                cf_scheduled_principal = np.array([cf.scheduled_principal for cf in cashflows])
                cf_scheduled_interest = np.array([cf.scheduled_interest for cf in cashflows])
                cf_prepayment = np.array([cf.prepayment for cf in cashflows])
                cf_loss = np.array([cf.loss for cf in cashflows])
                cf_recovery = np.array([cf.recovery for cf in cashflows])
                cf_balance = np.array([cf.balance for cf in cashflows])
                
                # Calculate total cashflows
                total_principal = np.sum(cf_scheduled_principal) + np.sum(cf_prepayment)
                total_interest = np.sum(cf_scheduled_interest)
                total_recovery = np.sum(cf_recovery)
                total_loss = np.sum(cf_loss)
                total_cashflow = total_principal + total_interest + total_recovery
                
                # Calculate present value
                cashflow = cf_scheduled_principal + cf_scheduled_interest + cf_prepayment + cf_recovery
                npv = np.sum(cashflow / (1 + discount_rate) ** cf_dates)
                
                # Calculate duration (Macaulay duration)
                weights = cashflow / np.sum(cashflow)
                duration = np.sum(weights * cf_dates)
                
                # Calculate modified duration
                modified_duration = duration / (1 + discount_rate)
                
                # Calculate weighted average life (WAL)
                weights_principal = (cf_scheduled_principal + cf_prepayment) / (total_principal)
                wal = np.sum(weights_principal * cf_dates)
                
                # Calculate IRR - iterative process
                # Set up cash flows for IRR calculation
                cashflow_irr = -cf_balance[0]  # Initial investment is negative
                for cf in cashflow[1:]:
                    cashflow_irr = np.append(cashflow_irr, cf)
                
                # Use numpy financial to estimate IRR
                try:
                    from numpy import irr
                    irr_value = float(irr(cashflow_irr))
                except:
                    irr_value = None
                
                # Create metrics object
                metrics = AssetPoolMetrics(
                    total_principal=float(total_principal),
                    total_interest=float(total_interest),
                    total_cashflow=float(total_cashflow),
                    npv=float(npv),
                    irr=float(irr_value) if irr_value is not None else None,
                    duration=float(duration),
                    modified_duration=float(modified_duration),
                    weighted_average_life=float(wal),
                    # Additional consumer credit specific metrics
                    convexity=None,  # Would require second derivative calculation
                    yield_to_maturity=None  # Would require solving for YTM
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
    
    def _run_consumer_credit_analytics(self, 
                                     consumer_assets: List[ConsumerCredit], 
                                     cashflows: List[AssetPoolCashflow],
                                     analysis_date: date) -> Dict[str, Any]:
        """
        Run specialized consumer credit analytics
        
        Args:
            consumer_assets: List of consumer credit assets
            cashflows: List of cashflow projections
            analysis_date: Date to start analysis from
            
        Returns:
            Dict containing specialized analytics
        """
        try:
            logger.info("Running specialized consumer credit analytics")
            
            # Calculate risk segmentation
            risk_segments = {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "unknown_risk": 0,
                "total_balance": 0
            }
            
            for asset in consumer_assets:
                balance = asset.balance
                risk_segments["total_balance"] += balance
                
                if asset.fico_score:
                    if asset.fico_score < 620:
                        risk_segments["high_risk"] += balance
                    elif asset.fico_score < 720:
                        risk_segments["medium_risk"] += balance
                    else:
                        risk_segments["low_risk"] += balance
                else:
                    risk_segments["unknown_risk"] += balance
            
            # Calculate purpose distribution
            purpose_distribution = {}
            for asset in consumer_assets:
                purpose = asset.loan_purpose.value
                if purpose not in purpose_distribution:
                    purpose_distribution[purpose] = 0
                
                purpose_distribution[purpose] += asset.balance
            
            # Calculate secured vs unsecured distribution
            secured_total = sum(a.balance for a in consumer_assets if a.is_secured)
            unsecured_total = sum(a.balance for a in consumer_assets if not a.is_secured)
            total_balance = sum(a.balance for a in consumer_assets)
            
            # Calculate expected losses
            expected_loss_rate = 0.0
            if total_balance > 0:
                # Estimate expected loss rate based on FICO scores if available
                if any(a.fico_score for a in consumer_assets):
                    fico_losses = {
                        range(300, 580): 0.15,    # Very high risk
                        range(580, 620): 0.10,    # High risk
                        range(620, 660): 0.05,    # Medium-high risk
                        range(660, 720): 0.02,    # Medium risk
                        range(720, 780): 0.01,    # Medium-low risk
                        range(780, 851): 0.005    # Low risk
                    }
                    
                    weighted_loss = 0.0
                    scored_balance = 0.0
                    
                    for asset in consumer_assets:
                        if asset.fico_score:
                            score = asset.fico_score
                            loss_rate = 0.05  # Default
                            
                            for score_range, rate in fico_losses.items():
                                if score in score_range:
                                    loss_rate = rate
                                    break
                            
                            weighted_loss += asset.balance * loss_rate
                            scored_balance += asset.balance
                    
                    if scored_balance > 0:
                        expected_loss_rate = weighted_loss / scored_balance
                else:
                    # If no FICO scores, use secured status
                    secured_loss_rate = 0.02    # Lower losses for secured loans
                    unsecured_loss_rate = 0.06  # Higher losses for unsecured
                    
                    expected_loss_rate = (
                        (secured_total * secured_loss_rate) + 
                        (unsecured_total * unsecured_loss_rate)
                    ) / total_balance
            
            # Create analytics report
            analytics = {
                "risk_segmentation": {
                    "high_risk_pct": risk_segments["high_risk"] / risk_segments["total_balance"] if risk_segments["total_balance"] > 0 else 0,
                    "medium_risk_pct": risk_segments["medium_risk"] / risk_segments["total_balance"] if risk_segments["total_balance"] > 0 else 0,
                    "low_risk_pct": risk_segments["low_risk"] / risk_segments["total_balance"] if risk_segments["total_balance"] > 0 else 0,
                    "unknown_risk_pct": risk_segments["unknown_risk"] / risk_segments["total_balance"] if risk_segments["total_balance"] > 0 else 0
                },
                "purpose_distribution": {k: v/total_balance for k, v in purpose_distribution.items()},
                "security_analysis": {
                    "secured_pct": secured_total / total_balance if total_balance > 0 else 0,
                    "unsecured_pct": unsecured_total / total_balance if total_balance > 0 else 0
                },
                "expected_loss_rate": expected_loss_rate,
                "expected_loss_amount": expected_loss_rate * total_balance
            }
            
            return analytics
            
        except Exception as e:
            logger.exception(f"Error in consumer credit analytics: {str(e)}")
            return {"error": str(e), "error_type": type(e).__name__}
    
    def _run_stress_tests(self, pool: Any, cashflows: List[AssetPoolCashflow], 
                         metrics: AssetPoolMetrics, analysis_date: date,
                         discount_rate: float) -> List[AssetPoolStressTest]:
        """
        Run stress tests on the consumer credit pool
        
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
            logger.info("Running consumer credit stress tests")
            
            stress_tests = []
            base_npv = metrics.npv
            
            # Define consumer credit specific stress scenarios
            scenarios = [
                {
                    "name": "high_default",
                    "description": "High default scenario (2x default rate)",
                    "default_multiplier": 2.0,
                    "prepayment_multiplier": 0.8
                },
                {
                    "name": "extreme_default",
                    "description": "Extreme default scenario (3x default rate)",
                    "default_multiplier": 3.0,
                    "prepayment_multiplier": 0.5
                },
                {
                    "name": "interest_rate_shock",
                    "description": "Interest rate shock (+300 bps)",
                    "discount_rate_delta": 0.03
                },
                {
                    "name": "combined_stress",
                    "description": "Combined stress (2x defaults, +200 bps rate shock)",
                    "default_multiplier": 2.0,
                    "discount_rate_delta": 0.02
                }
            ]
            
            # Run each stress scenario
            for scenario in scenarios:
                try:
                    # Apply stress to pool
                    stress_pool = pool  # Create a copy or reuse the same pool
                    
                    # Adjust discount rate if specified
                    scenario_discount_rate = discount_rate
                    if "discount_rate_delta" in scenario:
                        scenario_discount_rate += scenario["discount_rate_delta"]
                    
                    # Run cashflow projection with stressed parameters
                    stress_cf = self.absbox_service.run_cashflow_projection(
                        pool=stress_pool,
                        start_date=analysis_date,
                        prepayment_model="CPR",
                        prepayment_rate=0.08 * scenario.get("prepayment_multiplier", 1.0),
                        default_model="CDR",
                        default_rate=0.03 * scenario.get("default_multiplier", 1.0),
                        recovery_rate=0.35  # Keep recovery constant
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
                    stress_tests.append(
                        AssetPoolStressTest(
                            scenario_name=scenario["name"],
                            description=f"Error: {str(e)}",
                            npv=0.0,
                            npv_change=0.0,
                            npv_change_percent=0.0
                        )
                    )
            
            return stress_tests
            
        except Exception as e:
            logger.exception(f"Error running stress tests: {str(e)}")
            return []
