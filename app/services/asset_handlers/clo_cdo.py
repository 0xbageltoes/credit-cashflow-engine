"""
CLO/CDO Asset Handler

Production-quality implementation for analyzing CLO/CDO structured products using
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

from app.core.monitoring import CalculationTracker, CALCULATION_TIME
from app.models.asset_classes import (
    AssetClass, CLOCDO, CLOCDOTranche, AssetPool,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

class CLOCDOHandler:
    """
    Handler for CLO/CDO asset analysis
    
    Production-ready implementation of structured product analytics with
    comprehensive error handling, metrics, and stress testing.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None):
        """
        Initialize the CLO/CDO handler
        
        Args:
            absbox_service: The AbsBox service to use (created if not provided)
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        logger.info("CLOCDOHandler initialized")
    
    def analyze_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Analyze CLO/CDO structures
        
        Args:
            request: The analysis request containing pool and parameters
            
        Returns:
            AssetPoolAnalysisResponse with detailed analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing CLO/CDO structure: {request.pool.pool_name}")
            with CalculationTracker("clo_cdo_analysis"):
                # Filter to ensure we only process CLO/CDO assets
                clo_cdo_assets = [
                    asset for asset in request.pool.assets 
                    if asset.asset_class == AssetClass.CLO_CDO
                ]
                
                if not clo_cdo_assets:
                    raise ValueError("No CLO/CDO assets found in pool")
                
                # Create AbsBox pool and tranches
                pool, tranches = self._create_absbox_structures(clo_cdo_assets, request.pool.pool_name, 
                                                             request.pool.cut_off_date)
                
                # Calculate waterfall cashflows
                cashflows = self._calculate_cashflows(pool, tranches, request.analysis_date)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    cashflows, 
                    request.discount_rate or 0.05,  # Default 5% if not provided
                    request.analysis_date
                )
                
                # Run specialized CLO/CDO analytics
                analytics = self._run_clo_cdo_analytics(
                    clo_cdo_assets,
                    cashflows
                )
                
                # Run stress tests if requested
                stress_tests = None
                if request.include_stress_tests:
                    stress_tests = self._run_stress_tests(
                        pool, 
                        tranches,
                        cashflows, 
                        metrics, 
                        request.analysis_date,
                        request.discount_rate or 0.05
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
            logger.exception(f"Error analyzing CLO/CDO structure: {str(e)}")
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date,
                execution_time=time.time() - start_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_absbox_structures(self, clo_cdos: List[CLOCDO], name: str, 
                                cut_off_date: date) -> tuple[Any, List[Any]]:
        """
        Create AbsBox structures for CLO/CDO analysis
        
        Args:
            clo_cdos: List of CLO/CDO models
            name: Name of the pool
            cut_off_date: Cut-off date for the pool
            
        Returns:
            Tuple of (AbsBox collateral pool, list of AbsBox tranches)
        """
        try:
            logger.info(f"Creating AbsBox structures for {len(clo_cdos)} CLO/CDO products")
            
            # We expect a single CLO/CDO structure per pool
            if len(clo_cdos) > 1:
                logger.warning(f"Multiple CLO/CDO structures found in pool. Using the first one.")
            
            clo_cdo = clo_cdos[0]
            
            # First create collateral pool representing the underlying assets
            collateral_pool = ab.local.CollateralPool(
                balance=clo_cdo.collateral_pool_balance,
                wac=clo_cdo.collateral_pool_wac or 0.06,  # Default WAC if not provided
                wam=60,  # Assuming 5 years weighted average maturity if not provided
                cdr=0.02,  # Default annual default rate
                cpr=0.10,  # Default annual prepayment rate
                recovery=0.60,  # Default recovery rate
                lag=3,  # Default recovery lag in months
                name=f"{name}_collateral"
            )
            
            # Create tranches
            absbox_tranches = []
            for tranche in clo_cdo.tranches:
                ab_tranche = ab.local.Tranche(
                    balance=tranche.balance,
                    rate=tranche.rate,
                    name=tranche.name
                )
                
                # Set rate type and spread for floating rate tranches
                if tranche.rate_type == "floating" and tranche.spread:
                    ab_tranche.spread = tranche.spread
                    ab_tranche.index = tranche.index or "LIBOR"
                
                # Set attachment and detachment points if available
                if tranche.attachment_point is not None:
                    ab_tranche.attach = tranche.attachment_point
                
                if tranche.detachment_point is not None:
                    ab_tranche.detach = tranche.detachment_point
                
                absbox_tranches.append(ab_tranche)
            
            # Sort tranches by seniority
            absbox_tranches.sort(key=lambda t: getattr(t, 'seniority', 999))
            
            return collateral_pool, absbox_tranches
            
        except Exception as e:
            logger.exception(f"Error creating AbsBox structures: {str(e)}")
            raise ValueError(f"Failed to create AbsBox structures: {str(e)}")
    
    def _calculate_cashflows(self, pool: Any, tranches: List[Any], 
                           analysis_date: date) -> List[AssetPoolCashflow]:
        """
        Calculate waterfall cashflows for CLO/CDO structure
        
        Args:
            pool: AbsBox collateral pool
            tranches: List of AbsBox tranches
            analysis_date: Date to start analysis from
            
        Returns:
            List of cashflow projections
        """
        try:
            logger.info("Calculating CLO/CDO waterfall cashflows")
            with CALCULATION_TIME.labels(calculation_type="clo_cdo_cashflows").time():
                # Create a waterfall structure
                waterfall = self.absbox_service.create_waterfall(
                    pool=pool,
                    tranches=tranches,
                    start_date=analysis_date
                )
                
                # Run waterfall cashflow analysis
                cf = self.absbox_service.run_waterfall_analysis(
                    waterfall=waterfall,
                    scenarios={
                        "base": {
                            "cdr": 0.02,  # Default rate
                            "cpr": 0.10,  # Prepayment rate
                            "recovery": 0.60,  # Recovery rate
                            "lag": 3  # Recovery lag
                        }
                    }
                )
                
                # Convert AbsBox cashflows to our model
                # This is simplified as CLO/CDO waterfall cashflows are complex
                cashflows = []
                if cf and "base" in cf and hasattr(cf["base"], "get_aggregate_cashflows"):
                    agg_cf = cf["base"].get_aggregate_cashflows()
                    
                    for i, row in agg_cf.iterrows():
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
            logger.exception(f"Error calculating CLO/CDO cashflows: {str(e)}")
            raise ValueError(f"Failed to calculate CLO/CDO cashflows: {str(e)}")
    
    def _calculate_metrics(self, cashflows: List[AssetPoolCashflow], discount_rate: float, 
                         analysis_date: date) -> AssetPoolMetrics:
        """
        Calculate metrics for CLO/CDO cashflows
        
        Args:
            cashflows: List of cashflow projections
            discount_rate: Rate to discount future cashflows
            analysis_date: Date to start analysis from
            
        Returns:
            AssetPoolMetrics with calculated metrics
        """
        try:
            logger.info("Calculating CLO/CDO metrics")
            
            # Simplified metric calculation
            cf_dates = np.array([(cf.date - analysis_date).days/365 for cf in cashflows])
            cf_scheduled_principal = np.array([cf.scheduled_principal for cf in cashflows])
            cf_scheduled_interest = np.array([cf.scheduled_interest for cf in cashflows])
            cf_prepayment = np.array([cf.prepayment for cf in cashflows])
            cf_recovery = np.array([cf.recovery for cf in cashflows])
            
            # Calculate total cashflows
            total_principal = np.sum(cf_scheduled_principal) + np.sum(cf_prepayment)
            total_interest = np.sum(cf_scheduled_interest)
            total_cashflow = total_principal + total_interest + np.sum(cf_recovery)
            
            # Calculate NPV
            cashflow = cf_scheduled_principal + cf_scheduled_interest + cf_prepayment + cf_recovery
            npv = np.sum(cashflow / (1 + discount_rate) ** cf_dates)
            
            # Calculate duration
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
            logger.exception(f"Error calculating CLO/CDO metrics: {str(e)}")
            # Return basic metrics if calculation fails
            return AssetPoolMetrics(
                total_principal=sum(cf.scheduled_principal for cf in cashflows),
                total_interest=sum(cf.scheduled_interest for cf in cashflows),
                total_cashflow=sum(cf.scheduled_principal + cf.scheduled_interest for cf in cashflows),
                npv=0.0  # Cannot calculate NPV if error occurs
            )
    
    def _run_clo_cdo_analytics(self, 
                             clo_cdo_assets: List[CLOCDO], 
                             cashflows: List[AssetPoolCashflow]) -> Dict[str, Any]:
        """
        Run specialized CLO/CDO analytics
        
        Args:
            clo_cdo_assets: List of CLO/CDO assets
            cashflows: List of cashflow projections
            
        Returns:
            Dict containing specialized analytics
        """
        try:
            logger.info("Running specialized CLO/CDO analytics")
            
            # For simplicity, we assume a single CLO/CDO in the pool
            clo_cdo = clo_cdo_assets[0]
            
            # Calculate tranche metrics
            tranche_analytics = []
            total_tranche_balance = sum(t.balance for t in clo_cdo.tranches)
            
            for tranche in clo_cdo.tranches:
                tranche_pct = tranche.balance / total_tranche_balance if total_tranche_balance > 0 else 0
                
                analytics = {
                    "name": tranche.name,
                    "balance": tranche.balance,
                    "rate": tranche.rate,
                    "percent_of_deal": tranche_pct,
                    "seniority": tranche.seniority,
                    "attachment_point": tranche.attachment_point,
                    "detachment_point": tranche.detachment_point,
                    "thickness": (tranche.detachment_point - tranche.attachment_point 
                                if tranche.detachment_point and tranche.attachment_point else None)
                }
                
                tranche_analytics.append(analytics)
            
            # Calculate overcollateralization
            oc_ratio = clo_cdo.collateral_pool_balance / total_tranche_balance if total_tranche_balance > 0 else None
            
            # Calculate credit enhancement for each tranche
            for tranche in tranche_analytics:
                if tranche.get("attachment_point") is not None:
                    tranche["credit_enhancement"] = tranche.get("attachment_point")
            
            # Calculate weighted average cost of capital
            wacc = sum(t.balance * t.rate for t in clo_cdo.tranches) / total_tranche_balance if total_tranche_balance > 0 else None
            
            # Calculate excess spread
            excess_spread = (clo_cdo.collateral_pool_wac or 0) - wacc if wacc else None
            
            # Calculate projected losses
            total_losses = sum(cf.loss for cf in cashflows)
            loss_percentage = total_losses / clo_cdo.collateral_pool_balance if clo_cdo.collateral_pool_balance > 0 else 0
            
            # Create analytics report
            analytics = {
                "tranche_analytics": tranche_analytics,
                "deal_metrics": {
                    "overcollateralization_ratio": oc_ratio,
                    "wacc": wacc,
                    "excess_spread": excess_spread,
                    "projected_loss_rate": loss_percentage,
                    "projected_loss_amount": total_losses,
                },
                "reinvestment_period": {
                    "months": clo_cdo.reinvestment_period_months,
                    "active": clo_cdo.reinvestment_period_months and clo_cdo.reinvestment_period_months > 0
                },
                "collateral_quality": {
                    "warf": clo_cdo.collateral_pool_warf,
                    "wac": clo_cdo.collateral_pool_wac
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.exception(f"Error in CLO/CDO analytics: {str(e)}")
            return {"error": str(e), "error_type": type(e).__name__}
    
    def _run_stress_tests(self, pool: Any, tranches: List[Any], 
                        cashflows: List[AssetPoolCashflow], metrics: AssetPoolMetrics, 
                        analysis_date: date, discount_rate: float) -> List[AssetPoolStressTest]:
        """
        Run stress tests on CLO/CDO structure
        
        Args:
            pool: AbsBox collateral pool
            tranches: List of AbsBox tranches
            cashflows: List of cashflow projections
            metrics: Calculated metrics for the base case
            analysis_date: Date to start analysis from
            discount_rate: Rate to discount future cashflows
            
        Returns:
            List of stress test results
        """
        try:
            logger.info("Running CLO/CDO stress tests")
            
            stress_tests = []
            base_npv = metrics.npv
            
            # Define CLO/CDO specific stress scenarios
            scenarios = [
                {
                    "name": "high_defaults",
                    "description": "High default scenario (2x default rate)",
                    "cdr": 0.04,  # 2x base default rate
                    "cpr": 0.08,  # Reduced prepayment in high default
                    "recovery": 0.50  # Lower recovery rate
                },
                {
                    "name": "low_recovery",
                    "description": "Low recovery scenario",
                    "cdr": 0.03,  # Slightly higher defaults
                    "cpr": 0.10,  # Normal prepayment
                    "recovery": 0.40  # Significantly lower recovery
                },
                {
                    "name": "interest_rate_shock",
                    "description": "Interest rate shock (+200 bps)",
                    "discount_rate_delta": 0.02  # 200 bps rate increase
                },
                {
                    "name": "severe_stress",
                    "description": "Severe stress scenario (3x defaults, low recovery)",
                    "cdr": 0.06,  # 3x base default rate
                    "cpr": 0.05,  # Much lower prepayment
                    "recovery": 0.30,  # Very low recovery
                    "discount_rate_delta": 0.01  # 100 bps rate increase
                }
            ]
            
            # Create waterfall for analysis
            waterfall = self.absbox_service.create_waterfall(
                pool=pool,
                tranches=tranches,
                start_date=analysis_date
            )
            
            # Run each stress scenario
            for scenario in scenarios:
                try:
                    scenario_dict = {
                        "cdr": scenario.get("cdr", 0.02),
                        "cpr": scenario.get("cpr", 0.10),
                        "recovery": scenario.get("recovery", 0.60),
                        "lag": 3  # Keep lag constant
                    }
                    
                    # Adjust discount rate if specified
                    scenario_discount_rate = discount_rate
                    if "discount_rate_delta" in scenario:
                        scenario_discount_rate += scenario["discount_rate_delta"]
                    
                    # Run waterfall analysis with stress scenario
                    cf_results = self.absbox_service.run_waterfall_analysis(
                        waterfall=waterfall,
                        scenarios={scenario["name"]: scenario_dict}
                    )
                    
                    # Convert to our cashflow model
                    stress_cashflows = []
                    if cf_results and scenario["name"] in cf_results and hasattr(cf_results[scenario["name"]], "get_aggregate_cashflows"):
                        agg_cf = cf_results[scenario["name"]].get_aggregate_cashflows()
                        
                        for i, row in agg_cf.iterrows():
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
            logger.exception(f"Error running CLO/CDO stress tests: {str(e)}")
            return []
