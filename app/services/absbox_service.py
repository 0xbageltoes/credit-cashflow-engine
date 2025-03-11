"""Service for structured finance analytics using AbsBox"""
import logging
import os
import json
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date
import requests
from functools import lru_cache

# Import AbsBox libraries
import absbox as ab
from absbox.local.pool import Pool
from absbox.local.loan import Loan, FixedRateLoan, FloatingRateLoan
from absbox.local.deal import Deal
from absbox.local.engine import LiqEngine
from absbox.local.waterfall import Waterfall
from absbox.local.assumption import Assumption, DefaultAssumption
from absbox.local.rateAssumption import FlatCurve
from absbox.local.analytics import Analytics
from absbox.local.cashflow import Cashflow

from app.core.config import settings
from app.core.cache import RedisCache
from app.core.monitoring import CALCULATION_TIME, CalculationTracker
from app.models.structured_products import (
    StructuredDealRequest,
    StructuredDealResponse,
    LoanPoolConfig,
    WaterfallConfig,
    ScenarioConfig,
    AnalysisResult
)
from app.models.analytics import EnhancedAnalyticsResult, RiskMetrics, SensitivityAnalysis
from app.models.cashflow import CashflowProjection, LoanData, CashflowForecastResponse

logger = logging.getLogger(__name__)

class AbsBoxService:
    """Service for structured finance analytics using AbsBox"""
    
    def __init__(self, hastructure_url: Optional[str] = None, cache: Optional[RedisCache] = None):
        """Initialize the AbsBox service"""
        self.hastructure_url = hastructure_url or settings.HASTRUCTURE_URL
        self.cache = cache or RedisCache()
        self.engine = self._initialize_engine()
        self.analytics = Analytics()
        logger.info(f"AbsBox service initialized with Hastructure URL: {self.hastructure_url}")
    
    def _initialize_engine(self) -> LiqEngine:
        """Initialize the calculation engine"""
        try:
            # For local development and testing
            if not self.hastructure_url or settings.ENVIRONMENT in ["development", "test"]:
                logger.info("Using local LiqEngine for calculations")
                return LiqEngine()
            
            # For production, use remote Hastructure service
            logger.info(f"Connecting to Hastructure engine at {self.hastructure_url}")
            return ab.engine.RemoteEngine(self.hastructure_url)
        except Exception as e:
            logger.error(f"Failed to initialize AbsBox engine: {str(e)}")
            # Fallback to local engine if remote fails
            logger.warning("Falling back to local LiqEngine")
            return LiqEngine()
    
    def create_loan_pool(self, config: LoanPoolConfig) -> Pool:
        """Create a loan pool from configuration"""
        with CalculationTracker("create_loan_pool"):
            loans = []
            
            for loan_config in config.loans:
                if loan_config.rate_type.lower() == "fixed":
                    loan = FixedRateLoan(
                        balance=loan_config.balance,
                        rate=loan_config.rate,
                        originBalance=loan_config.original_balance or loan_config.balance,
                        originRate=loan_config.original_rate or loan_config.rate,
                        originTerm=loan_config.term,
                        remainTerm=loan_config.remaining_term or loan_config.term,
                        period=loan_config.payment_frequency,
                        delay=loan_config.payment_delay or 0,
                        startDate=loan_config.start_date,
                        status=loan_config.status or "current"
                    )
                else:
                    # Floating rate loan
                    loan = FloatingRateLoan(
                        balance=loan_config.balance,
                        rate=loan_config.rate,
                        originBalance=loan_config.original_balance or loan_config.balance,
                        originRate=loan_config.original_rate or loan_config.rate,
                        originTerm=loan_config.term,
                        remainTerm=loan_config.remaining_term or loan_config.term,
                        period=loan_config.payment_frequency,
                        delay=loan_config.payment_delay or 0,
                        startDate=loan_config.start_date,
                        margin=loan_config.margin or 0.0,
                        index=loan_config.index or "LIBOR",
                        reset=loan_config.reset_frequency or 12,
                        status=loan_config.status or "current"
                    )
                
                loans.append(loan)
            
            # Create pool with metadata
            pool = Pool(assets=loans)
            if config.pool_name:
                pool.setName(config.pool_name)
                
            return pool
    
    def create_waterfall(self, config: WaterfallConfig) -> Waterfall:
        """Create a waterfall structure from configuration"""
        with CalculationTracker("create_waterfall"):
            waterfall = Waterfall(config.start_date)
            
            # Add accounts
            for account in config.accounts:
                waterfall.addAccount(
                    name=account.name,
                    balance=account.initial_balance or 0.0,
                    rate=account.interest_rate or 0.0
                )
            
            # Add bonds/notes
            for bond in config.bonds:
                waterfall.addBond(
                    name=bond.name,
                    balance=bond.balance,
                    rate=bond.rate,
                    originBalance=bond.original_balance or bond.balance,
                    originRate=bond.original_rate or bond.rate,
                    startDate=bond.start_date or config.start_date,
                    rateType=bond.rate_type or "Fixed",
                    payment=bond.payment_frequency or "Monthly"
                )
            
            # Define waterfall rules
            for action in config.actions:
                waterfall.addRule(
                    trigger=action.trigger or True,
                    source=action.source or "CollectedInterest",
                    target=action.target,
                    amount=action.amount,
                    tag=action.tag or "Default"
                )
                
            return waterfall
    
    def create_assumption(self, config: ScenarioConfig) -> Assumption:
        """Create analysis assumptions from configuration"""
        if config.default_curve:
            # Create default curve
            default_curve = DefaultAssumption(
                lag=config.default_curve.lag or 0,
                defaultVector=config.default_curve.vector,
                recoveryVector=config.default_curve.recovery_vector or [0.6] * len(config.default_curve.vector),
                recoveryLag=config.default_curve.recovery_lag or 6
            )
        else:
            # Default assumption with no defaults
            default_curve = DefaultAssumption(
                defaultVector=[0.0],
                recoveryVector=[0.0]
            )
            
        # Create interest rate curve for floating rates
        if config.interest_rate_curve:
            rate_curve = FlatCurve(
                rateVector=config.interest_rate_curve.vector
            )
        else:
            rate_curve = FlatCurve([0.02])  # Default 2% flat curve
            
        # Create combined assumption
        assumption = Assumption(
            defaultCurve=default_curve,
            prepaymentCurve=None,  # Future enhancement
            rateCurve=rate_curve
        )
        
        return assumption
    
    def create_structured_deal(self, config: StructuredDealRequest) -> Deal:
        """Create a structured finance deal from configuration"""
        # Create loan pool
        pool = self.create_loan_pool(config.pool)
        
        # Create waterfall
        waterfall = self.create_waterfall(config.waterfall)
        
        # Create assumption
        assumption = self.create_assumption(config.scenario)
        
        # Create deal
        deal = Deal(
            pool=pool,
            waterfall=waterfall,
            assumption=assumption
        )
        
        if config.deal_name:
            deal.setName(config.deal_name)
            
        return deal
    
    def analyze_deal(self, deal_config: StructuredDealRequest) -> StructuredDealResponse:
        """Analyze a structured finance deal and return results"""
        cache_key = f"absbox:deal:{hash(json.dumps(deal_config.dict(), default=str))}"
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached result for deal analysis: {deal_config.deal_name}")
            return StructuredDealResponse(**cached_result)
        
        with CalculationTracker("analyze_structured_deal"):
            try:
                # Create deal
                deal = self.create_structured_deal(deal_config)
                
                # Run analysis
                result = self.engine.runDeal(deal)
                
                # Extract results
                bond_cf = result.bondFlow().reset_index()
                pool_cf = result.poolFlow().reset_index()
                pool_stats = result.poolStats()
                
                # Convert to structured response
                response = StructuredDealResponse(
                    deal_name=deal_config.deal_name,
                    execution_time=result.runTime(),
                    bond_cashflows=bond_cf.to_dict(orient="records"),
                    pool_cashflows=pool_cf.to_dict(orient="records"),
                    pool_statistics=pool_stats.to_dict() if pool_stats else {},
                    metrics={
                        "bond_metrics": result.bondMetrics().to_dict() if hasattr(result, "bondMetrics") else {},
                        "pool_metrics": result.poolMetrics().to_dict() if hasattr(result, "poolMetrics") else {},
                    },
                    status="success"
                )
                
                # Cache the result
                self.cache.set(cache_key, response.dict(), ttl=3600)
                
                return response
                
            except Exception as e:
                logger.error(f"Error analyzing deal: {str(e)}")
                return StructuredDealResponse(
                    deal_name=deal_config.deal_name,
                    execution_time=0.0,
                    bond_cashflows=[],
                    pool_cashflows=[],
                    pool_statistics={},
                    metrics={},
                    status="error",
                    error=str(e)
                )
    
    def run_scenario_analysis(self, deal_config: StructuredDealRequest, 
                             scenarios: List[ScenarioConfig]) -> List[AnalysisResult]:
        """Run multiple scenarios on a deal structure"""
        with CalculationTracker("scenario_analysis"):
            results = []
            
            # Create base deal
            base_deal = self.create_structured_deal(deal_config)
            
            for scenario in scenarios:
                try:
                    # Update assumption for this scenario
                    updated_assumption = self.create_assumption(scenario)
                    base_deal.setAssumption(updated_assumption)
                    
                    # Run analysis
                    result = self.engine.runDeal(base_deal)
                    
                    # Extract key metrics
                    bond_metrics = result.bondMetrics() if hasattr(result, "bondMetrics") else pd.DataFrame()
                    
                    # Create result object
                    scenario_result = AnalysisResult(
                        scenario_name=scenario.name or "Unnamed Scenario",
                        bond_metrics=bond_metrics.to_dict() if not bond_metrics.empty else {},
                        pool_metrics=result.poolMetrics().to_dict() if hasattr(result, "poolMetrics") else {},
                        execution_time=result.runTime(),
                        status="success"
                    )
                    
                    results.append(scenario_result)
                    
                except Exception as e:
                    logger.error(f"Error in scenario '{scenario.name}': {str(e)}")
                    results.append(AnalysisResult(
                        scenario_name=scenario.name or "Unnamed Scenario",
                        bond_metrics={},
                        pool_metrics={},
                        execution_time=0.0,
                        status="error",
                        error=str(e)
                    ))
            
            return results
    
    def calculate_enhanced_analytics(
        self, 
        cashflows: List[CashflowProjection], 
        discount_rate: float, 
        principal: float
    ) -> EnhancedAnalyticsResult:
        """
        Calculate enhanced analytics metrics using AbsBox's analytics module
        
        Args:
            cashflows: List of cashflow projections
            discount_rate: Discount rate to use for present value calculations
            principal: Original principal amount
            
        Returns:
            EnhancedAnalyticsResult object with analytics metrics
        """
        with CalculationTracker("calculate_enhanced_analytics"):
            try:
                # Convert cashflows to format expected by AbsBox
                dates = [datetime.strptime(cf.date, "%Y-%m-%d").date() for cf in cashflows]
                amounts = [cf.total_payment for cf in cashflows]
                principal_amts = [cf.principal for cf in cashflows]
                interest_amts = [cf.interest for cf in cashflows]
                
                # Create AbsBox Cashflow object
                cf = Cashflow(dates=dates, amounts=amounts)
                
                # Calculate basic metrics
                npv = self.analytics.npv(cf, discount_rate)
                irr = self.analytics.irr(cf)
                yield_val = self.analytics.yieldToMaturity(cf, principal)
                duration = self.analytics.modifiedDuration(cf, discount_rate)
                macaulay_dur = self.analytics.macaulayDuration(cf, discount_rate)
                convexity = self.analytics.convexity(cf, discount_rate)
                wal = self.analytics.weightedAverageLife(dates, principal_amts)
                
                # Calculate advanced metrics
                dscr = 0.0
                if sum(interest_amts) > 0:
                    dscr = sum(amounts) / sum(interest_amts)
                
                # Calculate spread metrics (if available)
                dm = None
                z_spread = None
                e_spread = None
                
                try:
                    # Attempt to calculate spread metrics
                    benchmark_curve = FlatCurve(rate=discount_rate)
                    dm = self.analytics.discountMargin(cf, principal, benchmark_curve)
                    z_spread = self.analytics.zSpread(cf, principal, benchmark_curve)
                    # E-spread/OAS would require option modeling, not included for simplicity
                except Exception as e:
                    logger.warning(f"Could not calculate spread metrics: {str(e)}")
                
                # Calculate sensitivity metrics
                sensitivity_metrics = {}
                try:
                    # Calculate price changes for 100bp rate movements
                    rate_up = discount_rate + 0.01
                    rate_down = max(0.0001, discount_rate - 0.01)  # Prevent negative rates
                    
                    npv_up = self.analytics.npv(cf, rate_up)
                    npv_down = self.analytics.npv(cf, rate_down)
                    
                    sensitivity_metrics = {
                        "rate_up_1pct": (npv_up - npv) / npv * 100,
                        "rate_down_1pct": (npv_down - npv) / npv * 100
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate sensitivity metrics: {str(e)}")
                
                return EnhancedAnalyticsResult(
                    npv=npv,
                    irr=irr,
                    yield_value=yield_val,
                    duration=duration,
                    macaulay_duration=macaulay_dur,
                    convexity=convexity,
                    discount_margin=dm,
                    z_spread=z_spread,
                    e_spread=None,  # Requires more complex modeling
                    weighted_average_life=wal,
                    debt_service_coverage=dscr,
                    interest_coverage_ratio=dscr,  # Simplification, could be different
                    sensitivity_metrics=sensitivity_metrics
                )
                
            except Exception as e:
                logger.error(f"Error calculating enhanced analytics: {str(e)}")
                # Return default values on error
                return EnhancedAnalyticsResult(
                    npv=0.0,
                    irr=0.0,
                    yield_value=0.0,
                    duration=0.0,
                    macaulay_duration=0.0,
                    convexity=0.0,
                    weighted_average_life=0.0,
                    debt_service_coverage=0.0,
                    interest_coverage_ratio=0.0
                )

    def generate_amortization_schedule(self, loan: LoanData) -> List[CashflowProjection]:
        """
        Generate amortization schedule using AbsBox
        
        Args:
            loan: Loan data object
            
        Returns:
            List of cashflow projections
        """
        with CalculationTracker("generate_amortization_schedule"):
            try:
                # Create AbsBox loan
                start_date = datetime.strptime(loan.start_date, "%Y-%m-%d").date()
                
                if loan.rate_type.lower() == "fixed":
                    abs_loan = FixedRateLoan(
                        balance=loan.principal,
                        rate=loan.interest_rate,
                        originBalance=loan.principal,
                        originRate=loan.interest_rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.term_months,
                        period="Monthly",
                        delay=0,
                        startDate=start_date,
                        status="current"
                    )
                else:
                    # For hybrid/floating rates
                    abs_loan = FloatingRateLoan(
                        balance=loan.principal,
                        rate=loan.interest_rate,
                        originBalance=loan.principal,
                        originRate=loan.interest_rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.term_months,
                        period="Monthly",
                        delay=0,
                        startDate=start_date,
                        margin=loan.rate_spread or 0.0,
                        index="LIBOR",  # Default, could be configurable
                        reset=1,  # Monthly reset
                        status="current"
                    )
                
                # Set interest-only periods if specified
                if loan.interest_only_periods and loan.interest_only_periods > 0:
                    abs_loan.setInterestOnlyTerm(loan.interest_only_periods)
                
                # Set balloon payment if specified
                if loan.balloon_payment and loan.balloon_payment > 0:
                    abs_loan.setBalloon(loan.balloon_payment)
                
                # Generate cashflows
                cashflows = abs_loan.getCashflow()
                
                # Convert to our model
                projections = []
                for i, row in cashflows.iterrows():
                    projection = CashflowProjection(
                        period=i,
                        date=row.get('date', start_date + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
                        principal=row.get('principal', 0.0),
                        interest=row.get('interest', 0.0),
                        total_payment=row.get('payment', 0.0),
                        remaining_balance=row.get('balance', 0.0),
                        is_interest_only=(i < loan.interest_only_periods) if loan.interest_only_periods else False,
                        is_balloon=(i == loan.term_months - 1 and loan.balloon_payment) if loan.balloon_payment else False,
                        rate=row.get('rate', loan.interest_rate)
                    )
                    projections.append(projection)
                
                return projections
                
            except Exception as e:
                logger.error(f"Error generating amortization schedule: {str(e)}")
                raise
    
    def forecast_loan_cashflows(
        self, 
        loan: LoanData, 
        discount_rate: float = 0.05,
        prepayment_curve: Optional[List[float]] = None,
        default_curve: Optional[List[float]] = None
    ) -> CashflowForecastResponse:
        """
        Forecast loan cashflows with prepayment and default assumptions
        
        Args:
            loan: Loan data object
            discount_rate: Discount rate for present value calculations
            prepayment_curve: Optional list of annual prepayment rates by period
            default_curve: Optional list of annual default rates by period
            
        Returns:
            CashflowForecastResponse with projections and analytics
        """
        cache_key = f"absbox:loan_forecast:{hash(json.dumps(loan.model_dump(), default=str))}"
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached result for loan forecast: {loan.principal}")
            return CashflowForecastResponse(**cached_result)
        
        with CalculationTracker("forecast_loan_cashflows"):
            try:
                # Generate base amortization schedule
                start_time = datetime.now()
                
                # Create loan pool with single loan
                start_date = datetime.strptime(loan.start_date, "%Y-%m-%d").date()
                
                if loan.rate_type.lower() == "fixed":
                    abs_loan = FixedRateLoan(
                        balance=loan.principal,
                        rate=loan.interest_rate,
                        originBalance=loan.principal,
                        originRate=loan.interest_rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.term_months,
                        period="Monthly",
                        delay=0,
                        startDate=start_date,
                        status="current"
                    )
                else:
                    # For hybrid/floating rates
                    abs_loan = FloatingRateLoan(
                        balance=loan.principal,
                        rate=loan.interest_rate,
                        originBalance=loan.principal,
                        originRate=loan.interest_rate,
                        originTerm=loan.term_months,
                        remainTerm=loan.term_months,
                        period="Monthly",
                        delay=0,
                        startDate=start_date,
                        margin=loan.rate_spread or 0.0,
                        index="LIBOR",  # Default, could be configurable
                        reset=1,  # Monthly reset
                        status="current"
                    )
                
                # Create pool
                pool = Pool(name="Single Loan Pool", loans=[abs_loan])
                
                # Create assumption
                assumption = Assumption()
                
                # Set prepayment curve if provided
                if prepayment_curve:
                    assumption.setPrepayment(prepayment_curve)
                elif loan.prepayment_assumption and loan.prepayment_assumption > 0:
                    # Use constant prepayment rate if specified
                    cpr = [loan.prepayment_assumption] * loan.term_months
                    assumption.setPrepayment(cpr)
                
                # Set default curve if provided
                if default_curve:
                    default_assumption = DefaultAssumption()
                    default_assumption.setDefault(default_curve)
                    default_assumption.setRecovery([0.6] * len(default_curve))  # Example recovery rate
                    assumption.setDefault(default_assumption)
                
                # Apply assumption to pool
                pool.setAssumption(assumption)
                
                # Run pool
                result = self.engine.runPool(pool)
                
                # Get cashflows
                cashflows = result.getFlow()
                
                # Convert to our model
                projections = []
                for i, row in cashflows.iterrows():
                    projection = CashflowProjection(
                        period=i,
                        date=row.get('date', start_date + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
                        principal=row.get('principal', 0.0),
                        interest=row.get('interest', 0.0),
                        total_payment=row.get('payment', 0.0),
                        remaining_balance=row.get('balance', 0.0),
                        is_interest_only=(i < loan.interest_only_periods) if loan.interest_only_periods else False,
                        is_balloon=(i == loan.term_months - 1 and loan.balloon_payment) if loan.balloon_payment else False,
                        rate=row.get('rate', loan.interest_rate)
                    )
                    projections.append(projection)
                
                # Calculate enhanced analytics
                cashflow_for_analytics = Cashflow(
                    dates=[datetime.strptime(cf.date, "%Y-%m-%d").date() for cf in projections],
                    amounts=[cf.total_payment for cf in projections]
                )
                
                analytics = self.analytics
                npv = analytics.npv(cashflow_for_analytics, discount_rate)
                irr = analytics.irr(cashflow_for_analytics)
                duration = analytics.modifiedDuration(cashflow_for_analytics, discount_rate)
                macaulay_duration = analytics.macaulayDuration(cashflow_for_analytics, discount_rate)
                convexity = analytics.convexity(cashflow_for_analytics, discount_rate)
                wal = analytics.weightedAverageLife(
                    [datetime.strptime(cf.date, "%Y-%m-%d").date() for cf in projections],
                    [cf.principal for cf in projections]
                )
                
                # Calculate total principal and interest
                total_principal = sum(cf.principal for cf in projections)
                total_interest = sum(cf.interest for cf in projections)
                
                # Create response
                end_time = datetime.now()
                computation_time = (end_time - start_time).total_seconds()
                
                response = CashflowForecastResponse(
                    projections=projections,
                    summary_metrics={
                        "total_principal": total_principal,
                        "total_interest": total_interest,
                        "total_payments": total_principal + total_interest,
                        "npv": npv,
                        "irr": irr,
                        "duration": duration,
                        "convexity": convexity,
                        "macaulay_duration": macaulay_duration,
                        "weighted_average_life": wal,
                        "interest_coverage_ratio": 0.0,  # Not applicable for single loan
                        "debt_service_coverage": 0.0,  # Not applicable for single loan
                    },
                    computation_time=computation_time
                )
                
                # Cache result
                self.cache.set(cache_key, response.model_dump(), ttl=3600)
                
                return response
                
            except Exception as e:
                logger.error(f"Error forecasting loan cashflows: {str(e)}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the AbsBox service and Hastructure engine"""
        status = {
            "absbox_version": ab.__version__,
            "engine_type": "remote" if self.hastructure_url else "local",
            "hastructure_status": "unknown"
        }
        
        # Check Hastructure connection if using remote engine
        if self.hastructure_url:
            try:
                response = requests.get(f"{self.hastructure_url}/health", timeout=5)
                if response.status_code == 200:
                    status["hastructure_status"] = "healthy"
                    status["hastructure_details"] = response.json()
                else:
                    status["hastructure_status"] = "unhealthy"
                    status["hastructure_error"] = f"HTTP {response.status_code}"
            except requests.RequestException as e:
                status["hastructure_status"] = "unavailable"
                status["hastructure_error"] = str(e)
        else:
            # Check local engine
            try:
                # Simple test to check if engine works
                test_loan = FixedRateLoan(balance=100000, rate=0.05, originTerm=360, remainTerm=360)
                test_pool = Pool(assets=[test_loan])
                test_result = self.engine.runPool(test_pool)
                status["local_engine_status"] = "healthy"
            except Exception as e:
                status["local_engine_status"] = "error"
                status["local_engine_error"] = str(e)
        
        return status
