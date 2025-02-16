"""Cashflow service for generating loan cash flow projections"""
import numpy as np
import numpy_financial as npf
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import asyncio
import logging

from app.models.cashflow import (
    CashflowProjection,
    ScenarioSaveRequest,
    ScenarioResponse,
    MonteCarloResults,
    LoanData,
    MonteCarloConfig,
    StressTestScenario,
    CashflowForecastRequest,
    CashflowForecastResponse,
    EconomicFactors
)
from app.models.analytics import AnalyticsResult
from app.core.config import settings
from app.services.analytics import AnalyticsService
from app.core.redis_cache import RedisCache
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class CashflowService:
    """Service for generating loan cash flow projections"""
    
    def __init__(self, supabase_client: Optional[Client] = None, redis_cache: Optional[RedisCache] = None, analytics_service: Optional[AnalyticsService] = None):
        """Initialize the CashflowService with optional dependencies for testing"""
        self.supabase = supabase_client or create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        self.analytics = analytics_service or AnalyticsService()
        self.cache = redis_cache or RedisCache()
        self.BATCH_SIZE = 1000

    def _vectorized_loan_calculations(
        self,
        loans: List[LoanData],
        economic_factors: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Vectorized calculation of loan amortization schedules with support for:
        - Interest-only periods
        - Balloon payments
        - Economic factor adjustments
        - Enhanced prepayment modeling
        """
        # Convert loans to arrays for vectorized operations
        principals = np.array([loan.principal for loan in loans])
        rates = np.array([loan.interest_rate for loan in loans])
        terms = np.array([loan.term_months for loan in loans])
        io_periods = np.array([loan.interest_only_periods or 0 for loan in loans])
        balloon_payments = np.array([loan.balloon_payment or 0 for loan in loans])
        
        # Adjust rates for economic factors if provided
        if economic_factors:
            # For hybrid rate loans, adjust based on market rate
            for i in range(len(loans)):
                if loans[i].rate_type == "hybrid":
                    market_rate = economic_factors.get('market_rate', rates[i])
                    spread = loans[i].rate_spread or 0
                    cap = loans[i].rate_cap or float('inf')
                    floor = loans[i].rate_floor or 0
                    
                    # Calculate new rate within bounds
                    rates[i] = min(cap, max(floor, market_rate + spread))
        
        # Initialize arrays
        max_term = int(np.max(terms))
        n_loans = len(loans)
        
        principal_payments = np.zeros((n_loans, max_term))
        interest_payments = np.zeros((n_loans, max_term))
        remaining_balance = np.zeros((n_loans, max_term))
        periods = [np.arange(1, term + 1) for term in terms]
        
        # Calculate monthly payments (excluding balloon)
        monthly_rates = rates / 12
        monthly_payments = np.zeros(n_loans)
        
        for i in range(n_loans):
            # Calculate regular monthly payment
            monthly_payments[i] = abs(npf.pmt(
                monthly_rates[i],
                terms[i],
                principals[i]
            ))
        
        # Calculate amortization schedule for each loan
        for i in range(n_loans):
            balance = principals[i]
            term = int(terms[i])
            
            for t in range(term):
                # Calculate interest portion
                interest = balance * monthly_rates[i]
                
                # Handle interest-only period
                if t < io_periods[i]:
                    principal_payment = 0
                    interest_payment = interest
                else:
                    # For regular payments
                    if t == term - 1:  # Last payment
                        if balloon_payments[i] > 0:
                            # Balloon payment
                            principal_payment = balloon_payments[i]
                        else:
                            # Regular final payment - pay off remaining balance
                            principal_payment = balance
                        interest_payment = interest
                    else:
                        # Regular amortization payment
                        principal_payment = monthly_payments[i] - interest
                        interest_payment = interest
                
                # Store payments and update balance
                principal_payments[i, t] = principal_payment
                interest_payments[i, t] = interest_payment
                remaining_balance[i, t] = balance
                
                # Update balance
                balance = max(0, balance - principal_payment)
                
                # For last payment, ensure balance is exactly zero
                if t == term - 1:
                    balance = 0
                    remaining_balance[i, t] = 0
        
        return principal_payments, interest_payments, remaining_balance, periods

    def _calculate_summary_metrics(self, projections: List[CashflowProjection], economic_factors: Optional[Dict] = None) -> Dict:
        """Calculate summary metrics for a list of projections with economic factor adjustments"""
        # Calculate total principal without counting prepayments
        total_principal = 0
        total_interest = 0
        total_payments = 0
        
        for proj in projections:
            total_principal += proj.principal
            total_interest += proj.interest
            total_payments += proj.total_payment
        
        # Calculate NPV using a base discount rate adjusted by economic factors
        base_rate = 0.05  # Base annual discount rate
        
        # Apply economic factor adjustments to discount rate
        if economic_factors:
            market_rate = economic_factors.get('market_rate', base_rate)
            inflation = economic_factors.get('inflation_rate', 0.02)
            unemployment = economic_factors.get('unemployment_rate', 0.05)
            gdp_growth = economic_factors.get('gdp_growth', 0.025)
            
            # Adjust discount rate based on economic conditions
            risk_premium = 0.02  # Base risk premium
            
            # Increase risk premium in adverse conditions
            if inflation > 0.03:
                risk_premium += (inflation - 0.03) * 2
            if unemployment > 0.06:
                risk_premium += (unemployment - 0.06) * 3
            if gdp_growth < 0:
                risk_premium += abs(gdp_growth) * 2
                
            # Final discount rate includes market rate and risk adjustments
            discount_rate = (market_rate + risk_premium) / 12  # Convert to monthly
        else:
            discount_rate = base_rate / 12
        
        # Calculate NPV with adjusted discount rate
        npv = 0
        for i, p in enumerate(projections, 1):
            # Add current payment discounted to present value
            discount_factor = (1 + discount_rate) ** i
            npv += p.total_payment / discount_factor
        
        return {
            "total_principal": float(total_principal),
            "total_interest": float(total_interest),
            "total_payments": float(total_payments),
            "npv": float(npv)
        }

    async def generate_forecast(
        self,
        request: CashflowForecastRequest,
        user_id: str
    ) -> CashflowForecastResponse:
        """Generate cash flow projections with enhanced features"""
        start_time = datetime.now()

        # Get economic factors
        economic_factors = None
        if request.economic_factors:
            economic_factors = request.economic_factors.model_dump()

        # Calculate base projections
        principal_payments, interest_payments, remaining_balance, periods = self._vectorized_loan_calculations(
            request.loans,
            economic_factors
        )

        # Create projections list
        projections = []
        
        # Get payment dates
        payment_dates = self._generate_payment_dates(request.loans[0].start_date, len(periods[0]))
        
        # Aggregate payments for each period
        for i in range(len(periods[0])):
            payment_date = payment_dates[i]
            
            # Sum payments across all loans
            total_principal = float(sum(principal_payments[j][i] for j in range(len(request.loans))))
            total_interest = float(sum(interest_payments[j][i] for j in range(len(request.loans))))
            total_balance = float(sum(remaining_balance[j][i] for j in range(len(request.loans))))
            
            # Check for special payment types
            is_interest_only = any(i < loan.interest_only_periods for loan in request.loans)
            is_balloon = i == len(periods[0]) - 1 and any(loan.balloon_payment for loan in request.loans)
            
            # Get effective rate for this period
            if economic_factors and request.loans[0].rate_type == "hybrid":
                market_rate = economic_factors.get('market_rate', request.loans[0].interest_rate)
                spread = request.loans[0].rate_spread or 0
                cap = request.loans[0].rate_cap or float('inf')
                floor = request.loans[0].rate_floor or 0
                effective_rate = min(cap, max(floor, market_rate + spread))
            else:
                effective_rate = request.loans[0].interest_rate
            
            # Create projection with validated values
            projections.append(CashflowProjection(
                period=i + 1,
                date=payment_date,
                principal=max(0.0, total_principal),
                interest=max(0.0, total_interest),
                total_payment=max(0.0, total_principal + total_interest),
                remaining_balance=max(0.0, total_balance),
                is_interest_only=is_interest_only,
                is_balloon=is_balloon,
                rate=min(1.0, max(0.0, float(effective_rate)))
            ))

        # Run Monte Carlo simulation if requested
        monte_carlo_results = None
        if getattr(request, 'run_monte_carlo', False):
            monte_carlo_results = await self._run_monte_carlo_simulation(request, economic_factors or {})

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(projections, economic_factors)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = CashflowForecastResponse(
            projections=projections,
            summary_metrics=summary_metrics,
            monte_carlo_results=monte_carlo_results,
            computation_time=execution_time
        )
        
        # Cache results
        if user_id != "simulation":
            await self._cache_results(user_id, response.model_dump())
        
        return response

    async def _run_monte_carlo_simulation(
        self,
        request: CashflowForecastRequest,
        economic_factors: Dict
    ) -> MonteCarloResults:
        """Run enhanced Monte Carlo simulation with stress scenarios"""
        # Initialize results
        npv_values = []
        default_scenarios = []
        prepayment_scenarios = []
        rate_scenarios = []
        stress_test_results = {}
        
        # Get Monte Carlo config
        config = request.monte_carlo_config
        if not config:
            config = MonteCarloConfig()  # Use default config
        
        # Run base simulations
        for _ in range(config.num_simulations):
            # Generate random scenario
            scenario = self._generate_random_scenario(economic_factors, config)
            
            # Create a copy of the request with the new scenario
            request_data = request.model_dump()
            request_data['run_monte_carlo'] = False  # Prevent recursion
            request_data['economic_factors'] = scenario
            
            # Run simulation with this scenario
            sim_request = CashflowForecastRequest.model_validate(request_data)
            sim_response = await self.generate_forecast(sim_request, "simulation")
            
            # Record results
            npv = sim_response.summary_metrics["npv"]
            npv_values.append(npv)
            
            # Record special scenarios
            if scenario.get('default_occurred', False):
                default_scenarios.append({
                    "npv": npv,
                    "rate": scenario.get('market_rate', 0.045)
                })
            
            if scenario.get('prepayment_occurred', False):
                prepayment_scenarios.append({
                    "npv": npv,
                    "rate": scenario.get('market_rate', 0.045)
                })
            
            rate_scenarios.append({
                "rate_shock": scenario.get('rate_shock', 0),
                "npv": npv
            })
        
        # Calculate confidence intervals
        npv_values = np.array(npv_values)
        confidence_intervals = {
            "npv": {
                "95": [float(np.percentile(npv_values, 2.5)), float(np.percentile(npv_values, 97.5))],
                "99": [float(np.percentile(npv_values, 0.5)), float(np.percentile(npv_values, 99.5))]
            }
        }
        
        # Run stress test scenarios if provided
        if hasattr(config, 'stress_scenarios') and config.stress_scenarios:
            for scenario in config.stress_scenarios:
                # Create stressed economic factors
                stressed_factors = economic_factors.copy()
                if scenario.economic_factors:
                    stressed_factors.update(scenario.economic_factors.model_dump())
                
                # Add stress test modifiers
                stressed_factors['market_rate'] = stressed_factors.get('market_rate', 0.045) + scenario.rate_shock
                stressed_factors['default_prob'] = config.default_prob * scenario.default_multiplier
                stressed_factors['prepay_prob'] = config.prepay_prob * scenario.prepay_multiplier
                
                # Run simulation with stressed scenario
                request_data = request.model_dump()
                request_data['run_monte_carlo'] = False
                request_data['economic_factors'] = stressed_factors
                stress_response = await self.generate_forecast(
                    CashflowForecastRequest.model_validate(request_data),
                    "simulation"
                )
                
                stress_test_results[scenario.name] = {
                    "npv": float(stress_response.summary_metrics["npv"]),
                    "rate_shock": scenario.rate_shock,
                    "default_multiplier": scenario.default_multiplier,
                    "prepay_multiplier": scenario.prepay_multiplier
                }
        
        # Calculate VaR metrics
        var_metrics = {
            "var_95": float(-np.percentile(npv_values, 5)),
            "var_99": float(-np.percentile(npv_values, 1)),
            "expected_shortfall": float(-np.mean(npv_values[npv_values < np.percentile(npv_values, 5)]))
        }
        
        # Sensitivity analysis
        rate_shocks = [r["rate_shock"] for r in rate_scenarios]
        npvs = [r["npv"] for r in rate_scenarios]
        sensitivity = {
            "rate_sensitivity": float(np.corrcoef(rate_shocks, npvs)[0, 1]) if len(rate_shocks) > 1 else 0.0
        }
        
        return MonteCarloResults(
            npv_distribution=[float(x) for x in npv_values],
            default_scenarios=default_scenarios,
            prepayment_scenarios=prepayment_scenarios,
            rate_scenarios=rate_scenarios,
            confidence_intervals=confidence_intervals,
            stress_test_results=stress_test_results,
            var_metrics=var_metrics,
            sensitivity_analysis=sensitivity
        )

    def _generate_random_scenario(self, base_scenario: Dict, config: MonteCarloConfig) -> Dict:
        """Generate a random economic scenario for Monte Carlo simulation"""
        scenario = {}
        
        # Generate base economic factors with significant random variation
        scenario['market_rate'] = max(0.001, np.random.normal(0.045, 0.02))
        scenario['inflation_rate'] = max(0, np.random.normal(0.02, 0.015))
        scenario['unemployment_rate'] = max(0.02, np.random.normal(0.05, 0.02))
        scenario['gdp_growth'] = np.random.normal(0.025, 0.02)
        scenario['house_price_appreciation'] = np.random.normal(0.03, 0.025)
        
        # Add extreme rate shocks with higher volatility
        rate_shock = float(np.random.normal(0, config.rate_volatility * 8))  # 8x volatility
        scenario['rate_shock'] = rate_shock
        
        # Apply rate shock to market rate
        scenario['market_rate'] = max(0.001, scenario['market_rate'] + rate_shock)
        
        # Add correlation effects between economic factors
        if scenario['market_rate'] > 0.06:  # High rates scenario
            scenario['inflation_rate'] *= 1.5
            scenario['gdp_growth'] *= 0.5
            scenario['unemployment_rate'] *= 1.3
        elif scenario['market_rate'] < 0.03:  # Low rates scenario
            scenario['inflation_rate'] *= 0.7
            scenario['gdp_growth'] *= 1.3
            scenario['unemployment_rate'] *= 0.8
        
        # Generate random events with increased probability in adverse conditions
        base_default_prob = config.default_prob * 2
        base_prepay_prob = config.prepay_prob * 2
        
        # Adjust probabilities based on economic conditions
        if scenario['market_rate'] > 0.07:
            base_default_prob *= 2
            base_prepay_prob *= 0.5
        elif scenario['market_rate'] < 0.03:
            base_default_prob *= 0.5
            base_prepay_prob *= 2
        
        if scenario['unemployment_rate'] > 0.07:
            base_default_prob *= 1.5
        
        if scenario['gdp_growth'] < 0:
            base_default_prob *= 1.5
            base_prepay_prob *= 0.7
        
        scenario['default_occurred'] = bool(np.random.random() < base_default_prob)
        scenario['prepayment_occurred'] = bool(np.random.random() < base_prepay_prob)
        
        return scenario

    async def _cache_results(self, user_id: str, response_data: Dict) -> None:
        """Cache forecast results"""
        await self.cache.set_forecast_result(user_id, response_data)

    async def _batch_save_projections(
        self,
        projections: List[CashflowProjection],
        forecast_id: str,
        user_id: str
    ) -> None:
        """Save projections in batches for better performance"""
        batches = [projections[i:i + self.BATCH_SIZE] 
                  for i in range(0, len(projections), self.BATCH_SIZE)]
        
        for batch in batches:
            projection_batch = [
                {
                    "forecast_id": forecast_id,
                    "user_id": user_id,
                    "period": p.period,
                    "date": p.date,
                    "principal": p.principal,
                    "interest": p.interest,
                    "total_payment": p.total_payment,
                    "remaining_balance": p.remaining_balance,
                    "is_interest_only": p.is_interest_only,
                    "is_balloon": p.is_balloon,
                    "rate": p.rate
                }
                for p in batch
            ]
            await self._save_batch(projection_batch)

    async def _save_batch(self, batch: List[Dict]) -> None:
        """Save a batch of records to Supabase"""
        await self.supabase.table("cashflow_projections").insert(batch).execute()

    async def _save_forecast(
        self,
        request: CashflowForecastRequest,
        projections: List[CashflowProjection],
        user_id: str,
        analytics_result: AnalyticsResult
    ) -> str:
        """Save forecast results to Supabase"""
        forecast_data = {
            "user_id": user_id,
            "scenario_name": request.scenario_name,
            "created_at": datetime.now().isoformat(),
            "loan_count": len(request.loans),
            "total_principal": sum(loan.principal for loan in request.loans),
            "npv": analytics_result.npv,
            "irr": analytics_result.irr,
            "duration": analytics_result.duration,
            "convexity": analytics_result.convexity,
            "monte_carlo_results": analytics_result.monte_carlo_results
        }
        
        result = await self.supabase.table("forecasts").insert(forecast_data).execute()
        forecast_id = result.data[0]["id"]
        
        # Save projections in batches
        await self._batch_save_projections(projections, forecast_id, user_id)
        
        # Cache the forecast result
        await self.cache.set_forecast_result(forecast_id, {
            "forecast": forecast_data,
            "projections": [p.dict() for p in projections]
        })
        
        return forecast_id

    async def _get_economic_factors(self) -> Dict:
        """Get economic factors from cache or external source"""
        cache_key = "economic_factors"
        
        # Try to get from cache first
        cached_factors = await self.cache.get(cache_key)
        if cached_factors:
            return cached_factors
        
        # If not in cache, get from Supabase
        result = await self.supabase.table("economic_factors").select("*").single().execute()
        
        if not result.data:
            # Return default values if no data found
            default_factors = {
                "market_rate": 0.045,
                "inflation_rate": 0.02,
                "unemployment_rate": 0.05,
                "gdp_growth": 0.025,
                "house_price_appreciation": 0.03,
                "month": datetime.now().month
            }
            
            # Cache default values
            await self.cache.set(cache_key, default_factors, expire=3600)  # Cache for 1 hour
            return default_factors
        
        factors = result.data
        
        # Cache the results
        await self.cache.set(cache_key, factors, expire=3600)  # Cache for 1 hour
        
        return factors

    def _adjust_rates_for_economic_factors(
        self,
        rates: np.ndarray,
        economic_factors: Dict
    ) -> np.ndarray:
        """Adjust interest rates based on economic factors"""
        adjusted_rates = rates.copy()
        
        if 'market_rate' in economic_factors:
            # Adjust floating rate portions
            rate_diff = economic_factors['market_rate'] - np.mean(rates)
            adjusted_rates += rate_diff * 0.5  # Assume 50% floating rate portion
        
        if 'inflation_rate' in economic_factors:
            # Add inflation premium
            adjusted_rates += economic_factors['inflation_rate'] * 0.2
        
        return adjusted_rates

    def _adjust_prepayment_for_economic_factors(
        self,
        prepay_rates: np.ndarray,
        economic_factors: Dict
    ) -> np.ndarray:
        """Adjust prepayment rates based on economic factors"""
        adjusted_rates = prepay_rates.copy()
        
        if 'unemployment_rate' in economic_factors:
            # Lower prepayment in high unemployment
            unemployment_factor = 1 - economic_factors['unemployment_rate'] * 0.5
            adjusted_rates *= unemployment_factor
        
        if 'gdp_growth' in economic_factors:
            # Higher prepayment in strong economy
            gdp_factor = 1 + max(0, economic_factors['gdp_growth']) * 0.3
            adjusted_rates *= gdp_factor
        
        return adjusted_rates

    def _calculate_prepayment_factors(
        self,
        base_prepay_rate: float,
        periods: np.ndarray,
        balances: np.ndarray,
        rate: float,
        economic_factors: Optional[Dict] = None
    ) -> np.ndarray:
        """Calculate dynamic prepayment factors based on loan characteristics and economic factors"""
        # Base prepayment curve (seasoning ramp)
        seasoning_ramp = 1 - np.exp(-periods / 12)  # Ramp up over first year
        
        # Loan size factor (larger loans have higher prepayment probability)
        balance_factor = np.log1p(balances) / np.log1p(np.max(balances))
        
        # Rate incentive (if market rates are lower, prepayment is more likely)
        if economic_factors and 'market_rate' in economic_factors:
            rate_diff = rate - economic_factors['market_rate']
            rate_factor = 1 + np.maximum(0, rate_diff) * 2  # Increase prepayment if rate is higher
        else:
            rate_factor = 1.0
        
        # Seasonal factors (higher in summer months)
        if economic_factors and 'month' in economic_factors:
            seasonal_factor = 1 + 0.2 * (economic_factors['month'] in [6, 7, 8])  # 20% higher in summer
        else:
            seasonal_factor = 1.0
        
        # Housing market factor
        if economic_factors and 'house_price_appreciation' in economic_factors:
            hpa = economic_factors['house_price_appreciation']
            housing_factor = 1 + np.maximum(0, hpa) * 0.5  # Higher HPA increases prepayment
        else:
            housing_factor = 1.0
        
        # Combine factors
        prepay_factors = (
            (1 - base_prepay_rate * seasoning_ramp * balance_factor * rate_factor * seasonal_factor * housing_factor)
            ** periods
        )
        
        return prepay_factors

    def _generate_payment_dates(self, start_date: str, num_periods: int) -> List[str]:
        """Generate payment dates for a given start date and number of periods"""
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        payment_dates = []
        
        for i in range(num_periods):
            payment_date = (start_date + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
            payment_dates.append(payment_date)
        
        return payment_dates
