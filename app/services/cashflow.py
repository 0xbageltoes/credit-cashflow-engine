import numpy as np
import numpy_financial as npf
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from app.models.cashflow import (
    CashflowForecastRequest,
    CashflowForecastResponse,
    CashflowProjection,
    ScenarioSaveRequest,
    ScenarioResponse,
    MonteCarloResults
)
from app.core.config import settings
from supabase import create_client, Client
from functools import lru_cache
from app.services.analytics import AnalyticsService, AnalyticsResult
from app.core.redis_cache import RedisCache
import asyncio
import logging

logger = logging.getLogger(__name__)

class CashflowService:
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        self.analytics = AnalyticsService()
        self.cache = RedisCache()
        self.BATCH_SIZE = 1000

    def _vectorized_loan_calculations(
        self,
        loans: List[Dict],
        economic_factors: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized calculation of loan amortization schedules with support for:
        - Interest-only periods
        - Balloon payments
        - Economic factor adjustments
        - Enhanced prepayment modeling
        """
        # Convert loan parameters to numpy arrays
        principals = np.array([float(loan.principal) for loan in loans])
        rates = np.array([float(loan.interest_rate) / 12 for loan in loans])
        terms = np.array([int(loan.term_months) for loan in loans])
        prepay_rates = np.array([float(loan.prepayment_assumption) for loan in loans])
        interest_only_periods = np.array([int(loan.interest_only_periods or 0) for loan in loans])
        balloon_payments = np.array([float(loan.balloon_payment or 0) for loan in loans])
        
        # Economic factors adjustment
        if economic_factors:
            rates = self._adjust_rates_for_economic_factors(rates, economic_factors)
            prepay_rates = self._adjust_prepayment_for_economic_factors(prepay_rates, economic_factors)
        
        # Calculate monthly payments (excluding interest-only periods and balloon)
        amort_terms = terms - interest_only_periods
        pmts = np.zeros_like(principals)
        for i in range(len(loans)):
            if balloon_payments[i] > 0:
                # Calculate payment with balloon
                pmts[i] = npf.pmt(
                    rates[i],
                    amort_terms[i],
                    -principals[i],
                    balloon_payments[i]
                )
            else:
                # Regular amortization
                pmts[i] = npf.pmt(
                    rates[i],
                    amort_terms[i],
                    -principals[i]
                )
        
        # Generate periods array for each loan
        max_term = max(terms)
        periods = np.arange(1, max_term + 1)
        periods_matrix = np.broadcast_to(periods, (len(loans), len(periods)))
        
        # Initialize payment arrays
        principal_payments = np.zeros((len(loans), max_term))
        interest_payments = np.zeros((len(loans), max_term))
        remaining_balance = np.zeros((len(loans), max_term))
        
        for i in range(len(loans)):
            term = terms[i]
            principal = principals[i]
            rate = rates[i]
            io_period = interest_only_periods[i]
            balloon = balloon_payments[i]
            
            # Handle interest-only period
            if io_period > 0:
                interest_payments[i, :io_period] = principal * rate
                principal_payments[i, :io_period] = 0
                remaining_balance[i, :io_period] = principal
            
            # Calculate amortization schedule
            balance = principal
            for j in range(io_period, term):
                interest = balance * rate
                if j == term - 1 and balloon > 0:
                    # Handle balloon payment
                    principal_payment = balloon
                else:
                    principal_payment = pmts[i] - interest
                
                interest_payments[i, j] = interest
                principal_payments[i, j] = principal_payment
                remaining_balance[i, j] = balance - principal_payment
                balance = remaining_balance[i, j]
        
        return principal_payments, interest_payments, remaining_balance, periods_matrix

    async def generate_forecast(
        self,
        request: CashflowForecastRequest,
        user_id: str
    ) -> CashflowForecastResponse:
        """Generate cash flow projections with enhanced features"""
        start_time = datetime.now()
        
        # Get economic factors
        economic_factors = request.economic_factors
        if not economic_factors:
            economic_factors = await self._get_economic_factors()
        
        # Convert economic factors to dict if needed
        if hasattr(economic_factors, 'dict'):
            economic_factors = economic_factors.dict()
        
        # Generate cash flows
        principal_payments, interest_payments, remaining_balance, periods = \
            self._vectorized_loan_calculations(request.loans, economic_factors)
        
        # Create projections
        projections = []
        start_date = datetime.strptime(request.loans[0].start_date, "%Y-%m-%d")
        
        for i in range(len(periods[0])):
            payment_date = (start_date + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
            
            # Sum payments across all loans for this period
            total_principal = np.sum(principal_payments[:, i])
            total_interest = np.sum(interest_payments[:, i])
            total_balance = np.sum(remaining_balance[:, i])
            
            # Determine if this is an interest-only or balloon payment
            is_interest_only = total_principal == 0 and total_interest > 0
            is_balloon = i == len(periods[0]) - 1 and any(loan.balloon_payment for loan in request.loans)
            
            # Get effective rate for this period
            if economic_factors:
                effective_rate = economic_factors['market_rate'] if 'market_rate' in economic_factors else request.loans[0].interest_rate
            else:
                effective_rate = request.loans[0].interest_rate
            
            projections.append(CashflowProjection(
                period=i + 1,
                date=payment_date,
                principal=float(total_principal),
                interest=float(total_interest),
                total_payment=float(total_principal + total_interest),
                remaining_balance=float(total_balance),
                is_interest_only=is_interest_only,
                is_balloon=is_balloon,
                rate=float(effective_rate)
            ))
        
        # Run analytics
        analytics_result = await self.analytics.analyze_cashflows(projections)
        
        # Run Monte Carlo simulation if requested
        monte_carlo_results = None
        if request.run_monte_carlo:
            monte_carlo_results = await self._run_monte_carlo_simulation(request, economic_factors)
        
        # Save results if user_id is provided
        forecast_id = None
        if user_id != "simulation":
            forecast_id = await self._save_forecast(request, projections, user_id, analytics_result)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return CashflowForecastResponse(
            scenario_id=forecast_id,
            projections=projections,
            summary_metrics={
                "total_principal": float(np.sum(principal_payments)),
                "total_interest": float(np.sum(interest_payments)),
                "total_payments": float(np.sum(principal_payments) + np.sum(interest_payments)),
                "npv": analytics_result.npv,
                "irr": analytics_result.irr,
                "duration": analytics_result.duration,
                "convexity": analytics_result.convexity
            },
            monte_carlo_results=monte_carlo_results,
            economic_scenario=economic_factors,
            computation_time=computation_time
        )

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

    def _generate_random_scenario(self, base_scenario: Dict, config: Dict) -> Dict:
        """Generate a random economic scenario for Monte Carlo simulation"""
        # Copy base scenario
        scenario = base_scenario.copy()
        
        # Generate correlated random shocks
        correlation_matrix = config.correlation_matrix if hasattr(config, 'correlation_matrix') else {}
        rate_volatility = config.rate_volatility if hasattr(config, 'rate_volatility') else 0.01
        
        # Generate random shocks
        rate_shock = np.random.normal(0, rate_volatility)
        default_shock = np.random.normal(0, 0.2)  # 20% volatility in default rates
        prepay_shock = np.random.normal(0, 0.3)   # 30% volatility in prepayment rates
        
        # Apply correlations
        if correlation_matrix:
            rate_default_corr = correlation_matrix.get('rate', {}).get('default', 0.3)
            rate_prepay_corr = correlation_matrix.get('rate', {}).get('prepay', -0.2)
            default_prepay_corr = correlation_matrix.get('default', {}).get('prepay', -0.1)
            
            # Adjust shocks based on correlations
            default_shock += rate_shock * rate_default_corr
            prepay_shock += rate_shock * rate_prepay_corr
            prepay_shock += default_shock * default_prepay_corr
        
        # Apply shocks to economic factors
        scenario['market_rate'] = max(0, base_scenario['market_rate'] + rate_shock)
        scenario['inflation_rate'] = max(-0.1, min(0.5, base_scenario['inflation_rate'] + rate_shock * 0.5))
        scenario['unemployment_rate'] = max(0, min(1, base_scenario['unemployment_rate'] + default_shock * 0.1))
        scenario['gdp_growth'] = max(-0.5, min(0.5, base_scenario['gdp_growth'] - default_shock * 0.2))
        scenario['house_price_appreciation'] = max(-0.5, min(0.5, base_scenario['house_price_appreciation'] + rate_shock * 0.3))
        
        # Record shock impacts
        scenario['rate_shock'] = rate_shock
        scenario['default_occurred'] = default_shock > 0.4  # 40% threshold for default event
        scenario['prepayment_occurred'] = prepay_shock > 0.6  # 60% threshold for prepayment event
        
        return scenario

    async def _run_single_simulation(
        self,
        request: CashflowForecastRequest,
        scenario: Dict
    ) -> Dict:
        """Run a single Monte Carlo simulation"""
        # Apply economic scenario
        request_copy = request.copy(deep=True)
        request_copy.economic_factors = scenario
        
        # Run simulation
        projections = await self.generate_forecast(request_copy, "simulation")
        
        # Calculate metrics
        analytics_result = await self.analytics.analyze_cashflows(projections)
        
        return {
            "npv": analytics_result.npv,
            "default_occurred": scenario['default_occurred'],
            "prepayment_occurred": scenario['prepayment_occurred'],
            "rate_shock": scenario['rate_shock']
        }

    async def _run_monte_carlo_simulation(
        self,
        request: CashflowForecastRequest,
        economic_factors: Dict
    ) -> MonteCarloResults:
        """Run enhanced Monte Carlo simulation with stress scenarios"""
        config = request.monte_carlo_config
        n_sims = config.num_simulations
        
        # Initialize result arrays
        npv_results = np.zeros(n_sims)
        default_scenarios = []
        prepayment_scenarios = []
        rate_scenarios = []
        
        # Base economic scenario
        base_scenario = economic_factors.copy()
        
        # Run simulations in parallel chunks
        chunk_size = 100
        n_chunks = (n_sims + chunk_size - 1) // chunk_size
        
        async def run_chunk(chunk_idx):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_sims)
            chunk_results = []
            
            for i in range(start_idx, end_idx):
                # Generate random economic scenario
                scenario = self._generate_random_scenario(base_scenario, config)
                
                # Run single simulation
                result = await self._run_single_simulation(request, scenario)
                chunk_results.append(result)
            
            return chunk_results
        
        # Run chunks in parallel
        tasks = [run_chunk(i) for i in range(n_chunks)]
        chunk_results = await asyncio.gather(*tasks)
        
        # Combine results
        all_results = []
        for chunk in chunk_results:
            all_results.extend(chunk)
        
        # Process results
        npv_distribution = [r['npv'] for r in all_results]
        default_scenarios = [r for r in all_results if r['default_occurred']]
        prepayment_scenarios = [r for r in all_results if r['prepayment_occurred']]
        rate_scenarios = [r for r in all_results if abs(r['rate_shock']) > 0.01]
        
        # Calculate confidence intervals
        confidence_intervals = {
            'npv': {
                '95': np.percentile(npv_distribution, [2.5, 97.5]).tolist(),
                '99': np.percentile(npv_distribution, [0.5, 99.5]).tolist()
            },
            'default_rate': {
                'mean': len(default_scenarios) / n_sims,
                'std': np.sqrt(len(default_scenarios) * (n_sims - len(default_scenarios)) / (n_sims ** 3))
            }
        }
        
        return MonteCarloResults(
            npv_distribution=npv_distribution,
            default_scenarios=default_scenarios,
            prepayment_scenarios=prepayment_scenarios,
            rate_scenarios=rate_scenarios,
            confidence_intervals=confidence_intervals
        )
