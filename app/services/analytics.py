import numpy as np
import numpy_financial as npf
from typing import List, Dict, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AnalyticsResult:
    npv: float
    irr: float
    duration: float
    convexity: float
    monte_carlo_results: Optional[Dict[str, List[float]]] = None

class AnalyticsService:
    def __init__(self, num_simulations: int = 1000, num_threads: int = 4):
        self.num_simulations = num_simulations
        self.num_threads = num_threads

    def calculate_npv(self, cashflows: np.ndarray, discount_rate: float) -> float:
        """Calculate Net Present Value of cash flows"""
        periods = np.arange(len(cashflows))
        return np.sum(cashflows / (1 + discount_rate) ** periods)

    def calculate_irr(self, cashflows: np.ndarray) -> float:
        """Calculate Internal Rate of Return"""
        try:
            return npf.irr(cashflows)
        except Exception:
            return np.nan

    def calculate_duration(self, cashflows: np.ndarray, discount_rate: float) -> float:
        """Calculate Macaulay Duration"""
        periods = np.arange(1, len(cashflows) + 1)
        pv_factors = 1 / (1 + discount_rate) ** periods
        pv_cashflows = cashflows * pv_factors
        weighted_time = np.sum(periods * pv_cashflows)
        return weighted_time / np.sum(pv_cashflows)

    def calculate_convexity(self, cashflows: np.ndarray, discount_rate: float) -> float:
        """Calculate convexity"""
        periods = np.arange(1, len(cashflows) + 1)
        pv_factors = 1 / (1 + discount_rate) ** periods
        pv_cashflows = cashflows * pv_factors
        weighted_squared_time = np.sum(periods * (periods + 1) * pv_cashflows)
        return weighted_squared_time / (np.sum(pv_cashflows) * (1 + discount_rate) ** 2)

    def _simulate_scenario(self, 
                         base_cashflows: np.ndarray,
                         base_default_prob: float,
                         base_prepay_prob: float,
                         rate_volatility: float) -> np.ndarray:
        """Simulate a single scenario with random shocks"""
        # Generate random shocks
        rate_shock = np.random.normal(0, rate_volatility, len(base_cashflows))
        default_shock = np.random.normal(0, 0.2)  # 20% volatility in default rates
        prepay_shock = np.random.normal(0, 0.3)   # 30% volatility in prepayment rates

        # Apply shocks
        adjusted_default_prob = max(0, min(1, base_default_prob + default_shock))
        adjusted_prepay_prob = max(0, min(1, base_prepay_prob + prepay_shock))
        
        # Apply default and prepayment impacts
        survival_prob = (1 - adjusted_default_prob) ** np.arange(len(base_cashflows))
        prepay_factor = (1 - adjusted_prepay_prob) ** np.arange(len(base_cashflows))
        
        # Combine all effects
        return base_cashflows * survival_prob * prepay_factor * (1 + rate_shock)

    def run_monte_carlo(self,
                       base_cashflows: np.ndarray,
                       default_prob: float = 0.02,
                       prepay_prob: float = 0.05,
                       rate_volatility: float = 0.01) -> Dict[str, List[float]]:
        """Run Monte Carlo simulation with parallel processing"""
        
        def simulate_batch(num_sims: int) -> List[np.ndarray]:
            return [
                self._simulate_scenario(base_cashflows, default_prob, prepay_prob, rate_volatility)
                for _ in range(num_sims)
            ]

        # Split simulations across threads
        sims_per_thread = self.num_simulations // self.num_threads
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            simulation_results = list(executor.map(
                simulate_batch,
                [sims_per_thread] * self.num_threads
            ))

        # Flatten results
        all_scenarios = [
            scenario
            for batch in simulation_results
            for scenario in batch
        ]

        # Calculate statistics
        scenario_array = np.array(all_scenarios)
        return {
            "mean": np.mean(scenario_array, axis=0).tolist(),
            "std": np.std(scenario_array, axis=0).tolist(),
            "percentile_5": np.percentile(scenario_array, 5, axis=0).tolist(),
            "percentile_95": np.percentile(scenario_array, 95, axis=0).tolist()
        }

    async def analyze_cashflows(self,
                         cashflows: np.ndarray,
                         discount_rate: float = 0.05,
                         run_monte_carlo: bool = True) -> AnalyticsResult:
        """Comprehensive analysis of cash flows including all metrics"""
        npv = self.calculate_npv(cashflows, discount_rate)
        irr = self.calculate_irr(cashflows)
        duration = self.calculate_duration(cashflows, discount_rate)
        convexity = self.calculate_convexity(cashflows, discount_rate)
        
        monte_carlo_results = None
        if run_monte_carlo:
            monte_carlo_results = self.run_monte_carlo(cashflows)
            
        return AnalyticsResult(
            npv=npv,
            irr=irr,
            duration=duration,
            convexity=convexity,
            monte_carlo_results=monte_carlo_results
        )
