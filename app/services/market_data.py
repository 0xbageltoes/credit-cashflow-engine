import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import httpx
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.cache import SQLiteCache, cache_response

class MarketDataService:
    def __init__(self):
        self.cache = SQLiteCache()
        self.fred_api_key = settings.FRED_API_KEY
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    async def get_treasury_yields(self) -> Dict[str, float]:
        """
        Fetch current Treasury yield curve from FRED
        Returns dict of tenor: yield pairs
        """
        series_ids = {
            "1M": "DGS1MO",
            "3M": "DGS3MO",
            "6M": "DGS6MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "5Y": "DGS5",
            "10Y": "DGS10",
            "30Y": "DGS30"
        }
        
        yields = {}
        async with httpx.AsyncClient() as client:
            for tenor, series_id in series_ids.items():
                url = f"{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&sort_order=desc&limit=1&file_type=json"
                response = await client.get(url)
                data = response.json()
                yields[tenor] = float(data["observations"][0]["value"])
        
        return yields

    @cache_response(ttl=3600)
    async def get_historical_rates(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch historical rate data from FRED
        """
        url = f"{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&observation_start={start_date}&observation_end={end_date}&file_type=json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            return data["observations"]

    def interpolate_yield_curve(
        self,
        yields: Dict[str, float],
        target_tenors: List[float]
    ) -> List[float]:
        """
        Interpolate yield curve for specific tenors
        """
        # Convert tenors to numeric months
        tenor_map = {
            "1M": 1/12, "3M": 3/12, "6M": 6/12,
            "1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30
        }
        
        x = np.array([tenor_map[k] for k in yields.keys()])
        y = np.array(list(yields.values()))
        
        return np.interp(target_tenors, x, y)

    async def get_prepayment_factors(
        self,
        current_rate: float,
        loan_age: int
    ) -> Dict[str, float]:
        """
        Calculate prepayment factors based on current market conditions
        """
        # Get current market rates
        yields = await self.get_treasury_yields()
        ten_year = yields.get("10Y", 0)
        
        # Calculate rate differential
        rate_diff = current_rate - ten_year
        
        # Base prepayment model
        base_prepay = 0.06  # 6% base CPR
        
        # Adjust for rate incentive
        if rate_diff > 1:  # Borrower has incentive to refinance
            rate_multiplier = 1 + (rate_diff * 0.5)
        else:
            rate_multiplier = 1
        
        # Adjust for seasoning
        if loan_age < 12:
            seasoning_multiplier = 0.2 + (loan_age * 0.067)  # Ramp up over first year
        else:
            seasoning_multiplier = 1
        
        return {
            "base_prepayment": base_prepay,
            "rate_multiplier": rate_multiplier,
            "seasoning_multiplier": seasoning_multiplier,
            "final_prepayment": base_prepay * rate_multiplier * seasoning_multiplier
        }

    async def get_historical_prepayment_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical prepayment data for model calibration
        """
        # Example using Fannie Mae 30Y MBS prepayment speeds
        series_id = "MORTGAGE30US"  # Example FRED series
        data = await self.get_historical_rates(series_id, start_date, end_date)
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"])
        
        return df

    def generate_stress_scenarios(self) -> List[Dict]:
        """
        Generate stress test scenarios based on historical events
        """
        return [
            {
                "name": "2008_financial_crisis",
                "description": "Simulate 2008 financial crisis conditions",
                "rate_shock": 0.03,  # 300bps increase in rates
                "prepay_shock": -0.5,  # 50% reduction in prepayment speeds
                "default_shock": 5.0,  # 5x increase in default probability
                "recovery_shock": -0.3,  # 30% reduction in recovery rates
                "duration_months": 24
            },
            {
                "name": "rising_rate_environment",
                "description": "Rapid interest rate increases",
                "rate_shock": 0.05,  # 500bps increase in rates
                "prepay_shock": -0.7,  # 70% reduction in prepayment speeds
                "default_shock": 0.5,  # 50% increase in default probability
                "recovery_shock": -0.1,  # 10% reduction in recovery rates
                "duration_months": 12
            },
            {
                "name": "credit_crisis",
                "description": "Severe credit market deterioration",
                "rate_shock": 0.02,  # 200bps increase in rates
                "prepay_shock": -0.3,  # 30% reduction in prepayment speeds
                "default_shock": 3.0,  # 3x increase in default probability
                "recovery_shock": -0.4,  # 40% reduction in recovery rates
                "duration_months": 18
            }
        ]
