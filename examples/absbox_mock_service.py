"""
AbsBox Mock Service

This module provides a simplified mock AbsBox service for demonstration purposes
when the full service dependencies aren't available.
"""

import os
import json
import logging
import random
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("absbox_mock_service")

class DateEncoder(json.JSONEncoder):
    """JSON encoder that handles date objects"""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

class AbsBoxMockService:
    """Mock AbsBox service for demonstration and testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock service
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        logger.info("Initialized AbsBox mock service")
    
    def analyze_deal(self, deal_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a structured deal
        
        Args:
            deal_config: Deal configuration dictionary
            
        Returns:
            Analysis result
        """
        deal_name = deal_config.get("deal_name", "Unknown Deal")
        logger.info(f"Analyzing deal: {deal_name}")
        
        # Generate mock analysis results
        result = self._generate_mock_results(deal_config)
        
        return result
    
    def _generate_mock_results(self, deal_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock analysis results
        
        Args:
            deal_config: Deal configuration
            
        Returns:
            Mock result dictionary
        """
        # Get deal parameters
        deal_name = deal_config.get("deal_name", "Mock Deal")
        pool_config = deal_config.get("pool_config", {})
        waterfall_config = deal_config.get("waterfall_config", {})
        
        # Get bond information
        bonds = waterfall_config.get("bonds", [])
        bond_ids = [bond.get("id", f"Bond_{i}") for i, bond in enumerate(bonds)]
        
        # Get loan information
        loans = pool_config.get("loans", [])
        total_pool_balance = sum(loan.get("balance", 0) for loan in loans)
        num_loans = len(loans)
        
        # Generate dates for cashflow projection
        start_date = date.today()
        periods = 60  # 5 years of monthly projections
        dates = [start_date + timedelta(days=30*i) for i in range(periods)]
        date_strs = [d.isoformat() for d in dates]
        
        # Generate mock bond cashflows
        bond_cashflows = {
            "dates": date_strs,
            "cashflows": {}
        }
        
        for bond_id in bond_ids:
            # Generate random cashflow series
            principal = [random.uniform(1000, 3000) for _ in range(periods)]
            interest = [random.uniform(500, 1500) for _ in range(periods)]
            fees = [random.uniform(50, 200) for _ in range(periods)]
            
            # Ensure declining trend for principal
            for i in range(1, periods):
                principal[i] = principal[i-1] * random.uniform(0.95, 0.99)
            
            bond_cashflows["cashflows"][bond_id] = {
                "principal": principal,
                "interest": interest,
                "fees": fees,
                "total": [p + i + f for p, i, f in zip(principal, interest, fees)]
            }
        
        # Generate mock pool cashflows
        pool_cashflows = {
            "dates": date_strs,
            "scheduled_principal": [random.uniform(3000, 6000) * 0.98**i for i in range(periods)],
            "unscheduled_principal": [random.uniform(500, 1500) * 1.02**i for i in range(periods)],
            "interest": [random.uniform(4000, 8000) * 0.99**i for i in range(periods)],
            "default": [random.uniform(100, 500) * 1.01**i for i in range(periods)],
            "recovery": [random.uniform(50, 300) * 1.005**i for i in range(periods)]
        }
        
        # Generate mock metrics
        bond_metrics = []
        for bond_id in bond_ids:
            bond_metrics.append({
                "id": bond_id,
                "balance": random.uniform(50000, 300000),
                "yield": random.uniform(0.03, 0.08),
                "duration": random.uniform(3, 8),
                "wac": random.uniform(0.04, 0.07),
                "wal": random.uniform(4, 10)
            })
        
        # Generate mock scenario comparison
        scenario_comparison = {
            "scenarios": ["base", "stress", "recovery"],
            "metrics": {
                "bond_yield": {
                    "Class_A": [0.035, 0.03, 0.04],
                    "Class_B": [0.05, 0.045, 0.055],
                    "Class_C": [0.065, 0.055, 0.07]
                },
                "bond_loss": {
                    "Class_A": [0.0, 0.01, 0.0],
                    "Class_B": [0.02, 0.05, 0.01],
                    "Class_C": [0.05, 0.15, 0.03]
                }
            }
        }
        
        # Assemble complete result
        result = {
            "status": "success",
            "deal_name": deal_name,
            "execution_time": random.uniform(0.5, 3.0),
            "analysis_date": date.today().isoformat(),
            "bond_cashflows": bond_cashflows,
            "pool_cashflows": pool_cashflows,
            "pool_summary": {
                "total_balance": total_pool_balance,
                "num_loans": num_loans,
                "wac": sum(loan.get("rate", 0) for loan in loans) / num_loans if num_loans > 0 else 0,
                "wal": random.uniform(5, 10)
            },
            "metrics": {
                "bond_metrics": bond_metrics,
                "scenario_comparison": scenario_comparison
            }
        }
        
        return result
    
    def create_sample_deal(self, name: str = "Mock Deal") -> Dict[str, Any]:
        """Create a sample deal for testing and demonstration
        
        Args:
            name: Deal name
            
        Returns:
            Sample deal configuration
        """
        return {
            "deal_name": name,
            "pool_config": {
                "pool_name": f"{name} Pool",
                "pool_type": "mortgage",
                "loans": [
                    {
                        "balance": 200000,
                        "rate": 0.045,
                        "term": 360,
                        "loan_type": "fixed"
                    },
                    {
                        "balance": 150000,
                        "rate": 0.05,
                        "term": 360,
                        "loan_type": "fixed"
                    },
                    {
                        "balance": 100000,
                        "rate": 0.04,
                        "term": 360,
                        "loan_type": "arm",
                        "initial_period": 60,
                        "rate_cap": 0.08,
                        "rate_floor": 0.03
                    }
                ]
            },
            "waterfall_config": {
                "bonds": [
                    {
                        "id": "Class_A",
                        "balance": 400000,
                        "rate": 0.04,
                        "priority": 1
                    },
                    {
                        "id": "Class_B",
                        "balance": 50000,
                        "rate": 0.06,
                        "priority": 2
                    }
                ],
                "waterfall_structure": {
                    "interest_allocation": "sequential",
                    "principal_allocation": "pro_rata",
                    "triggers": [
                        {
                            "type": "delinquency",
                            "threshold": 0.05,
                            "result": "sequential_all"
                        }
                    ]
                }
            },
            "scenario_config": {
                "default_curve": {
                    "type": "vector",
                    "vector": [0.001, 0.002, 0.003, 0.003, 0.002]
                },
                "prepayment_curve": {
                    "type": "vector",
                    "vector": [0.01, 0.02, 0.03, 0.04, 0.05]
                },
                "recovery_rate": 0.6,
                "recovery_lag": 12
            },
            "analysis_config": {
                "projection_periods": 120,
                "run_scenarios": True,
                "calculation_mode": "detailed"
            }
        }

# For demonstration purposes
if __name__ == "__main__":
    service = AbsBoxMockService()
    deal = service.create_sample_deal("Test Deal")
    result = service.analyze_deal(deal)
    
    # Save to file
    with open("mock_result.json", "w") as f:
        json.dump(result, f, cls=DateEncoder, indent=2)
    
    print(f"Mock result saved to mock_result.json")
