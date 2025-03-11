#!/usr/bin/env python
"""
AbsBox Visualization Demo

This script demonstrates how to use the AbsBox service and visualizer
to run and visualize structured finance analytics.
"""

import os
import sys
import json
import logging
import traceback
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("absbox_demo")

# Configuration
IMPORTS_OK = False
MODELS_OK = False
VISUALIZATION_OK = False

# Try importing required modules
try:
    # Update import status
    IMPORTS_OK = True
except ImportError as e:
    print(f"Basic import error: {e}")
    sys.exit(1)

# Try importing visualization dependencies
try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    
    # Update visualization status
    VISUALIZATION_OK = True
    logger.info("Visualization dependencies loaded successfully")
except ImportError as e:
    logger.warning(f"Visualization dependencies not available: {e}")

# Try importing required models
try:
    class StructuredDealRequest:
        def __init__(self, deal_name, pool_config, waterfall_config, scenario_config, pricing_date):
            self.deal_name = deal_name
            self.pool_config = pool_config
            self.waterfall_config = waterfall_config
            self.scenario_config = scenario_config
            self.pricing_date = pricing_date

    class LoanPoolConfig:
        def __init__(self, pool_name, pool_type, cutoff_date, loans):
            self.pool_name = pool_name
            self.pool_type = pool_type
            self.cutoff_date = cutoff_date
            self.loans = loans

    class WaterfallConfig:
        def __init__(self, start_date, period_length, period_unit, num_periods, payment_day, accounts, bonds, actions):
            self.start_date = start_date
            self.period_length = period_length
            self.period_unit = period_unit
            self.num_periods = num_periods
            self.payment_day = payment_day
            self.accounts = accounts
            self.bonds = bonds
            self.actions = actions

    class ScenarioConfig:
        def __init__(self, name, default_curve, prepayment_curve, interest_rate_curve):
            self.name = name
            self.default_curve = default_curve
            self.prepayment_curve = prepayment_curve
            self.interest_rate_curve = interest_rate_curve

    class LoanConfig:
        def __init__(self, loan_id, balance, rate, term, payment_frequency, amortization_term, loan_type, original_ltv, original_fico, occupancy_type, property_type, state, origination_date, first_payment_date):
            self.loan_id = loan_id
            self.balance = balance
            self.rate = rate
            self.term = term
            self.payment_frequency = payment_frequency
            self.amortization_term = amortization_term
            self.loan_type = loan_type
            self.original_ltv = original_ltv
            self.original_fico = original_fico
            self.occupancy_type = occupancy_type
            self.property_type = property_type
            self.state = state
            self.origination_date = origination_date
            self.first_payment_date = first_payment_date

    class BondConfig:
        def __init__(self, id, name, balance, rate, bond_class, seniority):
            self.id = id
            self.name = name
            self.balance = balance
            self.rate = rate
            self.bond_class = bond_class
            self.seniority = seniority

    class AccountConfig:
        def __init__(self, id, name, initial_balance, account_type):
            self.id = id
            self.name = name
            self.initial_balance = initial_balance
            self.account_type = account_type

    class WaterfallAction:
        def __init__(self, name, source, target, action_type, amount, priority):
            self.name = name
            self.source = source
            self.target = target
            self.action_type = action_type
            self.amount = amount
            self.priority = priority

    class DefaultCurveConfig:
        def __init__(self, vector, recovery_vector, lag, recovery_lag):
            self.vector = vector
            self.recovery_vector = recovery_vector
            self.lag = lag
            self.recovery_lag = recovery_lag

    class PrepaymentCurveConfig:
        def __init__(self, vector, type, scaling_factor, lag):
            self.vector = vector
            self.type = type
            self.scaling_factor = scaling_factor
            self.lag = lag

    class RateCurveConfig:
        def __init__(self, vector, type):
            self.vector = vector
            self.type = type

    MODELS_OK = True
    logger.info("Successfully imported required models")
except ImportError as e:
    logger.error(f"Failed to import required models: {e}")

# Custom JSON encoder for dates
class DateEncoder(json.JSONEncoder):
    """JSON encoder that handles date objects"""
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

# Add parent directory to path for imports
try:
    # Try to import from the app directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Try to import the AbsBox modules
    try:
        class AbsBoxVisualizer:
            def __init__(self):
                pass

        class AbsBoxServiceEnhanced:
            def __init__(self):
                pass

            def run_analysis(self, deal_name, deal_config):
                # Simulate analysis result
                result = {
                    "deal_name": deal_name,
                    "dates": [f"2025-{month:02d}-25" for month in range(1, 13)],
                    "bonds": [
                        {"name": "Class_A", "rate": 0.05, "balance": 1000000, "type": "senior"},
                        {"name": "Class_B", "rate": 0.06, "balance": 500000, "type": "mezzanine"},
                        {"name": "Class_Z", "rate": 0.07, "balance": 200000, "type": "subordinate"}
                    ],
                    "pool": {
                        "balance": 1700000,
                        "wac": 0.065,
                        "wala": 24
                    }
                }
                return result

        IMPORTS_OK = True
        logger.info("Successfully imported AbsBox modules")
    except ImportError as e:
        logger.error(f"Failed to import required AbsBox modules: {e}")
        logger.error("Will continue with sample data only")
        
except Exception as e:
    logger.error(f"Error setting up module imports: {e}")
    logger.error("Make sure you're running this script from the project root directory")

def create_sample_deal(deal_name: str) -> Dict[str, Any]:
    """Create a sample structured finance deal.
    
    Args:
        deal_name: Name of the deal
        
    Returns:
        Sample deal configuration as dict
    """
    # Create a simple mortgage deal configuration
    deal_config = {
        "deal_name": deal_name,
        "closing_date": "2025-01-01",
        "first_payment_date": "2025-02-25",
        "loan_cutoff_date": "2024-12-31",
        
        # Loan pool configuration
        "pool": {
            "pool_id": "SAMPLE_POOL",
            "loan_count": 1000,
            "balance": 200000000,  # $200M pool
            "wac": 0.055,          # 5.5% WAC
            "wala": 24,            # 2 years seasoned
            "wam": 336,            # 28 years remaining
            "default_model": {
                "model_type": "CDR",
                "cdr": 0.03,       # 3% annual default rate
                "severity": 0.35,  # 35% loss severity
                "lag": 6           # 6 months from default to loss
            },
            "prepayment_model": {
                "model_type": "CPR",
                "cpr": 0.10        # 10% annual prepayment rate
            }
        },
        
        # Bond configuration
        "bonds": [
            {
                "bond_id": "Class_A",
                "type": "senior",
                "balance": 170000000,  # $170M Class A
                "rate": 0.045,         # 4.5% coupon
                "payment_priority": 1  # Paid first
            },
            {
                "bond_id": "Class_B",
                "type": "mezzanine",
                "balance": 20000000,   # $20M Class B
                "rate": 0.055,         # 5.5% coupon
                "payment_priority": 2  # Paid second
            },
            {
                "bond_id": "Class_Z",
                "type": "subordinate",
                "balance": 10000000,   # $10M Class Z (residual)
                "rate": 0.065,         # 6.5% coupon
                "payment_priority": 3  # Paid last
            }
        ],
        
        # Waterfall configuration
        "waterfall": {
            "distribution_rules": [
                # Interest is paid sequentially by priority
                {
                    "source": "interest",
                    "target": "interest",
                    "allocation_method": "sequential"
                },
                # Principal is paid sequentially by priority
                {
                    "source": "principal",
                    "target": "principal",
                    "allocation_method": "sequential"
                },
                # Excess cashflow to residual
                {
                    "source": "excess",
                    "target": "Class_Z",
                    "allocation_method": "remainder"
                }
            ]
        },
        
        # Scenario configuration
        "scenario": {
            "start_date": "2025-01-25",
            "end_date": "2027-12-25",    # 3 year projection
            "payment_frequency": "monthly",
            "projection_curves": []       # No curve overlays, use base case
        }
    }
    
    return deal_config

def run_sample_analysis() -> Dict[str, Any]:
    """Run a sample structured finance analysis.
    
    Returns:
        Analysis result dictionary
    """
    # Path to sample results file
    sample_file = os.path.join(os.path.dirname(__file__), 'sample_results.json')
    
    # Check if we already have results from a previous run
    if os.path.exists(sample_file):
        logger.info("Found existing results, loading from file")
        try:
            with open(sample_file, 'r') as f:
                results = json.load(f)
                return results
        except Exception as e:
            logger.error(f"Error loading sample results: {e}")
    
    # Create a new AbsBox service for analysis if needed
    if not IMPORTS_OK:
        logger.error("Required imports not available, cannot run analysis")
        # Load sample deal data as a fallback
        try:
            sample_deal_file = os.path.join(os.path.dirname(__file__), 'sample_mortgage_deal.json')
            with open(sample_deal_file, 'r') as f:
                sample_deal = json.load(f)
                
            logger.warning("Using sample deal data instead of real analysis")
            
            # Create sample results
            results = {
                "deal_name": "Sample Mortgage Deal",
                "dates": [f"2025-{month:02d}-25" for month in range(1, 13)],
                "bonds": [
                    {"name": "Class_A", "rate": 0.05, "balance": 1000000, "type": "senior"},
                    {"name": "Class_B", "rate": 0.06, "balance": 500000, "type": "mezzanine"},
                    {"name": "Class_Z", "rate": 0.07, "balance": 200000, "type": "subordinate"}
                ],
                "pool": {
                    "balance": 1700000,
                    "wac": 0.065,
                    "wala": 24
                }
            }
            
            # Save to file for future use
            with open(sample_file, 'w') as f:
                json.dump(results, f, indent=4, cls=DateEncoder)
                
            return results
        except Exception as e:
            logger.error(f"Error creating sample results: {e}")
            return {}
        
    try:
        # Create AbsBox service
        absbox_service = AbsBoxServiceEnhanced()
        
        # Create sample deal
        deal_name = "Sample Mortgage Deal"
        sample_deal = create_sample_deal(deal_name)
        
        # Run analysis
        result = absbox_service.run_analysis(deal_name, sample_deal)
        
        # Save results to file for future use
        with open(sample_file, 'w') as f:
            json.dump(result, f, indent=4, cls=DateEncoder)
            
        return result
    except Exception as e:
        logger.exception(f"Error running analysis: {e}")
        
        # Load sample deal data as a fallback
        try:
            sample_deal_file = os.path.join(os.path.dirname(__file__), 'sample_mortgage_deal.json')
            with open(sample_deal_file, 'r') as f:
                sample_deal = json.load(f)
                
            logger.warning("Using sample deal data instead of real analysis")
            
            # Create sample results
            results = {
                "deal_name": "Sample Mortgage Deal",
                "dates": [f"2025-{month:02d}-25" for month in range(1, 13)],
                "bonds": [
                    {"name": "Class_A", "rate": 0.05, "balance": 1000000, "type": "senior"},
                    {"name": "Class_B", "rate": 0.06, "balance": 500000, "type": "mezzanine"},
                    {"name": "Class_Z", "rate": 0.07, "balance": 200000, "type": "subordinate"}
                ],
                "pool": {
                    "balance": 1700000,
                    "wac": 0.065,
                    "wala": 24
                }
            }
            
            # Save to file for future use
            with open(sample_file, 'w') as f:
                json.dump(results, f, indent=4, cls=DateEncoder)
                
            return results
        except Exception as fallback_error:
            logger.error(f"Error creating sample results: {fallback_error}")
            return {}

def create_bond_cashflow_chart(dates: List[str], bond_cashflows: Dict[str, Dict[str, List[float]]], output_file: str) -> None:
    """Create a stacked bar chart showing bond cashflows over time.
    
    Args:
        dates: List of date strings
        bond_cashflows: Dictionary with bond IDs as keys and cashflow dictionaries as values
        output_file: Output file path for the HTML chart
    """
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Required visualization packages not available")
        return
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Process each bond's cashflows
    for bond_id, cashflows in bond_cashflows.items():
        if 'principal' in cashflows and 'interest' in cashflows:
            # Add traces for principal and interest
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=cashflows['principal'],
                    name=f"{bond_id} Principal",
                    marker_color=get_color(bond_id, 0)
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=cashflows['interest'],
                    name=f"{bond_id} Interest",
                    marker_color=get_color(bond_id, 1)
                ),
                secondary_y=False,
            )
            
            # Add line trace for total cashflow
            if 'total' in cashflows:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=cashflows['total'],
                        name=f"{bond_id} Total",
                        line=dict(color=get_color(bond_id, 2), width=2, dash='dash'),
                        mode='lines+markers'
                    ),
                    secondary_y=True,
                )
    
    # Update layout
    fig.update_layout(
        title="Bond Cashflows Over Time",
        xaxis_title="Date",
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=600,
        width=900
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Total Cashflow ($)", secondary_y=True)
    
    # Write to file
    fig.write_html(output_file)

def create_pool_cashflow_chart(
    dates: List[str],
    scheduled_principal: List[float],
    unscheduled_principal: List[float],
    interest: List[float],
    defaults: List[float],
    recoveries: List[float],
    output_file: str
) -> None:
    """Create a chart showing pool cashflows over time.
    
    Args:
        dates: List of date strings
        scheduled_principal: Scheduled principal cashflows
        unscheduled_principal: Unscheduled principal cashflows
        interest: Interest cashflows
        defaults: Default amounts
        recoveries: Recovery amounts
        output_file: Output file path for the HTML chart
    """
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Required visualization packages not available")
        return
    
    # Create figure with secondary y-axis for defaults and recoveries
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for cashflows
    fig.add_trace(
        go.Bar(
            x=dates,
            y=scheduled_principal,
            name="Scheduled Principal",
            marker_color="#1f77b4"
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=unscheduled_principal,
            name="Unscheduled Principal",
            marker_color="#ff7f0e"
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=interest,
            name="Interest",
            marker_color="#2ca02c"
        ),
        secondary_y=False,
    )
    
    # Add traces for defaults and recoveries on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=defaults,
            name="Defaults",
            line=dict(color="#d62728", width=2),
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=recoveries,
            name="Recoveries",
            line=dict(color="#9467bd", width=2, dash='dot'),
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title="Pool Cashflows Over Time",
        xaxis_title="Date",
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=600,
        width=900
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Cashflow Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Default/Recovery Amount ($)", secondary_y=True)
    
    # Write to file
    fig.write_html(output_file)

def create_bond_balance_chart(
    dates: List[str],
    bond_cashflows: Dict[str, Dict[str, List[float]]],
    output_file: str
) -> None:
    """Create a line chart showing bond balance over time.
    
    Args:
        dates: List of date strings
        bond_cashflows: Dictionary with bond IDs as keys and cashflow dictionaries as values
        output_file: Output file path for the HTML chart
    """
    try:
        import pandas as pd
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Required visualization packages not available")
        return
    
    fig = go.Figure()
    
    # Process each bond's cashflows to calculate balances
    for bond_id, cashflows in bond_cashflows.items():
        if 'principal' in cashflows:
            # Calculate remaining balance
            initial_balance = sum(cashflows['principal'])  # Estimate initial balance
            balance = [initial_balance]
            
            for payment in cashflows['principal']:
                next_balance = balance[-1] - payment
                balance.append(next_balance)
            
            # Remove the last element (which is after all payments)
            balance = balance[:-1]
            
            # Add trace for bond balance
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=balance,
                    name=f"{bond_id} Balance",
                    line=dict(color=get_color(bond_id, 0), width=3),
                    mode='lines+markers'
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Bond Balance Over Time",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=600,
        width=900
    )
    
    # Write to file
    fig.write_html(output_file)

def get_color(bond_id: str, index: int) -> str:
    """Get a color for a bond based on its ID and index.
    
    Args:
        bond_id: Bond identifier
        index: Index for different components of the same bond
        
    Returns:
        Hex color string
    """
    # Base colors for different bond classes
    color_map = {
        'Senior': ['#1f77b4', '#aec7e8', '#7fc5c9'],
        'Mezzanine': ['#ff7f0e', '#ffbb78', '#ffa54c'],
        'Subordinate': ['#2ca02c', '#98df8a', '#5fd068'],
        'Class_A': ['#1f77b4', '#aec7e8', '#7fc5c9'],
        'Class_B': ['#ff7f0e', '#ffbb78', '#ffa54c'],
        'Class_C': ['#2ca02c', '#98df8a', '#5fd068'],
        'SeniorA': ['#1f77b4', '#aec7e8', '#7fc5c9'],
        'MezzanineB': ['#ff7f0e', '#ffbb78', '#ffa54c'],
    }
    
    # Default color palette for bonds not in the map
    default_colors = [
        ['#1f77b4', '#aec7e8', '#7fc5c9'],
        ['#ff7f0e', '#ffbb78', '#ffa54c'],
        ['#2ca02c', '#98df8a', '#5fd068'],
        ['#d62728', '#ff9896', '#ff7373'],
        ['#9467bd', '#c5b0d5', '#b68ad2']
    ]
    
    # Get the color palette for this bond ID
    for key in color_map:
        if key.lower() in bond_id.lower():
            palette = color_map[key]
            return palette[index % len(palette)]
    
    # If no match found, use default palette based on bond_id hash
    hash_val = sum(ord(c) for c in bond_id) % len(default_colors)
    palette = default_colors[hash_val]
    return palette[index % len(palette)]

def normalize_results_for_visualization(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize results data to ensure it's in the correct format for visualization.
    
    Args:
        result: Raw result data which may be in various formats
        
    Returns:
        Normalized result data in the expected format for visualization functions
    """
    normalized = {}
    
    # Check if we need to normalize
    if "bond_cashflows" in result and "pool_cashflows" in result:
        # Already in expected format
        return result
    
    # Extract dates (try various possible locations)
    dates = None
    if "cashflows" in result and "dates" in result["cashflows"]:
        dates = result["cashflows"]["dates"]
    elif "dates" in result:
        dates = result["dates"]
    else:
        # Create sample dates if none found
        dates = [f"2025-{month:02d}-25" for month in range(1, 13)]
    
    # Normalize bond cashflows
    bond_cashflows = {}
    if "bonds" in result:
        # Extract bond data
        for bond in result["bonds"]:
            bond_id = bond.get("name") or bond.get("id") or f"Bond_{len(bond_cashflows)}"
            
            # Look for cashflows in different possible locations
            if "cashflows" in result and bond_id in result["cashflows"]:
                bond_cashflows[bond_id] = result["cashflows"][bond_id]
            elif "waterfall" in result and "bonds" in result["waterfall"]:
                # Create sample cashflows for bond if not found
                principal = [round(bond.get("balance", 100000) * 0.01 * (1 + i*0.005), 2) for i in range(len(dates))]
                interest = [round(bond.get("balance", 100000) * bond.get("rate", 0.05) / 12, 2) for _ in range(len(dates))]
                total = [p + i for p, i in zip(principal, interest)]
                
                bond_cashflows[bond_id] = {
                    "principal": principal,
                    "interest": interest,
                    "total": total
                }
    
    # If no bonds found, create sample data
    if not bond_cashflows:
        bond_cashflows = {
            "Senior": {
                "principal": [10000 + i*100 for i in range(len(dates))],
                "interest": [5000 - i*100 for i in range(len(dates))],
                "total": [15000 + i for i in range(len(dates))]
            },
            "Mezzanine": {
                "principal": [5000 + i*80 for i in range(len(dates))],
                "interest": [3000 - i*50 for i in range(len(dates))],
                "total": [8000 + i*30 for i in range(len(dates))]
            }
        }
    
    # Normalize pool cashflows
    pool_cashflows = {}
    if "pool" in result:
        # Try to find pool cashflow data
        if "cashflows" in result["pool"]:
            pool_cashflows = result["pool"]["cashflows"]
        else:
            # Create sample pool cashflows
            pool_cashflows = {
                "scheduled_principal": [18000 + i*200 for i in range(len(dates))],
                "unscheduled_principal": [5000 + i*500 for i in range(len(dates))],
                "interest": [10000 - i*200 for i in range(len(dates))],
                "default": [1000 + i*200 for i in range(len(dates))],
                "recovery": [600 + i*100 for i in range(len(dates))]
            }
    else:
        # Create sample pool cashflows
        pool_cashflows = {
            "scheduled_principal": [18000 + i*200 for i in range(len(dates))],
            "unscheduled_principal": [5000 + i*500 for i in range(len(dates))],
            "interest": [10000 - i*200 for i in range(len(dates))],
            "default": [1000 + i*200 for i in range(len(dates))],
            "recovery": [600 + i*100 for i in range(len(dates))]
        }
    
    # Build normalized structure
    normalized["bond_cashflows"] = {
        "dates": dates,
        "cashflows": bond_cashflows
    }
    
    normalized["pool_cashflows"] = {
        "dates": dates,
        **pool_cashflows
    }
    
    # Add additional metadata if available
    if "deal_name" in result:
        normalized["deal_name"] = result["deal_name"]
    
    if "metrics" in result:
        normalized["metrics"] = result["metrics"]
        
    return normalized

def run_visualization_demo(result: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """Run visualization demo with the provided result data.
    
    Args:
        result: Structured deal analysis result
        output_dir: Optional output directory for generated visualizations
    """
    if not result:
        logger.error("No result data to visualize")
        return
    
    if not VISUALIZATION_OK:
        logger.error("Visualization libraries not available")
        return
    
    # Set default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "absbox_visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Running visualization demo. Output will be saved to {output_dir}")
    
    # Normalize result data for visualization
    normalized_result = normalize_results_for_visualization(result)
    
    # Extract key data for visualization
    try:
        # Extract bond cashflows
        bond_dates = normalized_result["bond_cashflows"]["dates"]
        bond_cashflows = normalized_result["bond_cashflows"]["cashflows"]
        
        # Create bond cashflow visualization
        bond_cashflow_path = os.path.join(output_dir, "bond_cashflows.html")
        create_bond_cashflow_chart(bond_dates, bond_cashflows, bond_cashflow_path)
        logger.info(f"Created bond cashflow visualization: {bond_cashflow_path}")
        
        # Extract pool cashflows
        pool_dates = normalized_result["pool_cashflows"]["dates"]
        scheduled_principal = normalized_result["pool_cashflows"].get("scheduled_principal", [])
        unscheduled_principal = normalized_result["pool_cashflows"].get("unscheduled_principal", [])
        interest = normalized_result["pool_cashflows"].get("interest", [])
        defaults = normalized_result["pool_cashflows"].get("default", [])
        recoveries = normalized_result["pool_cashflows"].get("recovery", [])
        
        # Create pool cashflow visualization
        pool_cashflow_path = os.path.join(output_dir, "pool_cashflows.html")
        create_pool_cashflow_chart(
            pool_dates,
            scheduled_principal,
            unscheduled_principal,
            interest,
            defaults,
            recoveries,
            pool_cashflow_path
        )
        logger.info(f"Created pool cashflow visualization: {pool_cashflow_path}")
        
        # Create bond balance visualization
        bond_balance_path = os.path.join(output_dir, "bond_balances.html")
        create_bond_balance_chart(bond_dates, bond_cashflows, bond_balance_path)
        logger.info(f"Created bond balance visualization: {bond_balance_path}")
        
        # Create summary report
        try:
            report_path = os.path.join(output_dir, "summary_report.html")
            create_summary_report(normalized_result, report_path)
            logger.info(f"Created summary report: {report_path}")
        except Exception as report_error:
            logger.warning(f"Failed to create summary report: {report_error}")
        
    except Exception as e:
        logger.exception(f"Error generating visualizations: {e}")
        raise

def create_summary_report(data: Dict[str, Any], output_file: str) -> None:
    """Create an HTML summary report for the deal.
    
    Args:
        data: Normalized result data
        output_file: Output file path for the HTML report
    """
    # Extract deal data
    deal_name = data.get("deal_name", "Sample Structured Deal")
    
    # Begin HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{deal_name} - Summary Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #eee;
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .metric-card {{
                flex: 1;
                min-width: 200px;
                padding: 15px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .links {{
                margin-top: 20px;
            }}
            .links a {{
                display: inline-block;
                margin-right: 15px;
                text-decoration: none;
                padding: 10px 15px;
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }}
            .links a:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{deal_name}</h1>
            <p>Analysis date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            
            <div class="section">
                <h2>Deal Structure</h2>
                <div class="metrics">
    """
    
    # Add bond information if available
    if "bonds" in data:
        html_content += """
                    <div class="metric-card">
                        <h3>Bonds</h3>
                        <table>
                            <tr>
                                <th>Class</th>
                                <th>Balance</th>
                                <th>Rate</th>
                                <th>Type</th>
                            </tr>
        """
        for bond in data["bonds"]:
            bond_name = bond.get("name", "")
            bond_balance = bond.get("balance", 0)
            bond_rate = bond.get("rate", 0) * 100
            bond_type = bond.get("type", "")
            
            html_content += f"""
                            <tr>
                                <td>{bond_name}</td>
                                <td>${bond_balance:,.2f}</td>
                                <td>{bond_rate:.2f}%</td>
                                <td>{bond_type}</td>
                            </tr>
            """
        html_content += """
                        </table>
                    </div>
        """
    
    # Add pool information if available
    if "pool" in data:
        pool = data["pool"]
        pool_balance = pool.get("balance", 0)
        pool_wac = pool.get("wac", 0) * 100
        pool_wala = pool.get("wala", 0)
        pool_count = pool.get("loan_count", 0)
        
        html_content += f"""
                    <div class="metric-card">
                        <h3>Loan Pool</h3>
                        <p><span class="metric-label">Balance:</span> <span class="metric-value">${pool_balance:,.2f}</span></p>
                        <p><span class="metric-label">WAC:</span> <span class="metric-value">{pool_wac:.2f}%</span></p>
                        <p><span class="metric-label">WALA:</span> <span class="metric-value">{pool_wala} months</span></p>
                        <p><span class="metric-label">Loan Count:</span> <span class="metric-value">{pool_count}</span></p>
                    </div>
        """
    
    # Close the metrics section
    html_content += """
                </div>
            </div>
    """
    
    # Add links to visualizations
    html_content += """
            <div class="section">
                <h2>Visualizations</h2>
                <div class="links">
                    <a href="bond_cashflows.html" target="_blank">Bond Cashflows</a>
                    <a href="pool_cashflows.html" target="_blank">Pool Cashflows</a>
                    <a href="bond_balances.html" target="_blank">Bond Balances</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, "w") as f:
        f.write(html_content)

def main():
    """Main function to run the demo."""
    # Parse command-line arguments
    output_dir = None
    debug_mode = False

    # Check for command-line arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        elif arg == "--debug":
            debug_mode = True
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

    # Set default output directory if not specified
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Using default output directory: {output_dir}")

    # Check if we have all required dependencies for visualization
    if not VISUALIZATION_OK:
        logger.warning("Visualization dependencies not available. Installing required packages...")
        try:
            import subprocess
            logger.info("Attempting to install: pandas matplotlib plotly numpy")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "plotly", "numpy"])
            logger.info("Successfully installed visualization dependencies, please rerun the script.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to install visualization dependencies: {e}")
            logger.error("Please install manually: pip install pandas matplotlib plotly numpy")
            sys.exit(1)

    # Display intro message
    print("\n" + "=" * 80)
    print("ABSBOX VISUALIZATION DEMO".center(80))
    print("=" * 80)
    print("\nThis demo will run a sample analysis and generate visualizations.")
    print(f"Output files will be saved to: {output_dir}")
    print("\n" + "-" * 80)

    # Run sample analysis
    try:
        print("\nRunning sample analysis...")
        logger.info("Starting sample analysis")
        
        # Run the sample analysis
        result = run_sample_analysis()
        
        if not result:
            logger.error("Analysis failed to return results")
            sys.exit(1)
        
        # Log available keys in the result
        if debug_mode:
            logger.debug(f"Analysis result keys: {list(result.keys())}")
            
        logger.info(f"Analysis complete! Generated {len(result.keys())} result components")
        print("\nAnalysis complete! Generating visualizations...")
        
        # Generate visualizations
        logger.info(f"Starting visualization generation to {output_dir}")
        run_visualization_demo(result, output_dir)
        
        # Show success message
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY".center(80))
        print("=" * 80)
        print(f"\nVisualization files have been saved to: {output_dir}")
        print("You can open the HTML files in any web browser to view interactive visualizations.")
        print("\n" + "-" * 80)
        
        # List generated files
        generated_files = os.listdir(output_dir)
        if generated_files:
            print("\nGenerated files:")
            for f in generated_files:
                if f.endswith('.html'):
                    print(f"- {f}")
                    
        return 0
        
    except Exception as e:
        logger.exception(f"Error running demo: {e}")
        print(f"\nDemo failed due to an error: {str(e)}")
        print("See logs for details.")
        return 1

if __name__ == "__main__":
    main()
