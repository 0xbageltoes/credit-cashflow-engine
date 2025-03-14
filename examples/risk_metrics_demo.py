"""
Risk Metrics and Stress Testing Demo

This script demonstrates the use of the enhanced Monte Carlo simulation service
with risk metrics calculation, stress testing, and sensitivity analysis.
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.monte_carlo_service import MonteCarloService
from app.models.monte_carlo import (
    MonteCarloSimulationRequest, SimulationVariable, 
    DistributionType, CorrelationMatrix
)
from app.utils.redis_service import RedisService
from app.utils.logging_utils import setup_logging

# Set up logging
setup_logging()

async def run_demo():
    """Run a comprehensive demonstration of risk metrics and stress testing"""
    print("\n===== Risk Metrics and Stress Testing Demo =====\n")
    
    # Initialize services
    redis_service = RedisService()
    monte_carlo_service = MonteCarloService(redis_service=redis_service)
    
    # Create a base simulation request
    base_request = create_sample_request()
    user_id = "demo_user"
    
    print("\n----- Running Base Simulation -----\n")
    base_result = await monte_carlo_service.run_enhanced_simulation(
        request=base_request,
        user_id=user_id,
        use_cache=True
    )
    
    # Display base simulation results
    print_simulation_results(base_result, "Base Scenario")
    
    # Display risk metrics
    print_risk_metrics(base_result.risk_metrics, "Base Scenario Risk Metrics")
    
    # Run stress tests
    print("\n----- Running Stress Tests -----\n")
    stress_scenarios = create_stress_scenarios()
    
    stress_results = monte_carlo_service.run_stress_test(
        base_request=base_request,
        stress_scenarios=stress_scenarios,
        user_id=user_id
    )
    
    # Display stress test results
    for scenario_name, result in stress_results.items():
        if scenario_name != "base_scenario":  # Already displayed base scenario
            print_simulation_results(result, scenario_name)
            print_risk_metrics(result.risk_metrics, f"{scenario_name} Risk Metrics")
    
    # Run sensitivity analysis
    print("\n----- Running Sensitivity Analysis -----\n")
    parameters_to_test = [
        {
            "name": "Default Rate",
            "path": "variables.0.parameters.mean",
            "values": [0.01, 0.03, 0.05, 0.07, 0.10]
        },
        {
            "name": "Recovery Rate",
            "path": "variables.1.parameters.mean",
            "values": [0.3, 0.4, 0.5, 0.6, 0.7]
        }
    ]
    
    sensitivity_results = monte_carlo_service.perform_sensitivity_analysis(
        base_request=base_request,
        parameters_to_test=parameters_to_test,
        user_id=user_id
    )
    
    # Display sensitivity analysis results
    print_sensitivity_results(sensitivity_results)
    
    # Generate and display visualizations
    print("\n----- Generating Visualizations -----\n")
    generate_visualizations(base_result, stress_results, sensitivity_results)
    
    print("\n===== Demo Complete =====\n")

def create_sample_request() -> MonteCarloSimulationRequest:
    """Create a sample simulation request for demonstration"""
    
    # Define simulation variables
    variables = [
        SimulationVariable(
            name="default_rate",
            description="Annual default rate",
            distribution=DistributionType.NORMAL,
            parameters={
                "mean": 0.03,
                "std_dev": 0.01
            }
        ),
        SimulationVariable(
            name="recovery_rate",
            description="Recovery rate on defaulted assets",
            distribution=DistributionType.BETA,
            parameters={
                "alpha": 5.0,
                "beta": 2.0,
                "min": 0.3,
                "max": 0.9
            }
        ),
        SimulationVariable(
            name="prepayment_rate",
            description="Annual prepayment rate",
            distribution=DistributionType.LOGNORMAL,
            parameters={
                "mean": -2.0,  # log of ~0.135
                "std_dev": 0.5
            }
        ),
        SimulationVariable(
            name="interest_rate",
            description="Annual interest rate",
            distribution=DistributionType.NORMAL,
            parameters={
                "mean": 0.04,
                "std_dev": 0.005
            }
        ),
        SimulationVariable(
            name="spread",
            description="Credit spread",
            distribution=DistributionType.NORMAL,
            parameters={
                "mean": 0.02,
                "std_dev": 0.004
            }
        )
    ]
    
    # Define correlations
    correlations = CorrelationMatrix(
        correlations={
            "default_rate:recovery_rate": -0.7,  # Negative correlation
            "default_rate:interest_rate": 0.3,   # Positive correlation
            "interest_rate:spread": 0.5,         # Positive correlation
            "prepayment_rate:interest_rate": -0.6  # Negative correlation
        }
    )
    
    # Define economic factors
    economic_factors = {
        "unemployment_rate": 0.05,
        "gdp_growth": 0.02,
        "housing_price_index_growth": 0.03
    }
    
    # Create the request
    request = MonteCarloSimulationRequest(
        name="Credit Portfolio Risk Analysis",
        description="Analysis of a sample credit portfolio under various economic scenarios",
        num_simulations=1000,
        variables=variables,
        correlation_matrix=correlations,
        asset_class="credit_portfolio",
        asset_parameters={
            "portfolio_size": 1000000000,  # $1B portfolio
            "average_duration": 5.0,
            "credit_quality": "investment_grade",
            "diversification": "moderate"
        },
        analysis_date=datetime.now().date(),
        projection_months=60,  # 5-year projection
        discount_rate=0.03,
        include_detailed_paths=True,
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    
    # Add economic factors
    request.economic_factors = economic_factors
    
    return request

def create_stress_scenarios() -> list:
    """Create a list of stress scenarios for testing"""
    return [
        {
            "name": "severe_recession",
            "description": "Severe economic downturn scenario",
            "economic_factors": {
                "unemployment_rate": 0.10,  # 10% unemployment
                "gdp_growth": -0.03,        # -3% GDP growth
                "housing_price_index_growth": -0.15  # 15% housing price decline
            },
            "variables": {
                "default_rate": {
                    "parameters": {
                        "mean": {"factor": 2.5}  # 2.5x default rate
                    }
                },
                "recovery_rate": {
                    "parameters": {
                        "min": {"factor": 0.8},  # 20% lower recovery minimums
                        "max": {"factor": 0.8}   # 20% lower recovery maximums
                    }
                },
                "spread": {
                    "parameters": {
                        "mean": {"factor": 2.0}  # 2x spread
                    }
                }
            }
        },
        {
            "name": "mild_recession",
            "description": "Mild economic slowdown scenario",
            "economic_factors": {
                "unemployment_rate": 0.07,   # 7% unemployment
                "gdp_growth": -0.01,         # -1% GDP growth
                "housing_price_index_growth": -0.05  # 5% housing price decline
            },
            "variables": {
                "default_rate": {
                    "parameters": {
                        "mean": {"factor": 1.5}  # 1.5x default rate
                    }
                },
                "recovery_rate": {
                    "parameters": {
                        "min": {"factor": 0.9},  # 10% lower recovery minimums
                        "max": {"factor": 0.9}   # 10% lower recovery maximums
                    }
                },
                "spread": {
                    "parameters": {
                        "mean": {"factor": 1.5}  # 1.5x spread
                    }
                }
            }
        },
        {
            "name": "rising_rates",
            "description": "Rising interest rate environment",
            "economic_factors": {
                "unemployment_rate": 0.04,   # 4% unemployment (healthy economy)
                "gdp_growth": 0.03,          # 3% GDP growth
                "housing_price_index_growth": 0.02  # 2% housing price growth
            },
            "variables": {
                "interest_rate": {
                    "parameters": {
                        "mean": {"add": 0.02}  # +2% interest rates
                    }
                },
                "prepayment_rate": {
                    "parameters": {
                        "mean": {"add": -0.5}  # Lower prepayment (log-scale)
                    }
                }
            }
        }
    ]

def print_simulation_results(result, scenario_name):
    """Print the key results from a simulation"""
    print(f"\n=== {scenario_name} Results ===")
    print(f"Simulation ID: {result.simulation_id}")
    print(f"Completed {result.completed_iterations}/{result.num_iterations} iterations in {result.calculation_time:.2f} seconds")
    
    print("\nNPV Statistics:")
    print(f"  Mean: ${result.npv_stats.mean:,.2f}")
    print(f"  Median: ${result.npv_stats.median:,.2f}")
    print(f"  Std Dev: ${result.npv_stats.std_dev:,.2f}")
    print(f"  Min: ${result.npv_stats.min_value:,.2f}")
    print(f"  Max: ${result.npv_stats.max_value:,.2f}")
    
    print("\nNPV Percentiles:")
    for p, value in sorted(result.npv_stats.percentiles.items(), key=lambda x: float(x[0])):
        print(f"  {p}%: ${float(value):,.2f}")
    
    if hasattr(result, 'expected_loss') and result.expected_loss is not None:
        print(f"\nExpected Loss: ${result.expected_loss:,.2f}")
    
    if hasattr(result, 'unexpected_loss') and result.unexpected_loss is not None:
        print(f"Unexpected Loss: ${result.unexpected_loss:,.2f}")
    
    if hasattr(result, 'economic_capital') and result.economic_capital is not None:
        print(f"Economic Capital (99.9%): ${result.economic_capital:,.2f}")

def print_risk_metrics(risk_metrics, title):
    """Print risk metrics in a formatted way"""
    if risk_metrics is None:
        print(f"\n{title}: No risk metrics available")
        return
        
    print(f"\n{title}:")
    
    print("\nValue at Risk (VaR):")
    for conf, value in sorted(risk_metrics.var.items(), key=lambda x: float(x[0])):
        print(f"  {float(conf)*100:.1f}% VaR: ${float(value):,.2f}")
    
    print("\nConditional Value at Risk (CVaR):")
    for conf, value in sorted(risk_metrics.cvar.items(), key=lambda x: float(x[0])):
        print(f"  {float(conf)*100:.1f}% CVaR: ${float(value):,.2f}")
    
    print("\nVolatility Measures:")
    print(f"  Volatility: ${risk_metrics.volatility:,.2f}")
    print(f"  Downside Deviation: ${risk_metrics.downside_deviation:,.2f}")
    
    print("\nPerformance Ratios:")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {risk_metrics.sortino_ratio:.4f}")
    
    print(f"\nMaximum Drawdown: ${risk_metrics.max_drawdown:,.2f}")

def print_sensitivity_results(sensitivity_results):
    """Print the results of sensitivity analysis"""
    print("\nSensitivity Analysis Results:")
    
    for param_name, param_results in sensitivity_results.items():
        if param_name == "base_scenario":
            continue
            
        print(f"\n--- Parameter: {param_name} ---")
        
        # Create a list of values and corresponding NPV means
        values = []
        means = []
        
        for value_str, result in sorted(param_results.items(), key=lambda x: float(x[0]) if x[0].replace('.', '', 1).isdigit() else x[0]):
            values.append(value_str)
            means.append(result.npv_stats.mean)
            
            print(f"  Value: {value_str}")
            print(f"    Mean NPV: ${result.npv_stats.mean:,.2f}")
            print(f"    Median NPV: ${result.npv_stats.median:,.2f}")
            if hasattr(result, 'risk_metrics') and result.risk_metrics:
                print(f"    95% VaR: ${result.risk_metrics.var.get('0.95', 0):,.2f}")
                print(f"    Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.4f}")
            print()

def generate_visualizations(base_result, stress_results, sensitivity_results):
    """Generate visualizations of the results"""
    try:
        # Create directory for visualizations if it doesn't exist
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. NPV Distribution Comparison
        plt.figure(figsize=(12, 8))
        
        # Add base scenario
        if hasattr(base_result, 'npv_stats') and base_result.npv_stats:
            # Create synthetic distribution based on percentiles
            x = list(base_result.npv_stats.percentiles.keys())
            y = list(base_result.npv_stats.percentiles.values())
            plt.plot(y, [float(p)/100 for p in x], label="Base Scenario", 
                     linewidth=2, marker='o')
        
        # Add stress scenarios
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        color_idx = 0
        for scenario_name, result in stress_results.items():
            if scenario_name != "base_scenario" and hasattr(result, 'npv_stats') and result.npv_stats:
                x = list(result.npv_stats.percentiles.keys())
                y = list(result.npv_stats.percentiles.values())
                plt.plot(y, [float(p)/100 for p in x], label=scenario_name,
                         linewidth=2, marker='o', color=colors[color_idx % len(colors)])
                color_idx += 1
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('NPV ($)')
        plt.ylabel('Cumulative Probability')
        plt.title('NPV Distribution Comparison Across Scenarios')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'npv_distribution_comparison.png'))
        
        # 2. VaR Comparison
        plt.figure(figsize=(10, 6))
        scenarios = ["Base Scenario"] + [s for s in stress_results.keys() if s != "base_scenario"]
        var_values = []
        
        for i, scenario_name in enumerate(scenarios):
            if i == 0:
                result = base_result
            else:
                result = stress_results[scenario_name]
                
            if hasattr(result, 'risk_metrics') and result.risk_metrics:
                var_values.append(float(result.risk_metrics.var.get('0.95', 0)))
            else:
                var_values.append(0)
        
        plt.bar(scenarios, var_values, color='darkred')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylabel('95% VaR ($)')
        plt.title('Value at Risk (95%) Comparison Across Scenarios')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'var_comparison.png'))
        
        # 3. Sensitivity Analysis
        for param_name, param_results in sensitivity_results.items():
            if param_name == "base_scenario":
                continue
                
            plt.figure(figsize=(10, 6))
            
            values = []
            means = []
            var_values = []
            
            for value_str, result in sorted(param_results.items(), key=lambda x: float(x[0]) if x[0].replace('.', '', 1).isdigit() else x[0]):
                values.append(value_str)
                means.append(result.npv_stats.mean)
                
                if hasattr(result, 'risk_metrics') and result.risk_metrics:
                    var_values.append(float(result.risk_metrics.var.get('0.95', 0)))
                else:
                    var_values.append(0)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('Mean NPV ($)', color=color)
            ax1.plot(values, means, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('95% VaR ($)', color=color)
            ax2.plot(values, var_values, marker='s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Sensitivity Analysis: Impact of {param_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'sensitivity_{param_name}.png'))
        
        print(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())
