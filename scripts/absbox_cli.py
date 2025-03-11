#!/usr/bin/env python
"""
Command-line interface for AbsBox integration

This CLI tool allows you to quickly test AbsBox functionality without
running the full API server.
"""
import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

# Import after setting path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import application modules
try:
    from app.services.absbox_service import AbsBoxService
    from app.models.structured_products import (
        StructuredDealRequest,
        LoanPoolConfig,
        LoanConfig,
        WaterfallConfig,
        AccountConfig,
        BondConfig,
        WaterfallAction,
        ScenarioConfig,
        DefaultCurveConfig,
        PrepaymentCurveConfig
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have installed the required dependencies with:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def health_check():
    """Check the health of the AbsBox service"""
    try:
        service = AbsBoxService()
        result = service.health_check()
        
        print("\n=== AbsBox Service Health Check ===")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"AbsBox Version: {result.get('version', 'unknown')}")
        
        engine_info = result.get('engine', {})
        print("\nEngine Information:")
        print(f"  Type: {engine_info.get('type', 'unknown')}")
        print(f"  URL: {engine_info.get('url', 'N/A')}")
        print(f"  Status: {engine_info.get('status', 'unknown')}")
        
        cache_info = result.get('cache', {})
        print("\nCache Information:")
        print(f"  Enabled: {cache_info.get('enabled', False)}")
        print(f"  Provider: {cache_info.get('provider', 'N/A')}")
        print(f"  Status: {cache_info.get('status', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def load_deal_from_file(file_path):
    """Load a deal definition from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            deal_data = json.load(f)
        
        # Convert the JSON to a StructuredDealRequest
        deal_request = StructuredDealRequest.parse_obj(deal_data)
        return deal_request
    except Exception as e:
        print(f"Error loading deal from file: {str(e)}")
        return None

def create_sample_deal():
    """Create a simple sample deal for testing"""
    today = date.today()
    start_date = today - timedelta(days=30)
    
    # Create a loan pool with two loans
    pool_config = LoanPoolConfig(
        pool_name="Sample Pool",
        loans=[
            LoanConfig(
                balance=200000.0,
                rate=0.05,
                term=360,
                start_date=start_date,
                rate_type="fixed"
            ),
            LoanConfig(
                balance=150000.0,
                rate=0.045,
                term=360,
                start_date=start_date,
                rate_type="fixed"
            )
        ]
    )
    
    # Calculate the total pool balance
    total_pool_balance = sum(loan.balance for loan in pool_config.loans)
    
    # Create a simple waterfall
    waterfall_config = WaterfallConfig(
        start_date=today,
        accounts=[
            AccountConfig(
                name="ReserveFund",
                initial_balance=total_pool_balance * 0.02  # 2% reserve
            )
        ],
        bonds=[
            BondConfig(
                name="ClassA",
                balance=total_pool_balance * 0.8,  # 80% senior
                rate=0.04
            ),
            BondConfig(
                name="ClassB",
                balance=total_pool_balance * 0.2,  # 20% junior
                rate=0.06
            )
        ],
        actions=[
            WaterfallAction(
                source="CollectedInterest",
                target="ClassA",
                amount="Interest"
            ),
            WaterfallAction(
                source="CollectedInterest",
                target="ClassB",
                amount="Interest"
            ),
            WaterfallAction(
                source="CollectedPrincipal",
                target="ClassA",
                amount="OutstandingPrincipal"
            ),
            WaterfallAction(
                source="CollectedPrincipal",
                target="ClassB",
                amount="OutstandingPrincipal"
            )
        ]
    )
    
    # Create default scenario
    scenario_config = ScenarioConfig(
        name="Base Scenario",
        default_curve=DefaultCurveConfig(
            vector=[0.01, 0.015, 0.02, 0.015, 0.01]
        ),
        prepayment_curve=PrepaymentCurveConfig(
            vector=[0.05, 0.06, 0.07, 0.08, 0.09]
        )
    )
    
    # Create the deal request
    deal_request = StructuredDealRequest(
        deal_name="Sample Deal",
        pool=pool_config,
        waterfall=waterfall_config,
        scenario=scenario_config
    )
    
    return deal_request

def save_deal_to_file(deal_request, file_path):
    """Save a deal definition to a JSON file"""
    try:
        # Convert the deal request to a dictionary
        deal_data = deal_request.dict()
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(deal_data, f, indent=2, default=str)
        
        print(f"Deal saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving deal to file: {str(e)}")
        return False

def analyze_deal(deal_request, output_file=None, plot=False):
    """Analyze a structured deal"""
    try:
        # Initialize the service
        service = AbsBoxService()
        
        print(f"\nAnalyzing deal: {deal_request.deal_name}")
        print("Pool information:")
        print(f"  Name: {deal_request.pool.pool_name}")
        print(f"  Number of loans: {len(deal_request.pool.loans)}")
        print(f"  Total balance: ${sum(l.balance for l in deal_request.pool.loans):,.2f}")
        
        print("\nWaterfall information:")
        print(f"  Number of bonds: {len(deal_request.waterfall.bonds)}")
        print(f"  Bonds:")
        for bond in deal_request.waterfall.bonds:
            print(f"    {bond.name}: ${bond.balance:,.2f} at {bond.rate*100:.2f}%")
        
        print(f"\nScenario: {deal_request.scenario.name}")
        
        # Time the analysis
        start_time = time.time()
        result = service.analyze_deal(deal_request)
        elapsed = time.time() - start_time
        
        # Print results
        print(f"\nAnalysis completed in {elapsed:.2f} seconds (reported: {result.execution_time:.2f}s)")
        print(f"Status: {result.status}")
        
        if result.status == "success":
            print(f"Bond cashflows: {len(result.bond_cashflows)} periods")
            print(f"Pool cashflows: {len(result.pool_cashflows)} periods")
            
            # Convert to DataFrame
            df_bond = pd.DataFrame(result.bond_cashflows)
            df_pool = pd.DataFrame(result.pool_cashflows)
            
            # Print pool statistics
            print("\nPool statistics:")
            for key, value in result.pool_statistics.items():
                print(f"  {key}: {value}")
            
            # Print metrics
            if result.metrics:
                print("\nMetrics:")
                for category, metrics in result.metrics.items():
                    print(f"  {category}:")
                    for name, value in metrics.items():
                        if isinstance(value, dict):
                            print(f"    {name}:")
                            for k, v in value.items():
                                print(f"      {k}: {v}")
                        else:
                            print(f"    {name}: {value}")
            
            # Save results to file if requested
            if output_file:
                output_data = {
                    "deal_name": result.deal_name,
                    "execution_time": result.execution_time,
                    "status": result.status,
                    "pool_statistics": result.pool_statistics,
                    "metrics": result.metrics,
                    "bond_cashflows": result.bond_cashflows,
                    "pool_cashflows": result.pool_cashflows
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                print(f"\nResults saved to {output_file}")
            
            # Generate plots if requested
            if plot and not df_bond.empty and 'date' in df_bond.columns:
                # Convert date column
                df_bond['date'] = pd.to_datetime(df_bond['date'])
                
                # Create a directory for plots
                plots_dir = Path("absbox_plots")
                plots_dir.mkdir(exist_ok=True)
                
                # Plot bond cashflows
                plt.figure(figsize=(12, 6))
                
                # Get bond names
                bond_names = [bond.name for bond in deal_request.waterfall.bonds]
                
                # Plot each bond's cashflows
                for bond_name in bond_names:
                    if bond_name in df_bond.columns:
                        plt.plot(df_bond['date'], df_bond[bond_name], label=bond_name)
                
                plt.title(f"Bond Cashflows - {deal_request.deal_name}")
                plt.xlabel("Date")
                plt.ylabel("Cashflow Amount")
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                plot_file = plots_dir / f"{deal_request.deal_name.replace(' ', '_')}_bond_cashflows.png"
                plt.savefig(plot_file)
                print(f"Plot saved to {plot_file}")
                
                # Close the plot to free memory
                plt.close()
            
            return True
        else:
            print(f"Error analyzing deal: {result.error}")
            return False
    except Exception as e:
        print(f"Error analyzing deal: {str(e)}")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='AbsBox CLI for structured finance analysis')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check service health')
    
    # Create sample deal command
    create_parser = subparsers.add_parser('create-sample', help='Create a sample deal')
    create_parser.add_argument('--output', '-o', help='Output file to save the deal definition')
    
    # Analyze deal command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a structured deal')
    analyze_parser.add_argument('--input', '-i', help='Input file containing the deal definition')
    analyze_parser.add_argument('--output', '-o', help='Output file to save the results')
    analyze_parser.add_argument('--plot', '-p', action='store_true', help='Generate plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'health':
        health_check()
    elif args.command == 'create-sample':
        deal = create_sample_deal()
        if args.output:
            save_deal_to_file(deal, args.output)
        else:
            # Print the deal as JSON
            print(json.dumps(deal.dict(), indent=2, default=str))
    elif args.command == 'analyze':
        if args.input:
            deal = load_deal_from_file(args.input)
            if deal:
                analyze_deal(deal, args.output, args.plot)
        else:
            # If no input file, create and analyze a sample deal
            deal = create_sample_deal()
            analyze_deal(deal, args.output, args.plot)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
