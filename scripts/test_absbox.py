"""
Simple test script for validating the AbsBox integration
"""
import os
import sys
from pathlib import Path
from datetime import date, timedelta
import json
import pandas as pd
import time

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after environment variables are loaded
try:
    import absbox as ab
    from absbox.local.pool import Pool
    from absbox.local.loan import FixedRateLoan
    from absbox.local.deal import Deal
    from absbox.local.engine import LiqEngine
    from absbox.local.waterfall import Waterfall
    from absbox.local.assumption import Assumption, DefaultAssumption
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
        DefaultCurveConfig
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have installed the required dependencies with:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def create_test_deal():
    """Create a simple test deal structure"""
    print("Creating test deal structure...")
    
    today = date.today()
    start_date = today - timedelta(days=30)
    
    # Create a loan pool with two loans
    pool_config = LoanPoolConfig(
        pool_name="Test Pool",
        loans=[
            LoanConfig(
                balance=200000.0,
                rate=0.05,
                term=360,
                start_date=start_date,
                rate_type="fixed"
            ),
            LoanConfig(
                balance=100000.0,
                rate=0.06,
                term=240,
                start_date=start_date,
                rate_type="fixed"
            )
        ]
    )
    
    # Create a simple waterfall
    waterfall_config = WaterfallConfig(
        start_date=today,
        accounts=[
            AccountConfig(
                name="ReserveFund",
                initial_balance=5000.0
            )
        ],
        bonds=[
            BondConfig(
                name="ClassA",
                balance=250000.0,
                rate=0.04
            ),
            BondConfig(
                name="ClassB",
                balance=50000.0,
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
            vector=[0.01, 0.02, 0.02, 0.015, 0.01]
        )
    )
    
    # Create the deal request
    deal_request = StructuredDealRequest(
        deal_name="Test Deal",
        pool=pool_config,
        waterfall=waterfall_config,
        scenario=scenario_config
    )
    
    return deal_request

def test_absbox_direct():
    """Test AbsBox functionality directly"""
    print("\n=== Testing direct AbsBox functionality ===")
    
    try:
        print(f"AbsBox version: {ab.__version__}")
        
        # Create a simple loan
        loan = FixedRateLoan(
            balance=100000.0,
            rate=0.05,
            originTerm=360,
            remainTerm=360,
            period="Monthly",
            startDate=date.today() - timedelta(days=30)
        )
        
        # Create a pool with the loan
        pool = Pool(assets=[loan])
        
        # Run the pool
        engine = LiqEngine()
        result = engine.runPool(pool)
        
        # Print results
        cashflows = result.cashflow()
        print(f"Successfully generated {len(cashflows)} cashflow periods")
        print("First 3 cashflow periods:")
        print(cashflows.head(3))
        
        return True
    except Exception as e:
        print(f"Error testing direct AbsBox functionality: {e}")
        return False

def test_absbox_service():
    """Test the AbsBox service"""
    print("\n=== Testing AbsBox service integration ===")
    
    try:
        # Create AbsBox service
        service = AbsBoxService()
        print("AbsBox service initialized")
        
        # Check service health
        health = service.health_check()
        print("Health check results:")
        print(json.dumps(health, indent=2))
        
        # Create test deal
        deal_request = create_test_deal()
        
        # Analyze the deal
        print(f"Analyzing deal: {deal_request.deal_name}")
        start_time = time.time()
        result = service.analyze_deal(deal_request)
        elapsed = time.time() - start_time
        
        # Print results
        print(f"Deal analysis completed in {elapsed:.2f} seconds (reported: {result.execution_time:.2f}s)")
        print(f"Status: {result.status}")
        
        if result.status == "success":
            print(f"Bond cashflows: {len(result.bond_cashflows)} periods")
            print(f"Pool cashflows: {len(result.pool_cashflows)} periods")
            
            # Print first few bond cashflows
            df = pd.DataFrame(result.bond_cashflows)
            if not df.empty:
                print("\nSample bond cashflows:")
                print(df.head(3))
            
            # Print pool statistics
            print("\nPool statistics:")
            for key, value in result.pool_statistics.items():
                print(f"  {key}: {value}")
                
            return True
        else:
            print(f"Error analyzing deal: {result.error}")
            return False
    except Exception as e:
        print(f"Error testing AbsBox service: {e}")
        return False

def main():
    """Main test function"""
    print("=== AbsBox Integration Test ===")
    
    # Test direct AbsBox functionality
    direct_test_success = test_absbox_direct()
    
    if not direct_test_success:
        print("Direct AbsBox test failed. Please check your installation.")
        return
        
    # Test the AbsBox service
    service_test_success = test_absbox_service()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Direct AbsBox functionality: {'PASS' if direct_test_success else 'FAIL'}")
    print(f"AbsBox service integration: {'PASS' if service_test_success else 'FAIL'}")
    
    if direct_test_success and service_test_success:
        print("\nSuccess! The AbsBox integration is working correctly.")
    else:
        print("\nThere were issues with the AbsBox integration. Please check the logs.")

if __name__ == "__main__":
    main()
