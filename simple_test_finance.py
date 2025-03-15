"""
Simplified test for finance utilities
------------------------------------
This minimalist test focuses on just importing and running a basic test
to identify any environment or implementation issues.
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_test():
    """Execute a simple test of the financial utilities"""
    
    print("=" * 70)
    print("TESTING FINANCE UTILITIES - SIMPLIFIED TEST")
    print("=" * 70)
    
    try:
        # Import the functions for testing
        from app.utils.finance_utils import (
            calculate_loan_cashflows,
            calculate_npv_vectorized,
            calculate_irr_vectorized,
            calculate_amortization_schedule
        )
        
        print("✓ Successfully imported finance_utils module")
        
        # Create test data
        principals = np.array([100000.0])
        rates = np.array([0.05])
        terms = np.array([360])
        
        # Test loan cashflows
        print("\nTesting calculate_loan_cashflows...")
        principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
            principals, rates, terms
        )
        
        print(f"Output shapes: Principal {principal_payments.shape}, Interest {interest_payments.shape}, Balances {remaining_balances.shape}")
        print(f"First principal payment: ${principal_payments[0, 0]:.2f}")
        print(f"First interest payment: ${interest_payments[0, 0]:.2f}")
        print(f"First balance: ${remaining_balances[0, 0]:.2f}")
        print(f"Total first payment: ${principal_payments[0, 0] + interest_payments[0, 0]:.2f}")
        print(f"Last balance: ${remaining_balances[0, -1]:.2f}")
        
        # Test NPV
        print("\nTesting calculate_npv_vectorized...")
        cashflows = np.array([-100000, 25000, 30000, 45000, 50000])
        dates = np.array([
            np.datetime64('2023-01-01'),
            np.datetime64('2023-07-01'),
            np.datetime64('2024-01-01'),
            np.datetime64('2024-07-01'),
            np.datetime64('2025-01-01')
        ])
        
        npv = calculate_npv_vectorized(cashflows, dates, 0.08)
        print(f"NPV at 8%: ${npv:.2f}")
        
        # Test IRR
        print("\nTesting calculate_irr_vectorized...")
        irr = calculate_irr_vectorized(cashflows, dates)
        print(f"IRR: {irr*100:.2f}%")
        
        # Test amortization schedule
        print("\nTesting calculate_amortization_schedule...")
        schedule = calculate_amortization_schedule(
            principal=200000,
            rate=0.04,
            term=180,
            io_period=0,
            start_date='2023-01-01'
        )
        
        print(f"Schedule shape: {schedule.shape}")
        print(f"First payment: ${schedule['payment'].iloc[0]:.2f}")
        print(f"First interest: ${schedule['interest'].iloc[0]:.2f}")
        print(f"First principal: ${schedule['principal'].iloc[0]:.2f}")
        print(f"First balance: ${schedule['remaining_balance'].iloc[0]:.2f}")
        
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
