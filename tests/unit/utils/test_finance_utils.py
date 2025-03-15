"""Unit tests for finance_utils module"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from app.utils.finance_utils import (
    calculate_loan_cashflows,
    calculate_npv_vectorized,
    calculate_irr_vectorized,
    calculate_amortization_schedule
)


class TestFinanceUtils(unittest.TestCase):
    """Test cases for vectorized financial calculation utilities"""

    def test_calculate_loan_cashflows_basic(self):
        """Test basic loan cashflow calculation for a single loan"""
        # Set up test data for a 30-year, $100,000 loan at 5%
        principals = np.array([100000.0])
        rates = np.array([0.05])
        terms = np.array([360])

        # Calculate cashflows
        principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
            principals, rates, terms
        )

        # Verify dimensions
        self.assertEqual(principal_payments.shape, (1, 360))
        self.assertEqual(interest_payments.shape, (1, 360))
        self.assertEqual(remaining_balances.shape, (1, 360))

        # Expected monthly payment
        expected_payment = 536.82  # Result from financial calculator
        first_payment = principal_payments[0, 0] + interest_payments[0, 0]
        self.assertAlmostEqual(first_payment, expected_payment, delta=0.1)

        # Check initial balance
        self.assertEqual(remaining_balances[0, 0], 100000.0)

        # Verify principal is paid off at the end
        self.assertLess(remaining_balances[0, -1], 1.0)  # Close to zero

        # Verify interest decreases over time
        self.assertGreater(interest_payments[0, 0], interest_payments[0, -1])
        
        # Verify principal increases over time
        self.assertLess(principal_payments[0, 0], principal_payments[0, -1])

    def test_calculate_loan_cashflows_multiple_loans(self):
        """Test vectorized calculation with multiple loans of different terms"""
        # Set up test data for 3 loans
        principals = np.array([100000.0, 250000.0, 500000.0])
        rates = np.array([0.05, 0.04, 0.035])
        terms = np.array([360, 240, 180])
        io_periods = np.array([0, 60, 24])
        balloon_payments = np.array([0, 0, 100000])

        # Calculate cashflows
        principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
            principals, rates, terms, io_periods, balloon_payments
        )

        # Verify dimensions (should be max term)
        self.assertEqual(principal_payments.shape, (3, 360))
        
        # Verify interest-only period for second loan
        for i in range(60):
            self.assertEqual(principal_payments[1, i], 0)
            self.assertGreater(interest_payments[1, i], 0)
            
        # Verify balloon payment for third loan
        self.assertAlmostEqual(principal_payments[2, 179], 100000.0)
        
        # Verify zero balance after term
        for i in range(3):
            term = terms[i]
            self.assertLess(remaining_balances[i, term-1], 1.0)  # Close to zero
            # Values after term should be zero
            if term < 360:
                self.assertEqual(principal_payments[i, term], 0)
                self.assertEqual(interest_payments[i, term], 0)

    def test_calculate_npv_vectorized(self):
        """Test NPV calculation with known values"""
        # Set up test data
        cashflows = np.array([-100000, 25000, 30000, 45000, 50000])
        dates = np.array([
            np.datetime64('2023-01-01'),
            np.datetime64('2023-07-01'),
            np.datetime64('2024-01-01'),
            np.datetime64('2024-07-01'),
            np.datetime64('2025-01-01')
        ])
        
        # Calculate NPV with 8% discount rate
        npv = calculate_npv_vectorized(cashflows, dates, 0.08)
        
        # Verify result (calculated separately)
        expected_npv = 33633.44
        self.assertAlmostEqual(npv, expected_npv, delta=1.0)
        
        # Test with base_date parameter
        base_date = np.datetime64('2023-01-01')
        npv2 = calculate_npv_vectorized(cashflows, dates, 0.08, base_date)
        self.assertAlmostEqual(npv, npv2)  # Should be the same

    def test_calculate_irr_vectorized(self):
        """Test IRR calculation with known values"""
        # Set up test data
        cashflows = np.array([-100000, 25000, 30000, 45000, 50000])
        dates = np.array([
            np.datetime64('2023-01-01'),
            np.datetime64('2023-07-01'),
            np.datetime64('2024-01-01'),
            np.datetime64('2024-07-01'),
            np.datetime64('2025-01-01')
        ])
        
        # Calculate IRR
        irr = calculate_irr_vectorized(cashflows, dates)
        
        # Verify result (approximate expected value)
        expected_irr = 0.22  # Approximately 22%
        self.assertAlmostEqual(irr, expected_irr, delta=0.02)
        
        # Test with invalid cashflows (all positive)
        with self.assertRaises(ValueError):
            calculate_irr_vectorized(np.array([1000, 500, 200]), dates[:3])

    def test_calculate_amortization_schedule(self):
        """Test amortization schedule generation for a single loan"""
        # Generate schedule for a 15-year, $200,000 loan at 4%
        schedule = calculate_amortization_schedule(
            principal=200000,
            rate=0.04,
            term=180,
            io_period=0,
            start_date='2023-01-01'
        )
        
        # Verify DataFrame properties
        self.assertIsInstance(schedule, pd.DataFrame)
        self.assertEqual(len(schedule), 180)
        self.assertTrue(all(col in schedule.columns for col in [
            'payment_number', 'payment_date', 'payment', 'principal', 'interest', 'remaining_balance'
        ]))
        
        # Verify calculated values
        self.assertAlmostEqual(schedule['payment'].iloc[0], 1479.38, delta=0.1)  # First payment
        self.assertAlmostEqual(schedule['interest'].iloc[0], 666.67, delta=0.1)  # First interest payment
        self.assertAlmostEqual(schedule['principal'].iloc[0], 812.71, delta=0.1)  # First principal payment
        
        # Verify balance
        self.assertEqual(schedule['remaining_balance'].iloc[0], 200000)
        self.assertLess(schedule['remaining_balance'].iloc[-1], 1.0)  # Close to zero at end
        
        # Test with interest-only period
        schedule_io = calculate_amortization_schedule(
            principal=200000,
            rate=0.04,
            term=180,
            io_period=60,
            start_date='2023-01-01'
        )
        
        # Verify interest-only period
        for i in range(60):
            self.assertEqual(schedule_io['principal'].iloc[i], 0)
            self.assertAlmostEqual(schedule_io['interest'].iloc[i], 666.67, delta=0.1)  # Monthly interest only

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty arrays
        principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
            np.array([]), np.array([]), np.array([])
        )
        self.assertEqual(principal_payments.shape, (0,))
        
        # Zero NPV for empty cashflows
        self.assertEqual(calculate_npv_vectorized(np.array([]), np.array([])), 0.0)
        
        # Very small interest rate
        principals = np.array([100000.0])
        rates = np.array([0.0000001])  # Nearly zero rate
        terms = np.array([12])
        principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
            principals, rates, terms
        )
        # Should be approximately equal payments
        expected_payment = 100000 / 12
        self.assertAlmostEqual(principal_payments[0, 0], expected_payment, delta=0.1)
        
        # Invalid inputs for amortization schedule
        with self.assertRaises(ValueError):
            calculate_amortization_schedule(principal=-100000, rate=0.05, term=360)
        with self.assertRaises(ValueError):
            calculate_amortization_schedule(principal=100000, rate=-0.05, term=360)
        with self.assertRaises(ValueError):
            calculate_amortization_schedule(principal=100000, rate=0.05, term=0)
        with self.assertRaises(ValueError):
            calculate_amortization_schedule(principal=100000, rate=0.05, term=360, io_period=360)


if __name__ == '__main__':
    unittest.main()
