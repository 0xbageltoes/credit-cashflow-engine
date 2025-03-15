"""
Vectorized financial calculation utilities for credit cashflow engine.

This module provides highly optimized, vectorized implementations of common financial
calculations such as loan cashflows, NPV (Net Present Value), and IRR (Internal Rate of Return).
These functions are designed to handle large datasets efficiently by leveraging NumPy vectorization
instead of using loops.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime, timedelta


def calculate_loan_cashflows(
    principals: np.ndarray,
    rates: np.ndarray,
    terms: np.ndarray,
    io_periods: Optional[np.ndarray] = None,
    balloon_payments: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of loan cashflows
    
    Efficiently computes principal payments, interest payments, and remaining balances
    for multiple loans simultaneously using NumPy vectorization.
    
    Args:
        principals: Array of loan principals
        rates: Array of interest rates (annual)
        terms: Array of loan terms in months
        io_periods: Optional array of interest-only periods
        balloon_payments: Optional array of balloon payments
        
    Returns:
        Tuple of (principal_payments, interest_payments, remaining_balances)
        Each is a 2D array with shape (num_loans, max_term)
        
    Example:
        ```python
        principals = np.array([100000, 250000, 500000])
        rates = np.array([0.05, 0.04, 0.035])  # 5%, 4%, 3.5%
        terms = np.array([360, 240, 180])  # 30 year, 20 year, 15 year
        io_periods = np.array([0, 60, 24])  # No IO, 5 year IO, 2 year IO
        principal, interest, balances = calculate_loan_cashflows(
            principals, rates, terms, io_periods)
        ```
    """
    # Number of loans and maximum term
    n_loans = len(principals)
    max_term = int(np.max(terms)) if len(terms) > 0 else 0
    
    if max_term == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Default values for optional parameters
    if io_periods is None:
        io_periods = np.zeros(n_loans, dtype=int)
    
    if balloon_payments is None:
        balloon_payments = np.zeros(n_loans)
    
    # Initialize output arrays
    principal_payments = np.zeros((n_loans, max_term))
    interest_payments = np.zeros((n_loans, max_term))
    remaining_balances = np.zeros((n_loans, max_term))
    
    # Calculate monthly rates
    monthly_rates = rates / 12
    
    # Vectorized calculation of monthly payment amounts
    # Avoid division by zero with np.where
    non_zero_rate_mask = monthly_rates > 1e-10
    
    # Prepare the terms for vectorized operations
    terms_for_calc = terms.copy()
    
    # Calculate monthly payments using vectorized operations
    # For non-zero rates: P * r / (1 - (1 + r)^-n)
    # For zero/small rates: P / n
    monthly_payments = np.zeros(n_loans)
    monthly_payments[non_zero_rate_mask] = principals[non_zero_rate_mask] * monthly_rates[non_zero_rate_mask] / (
        1 - (1 + monthly_rates[non_zero_rate_mask]) ** -terms_for_calc[non_zero_rate_mask]
    )
    monthly_payments[~non_zero_rate_mask] = principals[~non_zero_rate_mask] / terms_for_calc[~non_zero_rate_mask]
    
    # Vectorized calculation of amortization schedules
    # We still need to loop over the time dimension, but we can vectorize across loans
    balances = principals.copy()
    
    for t in range(max_term):
        # Store current balances for all loans at this time point
        mask = t < terms
        remaining_balances[mask, t] = balances[mask]
        
        # Calculate interest payments for all loans
        current_interest = balances * monthly_rates
        interest_payments[mask, t] = current_interest[mask]
        
        # Determine which loans are in IO period
        io_mask = (t < io_periods) & mask
        amort_mask = (~io_mask) & mask
        
        # Handle interest-only periods
        principal_payments[io_mask, t] = 0
        
        # Handle final payments with balloons
        final_payment_mask = (t == terms - 1) & (balloon_payments > 0) & mask
        # For balloon payments, pay remaining balance plus balloon amount
        principal_payments[final_payment_mask, t] = balances[final_payment_mask] + balloon_payments[final_payment_mask]
        
        # Handle regular amortizing payments (not IO, not balloon)
        regular_amort_mask = amort_mask & (~final_payment_mask)
        
        # For regular payments that aren't the final payment
        non_final_regular_mask = regular_amort_mask & (t < terms - 1)
        principal_payments[non_final_regular_mask, t] = np.maximum(
            0, 
            monthly_payments[non_final_regular_mask] - current_interest[non_final_regular_mask]
        )
        
        # Handle final regular payment (not balloon)
        final_regular_mask = (t == terms - 1) & (~final_payment_mask) & mask
        # For final payment, pay remaining balance (ensure full payoff)
        principal_payments[final_regular_mask, t] = balances[final_regular_mask]
        
        # Update balances
        balances = np.maximum(0, balances - principal_payments[:, t])
    
    # Explicitly set remaining balances to zero at the end of each loan's term
    for i in range(n_loans):
        if terms[i] > 0 and terms[i] <= max_term:
            remaining_balances[i, terms[i]-1:] = 0.0
        
    return principal_payments, interest_payments, remaining_balances


def calculate_npv_vectorized(
    cashflows: np.ndarray,
    dates: np.ndarray,
    discount_rate: float,
    base_date: np.datetime64 = None
) -> float:
    """Vectorized calculation of Net Present Value
    
    Efficiently computes the NPV of a series of cashflows using NumPy vectorization.
    
    Args:
        cashflows: Array of cashflow amounts
        dates: Array of cashflow dates (as np.datetime64)
        discount_rate: Annual discount rate
        base_date: Optional base date (default: earliest date)
        
    Returns:
        NPV value
        
    Example:
        ```python
        cashflows = np.array([-100000, 25000, 30000, 45000, 50000])
        dates = np.array([
            np.datetime64('2023-01-01'),
            np.datetime64('2023-07-01'),
            np.datetime64('2024-01-01'),
            np.datetime64('2024-07-01'),
            np.datetime64('2025-01-01')
        ])
        npv = calculate_npv_vectorized(cashflows, dates, 0.08)
        ```
    """
    # Handle empty cashflows
    if len(cashflows) == 0:
        return 0.0
    
    # Input validation
    if not isinstance(cashflows, np.ndarray):
        cashflows = np.array(cashflows, dtype=float)
    
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates, dtype='datetime64')
    
    if len(cashflows) != len(dates):
        raise ValueError("Cashflows and dates arrays must have the same length")
    
    # Use earliest date as base date if not provided
    if base_date is None:
        base_date = np.min(dates)
    elif not isinstance(base_date, np.datetime64):
        base_date = np.datetime64(base_date)
    
    # Calculate time differences in years
    time_diffs = (dates - base_date).astype('timedelta64[D]').astype(float) / 365.25
    
    # Calculate discount factors efficiently
    discount_factors = 1.0 / np.power(1.0 + discount_rate, time_diffs)
    
    # Calculate NPV
    return float(np.sum(cashflows * discount_factors))


def calculate_irr_vectorized(
    cashflows: np.ndarray,
    dates: np.ndarray,
    initial_guess: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> float:
    """Vectorized calculation of IRR using Newton's method
    
    Efficiently computes the Internal Rate of Return for a series of cashflows
    using Newton's method with NumPy vectorization.
    
    Args:
        cashflows: Array of cashflow amounts
        dates: Array of cashflow dates (as np.datetime64)
        initial_guess: Initial guess for IRR
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        IRR value
        
    Example:
        ```python
        cashflows = np.array([-100000, 25000, 30000, 45000, 50000])
        dates = np.array([
            np.datetime64('2023-01-01'),
            np.datetime64('2023-07-01'),
            np.datetime64('2024-01-01'),
            np.datetime64('2024-07-01'),
            np.datetime64('2025-01-01')
        ])
        irr = calculate_irr_vectorized(cashflows, dates)
        ```
    """
    # Handle empty cashflows
    if len(cashflows) == 0:
        return 0.0
    
    # Input validation
    if not isinstance(cashflows, np.ndarray):
        cashflows = np.array(cashflows, dtype=float)
    
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates, dtype='datetime64')
    
    if len(cashflows) != len(dates):
        raise ValueError("Cashflows and dates arrays must have the same length")
    
    # Check if valid IRR calculation is possible (need at least one positive and one negative cashflow)
    if np.all(cashflows >= 0) or np.all(cashflows <= 0):
        raise ValueError("IRR calculation requires at least one positive and one negative cashflow")
    
    # Convert dates to years from earliest date
    base_date = np.min(dates)
    time_diffs = (dates - base_date).astype('timedelta64[D]').astype(float) / 365.25
    
    # Newton's method to find IRR
    rate = initial_guess
    
    for _ in range(max_iterations):
        # Calculate NPV at current rate
        discount_factors = 1.0 / np.power(1.0 + rate, time_diffs)
        npv = np.sum(cashflows * discount_factors)
        
        # If NPV is close enough to zero, we've found the IRR
        if abs(npv) < tolerance:
            break
        
        # Calculate derivative of NPV with respect to rate
        deriv = np.sum(-time_diffs * cashflows * discount_factors / (1.0 + rate))
        
        # Update rate using Newton's method
        if abs(deriv) > tolerance:
            delta = npv / deriv
            rate -= delta
            
            # Add dampening for stability in case of large jumps
            if abs(delta) > 0.5:
                rate = rate + 0.5 * np.sign(delta)
                
            # Handle negative rates with a floor
            if rate < -0.99:
                rate = -0.99
                
            # Check for convergence
            if abs(delta) < tolerance:
                break
        else:
            # If derivative is too small, use bisection or another method
            rate = max(0.0001, rate * 0.9)
    
    return float(rate)


def calculate_amortization_schedule(
    principal: float,
    rate: float,
    term: int,
    io_period: int = 0,
    balloon_payment: float = 0.0,
    start_date: Union[str, datetime, np.datetime64] = None
) -> pd.DataFrame:
    """Calculate complete amortization schedule for a single loan
    
    Creates a pandas DataFrame with the complete amortization schedule including all payment details.
    
    Args:
        principal: Loan principal amount
        rate: Annual interest rate (e.g., 0.05 for 5%)
        term: Loan term in months
        io_period: Interest-only period in months
        balloon_payment: Optional balloon payment amount
        start_date: Loan start date (optional)
        
    Returns:
        DataFrame with complete amortization schedule
        
    Example:
        ```python
        schedule = calculate_amortization_schedule(
            principal=250000,
            rate=0.04,
            term=360,
            io_period=60,
            start_date='2023-01-01'
        )
        ```
    """
    # Handle date conversion
    if start_date is None:
        start_date = datetime.now()
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    elif isinstance(start_date, np.datetime64):
        start_date = pd.Timestamp(start_date).to_pydatetime()
    
    # Input validation
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if rate < 0:
        raise ValueError("Interest rate cannot be negative")
    if term <= 0:
        raise ValueError("Term must be positive")
    if io_period < 0 or io_period >= term:
        raise ValueError("Interest-only period must be between 0 and term-1")
    if balloon_payment < 0:
        raise ValueError("Balloon payment cannot be negative")
    
    # Set up arrays for a single loan
    principals = np.array([principal])
    rates = np.array([rate])
    terms = np.array([term])
    io_periods = np.array([io_period])
    balloon_payments = np.array([balloon_payment])
    
    # Calculate cashflows
    principal_payments, interest_payments, remaining_balances = calculate_loan_cashflows(
        principals, rates, terms, io_periods, balloon_payments
    )
    
    # Extract data for the single loan
    principal_payment = principal_payments[0, :term]
    interest_payment = interest_payments[0, :term]
    balance = remaining_balances[0, :term]
    
    # Generate payment dates
    payment_dates = []
    current_date = start_date
    for _ in range(term):
        # Move to next month
        month = current_date.month + 1
        year = current_date.year + (month > 12)
        month = (month - 1) % 12 + 1
        
        # Create new date with same day (or last day of month if original day is invalid)
        try:
            payment_date = current_date.replace(year=year, month=month)
        except ValueError:
            # Handle edge case for month with fewer days (e.g., Feb 30 -> Feb 28/29)
            last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
            payment_date = last_day.to_pydatetime()
        
        payment_dates.append(payment_date)
        current_date = payment_date
    
    # Calculate total payments
    total_payment = principal_payment + interest_payment
    
    # Ensure the last balance is exactly zero for clean reporting
    if len(balance) > 0:
        balance[-1] = 0.0
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'payment_number': np.arange(1, term + 1),
        'payment_date': payment_dates,
        'payment': total_payment,
        'principal': principal_payment,
        'interest': interest_payment,
        'remaining_balance': balance
    })
    
    return df
