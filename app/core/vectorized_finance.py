"""
Vectorized Financial Calculations

This module provides high-performance vectorized implementations of core financial
calculations using NumPy/Pandas. These functions are optimized for:
- Handling large datasets efficiently
- Minimizing memory usage
- Maximizing computation speed
- Supporting parallel processing
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging

from app.core.exceptions import CalculationError


logger = logging.getLogger(__name__)


@dataclass
class DiscountingParameters:
    """Parameters for NPV calculation and discounting"""
    discount_rate: float
    compounding_frequency: int = 12  # Monthly by default
    day_count_convention: str = 'actual/365'


@dataclass
class LoanParameters:
    """Parameters for loan cashflow generation"""
    principal: float
    rate: float  # Annual rate
    term: int  # In months
    io_period: int = 0  # Interest-only period in months
    balloon_payment: float = 0.0  # Optional balloon payment
    prepayment_rate: float = 0.0  # Optional annual prepayment rate
    default_rate: float = 0.0  # Optional annual default rate
    recovery_rate: float = 0.0  # Optional recovery rate on defaults (0-1)


def calculate_loan_cashflows(
    loans: Union[List[LoanParameters], np.ndarray, pd.DataFrame],
    economic_factors: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of loan cashflows
    
    Args:
        loans: List of loan parameters, NumPy array, or DataFrame of loans
        economic_factors: Optional economic factors to adjust calculations
            
    Returns:
        Tuple of (principal_payments, interest_payments, remaining_balances)
        Each is a 2D array with shape (num_loans, max_term)
        
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Convert input to numpy arrays
        if isinstance(loans, list):
            n_loans = len(loans)
            principals = np.array([loan.principal for loan in loans])
            rates = np.array([loan.rate for loan in loans])
            terms = np.array([loan.term for loan in loans])
            io_periods = np.array([loan.io_period for loan in loans])
            balloon_payments = np.array([loan.balloon_payment for loan in loans])
            prepayment_rates = np.array([loan.prepayment_rate for loan in loans])
            default_rates = np.array([loan.default_rate for loan in loans])
            recovery_rates = np.array([loan.recovery_rate for loan in loans])
        elif isinstance(loans, pd.DataFrame):
            n_loans = len(loans)
            principals = loans['principal'].values
            rates = loans['rate'].values
            terms = loans['term'].values
            io_periods = loans.get('io_period', np.zeros(n_loans)).values
            balloon_payments = loans.get('balloon_payment', np.zeros(n_loans)).values
            prepayment_rates = loans.get('prepayment_rate', np.zeros(n_loans)).values
            default_rates = loans.get('default_rate', np.zeros(n_loans)).values
            recovery_rates = loans.get('recovery_rate', np.zeros(n_loans)).values
        elif isinstance(loans, np.ndarray):
            raise NotImplementedError("Direct NumPy array input not yet supported")
        else:
            raise TypeError(f"Unsupported loan data type: {type(loans)}")
        
        # Apply economic factors if provided
        if economic_factors:
            # Adjust rates based on economic scenario
            if 'interest_rate_shock' in economic_factors:
                rates = rates + economic_factors['interest_rate_shock']
            
            # Adjust prepayment assumptions based on economic scenario
            if 'prepayment_multiplier' in economic_factors:
                prepayment_rates = prepayment_rates * economic_factors['prepayment_multiplier']
            
            # Adjust default assumptions based on economic scenario
            if 'default_multiplier' in economic_factors:
                default_rates = default_rates * economic_factors['default_multiplier']
            
            # Adjust recovery assumptions based on economic scenario
            if 'recovery_multiplier' in economic_factors:
                recovery_rates = recovery_rates * economic_factors['recovery_multiplier']
                # Ensure recovery rates are between 0 and 1
                recovery_rates = np.clip(recovery_rates, 0.0, 1.0)
        
        # Maximum term across all loans
        max_term = int(np.max(terms)) if len(terms) > 0 else 0
        
        if max_term == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Initialize output arrays
        principal_payments = np.zeros((n_loans, max_term))
        interest_payments = np.zeros((n_loans, max_term))
        remaining_balances = np.zeros((n_loans, max_term + 1))  # Include initial balance
        
        # Set initial balances
        remaining_balances[:, 0] = principals
        
        # Calculate monthly rates
        monthly_rates = rates / 12
        monthly_prepay_rates = prepayment_rates / 12
        monthly_default_rates = default_rates / 12
        
        # Calculate payment amounts (excluding balloon)
        monthly_payments = np.zeros(n_loans)
        for i in range(n_loans):
            # Skip if term is zero
            if terms[i] <= 0:
                continue
                
            # Calculate amortizing payment for the portion after IO period
            amort_term = terms[i] - io_periods[i]
            if amort_term <= 0:
                # Interest-only loan
                monthly_payments[i] = principals[i] * monthly_rates[i]
            else:
                # Regular amortizing loan
                if monthly_rates[i] > 1e-10:  # Avoid division by zero
                    monthly_payments[i] = principals[i] * monthly_rates[i] / (1 - (1 + monthly_rates[i]) ** -amort_term)
                else:
                    # Zero interest rate (rare)
                    monthly_payments[i] = principals[i] / amort_term
        
        # Calculate cashflows for each time period
        for t in range(max_term):
            # Calculate interest based on current balance
            current_interest = remaining_balances[:, t] * monthly_rates
            
            # Calculate defaults for this period
            defaults = remaining_balances[:, t] * monthly_default_rates
            
            # Calculate prepayments for this period (applies to non-defaulted balance)
            non_defaulted_balance = remaining_balances[:, t] - defaults
            prepayments = non_defaulted_balance * monthly_prepay_rates
            
            # Calculate recoveries on defaults
            recoveries = defaults * recovery_rates
            
            # Calculate scheduled principal for loans in amortization period
            scheduled_principal = np.zeros(n_loans)
            for i in range(n_loans):
                if t >= io_periods[i] and t < terms[i]:  # In amortization period
                    # Regular payment minus interest
                    scheduled_principal[i] = min(
                        monthly_payments[i] - current_interest[i],
                        remaining_balances[i, t] - defaults[i] - prepayments[i]
                    )
                    
                    # Handle balloon payment at maturity
                    if t == terms[i] - 1 and balloon_payments[i] > 0:
                        scheduled_principal[i] = balloon_payments[i]
            
            # Total principal reduction (scheduled + prepayments + defaults net of recoveries)
            total_principal = scheduled_principal + prepayments + defaults - recoveries
            
            # Ensure principal payments don't exceed remaining balance
            total_principal = np.minimum(total_principal, remaining_balances[:, t])
            
            # Store payments
            interest_payments[:, t] = current_interest
            principal_payments[:, t] = total_principal
            
            # Update balances
            remaining_balances[:, t+1] = remaining_balances[:, t] - total_principal
        
        # Remove the initial balance column
        remaining_balances = remaining_balances[:, 1:]
        
        return principal_payments, interest_payments, remaining_balances
        
    except Exception as e:
        # Log detailed error and raise
        logger.error(
            f"Error in calculate_loan_cashflows: {str(e)}",
            exc_info=True,
            extra={
                "num_loans": len(loans) if loans else 0,
                "has_economic_factors": economic_factors is not None
            }
        )
        raise CalculationError(
            f"Failed to calculate loan cashflows: {str(e)}",
            context={
                "num_loans": len(loans) if loans else 0,
                "has_economic_factors": economic_factors is not None
            },
            cause=e
        )


def calculate_npv(
    cashflows: np.ndarray,
    discount_params: DiscountingParameters,
    time_points: Optional[np.ndarray] = None,
    economic_factors: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Calculate NPV for multiple cashflow streams
    
    Args:
        cashflows: Array of shape (n_streams, n_periods) or (n_periods,)
        discount_params: Discounting parameters
        time_points: Optional array of time points in years, if None assumes monthly
        economic_factors: Optional economic factors to adjust calculations
            
    Returns:
        Array of NPVs, shape (n_streams,) or scalar
        
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Handle single cashflow stream case
        single_stream = len(cashflows.shape) == 1
        if single_stream:
            cashflows = cashflows.reshape(1, -1)
        
        n_streams, n_periods = cashflows.shape
        
        # Apply economic factors if provided
        discount_rate = discount_params.discount_rate
        if economic_factors and 'discount_rate_adjustment' in economic_factors:
            discount_rate += economic_factors['discount_rate_adjustment']
        
        # Ensure positive discount rate (negative makes no financial sense)
        discount_rate = max(0.0001, discount_rate)
        
        # Create time points if not provided
        if time_points is None:
            # Assume monthly cashflows
            time_points = np.arange(n_periods) / 12
        
        # Calculate discount factors
        if discount_params.compounding_frequency == 0:
            # Continuous compounding
            discount_factors = np.exp(-discount_rate * time_points)
        else:
            # Discrete compounding
            periods_per_year = discount_params.compounding_frequency
            discount_factors = (1 + discount_rate / periods_per_year) ** (-time_points * periods_per_year)
        
        # Apply discount factors
        pv = cashflows * discount_factors.reshape(1, -1)
        
        # Sum across periods
        npv = np.sum(pv, axis=1)
        
        # Return result in original shape
        return npv[0] if single_stream else npv
        
    except Exception as e:
        logger.error(
            f"Error in calculate_npv: {str(e)}",
            exc_info=True,
            extra={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
                "has_economic_factors": economic_factors is not None
            }
        )
        raise CalculationError(
            f"Failed to calculate NPV: {str(e)}",
            context={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
                "has_economic_factors": economic_factors is not None
            },
            cause=e
        )


def calculate_irr(
    cashflows: np.ndarray,
    initial_guess: float = 0.05,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    """Calculate IRR for multiple cashflow streams
    
    Args:
        cashflows: Array of shape (n_streams, n_periods) or (n_periods,)
                  First cashflow is assumed to be negative (investment)
        initial_guess: Initial IRR guess
        max_iterations: Maximum iterations for Newton's method
        tolerance: Convergence tolerance
            
    Returns:
        Array of IRRs, shape (n_streams,) or scalar
        
    Raises:
        CalculationError: If calculation fails to converge
    """
    try:
        # Handle single cashflow stream case
        single_stream = len(cashflows.shape) == 1
        if single_stream:
            cashflows = cashflows.reshape(1, -1)
        
        n_streams, n_periods = cashflows.shape
        
        # Time points (0, 1, 2, ..., n_periods-1)
        time_points = np.arange(n_periods)
        
        # Initialize IRR array
        irrs = np.full(n_streams, initial_guess)
        
        # Initialize convergence tracking
        converged = np.zeros(n_streams, dtype=bool)
        iterations = np.zeros(n_streams, dtype=int)
        
        # Newton-Raphson method for each stream
        while not np.all(converged) and np.max(iterations) < max_iterations:
            # Only process non-converged streams
            active = ~converged
            
            if not np.any(active):
                break
            
            # Calculate discount factors for active streams
            # Shape: (n_active_streams, n_periods)
            discount_factors = (1 + irrs[active].reshape(-1, 1)) ** (-time_points.reshape(1, -1))
            
            # Calculate NPV and its derivative for current IRR guess
            npv = np.sum(cashflows[active] * discount_factors, axis=1)
            npv_prime = np.sum(
                -time_points.reshape(1, -1) * cashflows[active] *
                discount_factors / (1 + irrs[active].reshape(-1, 1)),
                axis=1
            )
            
            # Update IRR using Newton's method
            # Limit step size to prevent divergence
            step = np.clip(npv / npv_prime, -0.5, 0.5)
            irrs[active] = irrs[active] - step
            
            # Check convergence
            new_converged = np.abs(npv) < tolerance
            converged[active] = new_converged
            
            # Update iteration count
            iterations[active] += 1
        
        # Check for non-convergence
        non_converged = ~converged
        if np.any(non_converged):
            logger.warning(
                f"{np.sum(non_converged)} cashflow streams did not converge to IRR"
            )
            
            # Use simple binary search for non-converged streams as fallback
            active = non_converged
            high = np.full(n_streams, 1.0)[active]  # 100% upper bound
            low = np.full(n_streams, -0.99)[active]  # -99% lower bound
            
            for _ in range(50):  # 50 binary search iterations
                # Mid-point
                mid = (high + low) / 2
                
                # Calculate NPV at mid-point
                discount_factors = (1 + mid.reshape(-1, 1)) ** (-time_points.reshape(1, -1))
                npv = np.sum(cashflows[active] * discount_factors, axis=1)
                
                # Update bounds
                high_mask = npv < 0
                low_mask = ~high_mask
                
                high[high_mask] = mid[high_mask]
                low[low_mask] = mid[low_mask]
            
            # Update IRRs with binary search results
            irrs[active] = (high + low) / 2
        
        # Return result in original shape
        return irrs[0] if single_stream else irrs
        
    except Exception as e:
        logger.error(
            f"Error in calculate_irr: {str(e)}",
            exc_info=True,
            extra={
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
            }
        )
        raise CalculationError(
            f"Failed to calculate IRR: {str(e)}",
            context={
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
            },
            cause=e
        )


def calculate_duration(
    cashflows: np.ndarray,
    discount_params: DiscountingParameters,
    time_points: Optional[np.ndarray] = None,
    modified: bool = True
) -> np.ndarray:
    """Calculate Macaulay or modified duration for multiple cashflow streams
    
    Args:
        cashflows: Array of shape (n_streams, n_periods) or (n_periods,)
        discount_params: Discounting parameters
        time_points: Optional array of time points in years, if None assumes monthly
        modified: If True, return modified duration; otherwise, return Macaulay duration
            
    Returns:
        Array of durations, shape (n_streams,) or scalar
        
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Handle single cashflow stream case
        single_stream = len(cashflows.shape) == 1
        if single_stream:
            cashflows = cashflows.reshape(1, -1)
        
        n_streams, n_periods = cashflows.shape
        
        # Create time points if not provided
        if time_points is None:
            # Assume monthly cashflows
            time_points = np.arange(n_periods) / 12
        
        # Calculate discount factors
        if discount_params.compounding_frequency == 0:
            # Continuous compounding
            discount_factors = np.exp(-discount_params.discount_rate * time_points)
        else:
            # Discrete compounding
            periods_per_year = discount_params.compounding_frequency
            discount_factors = (1 + discount_params.discount_rate / periods_per_year) ** (-time_points * periods_per_year)
        
        # Calculate present values
        pv = cashflows * discount_factors.reshape(1, -1)
        
        # Calculate NPV for each stream
        npv = np.sum(pv, axis=1)
        
        # Calculate weighted time
        weighted_time = np.sum(
            time_points.reshape(1, -1) * pv,
            axis=1
        )
        
        # Calculate Macaulay duration
        macaulay_duration = weighted_time / npv
        
        # Calculate modified duration if requested
        if modified:
            if discount_params.compounding_frequency == 0:
                # Continuous compounding
                duration = macaulay_duration
            else:
                # Discrete compounding
                duration = macaulay_duration / (1 + discount_params.discount_rate / discount_params.compounding_frequency)
        else:
            duration = macaulay_duration
        
        # Return result in original shape
        return duration[0] if single_stream else duration
        
    except Exception as e:
        logger.error(
            f"Error in calculate_duration: {str(e)}",
            exc_info=True,
            extra={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
                "modified": modified
            }
        )
        raise CalculationError(
            f"Failed to calculate duration: {str(e)}",
            context={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
                "modified": modified
            },
            cause=e
        )


def calculate_convexity(
    cashflows: np.ndarray,
    discount_params: DiscountingParameters,
    time_points: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate convexity for multiple cashflow streams
    
    Args:
        cashflows: Array of shape (n_streams, n_periods) or (n_periods,)
        discount_params: Discounting parameters
        time_points: Optional array of time points in years, if None assumes monthly
            
    Returns:
        Array of convexities, shape (n_streams,) or scalar
        
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Handle single cashflow stream case
        single_stream = len(cashflows.shape) == 1
        if single_stream:
            cashflows = cashflows.reshape(1, -1)
        
        n_streams, n_periods = cashflows.shape
        
        # Create time points if not provided
        if time_points is None:
            # Assume monthly cashflows
            time_points = np.arange(n_periods) / 12
        
        # Calculate discount factors
        if discount_params.compounding_frequency == 0:
            # Continuous compounding
            discount_factors = np.exp(-discount_params.discount_rate * time_points)
        else:
            # Discrete compounding
            periods_per_year = discount_params.compounding_frequency
            discount_factors = (1 + discount_params.discount_rate / periods_per_year) ** (-time_points * periods_per_year)
        
        # Calculate present values
        pv = cashflows * discount_factors.reshape(1, -1)
        
        # Calculate NPV for each stream
        npv = np.sum(pv, axis=1)
        
        # Calculate weighted squared time
        weighted_time_squared = np.sum(
            (time_points.reshape(1, -1) ** 2) * pv,
            axis=1
        )
        
        # Calculate convexity
        if discount_params.compounding_frequency == 0:
            # Continuous compounding
            convexity = weighted_time_squared / npv
        else:
            # Discrete compounding
            r = discount_params.discount_rate
            m = discount_params.compounding_frequency
            convexity = weighted_time_squared / npv / (1 + r/m) ** 2
        
        # Return result in original shape
        return convexity[0] if single_stream else convexity
        
    except Exception as e:
        logger.error(
            f"Error in calculate_convexity: {str(e)}",
            exc_info=True,
            extra={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
            }
        )
        raise CalculationError(
            f"Failed to calculate convexity: {str(e)}",
            context={
                "discount_rate": discount_params.discount_rate,
                "n_periods": cashflows.shape[1] if len(cashflows.shape) > 1 else len(cashflows),
            },
            cause=e
        )


def calculate_risk_metrics(
    cashflows: np.ndarray,
    discount_params: DiscountingParameters,
    confidence_levels: List[float] = [0.95, 0.99],
    time_points: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate risk metrics for a set of cashflow scenarios
    
    Args:
        cashflows: Array of shape (n_scenarios, n_periods)
        discount_params: Discounting parameters
        confidence_levels: List of confidence levels for VaR and ES
        time_points: Optional array of time points in years, if None assumes monthly
            
    Returns:
        Dictionary of risk metrics
        
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Calculate NPV for all scenarios
        npvs = calculate_npv(
            cashflows, 
            discount_params, 
            time_points=time_points
        )
        
        # Initialize results
        metrics = {
            'mean': float(np.mean(npvs)),
            'median': float(np.median(npvs)),
            'std_dev': float(np.std(npvs)),
            'min': float(np.min(npvs)),
            'max': float(np.max(npvs)),
        }
        
        # Calculate VaR and Expected Shortfall (ES) for each confidence level
        for confidence in confidence_levels:
            # Calculate percentile for VaR
            var_percentile = 100 * (1 - confidence)
            var_value = float(np.percentile(npvs, var_percentile))
            metrics[f'var_{int(confidence*100)}'] = var_value
            
            # Calculate Expected Shortfall (ES)
            es_value = float(npvs[npvs <= var_value].mean())
            metrics[f'es_{int(confidence*100)}'] = es_value
        
        return metrics
        
    except Exception as e:
        logger.error(
            f"Error in calculate_risk_metrics: {str(e)}",
            exc_info=True,
            extra={
                "discount_rate": discount_params.discount_rate,
                "n_scenarios": cashflows.shape[0],
                "n_periods": cashflows.shape[1],
            }
        )
        raise CalculationError(
            f"Failed to calculate risk metrics: {str(e)}",
            context={
                "discount_rate": discount_params.discount_rate,
                "n_scenarios": cashflows.shape[0],
                "n_periods": cashflows.shape[1],
            },
            cause=e
        )
