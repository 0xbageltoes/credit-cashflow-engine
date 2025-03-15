"""
Unified AbsBox Service

This module provides a comprehensive service for AbsBox financial calculations,
consolidating functionality from previous implementations with improved:
- Error handling
- Caching integration
- Metrics collection
- Economic factor adjustments
- Consistent sync/async patterns
"""

import logging
import os
import json
import traceback
from typing import Optional, Dict, List, Any, Union, Callable
from datetime import datetime
import asyncio
import functools
import contextlib
import time

# Core app imports
from app.core.cache_service import CacheService
from app.core.monitoring import PrometheusMetrics, CalculationTracker
from app.core.error_tracking import log_exception

# Import AbsBox dynamically to handle unavailable scenarios
try:
    import absbox
    ABSBOX_AVAILABLE = True
except ImportError:
    ABSBOX_AVAILABLE = False

# Custom exceptions for better error handling
class CalculationError(Exception):
    """Exception raised for calculation errors"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        self.message = message
        self.context = context or {}
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        return f"{self.message} (Context: {self.context})"


class ConfigurationError(Exception):
    """Exception raised for configuration errors"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class NullMetricsService:
    """Null object implementation of metrics service"""
    
    def track_timing(self, name: str) -> contextlib.AbstractContextManager:
        """Track timing for an operation"""
        
        @contextlib.contextmanager
        def _null_context():
            yield
            
        return _null_context()
    
    def record_calculation_time(self, calculation_type: str, seconds: float) -> None:
        """Record calculation time"""
        pass
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter"""
        pass
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge value"""
        pass


def handle_errors(logger=None, default_error=Exception):
    """Decorator for handling errors in AbsBox service methods"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                # Get calling context for better error reporting
                method_name = func.__name__
                context = {
                    "method": method_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                
                # Log the error with context
                error_logger = logger or self.logger
                error_logger.error(
                    f"Error in {method_name}: {str(e)}",
                    extra={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "context": context,
                        "traceback": traceback.format_exc()
                    }
                )
                
                # Log to error tracking system
                log_exception(e, context=context)
                
                # Re-raise as a calculation error with context
                if isinstance(e, default_error):
                    raise
                else:
                    raise default_error(f"Error in {method_name}: {str(e)}", context=context, cause=e)
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get calling context for better error reporting
                method_name = func.__name__
                context = {
                    "method": method_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                
                # Log the error with context
                error_logger = logger or self.logger
                error_logger.error(
                    f"Error in {method_name}: {str(e)}",
                    extra={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "context": context,
                        "traceback": traceback.format_exc()
                    }
                )
                
                # Log to error tracking system
                log_exception(e, context=context)
                
                # Re-raise as a calculation error with context
                if isinstance(e, default_error):
                    raise
                else:
                    raise default_error(f"Error in {method_name}: {str(e)}", context=context, cause=e)
        
        # Return appropriate wrapper based on function signature
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class AbsBoxService:
    """Unified service for AbsBox financial calculations"""
    
    def __init__(
        self,
        cache_service: CacheService,
        metrics_service: Optional[Any] = None,
        logging_service: Optional[Any] = None,
        absbox_url: Optional[str] = None
    ):
        """Initialize with proper dependencies
        
        Args:
            cache_service: Service for caching results
            metrics_service: Optional service for metrics collection
            logging_service: Optional service for logging
            absbox_url: Optional URL for remote AbsBox instance
        """
        self.cache = cache_service
        self.metrics = metrics_service or NullMetricsService()
        
        # Configure logger
        if logging_service:
            self.logger = logging_service.get_logger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            
        # Store configuration
        self.absbox_url = absbox_url or os.environ.get("ABSBOX_URL")
        
        # Lazy-loaded client
        self._client = None
        
        # Track service state
        self._initialized = False
        self._last_error = None
        self._error_count = 0
        
        self.logger.info("AbsBoxService initialized")
    
    @property
    def client(self):
        """Lazy-loading property for AbsBox client"""
        if self._client is None:
            self._initialize_absbox_client()
        return self._client
    
    def _initialize_absbox_client(self):
        """Initialize AbsBox client (local or remote)"""
        try:
            # Check if AbsBox is available
            if not ABSBOX_AVAILABLE:
                self.logger.error("AbsBox library is not available")
                raise ImportError("AbsBox library is required but not installed")
            
            # Initialize client based on URL
            if self.absbox_url:
                # Remote client
                self.logger.info(f"Initializing remote AbsBox client at {self.absbox_url}")
                self._client = absbox.RemoteClient(self.absbox_url)
            else:
                # Local client
                self.logger.info("Initializing local AbsBox client")
                self._client = absbox.local.Client() if hasattr(absbox, 'local') else absbox.LocalClient()
                
            self.logger.info(f"AbsBox client initialized: {type(self._client).__name__}")
            self._initialized = True
            
        except ImportError as e:
            self.logger.error(f"Failed to import AbsBox library: {str(e)}")
            self._last_error = str(e)
            self._error_count += 1
            raise ImportError("AbsBox library is required") from e
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AbsBox client: {str(e)}")
            self._last_error = str(e)
            self._error_count += 1
            raise RuntimeError(f"AbsBox initialization failed: {str(e)}") from e
    
    async def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key for a calculation
        
        Args:
            prefix: Cache key prefix
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            Cache key string
        """
        # Convert args and kwargs to a serializable format
        key_components = [prefix]
        
        # Add args to key components
        for arg in args:
            if hasattr(arg, 'dict') and callable(getattr(arg, 'dict')):
                # Pydantic model or similar
                key_components.append(json.dumps(arg.dict(), sort_keys=True))
            elif isinstance(arg, (list, dict, tuple, set)):
                # Collection types
                key_components.append(json.dumps(arg, sort_keys=True))
            else:
                # Primitive types
                key_components.append(str(arg))
        
        # Add kwargs to key components
        for k, v in sorted(kwargs.items()):
            if v is not None:
                if hasattr(v, 'dict') and callable(getattr(v, 'dict')):
                    # Pydantic model or similar
                    key_components.append(f"{k}:{json.dumps(v.dict(), sort_keys=True)}")
                elif isinstance(v, (list, dict, tuple, set)):
                    # Collection types
                    key_components.append(f"{k}:{json.dumps(v, sort_keys=True)}")
                else:
                    # Primitive types
                    key_components.append(f"{k}:{v}")
        
        # Join components and compute hash for a fixed-length key
        key_string = ":".join(key_components)
        return f"absbox:{prefix}:{hash(key_string)}"
    
    def _prepare_cashflows(self, cashflows: List[Dict[str, Any]]) -> Any:
        """Convert cashflows to AbsBox format
        
        Args:
            cashflows: List of cashflow dictionaries
            
        Returns:
            AbsBox-compatible cashflow structure
        """
        # This implementation will depend on the specific AbsBox API
        # Here's a placeholder implementation
        try:
            # Convert each cashflow to the appropriate AbsBox format
            formatted_cashflows = []
            
            for cf in cashflows:
                # Extract required fields
                amount = cf.get("amount", 0.0)
                date = cf.get("date", None)
                
                # Create AbsBox cashflow object if AbsBox provides specific classes
                if hasattr(absbox, "Cashflow"):
                    formatted_cf = absbox.Cashflow(amount=amount, date=date)
                else:
                    # Otherwise use dict format
                    formatted_cf = {"amount": amount, "date": date}
                    
                # Add any additional fields required by AbsBox
                for key, value in cf.items():
                    if key not in ("amount", "date"):
                        if hasattr(formatted_cf, "set_property"):
                            formatted_cf.set_property(key, value)
                        else:
                            formatted_cf[key] = value
                
                formatted_cashflows.append(formatted_cf)
            
            return formatted_cashflows
            
        except Exception as e:
            self.logger.error(f"Error preparing cashflows: {str(e)}")
            raise CalculationError(
                f"Failed to prepare cashflows: {str(e)}",
                context={"cashflows_count": len(cashflows)},
                cause=e
            )
    
    def _adjust_discount_rate(
        self,
        base_rate: float,
        economic_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Adjust discount rate based on economic factors
        
        Args:
            base_rate: Base discount rate
            economic_factors: Optional economic factors
            
        Returns:
            Adjusted discount rate
        """
        if not economic_factors:
            return base_rate
        
        # Start with base rate
        adjusted_rate = base_rate
        
        # Apply market rate adjustment
        if "market_rate" in economic_factors:
            market_rate = economic_factors["market_rate"]
            # Blend base rate and market rate (70% market, 30% base)
            adjusted_rate = market_rate * 0.7 + base_rate * 0.3
        
        # Apply inflation premium
        if "inflation_rate" in economic_factors:
            inflation = economic_factors["inflation_rate"]
            if inflation > 0.02:  # Only add premium for above-target inflation
                adjusted_rate += (inflation - 0.02) * 0.5
        
        # Apply risk premium based on economic conditions
        risk_premium = 0
        
        if "unemployment_rate" in economic_factors:
            unemployment = economic_factors["unemployment_rate"]
            if unemployment > 0.05:  # Add premium for high unemployment
                risk_premium += (unemployment - 0.05) * 0.2
        
        if "gdp_growth" in economic_factors:
            gdp_growth = economic_factors["gdp_growth"]
            if gdp_growth < 0.01:  # Add premium for low growth
                risk_premium += (0.01 - gdp_growth) * 0.3
        
        # Apply risk premium
        adjusted_rate += risk_premium
        
        # Ensure rate is positive and reasonable
        return max(0.001, min(0.25, adjusted_rate))  # Cap at 25% to prevent extreme values
    
    @handle_errors(default_error=CalculationError)
    async def calculate_npv(
        self,
        cashflows: List[Dict[str, Any]],
        discount_rate: float,
        economic_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate Net Present Value
        
        Args:
            cashflows: List of cashflow dictionaries
            discount_rate: Annual discount rate
            economic_factors: Optional economic factors
            
        Returns:
            NPV value
        """
        # Generate cache key
        cache_key = await self._generate_cache_key(
            "npv", cashflows, discount_rate, economic_factors=economic_factors
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug("Cache hit for NPV calculation")
            self.metrics.increment_counter("cache_hits", {"calculation": "npv"})
            return cached_result
        
        self.metrics.increment_counter("cache_misses", {"calculation": "npv"})
        
        # Track calculation time
        start_time = datetime.now()
        
        try:
            # Process economic factors if provided
            adjusted_rate = self._adjust_discount_rate(discount_rate, economic_factors)
            
            # Convert cashflows to AbsBox format
            absbox_cashflows = self._prepare_cashflows(cashflows)
            
            # Calculate NPV
            with self.metrics.track_timing("absbox_npv_calculation"):
                result = await asyncio.to_thread(
                    self.client.calculate_npv,
                    absbox_cashflows,
                    adjusted_rate
                )
            
            # Round to 6 decimal places for consistency
            result = round(result, 6)
            
            # Cache the result
            await self.cache.set(cache_key, result, ttl=3600)
            
            # Track metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_calculation_time("npv", calculation_time)
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"NPV calculation failed: {str(e)}",
                extra={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows),
                    "has_economic_factors": economic_factors is not None
                }
            )
            raise CalculationError(
                f"NPV calculation failed: {str(e)}",
                context={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows)
                },
                cause=e
            )
    
    @handle_errors(default_error=CalculationError)
    async def calculate_irr(
        self,
        cashflows: List[Dict[str, Any]],
        initial_guess: float = 0.1,
        economic_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate Internal Rate of Return
        
        Args:
            cashflows: List of cashflow dictionaries
            initial_guess: Initial guess for IRR calculation
            economic_factors: Optional economic factors
            
        Returns:
            IRR value as a decimal (not percentage)
        """
        # Generate cache key
        cache_key = await self._generate_cache_key(
            "irr", cashflows, initial_guess, economic_factors=economic_factors
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug("Cache hit for IRR calculation")
            self.metrics.increment_counter("cache_hits", {"calculation": "irr"})
            return cached_result
        
        self.metrics.increment_counter("cache_misses", {"calculation": "irr"})
        
        # Track calculation time
        start_time = datetime.now()
        
        try:
            # Apply economic factor adjustments to the initial guess if applicable
            adjusted_guess = self._adjust_discount_rate(initial_guess, economic_factors)
            
            # Convert cashflows to AbsBox format
            absbox_cashflows = self._prepare_cashflows(cashflows)
            
            # Calculate IRR
            with self.metrics.track_timing("absbox_irr_calculation"):
                result = await asyncio.to_thread(
                    self.client.calculate_irr,
                    absbox_cashflows,
                    adjusted_guess
                )
            
            # Round to 6 decimal places for consistency
            result = round(result, 6)
            
            # Cache the result
            await self.cache.set(cache_key, result, ttl=3600)
            
            # Track metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_calculation_time("irr", calculation_time)
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"IRR calculation failed: {str(e)}",
                extra={
                    "cashflows_count": len(cashflows),
                    "initial_guess": initial_guess,
                    "has_economic_factors": economic_factors is not None
                }
            )
            raise CalculationError(
                f"IRR calculation failed: {str(e)}",
                context={
                    "cashflows_count": len(cashflows),
                    "initial_guess": initial_guess
                },
                cause=e
            )
    
    @handle_errors(default_error=CalculationError)
    async def calculate_duration(
        self,
        cashflows: List[Dict[str, Any]],
        discount_rate: float,
        economic_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate Macaulay and Modified Duration
        
        Args:
            cashflows: List of cashflow dictionaries
            discount_rate: Annual discount rate
            economic_factors: Optional economic factors
            
        Returns:
            Dictionary with duration metrics
        """
        # Generate cache key
        cache_key = await self._generate_cache_key(
            "duration", cashflows, discount_rate, economic_factors=economic_factors
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug("Cache hit for duration calculation")
            self.metrics.increment_counter("cache_hits", {"calculation": "duration"})
            return cached_result
        
        self.metrics.increment_counter("cache_misses", {"calculation": "duration"})
        
        # Track calculation time
        start_time = datetime.now()
        
        try:
            # Process economic factors if provided
            adjusted_rate = self._adjust_discount_rate(discount_rate, economic_factors)
            
            # Convert cashflows to AbsBox format
            absbox_cashflows = self._prepare_cashflows(cashflows)
            
            # Calculate duration
            with self.metrics.track_timing("absbox_duration_calculation"):
                result = await asyncio.to_thread(
                    self.client.calculate_duration,
                    absbox_cashflows,
                    adjusted_rate
                )
            
            # Round results for consistency
            processed_result = {
                "macaulay_duration": round(result.get("macaulay_duration", 0), 4),
                "modified_duration": round(result.get("modified_duration", 0), 4),
                "effective_duration": round(result.get("effective_duration", 0), 4) if "effective_duration" in result else None
            }
            
            # Cache the result
            await self.cache.set(cache_key, processed_result, ttl=3600)
            
            # Track metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_calculation_time("duration", calculation_time)
            
            return processed_result
            
        except Exception as e:
            self.logger.error(
                f"Duration calculation failed: {str(e)}",
                extra={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows),
                    "has_economic_factors": economic_factors is not None
                }
            )
            raise CalculationError(
                f"Duration calculation failed: {str(e)}",
                context={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows)
                },
                cause=e
            )
    
    @handle_errors(default_error=CalculationError)
    async def run_stress_test(
        self,
        cashflows: List[Dict[str, Any]],
        discount_rate: float,
        scenarios: List[Dict[str, Any]],
        economic_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run stress test with multiple scenarios
        
        Args:
            cashflows: List of cashflow dictionaries
            discount_rate: Base discount rate
            scenarios: List of scenario configurations
            economic_factors: Base economic factors
            
        Returns:
            Dictionary with stress test results
        """
        # Generate cache key for the entire stress test
        cache_key = await self._generate_cache_key(
            "stress_test", cashflows, discount_rate, scenarios, economic_factors=economic_factors
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug("Cache hit for stress test calculation")
            self.metrics.increment_counter("cache_hits", {"calculation": "stress_test"})
            return cached_result
        
        self.metrics.increment_counter("cache_misses", {"calculation": "stress_test"})
        
        # Track calculation time
        start_time = datetime.now()
        
        try:
            # Process base economic factors
            base_rate = self._adjust_discount_rate(discount_rate, economic_factors)
            
            # Convert cashflows to AbsBox format
            absbox_cashflows = self._prepare_cashflows(cashflows)
            
            # Run stress tests for each scenario
            results = {}
            for scenario in scenarios:
                scenario_name = scenario.get("name", f"Scenario {len(results) + 1}")
                
                # Get scenario-specific economic factors by merging base and scenario factors
                scenario_factors = economic_factors.copy() if economic_factors else {}
                scenario_factors.update(scenario.get("economic_factors", {}))
                
                # Adjust discount rate for this scenario
                scenario_rate = self._adjust_discount_rate(discount_rate, scenario_factors)
                
                # Apply scenario-specific cashflow adjustments if provided
                scenario_cashflows = absbox_cashflows
                if "cashflow_adjustments" in scenario:
                    # Apply adjustments to cashflows (implementation depends on AbsBox API)
                    pass
                
                # Calculate NPV for this scenario
                with self.metrics.track_timing(f"stress_test_scenario_{len(results)}"):
                    npv = await asyncio.to_thread(
                        self.client.calculate_npv,
                        scenario_cashflows,
                        scenario_rate
                    )
                
                # Add result to dictionary
                results[scenario_name] = {
                    "npv": round(npv, 6),
                    "discount_rate": round(scenario_rate, 6),
                    "change_from_base": round(npv - (await self.calculate_npv(cashflows, discount_rate, economic_factors)), 6)
                }
            
            # Prepare final result with summary
            final_result = {
                "scenarios": results,
                "summary": {
                    "scenario_count": len(scenarios),
                    "min_npv": min(s["npv"] for s in results.values()),
                    "max_npv": max(s["npv"] for s in results.values()),
                    "average_npv": sum(s["npv"] for s in results.values()) / len(results) if results else 0
                }
            }
            
            # Cache the result
            await self.cache.set(cache_key, final_result, ttl=3600)
            
            # Track metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_calculation_time("stress_test", calculation_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(
                f"Stress test calculation failed: {str(e)}",
                extra={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows),
                    "scenarios_count": len(scenarios),
                    "has_economic_factors": economic_factors is not None
                }
            )
            raise CalculationError(
                f"Stress test calculation failed: {str(e)}",
                context={
                    "discount_rate": discount_rate,
                    "cashflows_count": len(cashflows),
                    "scenarios_count": len(scenarios)
                },
                cause=e
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service health and status information
        
        Returns:
            Dictionary with service status details
        """
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "last_error": self._last_error,
            "error_count": self._error_count,
            "absbox_available": ABSBOX_AVAILABLE,
            "absbox_url": self.absbox_url or "local",
            "timestamp": datetime.now().isoformat(),
        }
