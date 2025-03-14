"""
Unified AbsBox Service

This module provides a comprehensive service for structured finance analytics using AbsBox,
combining and enhancing the functionality from both absbox_service.py
and absbox_service_enhanced.py into a single, optimized implementation with
robust error handling, efficient caching, and comprehensive monitoring.
"""

import os
import time
import json
import logging
import asyncio
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, date
from functools import wraps
from contextlib import contextmanager

# Numerical libraries for vectorized calculations
import numpy as np
import pandas as pd

# Import the UnifiedRedisService for caching
from app.services.unified_redis_service import UnifiedRedisService, RedisConfig

# Import models
from app.models.structured_products import (
    StructuredDealRequest,
    StructuredDealResponse,
    LoanPoolConfig,
    WaterfallConfig,
    ScenarioConfig,
    CashflowProjection,
    EnhancedAnalyticsResult,
    LoanData,
    CashflowForecastResponse,
    EnhancedAnalyticsRequest,
    RiskMetrics,
    SensitivityAnalysis
)

from app.core.config import settings
from app.core.monitoring import CALCULATION_TIME, CalculationTracker

# Setup logging
logger = logging.getLogger("absbox_service")

# Optional monitoring
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Histogram, Summary, Gauge
    
    # Define metrics
    ABSBOX_OPERATIONS = Counter(
        'absbox_operations_total',
        'Total number of AbsBox operations',
        ['operation', 'status']
    )
    
    ABSBOX_CALCULATION_TIME = Histogram(
        'absbox_calculation_duration_seconds',
        'AbsBox calculation duration in seconds',
        ['operation']
    )
    
    ABSBOX_CACHE_HITS = Counter(
        'absbox_cache_hits_total',
        'Total number of AbsBox cache hits',
        ['operation']
    )
    
    ABSBOX_CACHE_MISSES = Counter(
        'absbox_cache_misses_total',
        'Total number of AbsBox cache misses',
        ['operation']
    )
    
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# AbsBox imports - use proper error handling for imports
try:
    import absbox as ab
    
    # Try importing directly from absbox.local if available
    try:
        from absbox.local.pool import Pool
        from absbox.local.loan import Loan, FixedRateLoan, FloatingRateLoan
        from absbox.local.deal import Deal
        from absbox.local.engine import LiqEngine
        from absbox.local.waterfall import Waterfall
        from absbox.local.assumption import Assumption, DefaultAssumption
        from absbox.local.rateAssumption import FlatCurve
        from absbox.local.analytics import Analytics
        from absbox.local.cashflow import Cashflow
        
        ABSBOX_IMPORTS_OK = True
        ABSBOX_IMPORT_METHOD = "direct"
    except ImportError:
        # Otherwise use the main absbox module
        logger.warning("Using alternative absbox import structure")
        Pool = ab.local.Pool if hasattr(ab, 'local') and hasattr(ab.local, 'Pool') else None 
        Loan = ab.local.Loan if hasattr(ab, 'local') and hasattr(ab.local, 'Loan') else None
        Deal = ab.local.Deal if hasattr(ab, 'local') and hasattr(ab.local, 'Deal') else None
        LiqEngine = ab.local.LiqEngine if hasattr(ab, 'local') and hasattr(ab.local, 'LiqEngine') else None
        Waterfall = ab.local.Waterfall if hasattr(ab, 'local') and hasattr(ab.local, 'Waterfall') else None
        Assumption = ab.local.Assumption if hasattr(ab, 'local') and hasattr(ab.local, 'Assumption') else None
        DefaultAssumption = ab.local.DefaultAssumption if hasattr(ab, 'local') and hasattr(ab.local, 'DefaultAssumption') else None
        Analytics = ab.local.Analytics if hasattr(ab, 'local') and hasattr(ab.local, 'Analytics') else None
        FlatCurve = ab.local.FlatCurve if hasattr(ab, 'local') and hasattr(ab.local, 'FlatCurve') else None
        Cashflow = ab.local.Cashflow if hasattr(ab, 'local') and hasattr(ab.local, 'Cashflow') else None
        
        ABSBOX_IMPORTS_OK = True
        ABSBOX_IMPORT_METHOD = "indirect"
        
    # Try importing Hastructure APIs if available
    try:
        from absbox.apis import HastructureEngine
        HASTRUCTURE_AVAILABLE = True
    except ImportError:
        logger.warning("Hastructure engine not available")
        HASTRUCTURE_AVAILABLE = False
        
except ImportError:
    logger.error("AbsBox package not installed. Functionality will be limited.")
    ABSBOX_IMPORTS_OK = False
    HASTRUCTURE_AVAILABLE = False


class OperationContext:
    """Context manager for tracking operation metrics and logging"""
    
    def __init__(self, service, operation_name: str):
        self.service = service
        self.operation_name = operation_name
        self.start_time = None
        self.success = False
        
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.success = exc_type is None
        
        # Log duration and status
        if self.success:
            logger.debug(f"Operation {self.operation_name} completed successfully in {duration:.2f}s")
        else:
            logger.error(f"Operation {self.operation_name} failed after {duration:.2f}s: {exc_val}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        # Record metrics if enabled
        if METRICS_ENABLED:
            status = "success" if self.success else "failure"
            ABSBOX_OPERATIONS.labels(operation=self.operation_name, status=status).inc()
            ABSBOX_CALCULATION_TIME.labels(operation=self.operation_name).observe(duration)
            
        return False  # Don't suppress exceptions


class UnifiedAbsBoxService:
    """
    Unified service for structured finance analytics using AbsBox
    
    This service combines and enhances the functionality from both absbox_service.py
    and absbox_service_enhanced.py into a single, optimized implementation with
    robust error handling, efficient caching, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        redis_service: Optional[UnifiedRedisService] = None,
        hastructure_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the unified AbsBox service
        
        Args:
            redis_service: UnifiedRedisService for caching
            hastructure_url: URL for Hastructure engine
            config: Service configuration options
        """
        self.config = config or {}
        
        # Set up Redis service
        if redis_service:
            self.redis = redis_service
        else:
            # Create a dedicated Redis namespace for AbsBox cache
            redis_config = RedisConfig(namespace="absbox")
            self.redis = UnifiedRedisService(config=redis_config)
            
        # Set up Hastructure connection
        self.hastructure_url = hastructure_url or settings.HASTRUCTURE_URL
        
        # Initialize lazy-loaded components
        self._engine = None
        self._analytics = None
        self._hastructure = None
        
        # Configure caching
        self.use_cache = self.config.get("use_cache", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour default
        
        # Configure vectorized calculations
        self.use_vectorized = self.config.get("use_vectorized", True)
        
        # Log initialization
        logger.info(f"UnifiedAbsBoxService initialized with Hastructure URL: {self.hastructure_url}")
        logger.info(f"AbsBox import method: {ABSBOX_IMPORT_METHOD if ABSBOX_IMPORTS_OK else 'FAILED'}")
    
    @property
    def engine(self):
        """Lazy-loaded calculation engine"""
        if self._engine is None:
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                self._engine = self._initialize_engine()
                logger.info("AbsBox calculation engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AbsBox engine: {e}")
                raise RuntimeError(f"Error initializing AbsBox engine: {e}")
                
        return self._engine
    
    @property
    def analytics(self):
        """Lazy-loaded analytics component"""
        if self._analytics is None:
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                self._analytics = Analytics()
                logger.info("AbsBox analytics component initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AbsBox analytics: {e}")
                raise RuntimeError(f"Error initializing AbsBox analytics: {e}")
                
        return self._analytics
    
    @property
    def hastructure(self):
        """Lazy-loaded Hastructure engine client"""
        if self._hastructure is None:
            if not HASTRUCTURE_AVAILABLE:
                raise RuntimeError("Hastructure engine not available")
                
            try:
                self._hastructure = HastructureEngine(self.hastructure_url)
                logger.info(f"Connected to Hastructure engine at {self.hastructure_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Hastructure engine: {e}")
                raise RuntimeError(f"Error connecting to Hastructure engine: {e}")
                
        return self._hastructure
        
    def _initialize_engine(self):
        """Initialize the AbsBox calculation engine"""
        try:
            # Create a properly configured engine instance
            engine_config = {
                "precision": self.config.get("precision", 6),
                "max_iterations": self.config.get("max_iterations", 1000),
                "convergence_threshold": self.config.get("convergence_threshold", 1e-6)
            }
            
            return LiqEngine(**engine_config)
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            raise
            
    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Generate a deterministic cache key from operation and parameters
        
        This ensures that identical operations with identical parameters
        will have the same cache key, regardless of parameter order.
        
        Args:
            operation: Name of the operation
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            A deterministic cache key string
        """
        # Convert args to a consistent representation
        args_str = json.dumps([str(arg) for arg in args], sort_keys=True)
        
        # Convert kwargs to a consistent representation
        kwargs_str = json.dumps(
            {k: str(v) for k, v in sorted(kwargs.items())}, 
            sort_keys=True
        )
        
        # Combine everything into a deterministic key
        combined = f"{operation}:{args_str}:{kwargs_str}"
        
        # Create a hash of the combined string for a shorter key
        import hashlib
        hash_obj = hashlib.md5(combined.encode())
        hash_str = hash_obj.hexdigest()
        
        return f"absbox:{operation}:{hash_str}"
        
    @contextmanager
    def operation(self, name: str):
        """Context manager for tracking operations"""
        with OperationContext(self, name) as context:
            yield context
    
    async def get_cached_result(self, cache_key: str):
        """
        Try to get a cached result
        
        Args:
            cache_key: The cache key to retrieve
            
        Returns:
            Cached result or None if not found
        """
        if not self.use_cache:
            return None
            
        try:
            # Get from cache using async Redis method
            result = await self.redis.get(cache_key)
            
            # Record cache hit/miss metrics
            if METRICS_ENABLED:
                if result is not None:
                    ABSBOX_CACHE_HITS.labels(operation=cache_key.split(":")[1]).inc()
                else:
                    ABSBOX_CACHE_MISSES.labels(operation=cache_key.split(":")[1]).inc()
                    
            return result
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None
    
    async def cache_result(self, cache_key: str, result: Any, ttl: Optional[int] = None):
        """
        Cache a result
        
        Args:
            cache_key: The cache key to store under
            result: The result to cache
            ttl: Cache time-to-live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.use_cache:
            return False
            
        try:
            # Set TTL to configured default if not specified
            ttl = ttl or self.cache_ttl
            
            # Cache using async Redis method
            return await self.redis.set(cache_key, result, ttl)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the AbsBox service
        
        Returns:
            Dictionary with health status information
        """
        health_data = {
            "status": "unavailable",
            "absbox_available": ABSBOX_IMPORTS_OK,
            "hastructure_available": HASTRUCTURE_AVAILABLE,
            "cache_available": False,
            "error": None,
            "version": {
                "absbox": getattr(ab, "__version__", "unknown") if ABSBOX_IMPORTS_OK else None,
            }
        }
        
        try:
            # Check Redis connection
            redis_health = self.redis.health_check()
            health_data["cache_available"] = redis_health["connected"]
            
            # Check if AbsBox is available
            if not ABSBOX_IMPORTS_OK:
                health_data["error"] = "AbsBox package not properly installed"
                return health_data
                
            # Try a simple calculation to verify engine works
            with self.operation("health_check"):
                # Create a simple loan for testing
                test_loan = FixedRateLoan(
                    name="test_loan",
                    balance=100000,
                    rate=0.05,
                    term=60,
                    period=12
                )
                
                # Verify loan calculation works
                cashflows = test_loan.project()
                if cashflows is not None:
                    health_data["status"] = "healthy"
        except Exception as e:
            health_data["error"] = str(e)
            logger.error(f"Health check failed: {e}")
            
        return health_data

    def create_loan(self, loan_data: Union[Dict, LoanData]) -> Any:
        """
        Create an AbsBox loan object from loan data
        
        Args:
            loan_data: Loan configuration data
            
        Returns:
            AbsBox loan object
        """
        with self.operation("create_loan"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                loan_type = loan_data.get("type", "fixed")
                
                # Common loan parameters
                loan_args = {
                    "name": loan_data.get("name", "loan"),
                    "balance": float(loan_data.get("balance", 0)),
                    "term": int(loan_data.get("term", 0)),
                    "period": int(loan_data.get("period", 12)),
                    "origination": loan_data.get("origination", date.today().isoformat()),
                    "delay": int(loan_data.get("delay", 0)),
                }
                
                # Create specific loan type
                if loan_type.lower() == "fixed":
                    loan_args["rate"] = float(loan_data.get("rate", 0))
                    return FixedRateLoan(**loan_args)
                elif loan_type.lower() == "floating":
                    loan_args["margin"] = float(loan_data.get("margin", 0))
                    loan_args["index"] = loan_data.get("index", "LIBOR")
                    loan_args["index_value"] = float(loan_data.get("index_value", 0))
                    return FloatingRateLoan(**loan_args)
                else:
                    raise ValueError(f"Unsupported loan type: {loan_type}")
            except Exception as e:
                logger.error(f"Error creating loan: {e}")
                logger.debug(f"Loan data: {loan_data}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to create loan: {e}")
    
    def create_pool(self, pool_config: Union[Dict, LoanPoolConfig]) -> Any:
        """
        Create an AbsBox pool object from pool configuration
        
        Args:
            pool_config: Pool configuration data
            
        Returns:
            AbsBox pool object
        """
        with self.operation("create_pool"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Create pool with basic attributes
                pool_name = pool_config.get("name", "pool")
                pool_obj = Pool(name=pool_name)
                
                # Add loans to the pool
                loans = pool_config.get("loans", [])
                for loan_data in loans:
                    loan = self.create_loan(loan_data)
                    pool_obj.add_loan(loan)
                
                # If cutoff date is provided, apply it
                cutoff_date = pool_config.get("cutoff_date")
                if cutoff_date:
                    pool_obj = pool_obj.subset(as_of=cutoff_date)
                
                return pool_obj
            except Exception as e:
                logger.error(f"Error creating pool: {e}")
                logger.debug(f"Pool config: {pool_config}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to create loan pool: {e}")
    
    def create_waterfall(self, waterfall_config: Union[Dict, WaterfallConfig]) -> Any:
        """
        Create an AbsBox waterfall object from waterfall configuration
        
        Args:
            waterfall_config: Waterfall configuration data
            
        Returns:
            AbsBox waterfall object
        """
        with self.operation("create_waterfall"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Create waterfall with basic attributes
                waterfall_name = waterfall_config.get("name", "waterfall")
                waterfall_obj = Waterfall(name=waterfall_name)
                
                # Configure cashflow allocation rules
                rules = waterfall_config.get("rules", [])
                for rule in rules:
                    rule_type = rule.get("type", "")
                    
                    if rule_type == "interest":
                        waterfall_obj.add_interest(
                            tranche=rule.get("tranche", ""),
                            rate=float(rule.get("rate", 0)),
                            accrual=rule.get("accrual", "actual/360")
                        )
                    elif rule_type == "principal":
                        waterfall_obj.add_principal(
                            tranche=rule.get("tranche", ""),
                            amount=float(rule.get("amount", 0)) if "amount" in rule else None,
                            percent=float(rule.get("percent", 0)) if "percent" in rule else None
                        )
                    elif rule_type == "reserve":
                        waterfall_obj.add_reserve(
                            name=rule.get("name", "reserve"),
                            amount=float(rule.get("amount", 0)) if "amount" in rule else None,
                            percent=float(rule.get("percent", 0)) if "percent" in rule else None
                        )
                    else:
                        logger.warning(f"Unsupported waterfall rule type: {rule_type}")
                
                return waterfall_obj
            except Exception as e:
                logger.error(f"Error creating waterfall: {e}")
                logger.debug(f"Waterfall config: {waterfall_config}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to create waterfall: {e}")
    
    def create_deal(
        self,
        pool: Any,
        waterfall: Any,
        deal_config: Dict = None
    ) -> Any:
        """
        Create an AbsBox deal object from pool, waterfall, and deal configuration
        
        Args:
            pool: AbsBox pool object
            waterfall: AbsBox waterfall object
            deal_config: Deal configuration data
            
        Returns:
            AbsBox deal object
        """
        with self.operation("create_deal"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Create deal with basic attributes
                deal_name = deal_config.get("name", "deal") if deal_config else "deal"
                deal_obj = Deal(
                    name=deal_name,
                    pool=pool,
                    waterfall=waterfall
                )
                
                # Apply deal-level configuration
                if deal_config:
                    # Start date
                    if "start_date" in deal_config:
                        deal_obj.start_date = deal_config["start_date"]
                    
                    # Payment frequency
                    if "payment_frequency" in deal_config:
                        deal_obj.payment_frequency = deal_config["payment_frequency"]
                    
                    # Call option
                    if "call_option" in deal_config:
                        call_option = deal_config["call_option"]
                        deal_obj.call_option(
                            trigger=call_option.get("trigger", 0.1),
                            trigger_type=call_option.get("trigger_type", "percent")
                        )
                
                return deal_obj
            except Exception as e:
                logger.error(f"Error creating deal: {e}")
                logger.debug(f"Deal config: {deal_config}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to create deal: {e}")
    
    def create_assumption(self, scenario_config: Union[Dict, ScenarioConfig]) -> Any:
        """
        Create an AbsBox assumption object from scenario configuration
        
        Args:
            scenario_config: Scenario configuration data
            
        Returns:
            AbsBox assumption object
        """
        with self.operation("create_assumption"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Create base assumption
                assumption = Assumption(
                    name=scenario_config.get("name", "base")
                )
                
                # Configure default assumptions
                if "default" in scenario_config:
                    default_config = scenario_config["default"]
                    default_obj = DefaultAssumption(
                        cdr=float(default_config.get("cdr", 0)),
                        severity=float(default_config.get("severity", 0)),
                        lag=int(default_config.get("lag", 0))
                    )
                    assumption.default(default_obj)
                
                # Configure prepayment assumptions
                if "prepayment" in scenario_config:
                    prepayment_config = scenario_config["prepayment"]
                    assumption.cpr(float(prepayment_config.get("cpr", 0)))
                
                # Configure rate assumptions
                if "rates" in scenario_config:
                    rates_config = scenario_config["rates"]
                    curve_type = rates_config.get("type", "flat")
                    
                    if curve_type == "flat":
                        curve = FlatCurve(
                            value=float(rates_config.get("value", 0)),
                            forward=rates_config.get("forward", False)
                        )
                        assumption.rate_curve(curve)
                    else:
                        logger.warning(f"Unsupported rate curve type: {curve_type}")
                
                return assumption
            except Exception as e:
                logger.error(f"Error creating assumption: {e}")
                logger.debug(f"Scenario config: {scenario_config}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to create assumption: {e}")

    def project_cashflows(
        self,
        deal: Any,
        assumption: Any = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Project deal cashflows under the given assumption
        
        Args:
            deal: AbsBox deal object
            assumption: AbsBox assumption object (optional)
            start_date: Start date for projection (optional)
            end_date: End date for projection (optional)
            cache_key: Custom cache key (optional)
            
        Returns:
            Dictionary with projected cashflows and metrics
        """
        with self.operation("project_cashflows"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Check if results are in cache
                if cache_key and self.use_cache:
                    try:
                        cached_result = self.redis.get_sync(cache_key)
                        if cached_result:
                            logger.debug(f"Cache hit for {cache_key}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="project_cashflows").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="project_cashflows").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Project cashflows with or without assumption
                if assumption:
                    result = deal.project(assumption)
                else:
                    result = deal.project()
                
                # Convert projection to serializable format
                serialized_result = self._serialize_projection(result)
                
                # Cache the result
                if cache_key and self.use_cache:
                    try:
                        self.redis.set_sync(cache_key, serialized_result, self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return serialized_result
            except Exception as e:
                logger.error(f"Error projecting cashflows: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to project cashflows: {e}")
    
    async def project_cashflows_async(
        self,
        deal: Any,
        assumption: Any = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Project deal cashflows asynchronously under the given assumption
        
        Args:
            deal: AbsBox deal object
            assumption: AbsBox assumption object (optional)
            start_date: Start date for projection (optional)
            end_date: End date for projection (optional)
            cache_key: Custom cache key (optional)
            
        Returns:
            Dictionary with projected cashflows and metrics
        """
        with self.operation("project_cashflows_async"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Check if results are in cache
                if cache_key and self.use_cache:
                    try:
                        cached_result = await self.redis.get(cache_key)
                        if cached_result:
                            logger.debug(f"Cache hit for {cache_key}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="project_cashflows").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="project_cashflows").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Run the projection in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                
                if assumption:
                    result = await loop.run_in_executor(
                        None, lambda: deal.project(assumption)
                    )
                else:
                    result = await loop.run_in_executor(
                        None, deal.project
                    )
                
                # Convert projection to serializable format
                serialized_result = self._serialize_projection(result)
                
                # Cache the result
                if cache_key and self.use_cache:
                    try:
                        await self.redis.set(cache_key, serialized_result, self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return serialized_result
            except Exception as e:
                logger.error(f"Error projecting cashflows asynchronously: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to project cashflows: {e}")
    
    def _serialize_projection(self, projection: Any) -> Dict[str, Any]:
        """
        Convert AbsBox projection to serializable format
        
        Args:
            projection: AbsBox projection object
            
        Returns:
            Dictionary with projection data
        """
        try:
            data = {}
            
            # Check if projection exists
            if not projection:
                return {"error": "No projection data available"}
            
            # Extract cashflows by date
            cashflows_by_date = {}
            for date_str, cf in projection.cashflows_by_date().items():
                cashflows_by_date[date_str] = {
                    "principal": float(cf.get("principal", 0)),
                    "interest": float(cf.get("interest", 0)),
                    "default": float(cf.get("default", 0)),
                    "recovery": float(cf.get("recovery", 0)),
                    "prepayment": float(cf.get("prepayment", 0)),
                    "ending_balance": float(cf.get("ending_balance", 0))
                }
            data["cashflows_by_date"] = cashflows_by_date
            
            # Extract tranche cashflows
            tranche_cashflows = {}
            for tranche, cf in projection.tranche_cashflows().items():
                tranche_cashflows[tranche] = {}
                for date_str, values in cf.items():
                    tranche_cashflows[tranche][date_str] = {
                        "principal": float(values.get("principal", 0)),
                        "interest": float(values.get("interest", 0)),
                        "balance": float(values.get("balance", 0))
                    }
            data["tranche_cashflows"] = tranche_cashflows
            
            # Extract summary metrics
            metrics = {}
            try:
                metrics["yield"] = {
                    tranche: float(value)
                    for tranche, value in projection.yield_metric().items()
                }
            except Exception as e:
                logger.warning(f"Error extracting yield metrics: {e}")
                metrics["yield"] = {}
                
            try:
                metrics["duration"] = {
                    tranche: float(value)
                    for tranche, value in projection.duration().items()
                }
            except Exception as e:
                logger.warning(f"Error extracting duration metrics: {e}")
                metrics["duration"] = {}
                
            try:
                metrics["wal"] = {
                    tranche: float(value)
                    for tranche, value in projection.wal().items()
                }
            except Exception as e:
                logger.warning(f"Error extracting WAL metrics: {e}")
                metrics["wal"] = {}
            
            data["metrics"] = metrics
            
            return data
        except Exception as e:
            logger.error(f"Error serializing projection: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            return {"error": f"Failed to serialize projection: {e}"}
    
    def _serialize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert Pandas DataFrame to serializable format
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with DataFrame data
        """
        try:
            # Reset the index if it's not the default RangeIndex
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()
                
            # Convert dates to strings
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
            
            # Convert to nested dictionary
            result = {}
            result["columns"] = df.columns.tolist()
            result["data"] = df.values.tolist()
            result["index"] = df.index.tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error serializing DataFrame: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            return {"error": f"Failed to serialize DataFrame: {e}"}
    
    def calculate_risk_metrics(
        self,
        deal: Any,
        scenarios: List[Any],
        base_assumption: Any = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics across multiple scenarios
        
        Args:
            deal: AbsBox deal object
            scenarios: List of AbsBox assumption objects
            base_assumption: Base assumption for comparison (optional)
            cache_key: Custom cache key (optional)
            
        Returns:
            Dictionary with risk metrics for each scenario
        """
        with self.operation("calculate_risk_metrics"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Check if results are in cache
                if cache_key and self.use_cache:
                    try:
                        cached_result = self.redis.get_sync(cache_key)
                        if cached_result:
                            logger.debug(f"Cache hit for {cache_key}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="calculate_risk_metrics").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="calculate_risk_metrics").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Project base case if provided
                base_projection = None
                if base_assumption:
                    base_projection = deal.project(base_assumption)
                
                # Project each scenario
                scenario_results = {}
                for i, scenario in enumerate(scenarios):
                    scenario_name = getattr(scenario, "name", f"scenario_{i}")
                    try:
                        projection = deal.project(scenario)
                        metrics = {
                            "yield": {
                                tranche: float(value)
                                for tranche, value in projection.yield_metric().items()
                            },
                            "duration": {
                                tranche: float(value)
                                for tranche, value in projection.duration().items()
                            },
                            "wal": {
                                tranche: float(value)
                                for tranche, value in projection.wal().items()
                            }
                        }
                        
                        # Calculate differences from base case
                        if base_projection:
                            diffs = {}
                            for metric_name, base_values in {
                                "yield": base_projection.yield_metric(),
                                "duration": base_projection.duration(),
                                "wal": base_projection.wal()
                            }.items():
                                diffs[metric_name] = {}
                                for tranche, base_value in base_values.items():
                                    scenario_value = metrics[metric_name].get(tranche, 0)
                                    diffs[metric_name][tranche] = float(scenario_value) - float(base_value)
                            
                            metrics["diff_from_base"] = diffs
                            
                        scenario_results[scenario_name] = metrics
                    except Exception as e:
                        logger.error(f"Error calculating metrics for scenario {scenario_name}: {e}")
                        scenario_results[scenario_name] = {"error": str(e)}
                
                result = {"scenarios": scenario_results}
                
                # Cache the result
                if cache_key and self.use_cache:
                    try:
                        self.redis.set_sync(cache_key, result, self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return result
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to calculate risk metrics: {e}")
    
    async def calculate_risk_metrics_async(
        self,
        deal: Any,
        scenarios: List[Any],
        base_assumption: Any = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics across multiple scenarios asynchronously
        
        Args:
            deal: AbsBox deal object
            scenarios: List of AbsBox assumption objects
            base_assumption: Base assumption for comparison (optional)
            cache_key: Custom cache key (optional)
            
        Returns:
            Dictionary with risk metrics for each scenario
        """
        with self.operation("calculate_risk_metrics_async"):
            if not ABSBOX_IMPORTS_OK:
                raise RuntimeError("AbsBox package not properly installed")
                
            try:
                # Check if results are in cache
                if cache_key and self.use_cache:
                    try:
                        cached_result = await self.redis.get(cache_key)
                        if cached_result:
                            logger.debug(f"Cache hit for {cache_key}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="calculate_risk_metrics").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="calculate_risk_metrics").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Run the calculation in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Process each scenario concurrently
                base_projection = None
                if base_assumption:
                    base_projection = await loop.run_in_executor(
                        None, lambda: deal.project(base_assumption)
                    )
                
                async def process_scenario(scenario, index):
                    scenario_name = getattr(scenario, "name", f"scenario_{index}")
                    try:
                        projection = await loop.run_in_executor(
                            None, lambda: deal.project(scenario)
                        )
                        
                        metrics = {
                            "yield": {
                                tranche: float(value)
                                for tranche, value in projection.yield_metric().items()
                            },
                            "duration": {
                                tranche: float(value)
                                for tranche, value in projection.duration().items()
                            },
                            "wal": {
                                tranche: float(value)
                                for tranche, value in projection.wal().items()
                            }
                        }
                        
                        # Calculate differences from base case
                        if base_projection:
                            diffs = {}
                            for metric_name, base_values in {
                                "yield": base_projection.yield_metric(),
                                "duration": base_projection.duration(),
                                "wal": base_projection.wal()
                            }.items():
                                diffs[metric_name] = {}
                                for tranche, base_value in base_values.items():
                                    scenario_value = metrics[metric_name].get(tranche, 0)
                                    diffs[metric_name][tranche] = float(scenario_value) - float(base_value)
                            
                            metrics["diff_from_base"] = diffs
                            
                        return scenario_name, metrics
                    except Exception as e:
                        logger.error(f"Error calculating metrics for scenario {scenario_name}: {e}")
                        return scenario_name, {"error": str(e)}
                
                # Process scenarios concurrently but with a reasonable limit
                tasks = []
                for i, scenario in enumerate(scenarios):
                    tasks.append(process_scenario(scenario, i))
                
                scenario_results = {}
                for coro in asyncio.as_completed(tasks):
                    scenario_name, metrics = await coro
                    scenario_results[scenario_name] = metrics
                
                result = {"scenarios": scenario_results}
                
                # Cache the result
                if cache_key and self.use_cache:
                    try:
                        await self.redis.set(cache_key, result, self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return result
            except Exception as e:
                logger.error(f"Error calculating risk metrics asynchronously: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to calculate risk metrics: {e}")

    def forecast_cashflows(
        self,
        loan_data: Union[List[Dict], List[LoanData]],
        forecast_params: Dict[str, Any] = None,
        cache: bool = True
    ) -> CashflowForecastResponse:
        """
        Forecast cashflows for a collection of loans
        
        This is a higher-level convenience method that handles the entire process
        of creating a loan pool, applying assumptions, and forecasting cashflows.
        
        Args:
            loan_data: List of loan data to forecast
            forecast_params: Parameters controlling the forecast
            cache: Whether to use caching
            
        Returns:
            CashflowForecastResponse with forecasted cashflows
        """
        with self.operation("forecast_cashflows"):
            start_time = time.time()
            request_id = forecast_params.get("request_id", str(int(time.time())))
            response = CashflowForecastResponse(
                request_id=request_id,
                status="processing"
            )
            
            try:
                # Determine if we should use cache
                use_cache = cache and self.use_cache
                cache_key = None
                
                if use_cache:
                    # Generate deterministic cache key
                    loan_data_json = json.dumps(loan_data, sort_keys=True) if isinstance(loan_data[0], dict) else json.dumps([loan.dict() for loan in loan_data], sort_keys=True)
                    params_json = json.dumps(forecast_params or {}, sort_keys=True)
                    
                    import hashlib
                    hash_obj = hashlib.md5((loan_data_json + params_json).encode())
                    cache_key = f"absbox:forecast:{hash_obj.hexdigest()}"
                    
                    # Check cache
                    try:
                        cached_result = self.redis.get_sync(cache_key)
                        if cached_result:
                            logger.info(f"Cache hit for forecast request {response.request_id}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="forecast_cashflows").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="forecast_cashflows").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Extract forecast parameters
                params = forecast_params or {}
                horizon_months = params.get("horizon_months", 120)  # 10 years default
                prepayment_rate = params.get("prepayment_rate", 0.0)
                default_rate = params.get("default_rate", 0.0)
                severity_rate = params.get("severity_rate", 0.0)
                recovery_lag = params.get("recovery_lag", 0)
                interest_rate_shock = params.get("interest_rate_shock", 0.0)
                
                # Create a pool with the loans
                pool_config = {
                    "name": params.get("pool_name", "forecast_pool"),
                    "loans": loan_data
                }
                pool = self.create_pool(pool_config)
                
                # Create a simple pass-through waterfall
                waterfall_config = {
                    "name": "passthrough",
                    "rules": [
                        {
                            "type": "interest",
                            "tranche": "A",
                            "rate": 0.0
                        },
                        {
                            "type": "principal",
                            "tranche": "A",
                            "percent": 1.0
                        }
                    ]
                }
                waterfall = self.create_waterfall(waterfall_config)
                
                # Create a deal
                deal_config = {
                    "name": "forecast_deal",
                    "start_date": params.get("start_date", date.today().isoformat())
                }
                deal = self.create_deal(pool, waterfall, deal_config)
                
                # Create assumption
                scenario_config = {
                    "name": "forecast_scenario",
                    "prepayment": {
                        "cpr": prepayment_rate
                    },
                    "default": {
                        "cdr": default_rate,
                        "severity": severity_rate,
                        "lag": recovery_lag
                    },
                    "rates": {
                        "type": "flat",
                        "value": interest_rate_shock,
                        "forward": True
                    }
                }
                assumption = self.create_assumption(scenario_config)
                
                # Project cashflows
                projection = self.project_cashflows(deal, assumption, cache_key=cache_key)
                
                # Process cashflows into a more convenient format
                cashflow_data = self._process_projected_cashflows(projection, horizon_months)
                
                # Create response with results
                response.status = "success"
                response.cashflow_data = cashflow_data
                response.calculation_time = time.time() - start_time
                
                # Cache the result
                if use_cache and cache_key:
                    try:
                        self.redis.set_sync(cache_key, response.dict(), self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return response
            except Exception as e:
                logger.error(f"Error forecasting cashflows: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                response.status = "error"
                response.error = str(e)
                response.calculation_time = time.time() - start_time
                
                return response
    
    async def forecast_cashflows_async(
        self,
        loan_data: Union[List[Dict], List[LoanData]],
        forecast_params: Dict[str, Any] = None,
        cache: bool = True
    ) -> CashflowForecastResponse:
        """
        Forecast cashflows for a collection of loans asynchronously
        
        This is a higher-level convenience method that handles the entire process
        of creating a loan pool, applying assumptions, and forecasting cashflows.
        
        Args:
            loan_data: List of loan data to forecast
            forecast_params: Parameters controlling the forecast
            cache: Whether to use caching
            
        Returns:
            CashflowForecastResponse with forecasted cashflows
        """
        with self.operation("forecast_cashflows_async"):
            start_time = time.time()
            request_id = forecast_params.get("request_id", str(int(time.time())))
            response = CashflowForecastResponse(
                request_id=request_id,
                status="processing"
            )
            
            try:
                # Determine if we should use cache
                use_cache = cache and self.use_cache
                cache_key = None
                
                if use_cache:
                    # Generate deterministic cache key
                    loan_data_json = json.dumps(loan_data, sort_keys=True) if isinstance(loan_data[0], dict) else json.dumps([loan.dict() for loan in loan_data], sort_keys=True)
                    params_json = json.dumps(forecast_params or {}, sort_keys=True)
                    
                    import hashlib
                    hash_obj = hashlib.md5((loan_data_json + params_json).encode())
                    cache_key = f"absbox:forecast:{hash_obj.hexdigest()}"
                    
                    # Check cache
                    try:
                        cached_result = await self.redis.get(cache_key)
                        if cached_result:
                            logger.info(f"Cache hit for forecast request {response.request_id}")
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_HITS.labels(operation="forecast_cashflows").inc()
                            return cached_result
                        else:
                            if METRICS_ENABLED:
                                ABSBOX_CACHE_MISSES.labels(operation="forecast_cashflows").inc()
                    except Exception as e:
                        logger.warning(f"Error checking cache: {e}")
                
                # Extract forecast parameters
                params = forecast_params or {}
                horizon_months = params.get("horizon_months", 120)  # 10 years default
                prepayment_rate = params.get("prepayment_rate", 0.0)
                default_rate = params.get("default_rate", 0.0)
                severity_rate = params.get("severity_rate", 0.0)
                recovery_lag = params.get("recovery_lag", 0)
                interest_rate_shock = params.get("interest_rate_shock", 0.0)
                
                # Run the calculation in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Create a pool with the loans
                pool_config = {
                    "name": params.get("pool_name", "forecast_pool"),
                    "loans": loan_data
                }
                pool = await loop.run_in_executor(
                    None, lambda: self.create_pool(pool_config)
                )
                
                # Create a simple pass-through waterfall
                waterfall_config = {
                    "name": "passthrough",
                    "rules": [
                        {
                            "type": "interest",
                            "tranche": "A",
                            "rate": 0.0
                        },
                        {
                            "type": "principal",
                            "tranche": "A",
                            "percent": 1.0
                        }
                    ]
                }
                waterfall = await loop.run_in_executor(
                    None, lambda: self.create_waterfall(waterfall_config)
                )
                
                # Create a deal
                deal_config = {
                    "name": "forecast_deal",
                    "start_date": params.get("start_date", date.today().isoformat())
                }
                deal = await loop.run_in_executor(
                    None, lambda: self.create_deal(pool, waterfall, deal_config)
                )
                
                # Create assumption
                scenario_config = {
                    "name": "forecast_scenario",
                    "prepayment": {
                        "cpr": prepayment_rate
                    },
                    "default": {
                        "cdr": default_rate,
                        "severity": severity_rate,
                        "lag": recovery_lag
                    },
                    "rates": {
                        "type": "flat",
                        "value": interest_rate_shock,
                        "forward": True
                    }
                }
                assumption = await loop.run_in_executor(
                    None, lambda: self.create_assumption(scenario_config)
                )
                
                # Project cashflows
                projection = await self.project_cashflows_async(deal, assumption, cache_key=cache_key)
                
                # Process cashflows into a more convenient format
                cashflow_data = await loop.run_in_executor(
                    None, lambda: self._process_projected_cashflows(projection, horizon_months)
                )
                
                # Create response with results
                response.status = "success"
                response.cashflow_data = cashflow_data
                response.calculation_time = time.time() - start_time
                
                # Cache the result
                if use_cache and cache_key:
                    try:
                        await self.redis.set(cache_key, response.dict(), self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return response
            except Exception as e:
                logger.error(f"Error forecasting cashflows asynchronously: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                response.status = "error"
                response.error = str(e)
                response.calculation_time = time.time() - start_time
                
                return response
    
    def _process_projected_cashflows(
        self, 
        projection: Dict[str, Any],
        horizon_months: int = 120
    ) -> Dict[str, Any]:
        """
        Process projected cashflows into a more convenient format
        
        Args:
            projection: Cashflow projection data
            horizon_months: Maximum number of months to include
            
        Returns:
            Processed cashflow data
        """
        try:
            # Initialize result
            result = {
                "dates": [],
                "principal": [],
                "interest": [],
                "default": [],
                "recovery": [],
                "prepayment": [],
                "ending_balance": [],
                "metrics": projection.get("metrics", {})
            }
            
            # Extract cashflows by date
            cashflows_by_date = projection.get("cashflows_by_date", {})
            
            # Convert to time series
            dates = sorted(cashflows_by_date.keys())
            
            # Limit to horizon
            dates = dates[:min(len(dates), horizon_months)]
            
            # Extract data
            for date_str in dates:
                cf = cashflows_by_date[date_str]
                result["dates"].append(date_str)
                result["principal"].append(float(cf.get("principal", 0)))
                result["interest"].append(float(cf.get("interest", 0)))
                result["default"].append(float(cf.get("default", 0)))
                result["recovery"].append(float(cf.get("recovery", 0)))
                result["prepayment"].append(float(cf.get("prepayment", 0)))
                result["ending_balance"].append(float(cf.get("ending_balance", 0)))
            
            return result
        except Exception as e:
            logger.error(f"Error processing projected cashflows: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            return {"error": f"Failed to process cashflows: {e}"}
    
    def execute_batch_calculations(
        self,
        batch_requests: List[Dict[str, Any]],
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Execute a batch of calculations efficiently
        
        This method allows running multiple independent calculations in a batch,
        optionally executing them in parallel for higher throughput.
        
        Args:
            batch_requests: List of calculation requests
            parallel: Whether to execute in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping request IDs to calculation results
        """
        with self.operation("execute_batch_calculations"):
            start_time = time.time()
            batch_id = str(int(time.time()))
            
            try:
                results = {}
                
                if parallel:
                    import concurrent.futures
                    
                    # Use ThreadPoolExecutor for parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Map each request to a future
                        futures = {}
                        
                        for request in batch_requests:
                            request_type = request.get("type", "structured_deal")
                            request_id = request.get("request_id", str(int(time.time())))
                            
                            if request_type == "structured_deal":
                                future = executor.submit(
                                    self.calculate_structured_deal,
                                    request
                                )
                            elif request_type == "scenario_analysis":
                                future = executor.submit(
                                    self.run_scenario_analysis,
                                    request
                                )
                            elif request_type == "forecast":
                                future = executor.submit(
                                    self.forecast_cashflows,
                                    request.get("loan_data", []),
                                    request.get("forecast_params", {})
                                )
                            else:
                                results[request_id] = {
                                    "status": "error",
                                    "error": f"Unsupported request type: {request_type}"
                                }
                                continue
                                
                            futures[request_id] = future
                        
                        # Collect results as they complete
                        for request_id, future in futures.items():
                            try:
                                results[request_id] = future.result()
                            except Exception as e:
                                logger.error(f"Error in batch request {request_id}: {e}")
                                results[request_id] = {
                                    "status": "error",
                                    "error": str(e)
                                }
                else:
                    # Execute sequentially
                    for request in batch_requests:
                        request_type = request.get("type", "structured_deal")
                        request_id = request.get("request_id", str(int(time.time())))
                        
                        try:
                            if request_type == "structured_deal":
                                results[request_id] = self.calculate_structured_deal(request)
                            elif request_type == "scenario_analysis":
                                results[request_id] = self.run_scenario_analysis(request)
                            elif request_type == "forecast":
                                results[request_id] = self.forecast_cashflows(
                                    request.get("loan_data", []),
                                    request.get("forecast_params", {})
                                )
                            else:
                                results[request_id] = {
                                    "status": "error",
                                    "error": f"Unsupported request type: {request_type}"
                                }
                        except Exception as e:
                            logger.error(f"Error in batch request {request_id}: {e}")
                            results[request_id] = {
                                "status": "error",
                                "error": str(e)
                            }
                
                # Calculate batch summary
                success_count = sum(1 for result in results.values() if isinstance(result, dict) and result.get("status") == "success")
                error_count = len(results) - success_count
                
                return {
                    "batch_id": batch_id,
                    "total_requests": len(batch_requests),
                    "success_count": success_count,
                    "error_count": error_count,
                    "calculation_time": time.time() - start_time,
                    "results": results
                }
            except Exception as e:
                logger.error(f"Error executing batch calculations: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                return {
                    "batch_id": batch_id,
                    "status": "error",
                    "error": str(e),
                    "calculation_time": time.time() - start_time
                }
    
    async def execute_batch_calculations_async(
        self,
        batch_requests: List[Dict[str, Any]],
        max_concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        Execute a batch of calculations asynchronously
        
        This method runs multiple calculations concurrently using asyncio
        for high throughput and efficient resource utilization.
        
        Args:
            batch_requests: List of calculation requests
            max_concurrency: Maximum number of concurrent tasks
            
        Returns:
            Dictionary mapping request IDs to calculation results
        """
        with self.operation("execute_batch_calculations_async"):
            start_time = time.time()
            batch_id = str(int(time.time()))
            
            try:
                results = {}
                
                # Create a semaphore to limit concurrency
                semaphore = asyncio.Semaphore(max_concurrency)
                
                async def process_request(request):
                    async with semaphore:
                        request_type = request.get("type", "structured_deal")
                        request_id = request.get("request_id", str(int(time.time())))
                        
                        try:
                            if request_type == "structured_deal":
                                return request_id, await self.calculate_structured_deal_async(request)
                            elif request_type == "scenario_analysis":
                                return request_id, await self.run_scenario_analysis_async(request)
                            elif request_type == "forecast":
                                return request_id, await self.forecast_cashflows_async(
                                    request.get("loan_data", []),
                                    request.get("forecast_params", {})
                                )
                            else:
                                return request_id, {
                                    "status": "error",
                                    "error": f"Unsupported request type: {request_type}"
                                }
                        except Exception as e:
                            logger.error(f"Error in batch request {request_id}: {e}")
                            return request_id, {
                                "status": "error",
                                "error": str(e)
                            }
                
                # Create tasks for all requests
                tasks = [process_request(request) for request in batch_requests]
                
                # Execute all tasks and gather results
                for coro in asyncio.as_completed(tasks):
                    request_id, result = await coro
                    results[request_id] = result
                
                # Calculate batch summary
                success_count = sum(1 for result in results.values() if isinstance(result, dict) and result.get("status") == "success")
                error_count = len(results) - success_count
                
                return {
                    "batch_id": batch_id,
                    "total_requests": len(batch_requests),
                    "success_count": success_count,
                    "error_count": error_count,
                    "calculation_time": time.time() - start_time,
                    "results": results
                }
            except Exception as e:
                logger.error(f"Error executing batch calculations asynchronously: {e}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                return {
                    "batch_id": batch_id,
                    "status": "error",
                    "error": str(e),
                    "calculation_time": time.time() - start_time
                }
