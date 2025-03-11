"""
Enhanced AbsBox Service

This module provides an enhanced service for interfacing with the AbsBox library,
adding robust error handling, detailed logging, and performance metrics.
"""

import os
import time
import json
import logging
import traceback
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import contextlib
import functools

# Setup logging
from pythonjsonlogger import jsonlogger
logger = logging.getLogger("absbox_service")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
    rename_fields={"asctime": "timestamp"}
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Optional monitoring imports
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.warning("Prometheus client not installed, metrics collection disabled")

# Optional Sentry import
try:
    import sentry_sdk
    SENTRY_ENABLED = True
except ImportError:
    SENTRY_ENABLED = False
    logger.warning("Sentry SDK not installed, error tracking disabled")

# Import models and Redis client
try:
    from app.models.structured_products import (
        StructuredDealRequest, 
        StructuredDealResponse,
        LoanPoolConfig,
        WaterfallConfig,
        ScenarioConfig,
        LoanConfig,
        BondConfig,
        AccountConfig,
        WaterfallAction,
        DefaultCurveConfig,
        PrepaymentCurveConfig,
        RateCurveConfig
    )
    REDIS_IMPORTS_OK = False
    try:
        import redis
        # Import Redis client from services directory (confirmed location)
        from app.services.redis_client import RedisClient, RedisConfig
        REDIS_IMPORTS_OK = True
    except ImportError as e:
        logger.error(f"Failed to import Redis client: {e}")
    MODEL_IMPORTS_OK = True
except ImportError:
    logger.error("Failed to import required models or Redis client")
    MODEL_IMPORTS_OK = False

# Define metrics if enabled
if METRICS_ENABLED:
    REQUEST_COUNTER = Counter(
        'absbox_requests_total', 
        'Total number of AbsBox requests',
        ['method', 'status']
    )
    REQUEST_LATENCY = Histogram(
        'absbox_request_duration_seconds', 
        'AbsBox request duration in seconds',
        ['method']
    )
    CACHE_COUNTER = Counter(
        'absbox_cache_operations_total', 
        'AbsBox cache operations',
        ['operation', 'status']
    )
    ACTIVE_CALCULATIONS = Gauge(
        'absbox_active_calculations', 
        'Number of active AbsBox calculations'
    )
    ERROR_COUNTER = Counter(
        'absbox_errors_total', 
        'AbsBox errors by type',
        ['error_type']
    )

def log_execution_time(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            if METRICS_ENABLED:
                ACTIVE_CALCULATIONS.inc()
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(
                f"{func.__name__} completed",
                extra={
                    "execution_time": execution_time,
                    "method": func.__name__,
                    "status": "success"
                }
            )
            
            if METRICS_ENABLED:
                REQUEST_COUNTER.labels(method=func.__name__, status="success").inc()
                REQUEST_LATENCY.labels(method=func.__name__).observe(execution_time)
            
            # Add execution time to result if it's a dictionary or has an attribute
            if isinstance(result, dict):
                result["execution_time"] = execution_time
            elif hasattr(result, "execution_time"):
                result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                extra={
                    "execution_time": execution_time,
                    "method": func.__name__,
                    "status": "error",
                    "error_type": error_type,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            
            if METRICS_ENABLED:
                REQUEST_COUNTER.labels(method=func.__name__, status="error").inc()
                REQUEST_LATENCY.labels(method=func.__name__).observe(execution_time)
                ERROR_COUNTER.labels(error_type=error_type).inc()
            
            if SENTRY_ENABLED:
                sentry_sdk.capture_exception(e)
            
            raise
        finally:
            if METRICS_ENABLED:
                ACTIVE_CALCULATIONS.dec()
    
    return wrapper

class AbsBoxServiceEnhanced:
    """Enhanced service for interacting with AbsBox and Hastructure"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AbsBox Service
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("absbox_service")
        
        # Set up configuration
        self.use_cache = self.config.get("use_cache", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour default
        
        # Initialize Redis client
        self._initialize_redis_client()
        
        # Initialize AbsBox connection - defer until actually needed
        self._absbox = None
        self._hastructure = None
    
    def _initialize_redis_client(self) -> None:
        """Initialize Redis client for caching if enabled."""
        self._redis_client = None
        
        if not self.use_cache:
            logger.debug("Redis cache disabled")
            return
            
        if not REDIS_IMPORTS_OK:
            logger.warning("Redis imports not available, continuing without cache")
            return

        try:
            logger.debug("Initializing Redis client for caching")
            
            # Create Redis configuration with more resilient settings
            redis_config = RedisConfig(
                url=os.environ.get("REDIS_URL"),
                socket_timeout=10.0,  # More generous timeout
                socket_connect_timeout=10.0,
                retry_on_timeout=True,
                health_check_interval=60,
                max_connections=20,
                decode_responses=False
            )
            
            self._redis_client = RedisClient(config=redis_config)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            logger.warning("Continuing without Redis cache")
            self._redis_client = None
    
    def _check_redis_connection(self) -> bool:
        """Check if Redis connection is available.
        
        Returns:
            True if Redis is available, False otherwise
        """
        if not self._redis_client:
            return False
            
        try:
            return self._redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            return False
    
    @property
    def absbox(self):
        """Lazy-loaded AbsBox client"""
        if self._absbox is None:
            try:
                # Import AbsBox here to prevent startup issues if not installed
                import absbox
                self._absbox = absbox
                logger.info("AbsBox client loaded")
            except ImportError as e:
                logger.error(f"Failed to import AbsBox: {e}")
                raise RuntimeError("AbsBox library not available") from e
        return self._absbox
    
    @property
    def hastructure(self):
        """Lazy-loaded Hastructure engine"""
        if self._hastructure is None:
            try:
                # Only create the engine when needed
                hastructure_engine_url = os.environ.get("HASTRUCTURE_ENGINE_URL", "http://localhost:5000")
                
                # Connect to real Hastructure engine
                logger.info(f"Connecting to Hastructure engine at {hastructure_engine_url}")
                from absbox.apis import HastructureEngine
                self._hastructure = HastructureEngine(hastructure_engine_url)
            except Exception as e:
                logger.error(f"Failed to initialize Hastructure engine: {e}")
                raise RuntimeError("Hastructure engine initialization failed") from e
        
        return self._hastructure
    
    def _get_cache_key(self, deal):
        """Generate a cache key for a deal
        
        Args:
            deal: A structured deal object
            
        Returns:
            A string cache key
        """
        # Define a custom JSON encoder to handle dates
        class DateSafeJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                from datetime import date, datetime
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)
        
        # Convert deal to dict and use its JSON representation as the cache key
        deal_dict = deal.dict() if hasattr(deal, "dict") else deal
        try:
            # Try to create a stable hash from the deal dict
            deal_json = json.dumps(deal_dict, sort_keys=True, cls=DateSafeJSONEncoder)
            return f"absbox:deal:{hash(deal_json)}"
        except Exception as e:
            # Fallback to a simpler but less reliable cache key
            logger.warning(f"Failed to create cache key from deal: {e}")
            return f"absbox:deal:{hash(str(deal_dict))}"
    
    @contextlib.contextmanager
    def _cache_context(self, cache_key: str):
        """Context manager for caching results"""
        cache_hit = False
        result = None
        
        # Try to get from cache
        if self._redis_client and self.use_cache:
            try:
                cached_data = self._redis_client.get(cache_key)
                if cached_data:
                    result = json.loads(cached_data)
                    cache_hit = True
                    logger.info("Cache hit", extra={"cache_key": cache_key})
                    
                    if METRICS_ENABLED:
                        CACHE_COUNTER.labels(operation="get", status="hit").inc()
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")
                if METRICS_ENABLED:
                    CACHE_COUNTER.labels(operation="get", status="error").inc()
        
        try:
            # Yield cache hit status and cached result
            yield cache_hit, result
            
        except Exception:
            # Don't cache errors
            raise
    
    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache"""
        if self._redis_client and self.use_cache:
            try:
                self._redis_client.set(
                    cache_key, 
                    json.dumps(result if isinstance(result, dict) else result.dict()),
                    ex=self.cache_ttl
                )
                logger.info("Stored in cache", extra={"cache_key": cache_key})
                
                if METRICS_ENABLED:
                    CACHE_COUNTER.labels(operation="set", status="success").inc()
            except Exception as e:
                logger.warning(f"Cache storage error: {e}")
                if METRICS_ENABLED:
                    CACHE_COUNTER.labels(operation="set", status="error").inc()
    
    @log_execution_time
    def analyze_deal(self, deal):
        """Analyze a structured deal
        
        Args:
            deal: StructuredDealRequest or dict
            
        Returns:
            StructuredDealResponse with analysis results
        """
        # Get cache key
        cache_key = self._get_cache_key(deal)
        
        # Try to get from cache if enabled
        if self.use_cache and self._redis_client:
            if self._check_redis_connection():
                cached_result = self._redis_client.get(cache_key)
                if cached_result:
                    if METRICS_ENABLED:
                        CACHE_COUNTER.labels(operation="get", status="hit").inc()
                    logger.info(f"Cache hit for deal: {deal.deal_name if hasattr(deal, 'deal_name') else 'Unknown'}")
                    return cached_result
            
                if METRICS_ENABLED:
                    CACHE_COUNTER.labels(operation="get", status="miss").inc()
        
        # Get the analysis from AbsBox
        deal_name = deal.deal_name if hasattr(deal, "deal_name") else "Unknown Deal"
        logger.info(f"Analyzing deal: {deal_name}", extra={
            "deal_name": deal_name,
            "num_loans": len(deal.pool_config.loans) if hasattr(deal, "pool_config") and hasattr(deal.pool_config, "loans") else 0,
            "num_bonds": len(deal.waterfall_config.bonds) if hasattr(deal, "waterfall_config") and hasattr(deal.waterfall_config, "bonds") else 0
        })
        
        # Convert to dict if it's a Pydantic model
        deal_dict = deal.dict() if hasattr(deal, "dict") else deal
        
        # Get engine
        engine = self.hastructure
        
        try:
            # Run the analysis
            start_time = time.time()
            result = engine.analyze_structured_deal(deal_dict)
            execution_time = time.time() - start_time
            
            # Format the result
            response = StructuredDealResponse(
                deal_name=deal_dict.get("deal_name", "Unknown"),
                execution_time=execution_time,
                bond_cashflows=result.get("bond_cashflows", []),
                pool_cashflows=result.get("pool_cashflows", []),
                pool_statistics=result.get("pool_statistics", {}),
                metrics=result.get("metrics", {}),
                status="success"
            )
            
        except Exception as e:
            logger.error(f"AbsBox analysis failed: {e}")
            
            # Create error response
            response = StructuredDealResponse(
                deal_name=deal_dict.get("deal_name", "Unknown"),
                execution_time=0.0,
                bond_cashflows=[],
                pool_cashflows=[],
                pool_statistics={},
                metrics={},
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
        
        # Cache the result if caching is enabled
        if self.use_cache and self._redis_client:
            if self._check_redis_connection():
                try:
                    self._redis_client.set(cache_key, response, ttl=self.cache_ttl)
                    if METRICS_ENABLED:
                        CACHE_COUNTER.labels(operation="set", status="success").inc()
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")
                    if METRICS_ENABLED:
                        CACHE_COUNTER.labels(operation="set", status="error").inc()
        
        return response
    
    @log_execution_time
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the AbsBox service
        
        Returns:
            A dictionary with health status information
        """
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cache": {
                "status": "disabled",
                "type": None
            },
            "engine": {
                "status": "unknown",
                "type": None
            }
        }
        
        # Check cache
        if self._redis_client:
            try:
                ping_result = self._redis_client.ping()
                health_data["cache"] = {
                    "status": "connected" if ping_result else "error",
                    "type": "redis"
                }
            except Exception as e:
                health_data["cache"] = {
                    "status": "error",
                    "type": "redis",
                    "error": str(e)
                }
        
        # Check engine
        try:
            engine = self.hastructure
            engine_type = type(engine).__name__
            
            # Try a simple operation to verify connectivity
            is_mock = engine_type == "MockHastructureEngine"
            
            if is_mock:
                engine_status = "mock"
            else:
                # For real engine, attempt to get status
                try:
                    engine.get_status()
                    engine_status = "connected"
                except:
                    engine_status = "error"
            
            health_data["engine"] = {
                "status": engine_status,
                "type": engine_type
            }
            
        except Exception as e:
            health_data["engine"] = {
                "status": "error",
                "error": str(e)
            }
            health_data["status"] = "degraded"
        
        return health_data
    
    @log_execution_time
    def clear_cache(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """Clear the cache
        
        Args:
            pattern: Optional pattern to match keys (e.g., "absbox:deal:*")
        
        Returns:
            A dictionary with the operation result
        """
        result = {
            "status": "error",
            "keys_deleted": 0,
            "message": "Cache clearing failed"
        }
        
        if not self._redis_client:
            result["message"] = "Cache not enabled"
            return result
        
        try:
            # Use default pattern if none provided
            if pattern is None:
                pattern = "absbox:*"
            
            # Get keys matching pattern
            keys = self._redis_client.keys(pattern)
            
            if not keys:
                result.update({
                    "status": "success",
                    "keys_deleted": 0,
                    "message": f"No keys found matching pattern: {pattern}"
                })
                return result
            
            # Delete keys
            deleted = self._redis_client.delete(*keys)
            
            result.update({
                "status": "success",
                "keys_deleted": deleted,
                "message": f"Successfully deleted {deleted} keys"
            })
            
            if METRICS_ENABLED:
                CACHE_COUNTER.labels(operation="clear", status="success").inc()
            
            logger.info(f"Cache cleared: {deleted} keys deleted", extra={"pattern": pattern})
            
        except Exception as e:
            result.update({
                "message": f"Error clearing cache: {str(e)}"
            })
            
            if METRICS_ENABLED:
                CACHE_COUNTER.labels(operation="clear", status="error").inc()
            
            logger.error(f"Error clearing cache: {e}")
        
        return result
    
    @log_execution_time
    def create_sample_deal(self, complexity: str = "medium") -> Dict[str, Any]:
        """Create a sample deal for testing and development
        
        Args:
            complexity: Complexity level (simple, medium, complex)
            
        Returns:
            A sample structured deal configuration
        """
        logger.info(f"Creating sample {complexity} deal")
        
        if complexity == "simple":
            # Create a simple deal with minimal configuration
            return {
                "deal_name": "Simple Test Deal",
                "pool_config": {
                    "pool_name": "Simple Pool",
                    "pool_type": "mortgage",
                    "loans": [
                        {
                            "balance": 100000,
                            "rate": 0.05,
                            "term": 360,
                            "loan_type": "fixed"
                        }
                    ]
                },
                "waterfall_config": {
                    "bonds": [
                        {
                            "id": "A",
                            "balance": 90000,
                            "rate": 0.045
                        },
                        {
                            "id": "B",
                            "balance": 10000,
                            "rate": 0.07
                        }
                    ]
                }
            }
        elif complexity == "complex":
            # Create a complex deal with detailed configuration
            return {
                "deal_name": "Complex Test Deal",
                "pool_config": {
                    "pool_name": "Complex Pool",
                    "pool_type": "mortgage",
                    "loans": [
                        {
                            "balance": 500000,
                            "rate": 0.045,
                            "term": 360,
                            "loan_type": "fixed",
                            "original_ltv": 0.8,
                            "original_fico": 720
                        },
                        {
                            "balance": 350000,
                            "rate": 0.04,
                            "term": 360,
                            "loan_type": "arm",
                            "original_ltv": 0.75,
                            "original_fico": 760
                        },
                        {
                            "balance": 425000,
                            "rate": 0.0475,
                            "term": 360,
                            "loan_type": "fixed",
                            "original_ltv": 0.85,
                            "original_fico": 700
                        }
                    ]
                },
                "waterfall_config": {
                    "bonds": [
                        {
                            "id": "Class_A",
                            "balance": 1000000,
                            "rate": 0.035
                        },
                        {
                            "id": "Class_B",
                            "balance": 200000,
                            "rate": 0.05
                        },
                        {
                            "id": "Class_C",
                            "balance": 75000,
                            "rate": 0.07
                        }
                    ]
                },
                "scenario_config": {
                    "default_curve": {
                        "vector": [0.001, 0.002, 0.003, 0.003, 0.002]
                    },
                    "prepayment_curve": {
                        "vector": [0.01, 0.02, 0.03, 0.05, 0.07]
                    }
                }
            }
        else:
            # Medium complexity (default)
            return {
                "deal_name": "Medium Test Deal",
                "pool_config": {
                    "pool_name": "Medium Pool",
                    "pool_type": "mortgage",
                    "loans": [
                        {
                            "balance": 200000,
                            "rate": 0.045,
                            "term": 360,
                            "loan_type": "fixed"
                        },
                        {
                            "balance": 150000,
                            "rate": 0.05,
                            "term": 360,
                            "loan_type": "fixed"
                        }
                    ]
                },
                "waterfall_config": {
                    "bonds": [
                        {
                            "id": "A",
                            "balance": 300000,
                            "rate": 0.04
                        },
                        {
                            "id": "B",
                            "balance": 50000,
                            "rate": 0.06
                        }
                    ]
                },
                "scenario_config": {
                    "default_curve": {
                        "vector": [0.001, 0.002, 0.002]
                    },
                    "prepayment_curve": {
                        "vector": [0.01, 0.02, 0.03]
                    }
                }
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for the AbsBox service
        
        Returns:
            Dictionary with metrics data
        """
        metrics = {
            "enabled": METRICS_ENABLED,
            "timestamp": datetime.now().isoformat()
        }
        
        if not METRICS_ENABLED:
            return metrics
        
        # Collect metrics from Prometheus registry
        try:
            registry = prom.REGISTRY
            metrics["data"] = {}
            
            for metric in registry.collect():
                if metric.name.startswith("absbox_"):
                    metric_name = metric.name
                    metrics["data"][metric_name] = {}
                    
                    for sample in metric.samples:
                        sample_name = sample.name
                        sample_labels = tuple(sorted(sample.labels.items()))
                        sample_value = sample.value
                        
                        if sample_labels:
                            label_str = ", ".join(f"{k}={v}" for k, v in sample_labels)
                            key = f"{sample_name}{{{label_str}}}"
                        else:
                            key = sample_name
                        
                        metrics["data"][metric_name][key] = sample_value
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics

    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get a value from Redis cache
        
        Args:
            cache_key: The cache key to retrieve
            
        Returns:
            The cached value or None if not found
        """
        if not self.use_cache or not self._redis_client:
            return None
            
        try:
            cached_data = self._redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data.decode('utf-8'))
            else:
                logger.info(f"Cache miss for key: {cache_key}")
                return None
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
            return None
    
    def save_to_cache(self, cache_key: str, value: Any) -> bool:
        """
        Save a value to Redis cache
        
        Args:
            cache_key: The cache key to save under
            value: The value to save (must be JSON serializable)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.use_cache or not self._redis_client:
            return False
            
        try:
            serialized = json.dumps(value, default=self.default)
            self._redis_client.setex(cache_key, self.cache_ttl, serialized)
            logger.info(f"Saved to cache: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False
    
    def calculate_enhanced_metrics(self, request) -> Dict[str, Any]:
        """
        Calculate enhanced analytics metrics for a loan
        
        This method uses the AbsBox library to calculate advanced financial
        metrics for a loan, including duration, convexity, and spread metrics.
        
        Args:
            request: An EnhancedAnalyticsRequest object
            
        Returns:
            EnhancedAnalyticsResult with calculated metrics
        """
        logger.info(f"Calculating enhanced metrics for loan: {request.loan_id or 'new_loan'}")
        
        # Convert request to AbsBox-compatible format
        loan_config = self._prepare_loan_config(request)
        
        # Calculate basic cashflows
        cashflow_result = self.absbox.generate_cashflows(**loan_config)
        
        # Calculate enhanced metrics
        npv = self.absbox.calculate_npv(cashflow_result, request.discount_rate or request.interest_rate)
        irr = self.absbox.calculate_irr(cashflow_result)
        duration = self.absbox.calculate_duration(cashflow_result, request.interest_rate)
        macaulay_duration = self.absbox.calculate_macaulay_duration(cashflow_result, request.interest_rate)
        convexity = self.absbox.calculate_convexity(cashflow_result, request.interest_rate)
        wal = self.absbox.calculate_weighted_average_life(cashflow_result)
        
        # Calculate spread metrics if market rate is provided
        discount_margin = None
        z_spread = None
        e_spread = None
        if request.market_rate is not None:
            discount_margin = self.absbox.calculate_discount_margin(cashflow_result, request.market_rate)
            z_spread = self.absbox.calculate_z_spread(cashflow_result, request.market_rate)
            e_spread = self.absbox.calculate_option_adjusted_spread(cashflow_result, request.market_rate)
        
        # Calculate debt service coverage if applicable
        debt_service = None
        interest_coverage = None
        if hasattr(request, "net_operating_income") and request.net_operating_income:
            debt_service = self.absbox.calculate_dscr(cashflow_result, request.net_operating_income)
            interest_coverage = self.absbox.calculate_interest_coverage_ratio(cashflow_result, request.net_operating_income)
        
        # Calculate sensitivity metrics (rate shock scenarios)
        sensitivity_metrics = {
            "rate_up_1pct": self.absbox.calculate_price_change(cashflow_result, request.interest_rate + 0.01),
            "rate_down_1pct": self.absbox.calculate_price_change(cashflow_result, request.interest_rate - 0.01)
        }
        
        # Compile result
        result = {
            "npv": npv,
            "irr": irr,
            "yield_value": irr,  # Set yield to IRR for simplicity
            "duration": duration,
            "macaulay_duration": macaulay_duration,
            "convexity": convexity,
            "weighted_average_life": wal,
            "discount_margin": discount_margin,
            "z_spread": z_spread,
            "e_spread": e_spread,
            "debt_service_coverage": debt_service,
            "interest_coverage_ratio": interest_coverage,
            "sensitivity_metrics": sensitivity_metrics
        }
        
        logger.info(f"Completed enhanced metrics calculation for loan: {request.loan_id or 'new_loan'}")
        return result
    
    def calculate_risk_metrics(self, request) -> Dict[str, Any]:
        """
        Calculate risk metrics for a loan
        
        This method calculates risk metrics like VaR, Expected Shortfall,
        and stress test results.
        
        Args:
            request: An EnhancedAnalyticsRequest object
            
        Returns:
            RiskMetrics with calculated risk measures
        """
        logger.info(f"Calculating risk metrics for loan: {request.loan_id or 'new_loan'}")
        
        # Convert request to AbsBox-compatible format
        loan_config = self._prepare_loan_config(request)
        
        # Calculate basic cashflows
        cashflow_result = self.absbox.generate_cashflows(**loan_config)
        
        # Perform Monte Carlo simulation for risk metrics
        simulation_results = self.absbox.run_monte_carlo(
            cashflow_result, 
            num_scenarios=1000,
            rate_volatility=0.2,
            prepay_volatility=0.3,
            default_volatility=0.4
        )
        
        # Calculate risk metrics from simulation results
        var_95 = self.absbox.calculate_var(simulation_results, confidence=0.95)
        var_99 = self.absbox.calculate_var(simulation_results, confidence=0.99)
        expected_shortfall = self.absbox.calculate_expected_shortfall(simulation_results, confidence=0.95)
        stress_loss = self.absbox.calculate_stress_loss(simulation_results)
        volatility = self.absbox.calculate_return_volatility(simulation_results)
        
        # Compile results
        result = {
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "stress_loss": stress_loss,
            "volatility": volatility
        }
        
        logger.info(f"Completed risk metrics calculation for loan: {request.loan_id or 'new_loan'}")
        return result
    
    def calculate_sensitivity(self, request) -> Dict[str, Any]:
        """
        Calculate sensitivity metrics for a loan
        
        This method calculates how loan value changes in response to changes
        in key risk factors like interest rates, prepayment speeds, and default rates.
        
        Args:
            request: An EnhancedAnalyticsRequest object
            
        Returns:
            SensitivityAnalysis with calculated sensitivity measures
        """
        logger.info(f"Calculating sensitivity for loan: {request.loan_id or 'new_loan'}")
        
        # Convert request to AbsBox-compatible format
        loan_config = self._prepare_loan_config(request)
        
        # Calculate basic cashflows
        cashflow_result = self.absbox.generate_cashflows(**loan_config)
        
        # Calculate interest rate sensitivity
        rate_sensitivity = {
            "up_50bps": self.absbox.calculate_price_change(cashflow_result, request.interest_rate + 0.005),
            "up_100bps": self.absbox.calculate_price_change(cashflow_result, request.interest_rate + 0.01),
            "down_50bps": self.absbox.calculate_price_change(cashflow_result, request.interest_rate - 0.005),
            "down_100bps": self.absbox.calculate_price_change(cashflow_result, request.interest_rate - 0.01)
        }
        
        # Calculate prepayment sensitivity
        prepay_rate = request.prepayment_rate or 0.05  # Default 5% if not specified
        prepayment_sensitivity = {
            "up_50pct": self.absbox.calculate_prepay_sensitivity(cashflow_result, prepay_rate * 1.5),
            "up_100pct": self.absbox.calculate_prepay_sensitivity(cashflow_result, prepay_rate * 2.0),
            "down_50pct": self.absbox.calculate_prepay_sensitivity(cashflow_result, prepay_rate * 0.5),
            "down_100pct": self.absbox.calculate_prepay_sensitivity(cashflow_result, 0)
        }
        
        # Calculate default sensitivity
        default_rate = request.default_rate or 0.01  # Default 1% if not specified
        default_sensitivity = {
            "up_50pct": self.absbox.calculate_default_sensitivity(cashflow_result, default_rate * 1.5),
            "up_100pct": self.absbox.calculate_default_sensitivity(cashflow_result, default_rate * 2.0),
            "down_50pct": self.absbox.calculate_default_sensitivity(cashflow_result, default_rate * 0.5),
            "down_100pct": self.absbox.calculate_default_sensitivity(cashflow_result, 0)
        }
        
        # Calculate recovery sensitivity if applicable
        recovery_sensitivity = None
        if request.recovery_rate is not None:
            recovery_rate = request.recovery_rate
            recovery_sensitivity = {
                "up_25pct": self.absbox.calculate_recovery_sensitivity(cashflow_result, min(1.0, recovery_rate * 1.25)),
                "down_25pct": self.absbox.calculate_recovery_sensitivity(cashflow_result, max(0.0, recovery_rate * 0.75))
            }
        
        # Compile results
        result = {
            "rate_sensitivity": rate_sensitivity,
            "prepayment_sensitivity": prepayment_sensitivity,
            "default_sensitivity": default_sensitivity,
            "recovery_sensitivity": recovery_sensitivity
        }
        
        logger.info(f"Completed sensitivity analysis for loan: {request.loan_id or 'new_loan'}")
        return result

    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the AbsBox service
        
        Returns:
            Dictionary with health information
        """
        try:
            # Check AbsBox connection
            absbox_ok = self.absbox is not None
            
            # Check Redis connection if enabled
            redis_ok = False
            if self.use_cache:
                redis_ok = self._check_redis_connection()
            
            # Get memory usage
            import psutil
            memory_info = psutil.Process().memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            
            # Get version info
            version = getattr(self.absbox, "__version__", "unknown")
            
            # Get uptime (if tracked)
            uptime = None
            if hasattr(self, "_start_time"):
                uptime = time.time() - self._start_time
            
            return {
                "status": "ok" if absbox_ok else "error",
                "version": version,
                "absbox_connected": absbox_ok,
                "cache_status": "connected" if redis_ok else "disconnected",
                "memory_usage": f"{memory_usage_mb:.2f} MB",
                "uptime": f"{uptime:.2f} seconds" if uptime else "unknown"
            }
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _prepare_loan_config(self, request) -> Dict[str, Any]:
        """
        Convert an EnhancedAnalyticsRequest to AbsBox-compatible configuration
        
        Args:
            request: An EnhancedAnalyticsRequest object
            
        Returns:
            Dictionary with AbsBox configuration parameters
        """
        # Basic loan parameters
        config = {
            "principal": request.principal,
            "interest_rate": request.interest_rate,
            "term_months": request.term_months,
            "start_date": request.start_date.isoformat()
        }
        
        # Add optional parameters if present
        if request.prepayment_rate is not None:
            config["prepayment_rate"] = request.prepayment_rate
        
        if request.default_rate is not None:
            config["default_rate"] = request.default_rate
            
        if request.recovery_rate is not None:
            config["recovery_rate"] = request.recovery_rate
            
        if request.balloon_payment is not None:
            config["balloon_payment"] = request.balloon_payment
            
        if request.interest_only_periods is not None and request.interest_only_periods > 0:
            config["interest_only_periods"] = request.interest_only_periods
            
        if request.payment_frequency is not None:
            config["payment_frequency"] = request.payment_frequency
        
        return config
    
    def analyze_structured_deal(self, deal) -> Dict[str, Any]:
        """
        Analyze a structured finance deal
        
        This method uses hastructure to analyze a structured deal, calculating
        bond cashflows, performance metrics, and key analysis points.
        
        Args:
            deal: A StructuredDealRequest object
            
        Returns:
            StructuredDealResponse with analysis results
        """
        logger.info(f"Analyzing structured deal: {deal.deal_name}")
        
        try:
            # Generate unique cache key based on deal parameters
            cache_key = f"structured_deal:{hash(json.dumps(deal.dict(), default=self.default))}"
            cached_result = self.get_from_cache(cache_key)
            
            if cached_result:
                logger.info(f"Using cached result for deal: {deal.deal_name}")
                return cached_result
                
            # Use hastructure for structured deal analysis
            result = self.hastructure.analyze_deal(deal.dict())
            
            # Save result to cache
            self.save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing structured deal: {str(e)}")
            raise
