"""Enhanced service for structured finance analytics using AbsBox with advanced caching and performance optimizations"""
import logging
import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date
import requests
from functools import lru_cache

# Import AbsBox libraries
import absbox as ab

# Setup logging
logger = logging.getLogger(__name__)

# Use the proper import structure based on the installed absbox package
try:
    # Try importing directly from absbox.local if available
    from absbox.local.pool import Pool
    from absbox.local.loan import Loan, FixedRateLoan, FloatingRateLoan
    from absbox.local.deal import Deal
    from absbox.local.engine import LiqEngine
    from absbox.local.waterfall import Waterfall
    from absbox.local.assumption import Assumption, DefaultAssumption
    from absbox.local.rateAssumption import FlatCurve
    from absbox.local.analytics import Analytics
    from absbox.local.cashflow import Cashflow
except ImportError:
    # Otherwise use the main absbox module
    # The absbox module has the required classes and functionality, but with a different structure
    logger.warning("Using alternative absbox import structure")
    # These will need to be imported from the appropriate modules or constructed as needed
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

from app.core.config import settings
from app.core.cache_service import CacheService
# Try importing the monitoring classes we need, with a fallback implementation
try:
    from app.core.monitoring import CALCULATION_TIME, CalculationTracker
except ImportError:
    # Create fallback implementations if the imports fail
    import time
    logger.warning("CalculationTracker not found in monitoring module, using fallback implementation")
    
    # Fallback implementation for the CALCULATION_TIME metric
    class DummySummary:
        def labels(self, operation_type):
            return self
            
        def observe(self, duration):
            pass
    
    # Use a global dummy summary
    CALCULATION_TIME = DummySummary()
    
    # Fallback implementation for the CalculationTracker
    class CalculationTracker:
        """
        Fallback implementation of the CalculationTracker class.
        Provides basic timing functionality without Prometheus integration.
        """
        def __init__(self, operation_type: str, enable_logging: bool = True):
            self.operation_type = operation_type
            self.enable_logging = enable_logging
            self.start_time = None
            self.end_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            if self.enable_logging:
                logger.debug(f"Starting calculation: {self.operation_type}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            
            if self.enable_logging:
                if exc_type:
                    logger.error(f"Calculation '{self.operation_type}' failed after {duration:.3f}s: {exc_val}")
                else:
                    logger.info(f"Calculation '{self.operation_type}' completed in {duration:.3f}s")
                    
            self.duration = duration
            return False
        
        @property
        def elapsed_time(self) -> float:
            if self.start_time is None:
                return 0.0
                
            end = self.end_time if self.end_time is not None else time.time()
            return end - self.start_time

from app.models.structured_products import (
    LoanPoolConfig, WaterfallConfig, ScenarioConfig, StructuredDealRequest,
    CashflowProjection, EnhancedAnalyticsResult, LoanData, CashflowForecastResponse
)
from app.services.unified_redis_service import UnifiedRedisService, RedisConfig


class AbsBoxServiceEnhanced:
    """Enhanced service for structured finance analytics using AbsBox with advanced features"""
    
    def __init__(self, hastructure_url: Optional[str] = None, cache_service: Optional[Union[CacheService, UnifiedRedisService]] = None):
        """
        Initialize the Enhanced AbsBox service with improved caching and performance optimizations
        
        Args:
            hastructure_url: URL endpoint for Hastructure services
            cache_service: Cache service for improved performance (Redis preferred for production)
        """
        self.hastructure_url = hastructure_url or settings.HASTRUCTURE_URL
        
        # Use the provided cache service or initialize a new one
        # In production, this should be a Redis-backed service for better scalability
        if cache_service:
            self.cache = cache_service
        else:
            # Initialize Redis cache for production use
            redis_config = RedisConfig()
            if settings.ENVIRONMENT == "production":
                logger.info("Initializing enhanced AbsBox service with production Redis cache")
                self.cache = UnifiedRedisService(redis_config)
            else:
                logger.info("Initializing enhanced AbsBox service with standard cache")
                self.cache = CacheService()
        
        # Initialize engine with performance optimizations
        self.engine = self._initialize_engine()
        self.analytics = Analytics()
        
        # Additional performance tracking for production monitoring
        self.performance_metrics = {
            "calculations_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_calculation_time_ms": 0
        }
        
        logger.info(f"Enhanced AbsBox service initialized with Hastructure URL: {self.hastructure_url}")
        
    def _initialize_engine(self) -> LiqEngine:
        """
        Initialize the calculation engine with production-optimized settings
        
        Returns:
            Configured LiqEngine instance
        """
        try:
            # Create engine with optimized settings for production use
            engine = LiqEngine()
            
            # Apply performance optimizations based on environment
            if settings.ENVIRONMENT == "production":
                # In production, optimize for throughput and stability
                engine.max_threads = min(os.cpu_count() or 4, 8)  # Limit to avoid resource exhaustion
                engine.precision = "high"  # Use high precision in production
            else:
                # In development, optimize for responsiveness
                engine.max_threads = 2
                engine.precision = "medium"
                
            logger.info(f"AbsBox engine initialized with {engine.max_threads} threads and {engine.precision} precision")
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize AbsBox engine: {e}")
            # Fallback to basic engine configuration
            return LiqEngine()
            
    def create_loan_pool(self, config: LoanPoolConfig) -> Pool:
        """
        Create a loan pool from configuration with enhanced validation and error handling
        
        Args:
            config: Loan pool configuration
            
        Returns:
            Configured Pool instance
        """
        with CalculationTracker() as tracker:
            try:
                # Cache key based on the normalized configuration
                cache_key = f"loan_pool:{hash(json.dumps(config.dict(), sort_keys=True))}"
                
                # Check cache first for better performance
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.performance_metrics["cache_hits"] += 1
                    logger.debug(f"Retrieved loan pool from cache with key: {cache_key}")
                    return cached_result
                
                self.performance_metrics["cache_misses"] += 1
                
                # Create the pool with improved error handling for production
                pool = Pool()
                
                # Add loans to the pool with validation
                for loan_config in config.loans:
                    try:
                        # Additional validation for production readiness
                        if loan_config.interest_rate < 0 or loan_config.principal <= 0:
                            logger.warning(f"Invalid loan parameters: interest_rate={loan_config.interest_rate}, principal={loan_config.principal}")
                            continue
                            
                        # Create the appropriate loan type
                        if loan_config.loan_type == "fixed":
                            loan = FixedRateLoan(
                                principal=loan_config.principal,
                                rate=loan_config.interest_rate,
                                term=loan_config.term,
                                payment_freq=loan_config.payment_frequency
                            )
                        elif loan_config.loan_type == "floating":
                            loan = FloatingRateLoan(
                                principal=loan_config.principal,
                                margin=loan_config.margin,
                                index=loan_config.index,
                                term=loan_config.term,
                                payment_freq=loan_config.payment_frequency
                            )
                        else:
                            logger.warning(f"Unsupported loan type: {loan_config.loan_type}")
                            continue
                            
                        pool.add(loan)
                    except Exception as e:
                        logger.error(f"Error adding loan to pool: {e}")
                        # Continue processing other loans in production for resilience
                        continue
                
                # Cache the result for future use
                self.cache.set(cache_key, pool, ttl=3600)  # 1 hour TTL
                
                return pool
            except Exception as e:
                logger.error(f"Failed to create loan pool: {e}")
                # Return an empty pool in production to allow continued operation
                return Pool()
            finally:
                # Update performance metrics
                self.performance_metrics["calculations_total"] += 1
                self._update_performance_metrics(tracker.elapsed_time)

    def _update_performance_metrics(self, elapsed_time: float) -> None:
        """
        Update internal performance metrics for monitoring
        
        Args:
            elapsed_time: Elapsed time in milliseconds for the operation
        """
        # Update average calculation time with exponential moving average
        prev_avg = self.performance_metrics["avg_calculation_time_ms"]
        count = self.performance_metrics["calculations_total"]
        
        if count > 1:
            # Use an exponential moving average for more stable metrics
            alpha = 0.1  # Smoothing factor
            new_avg = prev_avg * (1 - alpha) + elapsed_time * alpha
        else:
            new_avg = elapsed_time
            
        self.performance_metrics["avg_calculation_time_ms"] = new_avg

    def health_check(self) -> Dict[str, Any]:
        """
        Enhanced health check for the AbsBox service and its dependencies
        
        Returns:
            Health status information with detailed metrics
        """
        status = {
            "service": "absbox_enhanced",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "engine": "unknown",
            "cache": "unknown",
            "hastructure": "unknown",
            "performance": self.performance_metrics,
            "details": {}
        }
        
        # Check engine health
        try:
            # Simple calculation to verify the engine is working
            test_loan = FixedRateLoan(principal=100000, rate=0.05, term=360, payment_freq="monthly")
            result = test_loan.calculate_payment()
            if result > 0:
                status["engine"] = "healthy"
            else:
                status["engine"] = "degraded"
                status["details"]["engine_error"] = "Engine returned invalid result"
        except Exception as e:
            status["engine"] = "unhealthy"
            status["details"]["engine_error"] = str(e)
            status["status"] = "degraded"
            
        # Check cache health
        try:
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            self.cache.set(test_key, test_value, ttl=60)
            retrieved = self.cache.get(test_key)
            
            if retrieved and retrieved.get("test") is True:
                status["cache"] = "healthy"
            else:
                status["cache"] = "degraded"
                status["details"]["cache_error"] = "Cache retrieval failed"
        except Exception as e:
            status["cache"] = "unhealthy"
            status["details"]["cache_error"] = str(e)
            status["status"] = "degraded"
            
        # Check Hastructure API health if configured
        if self.hastructure_url:
            try:
                response = requests.get(f"{self.hastructure_url}/health", timeout=5)
                if response.status_code == 200:
                    status["hastructure"] = "healthy"
                else:
                    status["hastructure"] = "degraded"
                    status["details"]["hastructure_error"] = f"HTTP {response.status_code}"
            except Exception as e:
                status["hastructure"] = "unhealthy"
                status["details"]["hastructure_error"] = str(e)
                status["status"] = "degraded"
        else:
            status["hastructure"] = "not_configured"
            
        # Overall status determination
        if "unhealthy" in [status["engine"], status["cache"], status["hastructure"]]:
            status["status"] = "unhealthy"
        elif "degraded" in [status["engine"], status["cache"], status["hastructure"]]:
            status["status"] = "degraded"
            
        return status