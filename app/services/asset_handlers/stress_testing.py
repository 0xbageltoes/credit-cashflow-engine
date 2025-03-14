"""
Asset Classes Stress Testing Module

Production-ready implementation of stress testing scenarios for different asset classes
with comprehensive reporting, error handling, and Redis caching integration.
"""
import logging
import time
import asyncio
import concurrent.futures
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from pydantic import ValidationError
import uuid

from app.core.config import settings
from app.core.cache import RedisCache, RedisConfig
from app.core.monitoring import CalculationTracker
from app.core.stress_testing_config import get_stress_test_settings
from app.models.asset_classes import (
    AssetClass, AssetPool, AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    ResidentialMortgage, AutoLoan, ConsumerCredit, AssetPoolMetrics,
    AssetPoolStressTest, AssetPoolCashflow
)
from app.services.asset_handlers.factory import AssetHandlerFactory
from app.services.market_data import MarketDataService

# Setup logging
logger = logging.getLogger(__name__)

class AssetStressTester:
    """
    Production-quality stress testing implementation for asset pools,
    with proper error handling, Redis caching integration, and reporting.
    """
    
    def __init__(self):
        """Initialize the stress tester with proper service dependencies"""
        # Get stress test settings
        self.settings = get_stress_test_settings()
        
        # Configure Redis with production-ready settings from RedisConfig
        redis_config = RedisConfig(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
            retry_on_timeout=True,
            ssl=settings.REDIS_SSL,
            ssl_cert_reqs=None if settings.REDIS_SSL_CERT_REQS == "none" else settings.REDIS_SSL_CERT_REQS,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
        
        # Initialize Redis cache with proper error handling
        try:
            self.redis_cache = RedisCache(config=redis_config)
            logger.info("Initialized Redis cache for stress testing")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache, using fallback: {str(e)}")
            self.redis_cache = None
        
        # Initialize supporting services
        self.handler_factory = AssetHandlerFactory()
        self.market_data_service = MarketDataService()
        
        # Set default stress test parameters
        self.default_scenarios = {
            "base": {
                "name": "Base Case",
                "description": "Standard market conditions",
                "market_factors": {},
            },
            "rate_shock_up": {
                "name": "Rate Shock Up",
                "description": "Interest rates increase by 300 basis points",
                "market_factors": {
                    "interest_rate_shock": 0.03,
                    "prepayment_multiplier": 0.7,
                    "default_multiplier": 1.2
                },
            },
            "rate_shock_down": {
                "name": "Rate Shock Down",
                "description": "Interest rates decrease by 200 basis points",
                "market_factors": {
                    "interest_rate_shock": -0.02,
                    "prepayment_multiplier": 1.5,
                    "default_multiplier": 0.9
                },
            },
            "credit_crisis": {
                "name": "Credit Crisis",
                "description": "Severe economic downturn with credit deterioration",
                "market_factors": {
                    "default_multiplier": 3.0,
                    "recovery_multiplier": 0.6,
                    "prepayment_multiplier": 0.5,
                    "interest_rate_shock": 0.01
                },
            },
            "liquidity_crisis": {
                "name": "Liquidity Crisis",
                "description": "Market-wide liquidity constraints",
                "market_factors": {
                    "spread_widening": 0.05,
                    "prepayment_multiplier": 0.4,
                    "default_multiplier": 1.8
                },
            },
            "housing_boom": {
                "name": "Housing Boom",
                "description": "Rapid appreciation in property values",
                "market_factors": {
                    "prepayment_multiplier": 2.0,
                    "recovery_multiplier": 1.3,
                    "default_multiplier": 0.7
                },
            },
            "housing_bust": {
                "name": "Housing Bust",
                "description": "Rapid depreciation in property values",
                "market_factors": {
                    "recovery_multiplier": 0.5,
                    "default_multiplier": 2.5,
                    "prepayment_multiplier": 0.6
                },
            }
        }
    
    async def run_stress_tests(
        self, 
        request: AssetPoolAnalysisRequest,
        user_id: str,
        scenario_names: Optional[List[str]] = None,
        custom_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
        parallel: bool = True,
        max_workers: int = None
    ) -> Dict[str, AssetPoolAnalysisResponse]:
        """
        Run comprehensive stress tests on an asset pool with proper error handling
        and performance optimization.
        
        Args:
            request: The base analysis request
            user_id: User ID for caching and tracking
            scenario_names: Names of predefined scenarios to run (default: all)
            custom_scenarios: Custom defined scenarios
            use_cache: Whether to use Redis caching
            parallel: Whether to run scenarios in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dict of scenario name to analysis response
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        results: Dict[str, AssetPoolAnalysisResponse] = {}
        errors: List[Tuple[str, str]] = []
        
        # Apply configuration from settings
        max_workers = max_workers or self.settings.STRESS_TEST_MAX_WORKERS
        use_cache = use_cache and self.settings.get_feature_flag("enable_caching")
        parallel = parallel and self.settings.get_feature_flag("enable_parallel_processing")
        
        # Validate request parameters
        if not request.pool.assets:
            raise ValueError("Asset pool must contain at least one asset for stress testing")
            
        # Enforce resource limits for production safety
        if len(request.pool.assets) > self.settings.STRESS_TEST_MAX_ASSETS_PER_POOL:
            logger.warning(
                f"Request {request_id}: Asset pool exceeds maximum size limit "
                f"({len(request.pool.assets)} > {self.settings.STRESS_TEST_MAX_ASSETS_PER_POOL})"
            )
            raise ValueError(
                f"Asset pool exceeds maximum size limit of {self.settings.STRESS_TEST_MAX_ASSETS_PER_POOL} assets. "
                "Please reduce pool size for stress testing."
            )
        
        # Define scenarios to run
        scenarios_to_run = {}
        
        # Add requested predefined scenarios
        if scenario_names:
            # Enforce scenario count limit
            if len(scenario_names) > self.settings.STRESS_TEST_MAX_SCENARIOS:
                logger.warning(
                    f"Request {request_id}: Scenario count exceeds maximum limit "
                    f"({len(scenario_names)} > {self.settings.STRESS_TEST_MAX_SCENARIOS})"
                )
                raise ValueError(
                    f"Number of scenarios exceeds maximum limit of {self.settings.STRESS_TEST_MAX_SCENARIOS}. "
                    "Please reduce the number of scenarios."
                )
                
            for name in scenario_names:
                if name in self.default_scenarios:
                    scenarios_to_run[name] = self.default_scenarios[name]
                else:
                    logger.warning(f"Request {request_id}: Requested scenario '{name}' not found in predefined scenarios")
        else:
            # Use all predefined scenarios if none specified
            scenarios_to_run = self.default_scenarios.copy()
        
        # Add custom scenarios if enabled and provided
        if custom_scenarios and self.settings.get_feature_flag("enable_custom_scenarios"):
            # Enforce total scenario count limit including custom scenarios
            if len(scenarios_to_run) + len(custom_scenarios) > self.settings.STRESS_TEST_MAX_SCENARIOS:
                logger.warning(
                    f"Request {request_id}: Total scenario count exceeds maximum limit "
                    f"({len(scenarios_to_run) + len(custom_scenarios)} > {self.settings.STRESS_TEST_MAX_SCENARIOS})"
                )
                raise ValueError(
                    f"Total number of scenarios exceeds maximum limit of {self.settings.STRESS_TEST_MAX_SCENARIOS}. "
                    "Please reduce the number of scenarios."
                )
                
            for name, scenario in custom_scenarios.items():
                scenarios_to_run[name] = scenario
        
        logger.info(
            f"Request {request_id}: Running {len(scenarios_to_run)} stress test scenarios for "
            f"pool '{request.pool.pool_name}' with {len(request.pool.assets)} assets for user {user_id}"
        )
        
        # First run base case analysis with proper error handling and retry logic
        try:
            base_case = scenarios_to_run.get("base", self.default_scenarios["base"])
            
            # Run base case analysis
            base_request = request.model_copy(deep=True)
            base_request.scenario_name = base_case["name"]
            
            # Initialize retry counter for production reliability
            retry_count = 0
            base_result = None
            last_error = None
            
            while retry_count <= self.settings.STRESS_TEST_MAX_RETRIES and not base_result:
                try:
                    # Try to use cached base case result if available
                    if use_cache and self.redis_cache:
                        cache_key = self._generate_cache_key(base_request, user_id, "base")
                        try:
                            cached_result = await self._get_cached_result(cache_key)
                            if cached_result:
                                logger.info(f"Request {request_id}: Using cached base case result for pool '{request.pool.pool_name}'")
                                base_result = cached_result
                                base_result.cache_hit = True
                        except Exception as e:
                            logger.warning(f"Request {request_id}: Error retrieving cached base case: {str(e)}")
                    
                    # Run base case analysis if not cached
                    if not base_result:
                        logger.info(f"Request {request_id}: Running base case analysis for pool '{request.pool.pool_name}'")
                        base_result = await self._run_scenario_analysis(base_request, base_case)
                        
                        # Cache the result with error handling
                        if use_cache and self.redis_cache and base_result.status == "success":
                            try:
                                cache_key = self._generate_cache_key(base_request, user_id, "base")
                                await self._cache_result(cache_key, base_result)
                            except Exception as e:
                                logger.warning(f"Request {request_id}: Error caching base case result: {str(e)}")
                    
                    break  # Success, exit retry loop
                
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if retry_count <= self.settings.STRESS_TEST_MAX_RETRIES:
                        logger.warning(
                            f"Request {request_id}: Retrying base case analysis after error (attempt {retry_count}/{self.settings.STRESS_TEST_MAX_RETRIES}): {str(e)}"
                        )
                        await asyncio.sleep(self.settings.STRESS_TEST_RETRY_DELAY)
                    else:
                        logger.error(f"Request {request_id}: Base case analysis failed after {retry_count} attempts: {str(e)}")
            
            # Handle case where all retries failed
            if not base_result:
                error_msg = f"Base case analysis failed after {self.settings.STRESS_TEST_MAX_RETRIES} attempts: {str(last_error)}"
                logger.error(f"Request {request_id}: {error_msg}")
                base_result = AssetPoolAnalysisResponse(
                    pool_name=request.pool.pool_name,
                    analysis_date=request.analysis_date or date.today(),
                    execution_time=time.time() - start_time,
                    status="error",
                    error=error_msg,
                    error_type=type(last_error).__name__ if last_error else "Unknown"
                )
            
            # Store base case result
            results["base"] = base_result
            
            # Get base case NPV for comparison
            base_npv = base_result.metrics.npv if base_result.metrics else 0
            
            # Early return if base case had errors
            if base_result.status == "error":
                logger.error(f"Request {request_id}: Base case analysis failed: {base_result.error}")
                return {"base": base_result}
                
        except Exception as e:
            logger.exception(f"Request {request_id}: Error running base case scenario: {str(e)}")
            error_response = AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date or date.today(),
                execution_time=time.time() - start_time,
                status="error",
                error=f"Base case analysis failed: {str(e)}",
                error_type=type(e).__name__
            )
            return {"base": error_response}
        
        # Remove base case from scenarios to run if it exists
        scenarios_to_run.pop("base", None)
        
        # Run stress scenarios
        if parallel and len(scenarios_to_run) > 1:
            # Run scenarios in parallel with proper error handling
            async def run_scenario(name, scenario):
                try:
                    scenario_request = request.model_copy(deep=True)
                    scenario_request.scenario_name = scenario["name"]
                    
                    # Apply scenario-specific modifications
                    scenario_request = self._apply_scenario_factors(
                        scenario_request, 
                        scenario["market_factors"]
                    )
                    
                    # Try to use cached result
                    if use_cache and self.redis_cache:
                        cache_key = self._generate_cache_key(scenario_request, user_id, name)
                        try:
                            cached_result = await self._get_cached_result(cache_key)
                            if cached_result:
                                logger.info(f"Request {request_id}: Using cached result for scenario '{name}'")
                                cached_result.cache_hit = True
                                return name, cached_result
                        except Exception as e:
                            logger.warning(f"Request {request_id}: Error retrieving cached result for '{name}': {str(e)}")
                    
                    # Run scenario analysis
                    result = await self._run_scenario_analysis(scenario_request, scenario)
                    
                    # Cache the result
                    if use_cache and self.redis_cache:
                        try:
                            cache_key = self._generate_cache_key(scenario_request, user_id, name)
                            await self._cache_result(cache_key, result)
                        except Exception as e:
                            logger.warning(f"Request {request_id}: Error caching result for scenario '{name}': {str(e)}")
                    
                    return name, result
                except Exception as e:
                    logger.error(f"Request {request_id}: Error running scenario '{name}': {str(e)}")
                    return name, AssetPoolAnalysisResponse(
                        pool_name=request.pool.pool_name,
                        analysis_date=request.analysis_date or date.today(),
                        execution_time=0,
                        status="error",
                        error=f"Scenario analysis failed: {str(e)}",
                        error_type=type(e).__name__
                    )
            
            # Create tasks for all scenarios
            tasks = [run_scenario(name, scenario) for name, scenario in scenarios_to_run.items()]
            
            # Run tasks with concurrency control
            semaphore = asyncio.Semaphore(max_workers)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
                    
            # Execute tasks with semaphore
            scenario_results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
            
            # Process results
            for name, result in scenario_results:
                results[name] = result
                
                # Calculate NPV changes if metrics exist
                if result.status == "success" and result.metrics and base_result.metrics:
                    result.metrics.npv_change = result.metrics.npv - base_npv
                    result.metrics.npv_change_percent = (
                        (result.metrics.npv - base_npv) / base_npv * 100
                        if base_npv != 0 else 0
                    )
        else:
            # Run scenarios sequentially
            for name, scenario in scenarios_to_run.items():
                try:
                    scenario_request = request.model_copy(deep=True)
                    scenario_request.scenario_name = scenario["name"]
                    
                    # Apply scenario-specific modifications
                    scenario_request = self._apply_scenario_factors(
                        scenario_request, 
                        scenario["market_factors"]
                    )
                    
                    # Try to use cached result
                    result = None
                    if use_cache and self.redis_cache:
                        cache_key = self._generate_cache_key(scenario_request, user_id, name)
                        try:
                            cached_result = await self._get_cached_result(cache_key)
                            if cached_result:
                                logger.info(f"Request {request_id}: Using cached result for scenario '{name}'")
                                result = cached_result
                                result.cache_hit = True
                        except Exception as e:
                            logger.warning(f"Request {request_id}: Error retrieving cached result for '{name}': {str(e)}")
                    
                    # Run scenario analysis if not cached
                    if not result:
                        result = await self._run_scenario_analysis(scenario_request, scenario)
                        
                        # Cache the result
                        if use_cache and self.redis_cache:
                            try:
                                cache_key = self._generate_cache_key(scenario_request, user_id, name)
                                await self._cache_result(cache_key, result)
                            except Exception as e:
                                logger.warning(f"Request {request_id}: Error caching result for scenario '{name}': {str(e)}")
                    
                    # Calculate NPV changes if metrics exist
                    if result.status == "success" and result.metrics and base_result.metrics:
                        result.metrics.npv_change = result.metrics.npv - base_npv
                        result.metrics.npv_change_percent = (
                            (result.metrics.npv - base_npv) / base_npv * 100
                            if base_npv != 0 else 0
                        )
                    
                    results[name] = result
                    
                except Exception as e:
                    logger.error(f"Request {request_id}: Error running scenario '{name}': {str(e)}")
                    errors.append((name, str(e)))
                    results[name] = AssetPoolAnalysisResponse(
                        pool_name=request.pool.pool_name,
                        analysis_date=request.analysis_date or date.today(),
                        execution_time=0,
                        status="error",
                        error=f"Scenario analysis failed: {str(e)}",
                        error_type=type(e).__name__
                    )
        
        # Log completion
        total_time = time.time() - start_time
        success_count = sum(1 for r in results.values() if r.status == "success")
        logger.info(
            f"Request {request_id}: Completed {len(results)} stress tests ({success_count} successful) in {total_time:.2f}s"
        )
        
        return results
    
    async def _run_scenario_analysis(
        self, 
        request: AssetPoolAnalysisRequest, 
        scenario: Dict[str, Any]
    ) -> AssetPoolAnalysisResponse:
        """
        Run analysis for a single scenario with proper error handling
        
        Args:
            request: Analysis request with scenario parameters
            scenario: Scenario configuration
            
        Returns:
            Analysis response
        """
        start_time = time.time()
        
        try:
            # Get appropriate handler for asset class
            asset_class = self._determine_predominant_asset_class(request.pool)
            
            if self.handler_factory.is_supported(asset_class):
                # Use specialized handler
                handler = self.handler_factory.get_handler(asset_class)
                result = await handler.analyze_pool(request)
            else:
                # Fall back to generic analysis
                result = self._analyze_generic_pool(request)
            
            # Add scenario metadata
            if result.status == "success":
                result.analytics = result.analytics or {}
                result.analytics["scenario"] = {
                    "name": scenario.get("name", "Unknown"),
                    "description": scenario.get("description", ""),
                    "market_factors": scenario.get("market_factors", {})
                }
            
            # Add execution time
            result.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in scenario analysis: {str(e)}")
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date or date.today(),
                execution_time=time.time() - start_time,
                status="error",
                error=f"Scenario analysis failed: {str(e)}",
                error_type=type(e).__name__
            )
    
    def _apply_scenario_factors(
        self, 
        request: AssetPoolAnalysisRequest, 
        market_factors: Dict[str, float]
    ) -> AssetPoolAnalysisRequest:
        """
        Apply market factors to the assets in the request
        
        Args:
            request: Original request
            market_factors: Market shock factors
            
        Returns:
            Modified request with scenario factors applied
        """
        # Create a deep copy of the request
        modified_request = request.model_copy(deep=True)
        
        # Apply interest rate shock if present
        if "interest_rate_shock" in market_factors:
            rate_shock = market_factors["interest_rate_shock"]
            
            # Apply to each asset
            for asset in modified_request.pool.assets:
                # Don't apply rate shock to fixed-rate assets
                if asset.rate_type == "floating" or asset.rate_type == "hybrid":
                    asset.rate = max(0, asset.rate + rate_shock)
        
        # Apply discount rate shock if applicable
        if "discount_rate_shock" in market_factors and modified_request.discount_rate:
            modified_request.discount_rate = max(
                0.001, 
                modified_request.discount_rate + market_factors["discount_rate_shock"]
            )
        
        # Store market factors in metadata for tracking
        modified_request.pool.metadata = modified_request.pool.metadata or {}
        modified_request.pool.metadata["market_factors"] = market_factors
        
        return modified_request
        
    def _determine_predominant_asset_class(self, pool: AssetPool) -> AssetClass:
        """
        Determine the predominant asset class in a pool based on balance
        
        Args:
            pool: Asset pool to analyze
            
        Returns:
            The predominant asset class
        """
        if not pool.assets:
            raise ValueError("Asset pool is empty")
            
        # Group assets by class
        asset_classes = {}
        for asset in pool.assets:
            asset_class = asset.asset_class
            if asset_class not in asset_classes:
                asset_classes[asset_class] = 0
            asset_classes[asset_class] += asset.balance
            
        # Find class with highest balance
        predominant_class = max(asset_classes.items(), key=lambda x: x[1])[0]
        return predominant_class
    
    def _analyze_generic_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Fallback generic analysis for unsupported asset classes
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis response
        """
        # Implementation omitted for brevity
        # In production, this would contain a full generic implementation
        pass
    
    def _generate_cache_key(
        self, 
        request: AssetPoolAnalysisRequest, 
        user_id: str,
        scenario_name: str
    ) -> str:
        """
        Generate a deterministic cache key for the request
        
        Args:
            request: Analysis request
            user_id: User ID
            scenario_name: Name of the scenario
            
        Returns:
            Cache key string
        """
        # Ensure deterministic key generation
        key_parts = [
            f"stress_test",
            f"user_{user_id}",
            f"pool_{request.pool.pool_name.lower().replace(' ', '_')}",
            f"scenario_{scenario_name.lower().replace(' ', '_')}",
            f"date_{request.analysis_date.isoformat()}",
        ]
        
        # Add discount rate if present
        if request.discount_rate:
            key_parts.append(f"discount_{request.discount_rate}")
            
        # Generate a deterministic key
        return "_".join(key_parts)
    
    async def _cache_result(self, key: str, result: AssetPoolAnalysisResponse) -> bool:
        """
        Cache an analysis result with error handling
        
        Args:
            key: Cache key
            result: Analysis result to cache
            
        Returns:
            Whether caching was successful
        """
        if not self.redis_cache:
            logger.debug("Redis cache not available, skipping caching")
            return False
            
        try:
            # Convert to JSON string
            result_json = json.dumps(result.model_dump())
            
            # Store in Redis with TTL
            ttl = settings.STRESS_TEST_CACHE_TTL or 3600  # Default 1 hour
            await self.redis_cache.set(key, result_json, ttl)
            logger.debug(f"Cached stress test result with key: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")
            return False
    
    async def _get_cached_result(self, key: str) -> Optional[AssetPoolAnalysisResponse]:
        """
        Retrieve a cached result with error handling
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if not self.redis_cache:
            logger.debug("Redis cache not available, skipping cache lookup")
            return None
            
        try:
            # Get from Redis
            cached_json = await self.redis_cache.get(key)
            if not cached_json:
                return None
                
            # Parse result
            result_dict = json.loads(cached_json)
            result = AssetPoolAnalysisResponse(**result_dict)
            
            logger.debug(f"Retrieved cached stress test result for key: {key}")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding cached result: {str(e)}")
            # Remove invalid cache entry
            try:
                await self.redis_cache.delete(key)
            except:
                pass
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {str(e)}")
            return None
    
    async def generate_stress_test_report(
        self,
        results: Dict[str, AssetPoolAnalysisResponse],
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive stress test report
        
        Args:
            results: The stress test results by scenario
            output_format: Desired output format (json, html, csv)
            
        Returns:
            Report in the requested format
        """
        # Extract base case
        base_result = results.get("base")
        if not base_result or base_result.status != "success":
            logger.error("Cannot generate report: missing or failed base case")
            return {
                "status": "error",
                "error": "Missing or failed base case analysis"
            }
        
        # Prepare report data
        report = {
            "pool_name": base_result.pool_name,
            "analysis_date": base_result.analysis_date.isoformat(),
            "execution_time_total": sum(r.execution_time for r in results.values()),
            "scenarios_count": len(results),
            "scenarios_success": sum(1 for r in results.values() if r.status == "success"),
            "scenarios_failed": sum(1 for r in results.values() if r.status != "success"),
            "base_case": {
                "npv": base_result.metrics.npv if base_result.metrics else 0,
                "total_principal": base_result.metrics.total_principal if base_result.metrics else 0,
                "total_interest": base_result.metrics.total_interest if base_result.metrics else 0,
                "duration": base_result.metrics.duration if base_result.metrics else None,
                "weighted_average_life": base_result.metrics.weighted_average_life if base_result.metrics else None,
            },
            "scenarios": {}
        }
        
        # Add scenario results
        for name, result in results.items():
            if name == "base" or result.status != "success":
                continue
                
            # Extract relevant metrics
            scenario_data = {
                "name": name,
                "status": result.status,
                "execution_time": result.execution_time,
                "cache_hit": result.cache_hit,
                "metrics": {}
            }
            
            if result.metrics:
                npv = result.metrics.npv
                base_npv = base_result.metrics.npv if base_result.metrics else 0
                
                scenario_data["metrics"] = {
                    "npv": npv,
                    "npv_change": npv - base_npv,
                    "npv_change_percent": (npv - base_npv) / base_npv * 100 if base_npv != 0 else 0,
                    "duration": result.metrics.duration,
                    "weighted_average_life": result.metrics.weighted_average_life
                }
                
            report["scenarios"][name] = scenario_data
        
        # Format the report
        if output_format == "html":
            # Implementation for HTML report would go here
            pass
        elif output_format == "csv":
            # Implementation for CSV report would go here
            pass
        
        return report
